/**
 * @file angle_encoder.cpp
 * @brief AngleEncoder 类实现，包含 TensorRT 推理和基于 FFT 互相关的角度计算。
 */

#include "angle_encoder.h"
#include "cuda_utils.h"
#include "macros.h"
#include "common.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// 辅助函数：提取不带扩展名的文件名
// ---------------------------------------------------------------------------
static std::string getFilenameWithoutExtension(const std::string& path)
{
    try {
        fs::path p(path);
        std::string filename = p.filename().string();
        size_t dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos) return filename;
        return filename.substr(0, dotPos);
    } catch (...) {
        return "";
    }
}

// ---------------------------------------------------------------------------
// 构造函数 / 析构函数
// ---------------------------------------------------------------------------

AngleEncoder::AngleEncoder(string model_path, nvinfer1::ILogger& logger)
{
    // 判断：加载序列化引擎还是从 ONNX 构建
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, logger);
    }
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
    }

    // 从引擎读取输入尺寸
#if NV_TENSORRT_MAJOR < 10
    auto input_dims = engine_->getBindingDimensions(0);
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];
#else
    auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(0));
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];
#endif

    // 从引擎读取输出尺寸  [1, C, 1, W]
#if NV_TENSORRT_MAJOR < 10
    auto output_dims = engine_->getBindingDimensions(1);
    feature_channels_ = output_dims.d[1];
    feature_width_ = output_dims.d[3];
    output_size_ = feature_channels_ * feature_width_;
#else
    auto output_dims = engine_->getTensorShape(engine_->getIOTensorName(1));
    feature_channels_ = output_dims.d[1];
    feature_width_ = output_dims.d[3];
    output_size_ = feature_channels_ * feature_width_;
#endif

    // 分配 GPU 缓冲区：[0]=输入图像（3×H×W 个 float），[1]=输出特征
    CUDA_CHECK(cudaMalloc((void**)&gpu_buffers_[0], 3 * input_w_ * input_h_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&gpu_buffers_[1], output_size_ * sizeof(float)));

    // 分配 CPU 缓冲区
    cpu_output_buffer_ = new float[output_size_];
    cpu_input_buffer_.resize(3 * input_w_ * input_h_);

    // 创建 CUDA 流
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // 设置 TensorRT 10+ 的张量地址
#if NV_TENSORRT_MAJOR >= 10
    context_->setTensorAddress(engine_->getIOTensorName(0), gpu_buffers_[0]);
    context_->setTensorAddress(engine_->getIOTensorName(1), gpu_buffers_[1]);
#endif

    // 预热
    for (int i = 0; i < 3; i++) {
        this->infer();
    }
    std::cout << "AngleEncoder 预热完成" << std::endl;
}

AngleEncoder::~AngleEncoder()
{
    if (stream_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaStreamDestroy(stream_));
        stream_ = nullptr;
    }

    for (int i = 0; i < 2; i++) {
        if (gpu_buffers_[i]) {
            CUDA_CHECK(cudaFree(gpu_buffers_[i]));
            gpu_buffers_[i] = nullptr;
        }
    }

    delete[] cpu_output_buffer_;
    cpu_output_buffer_ = nullptr;

    delete context_;
    delete engine_;
    delete runtime_;

    context_ = nullptr;
    engine_ = nullptr;
    runtime_ = nullptr;
}

// ---------------------------------------------------------------------------
// 引擎初始化（加载序列化引擎）
// ---------------------------------------------------------------------------

void AngleEncoder::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    std::ifstream engineStream(engine_path, std::ios::binary);
    if (!engineStream.is_open()) {
        std::cerr << "错误：无法打开引擎文件：" << engine_path << std::endl;
        return;
    }

    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);

    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    runtime_ = createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(engineData.get(), modelSize);
    if (!engine_) {
        std::cerr << "错误：CUDA 引擎反序列化失败" << std::endl;
        return;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "错误：创建执行上下文失败" << std::endl;
        return;
    }

    std::cout << "AngleEncoder 已加载引擎：" << engine_path << std::endl;
}

// ---------------------------------------------------------------------------
// 从 ONNX 构建引擎
// ---------------------------------------------------------------------------

void AngleEncoder::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(),
                                         static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) {
        std::cerr << "错误：ONNX 模型解析失败：" << onnxPath << std::endl;
        delete parser;
        delete config;
        delete network;
        delete builder;
        return;
    }

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel) {
        std::cerr << "错误：从 ONNX 构建序列化网络失败" << std::endl;
        delete parser;
        delete config;
        delete network;
        delete builder;
        return;
    }

    runtime_ = createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
    context_ = engine_->createExecutionContext();

    delete serializedModel;
    delete parser;
    delete config;
    delete network;
    delete builder;

    std::cout << "AngleEncoder 从 ONNX 构建引擎完成：" << onnxPath << std::endl;
}

bool AngleEncoder::saveEngine(const std::string& filename)
{
    if (!engine_) return false;

    IHostMemory* serializedModel = engine_->serialize();
    if (!serializedModel) return false;

    std::string outName = filename;
    size_t dotPos = outName.find_last_of('.');
    if (dotPos != std::string::npos) {
        outName = outName.substr(0, dotPos);
    }
    outName += ".engine";

    std::ofstream outFile(outName, std::ios::binary);
    if (!outFile.is_open()) {
        delete serializedModel;
        return false;
    }

    outFile.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
    outFile.close();
    delete serializedModel;

    std::cout << "AngleEncoder 引擎已保存至：" << outName << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// 预处理：缩放、BGR 转 RGB、归一化、CHW 重排、拷贝至 GPU
// ---------------------------------------------------------------------------

void AngleEncoder::preprocess(Mat& image)
{
    // 缩放到模型输入尺寸（宽×高 = 8192×128）
    Mat resized;
    cv::resize(image, resized, cv::Size(input_w_, input_h_));

    // BGR 转 RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 转为 float 并归一化到 [0, 1]
    Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // HWC 重排为 CHW
    // 输入布局：H×W×C (128×8192×3)，目标布局：C×H×W (3×128×8192)
    float* dst = cpu_input_buffer_.data();
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_h_; h++) {
            const float* src_row = float_img.ptr<float>(h);
            for (int w = 0; w < input_w_; w++) {
                dst[c * input_h_ * input_w_ + h * input_w_ + w] = src_row[w * 3 + c];
            }
        }
    }

    // 拷贝到 GPU
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers_[0],
                                cpu_input_buffer_.data(),
                                cpu_input_buffer_.size() * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---------------------------------------------------------------------------
// 推理
// ---------------------------------------------------------------------------

void AngleEncoder::infer()
{
#if NV_TENSORRT_MAJOR < 10
    context_->enqueueV2((void**)gpu_buffers_, stream_, nullptr);
#else
    context_->enqueueV3(stream_);
#endif
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---------------------------------------------------------------------------
// 公开接口：将图像编码为特征向量
// ---------------------------------------------------------------------------

vector<float> AngleEncoder::encode(Mat& image, int r_min, int r_max)
{
    // 如果 r_min 和 r_max 均为 -1，按原逻辑处理（整幅图直接缩放送入推理）
    if (r_min == -1 && r_max == -1)
    {
        preprocess(image);
    }
    else
    {
        // 以图像中心为圆心，提取 [r_min, r_max] 圆环区域并极坐标展开
        int cx = image.cols / 2;
        int cy = image.rows / 2;

        // 参数校验：r_max 不能超过图像内切圆半径
        int max_possible = std::min({cx, cy, image.cols - cx, image.rows - cy});
        if (r_max <= 0 || r_min < 0 || r_min >= r_max) {
            std::cerr << "错误：无效的圆环参数 r_min=" << r_min
                      << " r_max=" << r_max << std::endl;
            return vector<float>();
        }
        if (r_max > max_possible) {
            r_max = max_possible;
        }

        // 极坐标展开：输出列=半径(0..r_max)，行=角度(0..360°)，角度分辨率取 8192
        Mat polar_img;
        cv::warpPolar(image, polar_img,
                      cv::Size(r_max, 8192),              // 宽=半径, 高=角度
                      cv::Point2f((float)cx, (float)cy),  // 展开中心
                      (double)r_max,                       // 最大半径
                      cv::WARP_POLAR_LINEAR | cv::WARP_FILL_OUTLIERS);

        // 转置：宽度变为角度，高度变为半径 → (r_max 行, 8192 列)
        cv::transpose(polar_img, polar_img);

        // 裁剪半径范围 [r_min, r_max)，得到圆环区域 → ((r_max-r_min) 行, 8192 列)
        cv::Rect ring_roi(0, r_min, 8192, r_max - r_min);
        Mat ring = polar_img(ring_roi);

        // 将圆环图像送入预处理管线（缩放至 128×8192 等）
        preprocess(ring);
    }

    infer();

    // 将输出从 GPU 拷贝到 CPU
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_,
                                gpu_buffers_[1],
                                output_size_ * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return vector<float>(cpu_output_buffer_, cpu_output_buffer_ + output_size_);
}

// ---------------------------------------------------------------------------
// 静态方法：通过 FFT 互相关计算角度偏差（等价于 CorrFFTHead）
// ---------------------------------------------------------------------------

float AngleEncoder::compute_angle(const vector<float>& template_feat,
                                   const vector<float>& query_feat,
                                   int bins)
{
    const int channels = static_cast<int>(template_feat.size()) / bins;
    if (channels * bins != (int)template_feat.size() ||
        (int)query_feat.size() != channels * bins) {
        std::cerr << "错误：compute_angle 中特征尺寸不匹配" << std::endl;
        return 0.0f;
    }

    // --- 第 1 步：沿通道维度逐列 L2 归一化 ---
    // 等价于 PyTorch：F.normalize(fx1, dim=1, eps=1e-6)
    // 对每个宽度位置 j，将各通道组成的 64 维向量归一化为单位长度

    auto normalize_channels = [&](const vector<float>& src, vector<float>& dst) {
        vector<float> col_norms(bins, 0.0f);
        for (int c = 0; c < channels; c++) {
            const float* src_c = src.data() + c * bins;
            for (int j = 0; j < bins; j++) {
                col_norms[j] += src_c[j] * src_c[j];
            }
        }
        for (int j = 0; j < bins; j++) {
            col_norms[j] = std::sqrt(col_norms[j]) + 1e-6f;
        }

        dst.resize(src.size());
        for (int c = 0; c < channels; c++) {
            const float* src_c = src.data() + c * bins;
            float* dst_c = dst.data() + c * bins;
            for (int j = 0; j < bins; j++) {
                dst_c[j] = src_c[j] / col_norms[j];
            }
        }
    };

    vector<float> fx_norm, fy_norm;
    normalize_channels(template_feat, fx_norm);
    normalize_channels(query_feat, fy_norm);

    // --- 第 2 步：沿每一行（宽度维度）做 DFT ---
    // 布局：(channels, bins)，每行为一个通道
    cv::Mat fx_mat(channels, bins, CV_32F, fx_norm.data());
    cv::Mat fy_mat(channels, bins, CV_32F, fy_norm.data());

    cv::Mat fx_dft, fy_dft;
    cv::dft(fx_mat, fx_dft, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    cv::dft(fy_mat, fy_dft, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    // fx_dft, fy_dft：(channels, bins)，双通道（实部，虚部）

    // --- 第 3 步：频域互相关 ---
    // prod[j] = sum_over_channels( F_fx[c,j] * conj(F_fy[c,j]) )
    // 然后将 DC 分量置零

    cv::Mat prod_dft(1, bins, CV_32FC2, cv::Scalar(0, 0));
    float* prod_ptr = prod_dft.ptr<float>(0);

    for (int c = 0; c < channels; c++) {
        const float* fx_row = fx_dft.ptr<float>(c);
        const float* fy_row = fy_dft.ptr<float>(c);
        for (int j = 0; j < bins; j++) {
            float fx_real = fx_row[2 * j];
            float fx_imag = fx_row[2 * j + 1];
            float fy_real = fy_row[2 * j];
            float fy_imag = fy_row[2 * j + 1];
            // conj(F_fy)：(fy_real, -fy_imag)
            prod_ptr[2 * j]     += fx_real * fy_real + fx_imag * fy_imag;
            prod_ptr[2 * j + 1] += fx_imag * fy_real - fx_real * fy_imag;
        }
    }

    // DC 分量（索引 0）置零
    prod_ptr[0] = 0.0f;
    prod_ptr[1] = 0.0f;

    // --- 第 4 步：逆 DFT 得到互相关 logits ---
    cv::Mat corr_logits;
    cv::dft(prod_dft, corr_logits, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    // corr_logits：(1, bins)，实数值

    // --- 第 5 步：三点抛物线插值求亚像素峰值 ---
    const float* logits = corr_logits.ptr<float>(0);

    int peak_idx = 0;
    float peak_val = logits[0];
    for (int j = 1; j < bins; j++) {
        if (logits[j] > peak_val) {
            peak_val = logits[j];
            peak_idx = j;
        }
    }

    // 三点抛物线插值
    int left_idx = (peak_idx - 1 + bins) % bins;
    int right_idx = (peak_idx + 1) % bins;
    float y_left = logits[left_idx];
    float y_center = logits[peak_idx];
    float y_right = logits[right_idx];

    float denom = 2.0f * (y_left + y_right - 2.0f * y_center);
    float offset = 0.0f;
    if (std::abs(denom) > 1e-10f) {
        offset = (y_left - y_right) / denom;
    }

    float peak_float = static_cast<float>(peak_idx) + offset;
    // 折回到 [0, bins) 范围
    peak_float = std::fmod(peak_float + static_cast<float>(bins), static_cast<float>(bins));

    // --- 第 6 步：将偏移索引转换为角度（度） ---
    float angle_deg = peak_float * 360.0f / static_cast<float>(bins);

    return angle_deg;
}
