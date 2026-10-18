/**
 * @file angle_encoder.h
 * @brief AngleEncoder 类头文件，基于 TensorRT 和 OpenCV。
 *
 * 该类封装了 TensorRT 编码器引擎的加载、极坐标展开图像的特征提取，
 * 以及通过 FFT 互相关计算两幅特征图之间的角度偏差。
 *
 * 管线：
 *   1. 极坐标展开图 (128x8192) -> 编码器 -> 特征 (64x1x8192)
 *   2. 模板特征 + 查询特征 -> FFT 互相关 -> 角度（度）
 */

#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace nvinfer1;
using namespace std;
using namespace cv;

/**
 * @class AngleEncoder
 * @brief 图像编码器类，使用 TensorRT 进行推理，并通过 FFT 互相关计算
 *        两幅编码特征图之间的角度偏差。
 *
 * 用法：
 *   AngleEncoder encoder("models/xike.m", logger);
 *   vector<float> tmpl_feat = encoder.encode(template_image);
 *   vector<float> query_feat = encoder.encode(query_image);
 *   float angle_deg = AngleEncoder::compute_angle(tmpl_feat, query_feat, 8192);
 */
class AngleEncoder
{
public:
    /**
     * @brief 构造函数，初始化 AngleEncoder 对象。
     *
     * 加载 TensorRT 引擎并分配 GPU 缓冲区。
     *
     * @param model_path TensorRT 序列化引擎文件（.m）或 ONNX 文件路径。
     * @param logger TensorRT 日志记录器引用，用于错误报告。
     */
    AngleEncoder(string model_path, nvinfer1::ILogger& logger);

    /**
     * @brief 析构函数，释放资源。
     */
    ~AngleEncoder();

    /**
     * @brief 将图像编码为特征向量。
     *
     * 预处理图像（BGR 转 RGB、归一化、缩放至 128×8192），
     * 执行 TensorRT 推理，返回展平的特征向量。
     *
     * 当 r_min 和 r_max 均为 -1 时，整幅图像直接缩放送入推理。
     * 否则以图像中心为圆心，提取 [r_min, r_max] 圆环区域做极坐标展开，
     * 再缩放至 128×8192 送入推理。
     *
     * @param image 输入图像。
     * @param r_min 圆环内半径（像素），-1 表示不裁剪圆环。
     * @param r_max 圆环外半径（像素），-1 表示不裁剪圆环。
     * @return 特征向量，大小为 feature_channels * feature_width（64 × 8192 = 524288）。
     */
    vector<float> encode(Mat& image, int r_min = -1, int r_max = -1);

    /**
     * @brief 计算模板特征与查询特征之间的角度偏差。
     *
     * 使用基于 FFT 的互相关（CorrFFTHead）：
     *   1. 沿通道维度 L2 归一化
     *   2. 沿宽度维度做 rfft
     *   3. 频域互相关（共轭相乘）
     *   4. irfft 得到互相关 logits
     *   5. 三点抛物线插值求亚像素峰值
     *   6. 将偏移索引转换为角度（度）
     *
     * @param template_feat 零度模板图像经 encode() 得到的展平特征向量。
     * @param query_feat 查询图像经 encode() 得到的展平特征向量。
     * @param bins 角度分箱数（特征图宽度，默认 8192）。
     * @return 角度偏差（度），范围 [0, 360)。
     */
    static float compute_angle(const vector<float>& template_feat,
                               const vector<float>& query_feat,
                               int bins = 8192);

    /**
     * @brief 获取编码器期望的输入图像宽度。
     */
    int get_input_w() const { return input_w_; }

    /**
     * @brief 获取编码器期望的输入图像高度。
     */
    int get_input_h() const { return input_h_; }

    /**
     * @brief 获取编码器输出的特征通道数。
     */
    int get_feature_channels() const { return feature_channels_; }

    /**
     * @brief 获取编码器输出的特征宽度。
     */
    int get_feature_width() const { return feature_width_; }

private:
    /**
     * @brief 从引擎文件初始化 TensorRT 组件。
     */
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    /**
     * @brief 从 ONNX 模型构建 TensorRT 引擎。
     */
    void build(std::string onnxPath, nvinfer1::ILogger& logger);

    /**
     * @brief 将构建好的 TensorRT 引擎保存到文件。
     */
    bool saveEngine(const std::string& filename);

    /**
     * @brief 预处理图像并拷贝到 GPU 缓冲区。
     */
    void preprocess(Mat& image);

    /**
     * @brief 执行 TensorRT 推理。
     */
    void infer();

    // TensorRT 对象
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;

    // GPU 缓冲区：[0]=输入图像，[1]=输出特征
    float* gpu_buffers_[2] = {nullptr, nullptr};
    // CPU 输出缓冲区，用于回读特征
    float* cpu_output_buffer_ = nullptr;

    // 模型输入输出尺寸
    int input_w_ = 8192;
    int input_h_ = 128;
    int feature_channels_ = 64;
    int feature_width_ = 8192;
    int output_size_ = 0;  // feature_channels_ * feature_width_

    // CPU 预处理输入缓冲区
    vector<float> cpu_input_buffer_;
};
