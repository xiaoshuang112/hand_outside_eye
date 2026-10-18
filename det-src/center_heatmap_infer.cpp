#include "center_heatmap_infer.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>

#if !defined(USE_X3_NPU)
#include <opencv2/dnn.hpp>
#endif

#if defined(USE_X3_NPU)
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#endif

namespace {

struct LetterboxMeta {
    int orig_w = 0, orig_h = 0, size = 0, left = 0, top = 0;
    double scale_x = 1, scale_y = 1;
    cv::Point2d to_original(double x, double y) const {
        return {(x - left) / scale_x, (y - top) / scale_y};
    }
};

cv::Mat to_gray8(const cv::Mat& gray) {
    CV_Assert(!gray.empty());
    if (gray.type() == CV_8UC1) return gray;
    if (gray.type() == CV_32FC1) {
        double minv = 0, maxv = 0;
        cv::minMaxLoc(gray, &minv, &maxv);
        cv::Mat out;
        if (maxv <= 1.5) gray.convertTo(out, CV_8U, 255.0);
        else gray.convertTo(out, CV_8U);
        return out;
    }
    cv::Mat out;
    if (gray.channels() == 3) cv::cvtColor(gray, out, cv::COLOR_BGR2GRAY);
    else gray.convertTo(out, CV_8U);
    return out;
}

cv::Mat letterbox_gray(const cv::Mat& gray, int size, LetterboxMeta& meta) {
    cv::Mat src8 = to_gray8(gray);
    meta.orig_w = src8.cols;
    meta.orig_h = src8.rows;
    meta.size = size;
    if (meta.orig_w <= 0 || meta.orig_h <= 0)
        throw cv::Exception(0, "empty ROI", "letterbox_gray", __FILE__, __LINE__);
    const double scale = std::min(static_cast<double>(size) / meta.orig_w,
                                  static_cast<double>(size) / meta.orig_h);
    const int new_w = std::max(1, static_cast<int>(std::lround(meta.orig_w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(meta.orig_h * scale)));
    meta.left = (size - new_w) / 2;
    meta.top = (size - new_h) / 2;
    meta.scale_x = static_cast<double>(new_w) / meta.orig_w;
    meta.scale_y = static_cast<double>(new_h) / meta.orig_h;
    cv::Mat resized;
    cv::resize(src8, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(size, size, CV_8UC1, cv::Scalar(0));
    resized.copyTo(canvas(cv::Rect(meta.left, meta.top, new_w, new_h)));
    return canvas;
}

bool decode_heatmap_chw(const float* logits, int h, int w, int input_size, int radius,
                        cv::Point2d& center_model, double& mass_out) {
    if (!logits || h < 1 || w < 1) return false;
    // softmax
    double maxv = logits[0];
    for (int i = 1; i < h * w; ++i) maxv = std::max(maxv, static_cast<double>(logits[i]));
    std::vector<double> probs(static_cast<size_t>(h) * w);
    double sum = 0;
    for (int i = 0; i < h * w; ++i) {
        probs[i] = std::exp(static_cast<double>(logits[i]) - maxv);
        sum += probs[i];
    }
    if (!(sum > 0) || !std::isfinite(sum)) return false;
    for (double& p : probs) p /= sum;
    int peak = 0;
    for (int i = 1; i < h * w; ++i) if (probs[i] > probs[peak]) peak = i;
    const int x0 = peak % w, y0 = peak / w;
    const int left = std::max(0, x0 - radius);
    const int right = std::min(w, x0 + radius + 1);
    const int top = std::max(0, y0 - radius);
    const int bottom = std::min(h, y0 + radius + 1);
    mass_out = 0;
    double cx = 0, cy = 0;
    for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) {
            const double p = probs[static_cast<size_t>(y) * w + x];
            mass_out += p;
            cx += p * x;
            cy += p * y;
        }
    }
    if (!(mass_out > 1e-12) || !std::isfinite(mass_out)) return false;
    cx /= mass_out;
    cy /= mass_out;
    center_model.x = cx * (static_cast<double>(input_size) / w);
    center_model.y = cy * (static_cast<double>(input_size) / h);
    return std::isfinite(center_model.x) && std::isfinite(center_model.y);
}

#if !defined(USE_X3_NPU)
struct CpuOnnxBackend {
    cv::dnn::Net net;
    bool load(const std::string& path) {
        try { net = cv::dnn::readNetFromONNX(path); }
        catch (...) { return false; }
        if (net.empty()) return false;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return true;
    }
    bool infer(const cv::Mat& canvas8, int size, std::vector<float>& logits_out, int& hout, int& wout) {
        cv::Mat f32; canvas8.convertTo(f32, CV_32F, 1.0 / 255.0);
        const int sizes[4] = {1, 1, size, size};
        cv::Mat blob(4, sizes, CV_32F);
        std::memcpy(blob.ptr<float>(), f32.ptr<float>(), sizeof(float) * static_cast<size_t>(size) * size);
        net.setInput(blob);
        cv::Mat out = net.forward();
        if (out.dims == 4) { hout = out.size[2]; wout = out.size[3]; }
        else if (out.dims == 3) { hout = out.size[1]; wout = out.size[2]; }
        else if (out.dims == 2) { hout = out.rows; wout = out.cols; }
        else return false;
        logits_out.resize(static_cast<size_t>(hout) * wout);
        std::memcpy(logits_out.data(), out.ptr<float>(), sizeof(float) * logits_out.size());
        return true;
    }
};
#endif

#if defined(USE_X3_NPU)
struct BpuBackend {
    hbPackedDNNHandle_t packed = nullptr;
    hbDNNHandle_t model = nullptr;
    hbDNNTensor input{};
    hbDNNTensor output{};
    bool input_nhwc = false;
    bool input_f32 = true;
    bool output_f32 = true;
    float input_scale = 1.f;
    float output_scale = 1.f;
    int in_h = 256, in_w = 256;
    int out_h = 128, out_w = 128;

    static void check(int code, const char* op) {
        if (code) throw std::runtime_error(std::string(op) + " failed: " + std::to_string(code));
    }
    ~BpuBackend() { close(); }
    void close() {
        if (input.sysMem[0].virAddr) { hbSysFreeMem(&input.sysMem[0]); input.sysMem[0].virAddr = nullptr; }
        if (output.sysMem[0].virAddr) { hbSysFreeMem(&output.sysMem[0]); output.sysMem[0].virAddr = nullptr; }
        if (packed) { hbDNNRelease(packed); packed = nullptr; model = nullptr; }
    }
    bool load(const std::string& path) {
        try {
            const char* file = path.c_str();
            check(hbDNNInitializeFromFiles(&packed, &file, 1), "hbDNNInitializeFromFiles");
            const char** names = nullptr; int count = 0;
            check(hbDNNGetModelNameList(&names, &count, packed), "hbDNNGetModelNameList");
            if (count < 1) return false;
            check(hbDNNGetModelHandle(&model, packed, names[0]), "hbDNNGetModelHandle");
            check(hbDNNGetInputTensorProperties(&input.properties, model, 0), "input props");
            check(hbDNNGetOutputTensorProperties(&output.properties, model, 0), "output props");
            const auto& ip = input.properties;
            const auto& op = output.properties;
            input_nhwc = (ip.tensorLayout == HB_DNN_LAYOUT_NHWC);
            input_f32 = (ip.tensorType == HB_DNN_TENSOR_TYPE_F32);
            output_f32 = (op.tensorType == HB_DNN_TENSOR_TYPE_F32);
            if (ip.validShape.numDimensions != 4 || op.validShape.numDimensions != 4) return false;
            if (input_nhwc) {
                in_h = ip.validShape.dimensionSize[1];
                in_w = ip.validShape.dimensionSize[2];
            } else {
                in_h = ip.validShape.dimensionSize[2];
                in_w = ip.validShape.dimensionSize[3];
            }
            // Prefer NCHW readout for heatmap; if NHWC, still H/W from dims.
            if (op.tensorLayout == HB_DNN_LAYOUT_NHWC) {
                out_h = op.validShape.dimensionSize[1];
                out_w = op.validShape.dimensionSize[2];
            } else {
                out_h = op.validShape.dimensionSize[2];
                out_w = op.validShape.dimensionSize[3];
            }
            if (ip.quantiType == SCALE && ip.scale.scaleLen >= 1) input_scale = ip.scale.scaleData[0];
            if (op.quantiType == SCALE && op.scale.scaleLen >= 1) output_scale = op.scale.scaleData[0];
            check(hbSysAllocCachedMem(&input.sysMem[0], ip.alignedByteSize), "alloc input");
            check(hbSysAllocCachedMem(&output.sysMem[0], op.alignedByteSize), "alloc output");
            std::memset(input.sysMem[0].virAddr, 0, ip.alignedByteSize);
            return true;
        } catch (...) {
            close();
            return false;
        }
    }
    bool infer(const cv::Mat& canvas8, int /*size*/, std::vector<float>& logits_out, int& hout, int& wout) {
        try {
            // Prepare input as float 0..1 matching training / hb_mapper calibration.
            cv::Mat f32; canvas8.convertTo(f32, CV_32F, 1.0 / 255.0);
            if (input_f32) {
                if (input_nhwc) {
                    // 1xHxWx1
                    auto* dst = static_cast<float*>(input.sysMem[0].virAddr);
                    for (int y = 0; y < in_h; ++y) {
                        const float* row = f32.ptr<float>(y);
                        for (int x = 0; x < in_w; ++x) dst[y * in_w + x] = row[x];
                    }
                } else {
                    // 1x1xHxW
                    std::memcpy(input.sysMem[0].virAddr, f32.ptr<float>(),
                                sizeof(float) * static_cast<size_t>(in_h) * in_w);
                }
            } else if (input.properties.tensorType == HB_DNN_TENSOR_TYPE_S8) {
                // quantize with scale: q = round(x / scale)
                auto* dst = static_cast<int8_t*>(input.sysMem[0].virAddr);
                const float inv = (input_scale != 0.f) ? (1.f / input_scale) : 1.f;
                if (input_nhwc) {
                    std::memset(dst, 0, input.properties.alignedByteSize);
                    for (int y = 0; y < in_h; ++y) {
                        const float* row = f32.ptr<float>(y);
                        for (int x = 0; x < in_w; ++x) {
                            int q = static_cast<int>(std::lround(row[x] * inv));
                            q = std::max(-128, std::min(127, q));
                            dst[y * input.properties.alignedShape.dimensionSize[2] * input.properties.alignedShape.dimensionSize[3]
                                + x * input.properties.alignedShape.dimensionSize[3] + 0] = static_cast<int8_t>(q);
                        }
                    }
                } else {
                    for (int i = 0; i < in_h * in_w; ++i) {
                        int q = static_cast<int>(std::lround(f32.ptr<float>()[i] * inv));
                        dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
                    }
                }
            } else {
                return false;
            }
            check(hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN), "input flush");
            hbDNNTaskHandle_t task = nullptr;
            hbDNNTensor* outputs = &output;
            hbDNNInferCtrlParam control; HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
            int code = hbDNNInfer(&task, &outputs, &input, model, &control);
            if (code) { if (task) hbDNNReleaseTask(task); check(code, "hbDNNInfer"); }
            code = hbDNNWaitTaskDone(task, 1000);
            hbDNNReleaseTask(task);
            check(code, "hbDNNWaitTaskDone");
            check(hbSysFlushMem(&output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE), "output flush");

            hout = out_h; wout = out_w;
            logits_out.resize(static_cast<size_t>(hout) * wout);
            if (output_f32) {
                if (output.properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
                    // 1xHxWx1
                    auto* src = static_cast<float*>(output.sysMem[0].virAddr);
                    for (int y = 0; y < hout; ++y)
                        for (int x = 0; x < wout; ++x)
                            logits_out[static_cast<size_t>(y) * wout + x] = src[y * wout + x];
                } else {
                    std::memcpy(logits_out.data(), output.sysMem[0].virAddr,
                                sizeof(float) * logits_out.size());
                }
            } else if (output.properties.tensorType == HB_DNN_TENSOR_TYPE_S32) {
                auto* src = static_cast<int32_t*>(output.sysMem[0].virAddr);
                for (size_t i = 0; i < logits_out.size(); ++i)
                    logits_out[i] = static_cast<float>(src[i]) * output_scale;
            } else if (output.properties.tensorType == HB_DNN_TENSOR_TYPE_S8) {
                auto* src = static_cast<int8_t*>(output.sysMem[0].virAddr);
                for (size_t i = 0; i < logits_out.size(); ++i)
                    logits_out[i] = static_cast<float>(src[i]) * output_scale;
            } else {
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }
};
#endif

} // namespace

bool CenterHeatmapInfer::ready() const { return impl_ != nullptr; }

void CenterHeatmapInfer::close() {
#if defined(USE_X3_NPU)
    if (impl_) { delete static_cast<BpuBackend*>(impl_); impl_ = nullptr; }
#else
    if (impl_) { delete static_cast<CpuOnnxBackend*>(impl_); impl_ = nullptr; }
#endif
    backend_ = "none";
}

bool CenterHeatmapInfer::load(const std::string& model_path, int image_size, int decode_radius) {
    close();
    if (image_size < 32 || (image_size % 16) != 0 || decode_radius < 0) return false;
    image_size_ = image_size;
    decode_radius_ = decode_radius;
    path_ = model_path;
#if defined(USE_X3_NPU)
    auto* b = new BpuBackend();
    if (!b->load(model_path)) { delete b; return false; }
    impl_ = b;
    backend_ = "bpu_hb_dnn";
    return true;
#else
    // Prefer ONNX via OpenCV DNN on desktop / Jetson native builds.
    auto* b = new CpuOnnxBackend();
    if (!b->load(model_path)) { delete b; return false; }
    impl_ = b;
    backend_ = "opencv_dnn_cpu";
    return true;
#endif
}

bool CenterHeatmapInfer::predict(const cv::Mat& gray_roi, cv::Point2d& center_roi_xy, double* mass) const {
    if (!ready() || gray_roi.empty()) return false;
    try {
        auto t0 = std::chrono::steady_clock::now();
        LetterboxMeta meta;
        cv::Mat canvas = letterbox_gray(gray_roi, image_size_, meta);
        std::vector<float> logits;
        int hout = 0, wout = 0;
#if defined(USE_X3_NPU)
        auto* b = static_cast<BpuBackend*>(impl_);
        if (!b->infer(canvas, image_size_, logits, hout, wout)) return false;
#else
        auto* b = static_cast<CpuOnnxBackend*>(impl_);
        if (!b->infer(canvas, image_size_, logits, hout, wout)) return false;
#endif
        cv::Point2d center_model;
        double mass_val = 0;
        if (!decode_heatmap_chw(logits.data(), hout, wout, image_size_, decode_radius_, center_model, mass_val))
            return false;
        center_roi_xy = meta.to_original(center_model.x, center_model.y);
        if (mass) *mass = mass_val;
        last_infer_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return std::isfinite(center_roi_xy.x) && std::isfinite(center_roi_xy.y);
    } catch (...) {
        return false;
    }
}