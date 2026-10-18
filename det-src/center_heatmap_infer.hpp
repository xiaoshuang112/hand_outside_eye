#pragma once
#include <opencv2/core.hpp>
#include <string>

// Center-only heatmap: OpenCV DNN (.onnx) on desktop; hb_dnn (.bin) on X3 BPU.
// Letterbox + local-centroid decode match center_angle training/export.
struct CenterHeatmapInfer {
    bool load(const std::string& model_path, int image_size = 256, int decode_radius = 4);
    bool ready() const;
    // gray ROI (8U or 32F 0..1/0..255). Returns center in ROI pixel coords.
    bool predict(const cv::Mat& gray_roi, cv::Point2d& center_roi_xy, double* mass = nullptr) const;
    int image_size() const { return image_size_; }
    std::string path() const { return path_; }
    std::string backend() const { return backend_; }
    // Last successful infer wall time in milliseconds (letterbox+infer+decode).
    double last_infer_ms() const { return last_infer_ms_; }
    void close();
    ~CenterHeatmapInfer() { close(); }
    CenterHeatmapInfer() = default;
    CenterHeatmapInfer(const CenterHeatmapInfer&) = delete;
    CenterHeatmapInfer& operator=(const CenterHeatmapInfer&) = delete;

private:
    std::string path_;
    std::string backend_ = "none";
    int image_size_ = 256;
    int decode_radius_ = 4;
    mutable double last_infer_ms_ = 0;
    void* impl_ = nullptr; // opaque backend state
};