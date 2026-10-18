#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "hb_dnn.h"

using namespace std;
using namespace cv;

struct Detection {
    float conf;
    int class_id;
    cv::Rect bbox;
};

struct template_circle {
    cv::Point3f centor = {0, 0, 0};
    float radius = 100.0;
    std::vector<cv::Point2f> pts;
    float angle;
    float angle_det;
};

struct Detection_pan {
    float conf = 0;
    int class_id = -1;
    cv::Rect2f bbox = {0, 0, 0, 0};
    std::vector<cv::Point2f> pts;
    std::vector<cv::Size2f> sizes;
    std::vector<cv::Point2f> pts_new;
    std::vector<float> radius;
    float angle = 0.0;
    template_circle c;
};

// Per-scale decode info for grid-based detection heads
struct GridScale {
    int stride;
    int grid_h;
    int grid_w;
    int bbox_output_idx;  // index in output tensor array
    int cls_output_idx;   // index in output tensor array
    float* bbox_data;     // dequantized float data
    float* cls_data;
    int bbox_elements;
    int cls_elements;
    float bbox_scale;
    float cls_scale;
    bool bbox_is_int8;
    bool cls_is_int8;
};

class YOLOv11RDK {
public:
    YOLOv11RDK(const std::string& model_path);
    ~YOLOv11RDK();

    void preprocess(cv::Mat& image);
    void infer();
    void postprocess_pan(cv::Mat& image,
                         std::vector<std::vector<Detection_pan>>& outputs);

    void restore_real_xy(cv::Mat& image, std::vector<Detection_pan>& output);
    void draw_pan(cv::Mat& image, std::vector<Detection_pan>& output);
    void draw_pan(cv::Mat& image, Detection_pan& detection);

    int resizeKeepAspectRatio = 1;
    float m_scale_h = 1;
    float m_scale_w = 1;
    float m_margin_top = 0;
    float m_margin_left = 0;

    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes;
    std::string m_model_name;

private:
    // Grid-based decode (multi-output model: bbox_P3/P4/P5 + cls_P3/P4/P5)
    void decode_grid_postprocess(
        cv::Mat& image,
        std::vector<std::vector<Detection_pan>>& outputs);

    // Legacy flat decode (single output: [1, 7, N])
    void decode_postprocess_feilun(cv::Mat& image, const float* output_data,
                                   std::vector<std::vector<Detection_pan>>& outputs);

    hbPackedDNNHandle_t packed_handle_;
    hbDNNHandle_t dnn_handle_;

    // Multi-output support
    int num_outputs_;
    std::vector<hbDNNTensor> output_tensors_;
    std::vector<hbSysMem> output_mems_;
    std::vector<GridScale> grid_scales_;
    bool use_grid_decode_;

    // Legacy single-output
    hbSysMem input_mem_;
    hbDNNTensor input_tensor_;
    float* output_float_;
    float output_scale_;
    bool output_is_int8_;
    int output_elements_;

    float conf_threshold = 0.20f;
    float nms_threshold = 0.65f;

    // Helper
    static inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    bool decode_raw_ = true;  // no sigmoid on bbox (dist2bbox raw)
    bool no_letterbox_ = false;  // straight resize, no padding
};
