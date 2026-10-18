#include "yolov11_rdk.h"
#include "nv12_utils.h"
#include "common.h"
#include "cryptohash-linux.h"
#include <cstring>
#include <algorithm>
#include <set>
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>

YOLOv11RDK::YOLOv11RDK(const std::string& model_path) {
    use_grid_decode_ = false;
    output_float_ = nullptr;

    // Derive model name from filename (without extension)
    {
        std::filesystem::path p(model_path);
        std::string stem = p.stem().string();
        if (!stem.empty()) {
            m_model_name = stem;
        } else {
            m_model_name = "feilun_rdk";
        }
    }

    // 1. Read model file into memory
    std::ifstream model_stream(model_path, std::ios::binary);
    if (!model_stream.is_open()) {
        throw std::runtime_error("Failed to open model file: " + model_path);
    }
    model_stream.seekg(0, std::ios::end);
    const size_t modelSize = model_stream.tellg();
    model_stream.seekg(0, std::ios::beg);
    std::vector<char> engineData(modelSize);
    model_stream.read(engineData.data(), modelSize);
    model_stream.close();

    // 看看是否加密模型
    int size_bytes = sizeof(unsigned long long);
    unsigned long long data_size = *((unsigned long long*)engineData.data());
    std::vector<char> data_buf;

    const void* model_data;
    int32_t model_data_len;

    if (modelSize == (data_size + size_bytes))
    {
        // 这是加密模型
        std::vector<std::string> mac_list = get_mac_address_list();
        for (auto& mac : mac_list)
        {
            int ret = decode_buf(engineData.data(), modelSize, mac, m_model_name, data_buf);
            if (0 == ret) break;
        }

        if (data_buf.size() == 0)
        {
            std::cerr << "Error read data OR UNAUTHORIZED device" << std::endl;
            std::cerr << "Error read data OR UNAUTHORIZED device" << std::endl;
            std::cerr << "Error read data OR UNAUTHORIZED device" << std::endl;
            throw std::runtime_error("Error read data OR UNAUTHORIZED device: " + model_path);
        }

        model_data = data_buf.data();
        model_data_len = (int32_t)data_buf.size();
        std::cout << "[YOLOv11RDK] encrypted model decoded, model_name=" << m_model_name << std::endl;
    }
    else
    {
        // 普通模型
        model_data = engineData.data();
        model_data_len = (int32_t)modelSize;
        std::cout << "[YOLOv11RDK] normal model, model_name=" << m_model_name << std::endl;
    }

    // Load model from memory via hbDNN API
    const void* model_data_arr[] = { model_data };
    int32_t model_data_lens[] = { model_data_len };
    int ret = hbDNNInitializeFromDDR(&packed_handle_, model_data_arr, model_data_lens, 1);
    if (ret != 0) {
        std::cerr << "hbDNNInitializeFromDDR failed: " << ret << std::endl;
        throw std::runtime_error("Failed to load model: " + model_path);
    }

    // 2. Get model name
    const char** name_list;
    int name_count;
    hbDNNGetModelNameList(&name_list, &name_count, packed_handle_);
    std::string model_name = (name_count > 0) ? name_list[0] : "feilun";

    // 3. Get model handle
    ret = hbDNNGetModelHandle(&dnn_handle_, packed_handle_, model_name.c_str());
    if (ret != 0) {
        std::cerr << "hbDNNGetModelHandle failed: " << ret << std::endl;
        throw std::runtime_error("Failed to get model handle");
    }

    // 4. Get input properties
    hbDNNTensorProperties input_prop;
    ret = hbDNNGetInputTensorProperties(&input_prop, dnn_handle_, 0);
    if (ret != 0) {
        throw std::runtime_error("Failed to get input properties");
    }

    int input_size = input_prop.alignedByteSize;
    input_h = input_prop.validShape.dimensionSize[2];
    input_w = input_prop.validShape.dimensionSize[3];

    // 5. Get output count
    ret = hbDNNGetOutputCount(&num_outputs_, dnn_handle_);
    if (ret != 0 || num_outputs_ <= 0) {
        throw std::runtime_error("Failed to get output count");
    }

    std::cout << "[YOLOv11RDK] input: " << input_w << "x" << input_h
              << " NV12, num_outputs: " << num_outputs_ << std::endl;

    // 6. Allocate input memory
    ret = hbSysAllocCachedMem(&input_mem_, input_size);
    if (ret != 0) {
        throw std::runtime_error("Failed to allocate input memory");
    }

    input_tensor_.properties = input_prop;
    input_tensor_.sysMem[0] = input_mem_;

    // 7. Determine decode mode from output shape
    // Grid format: output count >= 4 (bbox_P3, cls_P3, bbox_P4, cls_P4, ...)
    // Flat format: output count == 1, shape [1, K, N]
    bool is_grid = (num_outputs_ >= 4);

    if (is_grid) {
        use_grid_decode_ = true;
        output_tensors_.resize(num_outputs_);
        output_mems_.resize(num_outputs_);
        output_elements_ = 0;

        for (int i = 0; i < num_outputs_; i++) {
            hbDNNTensorProperties prop;
            ret = hbDNNGetOutputTensorProperties(&prop, dnn_handle_, i);
            if (ret != 0) {
                throw std::runtime_error("Failed to get output properties for idx " + std::to_string(i));
            }

            int elements = 1;
            for (int d = 0; d < prop.validShape.numDimensions; d++) {
                elements *= prop.validShape.dimensionSize[d];
            }
            int mem_size = prop.alignedByteSize;

            ret = hbSysAllocCachedMem(&output_mems_[i], mem_size);
            if (ret != 0) {
                throw std::runtime_error("Failed to allocate output memory");
            }

            output_tensors_[i].properties = prop;
            output_tensors_[i].sysMem[0] = output_mems_[i];

            float scale = 1.0f;
            bool is_int8 = false;
            if (prop.quantiType == SCALE && prop.scale.scaleData != nullptr) {
                scale = prop.scale.scaleData[0];
                is_int8 = true;
            }

            int h = prop.validShape.dimensionSize[2];
            int w = prop.validShape.dimensionSize[3];

            std::cout << "  output[" << i << "]: shape=[";
            for (int d = 0; d < prop.validShape.numDimensions; d++) {
                if (d > 0) std::cout << "x";
                std::cout << prop.validShape.dimensionSize[d];
            }
            std::cout << "] " << (is_int8 ? "int8" : "fp32")
                      << " scale=" << scale
                      << " H=" << h << " W=" << w << std::endl;

            output_elements_ += elements;
        }

        // 8. Pair bbox/cls outputs into scales
        // Output naming pattern: bbox_P3, cls_P3, bbox_P4, cls_P4, ...
        // We pair by: output[0]=bbox, output[1]=cls for same scale, etc.
        int num_scales = num_outputs_ / 2;
        grid_scales_.resize(num_scales);
        for (int s = 0; s < num_scales; s++) {
            int bi = s * 2;      // bbox index
            int ci = s * 2 + 1;  // cls index

            auto& bprop = output_tensors_[bi].properties;
            auto& cprop = output_tensors_[ci].properties;

            int bh = bprop.validShape.dimensionSize[2];
            int bw = bprop.validShape.dimensionSize[3];
            int ch = cprop.validShape.dimensionSize[2];
            int cw = cprop.validShape.dimensionSize[3];

            // Stride = input_size / feature_map_size
            int stride = input_w / bw;

            grid_scales_[s].stride = stride;
            grid_scales_[s].grid_h = bh;
            grid_scales_[s].grid_w = bw;
            grid_scales_[s].bbox_output_idx = bi;
            grid_scales_[s].cls_output_idx = ci;

            int be = 1;
            for (int d = 0; d < bprop.validShape.numDimensions; d++)
                be *= bprop.validShape.dimensionSize[d];
            int ce = 1;
            for (int d = 0; d < cprop.validShape.numDimensions; d++)
                ce *= cprop.validShape.dimensionSize[d];

            grid_scales_[s].bbox_elements = be;
            grid_scales_[s].cls_elements = ce;

            // Dequant scale
            float bs = 1.0f, cs = 1.0f;
            bool bi8 = false, ci8 = false;
            if (bprop.quantiType == SCALE && bprop.scale.scaleData != nullptr) {
                bs = bprop.scale.scaleData[0]; bi8 = true;
            }
            if (cprop.quantiType == SCALE && cprop.scale.scaleData != nullptr) {
                cs = cprop.scale.scaleData[0]; ci8 = true;
            }
            grid_scales_[s].bbox_scale = bs;
            grid_scales_[s].cls_scale = cs;
            grid_scales_[s].bbox_is_int8 = bi8;
            grid_scales_[s].cls_is_int8 = ci8;

            // Allocate float buffers
            grid_scales_[s].bbox_data = new float[be];
            grid_scales_[s].cls_data = new float[ce];

            std::cout << "  scale[" << s << "]: stride=" << stride
                      << " grid=" << bw << "x" << bh
                      << " bbox_idx=" << bi << " cls_idx=" << ci << std::endl;
        }

        // Store num_classes from cls channel count (first scale)
        num_classes = output_tensors_[1].properties.validShape.dimensionSize[1];
        detection_attribute_size = num_classes + 4;
        num_detections = 0;
        for (auto& gs : grid_scales_) {
            num_detections += gs.grid_h * gs.grid_w;
        }
    } else {
        // Legacy single-output flat format
        use_grid_decode_ = false;
        output_tensors_.resize(1);
        output_mems_.resize(1);

        hbDNNTensorProperties output_prop;
        ret = hbDNNGetOutputTensorProperties(&output_prop, dnn_handle_, 0);
        if (ret != 0) {
            throw std::runtime_error("Failed to get output properties");
        }

        output_elements_ = 1;
        for (int i = 0; i < output_prop.validShape.numDimensions; i++) {
            output_elements_ *= output_prop.validShape.dimensionSize[i];
        }
        detection_attribute_size = output_prop.validShape.dimensionSize[1];
        num_detections = output_prop.validShape.dimensionSize[2];
        num_classes = detection_attribute_size - 4;
        int output_size = output_prop.alignedByteSize;

        output_scale_ = 1.0f;
        output_is_int8_ = false;
        if (output_prop.quantiType == SCALE && output_prop.scale.scaleData != nullptr) {
            output_scale_ = output_prop.scale.scaleData[0];
            output_is_int8_ = true;
        }

        ret = hbSysAllocCachedMem(&output_mems_[0], output_size);
        if (ret != 0) {
            hbSysFreeMem(&input_mem_);
            throw std::runtime_error("Failed to allocate output memory");
        }

        output_tensors_[0].properties = output_prop;
        output_tensors_[0].sysMem[0] = output_mems_[0];

        output_float_ = new float[output_elements_];

        std::cout << "[YOLOv11RDK] flat output: " << detection_attribute_size
                  << "x" << num_detections
                  << (output_is_int8_ ? " int8" : " float32")
                  << ", scale=" << output_scale_ << std::endl;
    }
}

YOLOv11RDK::~YOLOv11RDK() {
    if (use_grid_decode_) {
        for (auto& gs : grid_scales_) {
            if (gs.bbox_data) delete[] gs.bbox_data;
            if (gs.cls_data) delete[] gs.cls_data;
        }
        for (auto& mem : output_mems_) {
            hbSysFreeMem(&mem);
        }
    } else {
        if (output_float_) delete[] output_float_;
        for (auto& mem : output_mems_) {
            hbSysFreeMem(&mem);
        }
    }
    hbSysFreeMem(&input_mem_);
    hbDNNRelease(packed_handle_);
}

void YOLOv11RDK::preprocess(cv::Mat& image) {
    bgr_to_nv12(image, (uint8_t*)input_mem_.virAddr,
                input_w, input_h, resizeKeepAspectRatio > 0,
                m_scale_w, m_scale_h, m_margin_left, m_margin_top);
}

void YOLOv11RDK::infer() {
    hbSysFlushMem(&input_mem_, HB_SYS_MEM_CACHE_CLEAN);

    hbDNNInferCtrlParam ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);

    hbDNNTaskHandle_t task = nullptr;

    // Build output tensor pointer array
    std::vector<hbDNNTensor*> output_ptrs(output_tensors_.size());
    for (size_t i = 0; i < output_tensors_.size(); i++) {
        output_ptrs[i] = &output_tensors_[i];
    }

    int ret = hbDNNInfer(&task, output_ptrs.data(), &input_tensor_,
                         dnn_handle_, &ctrl);
    if (ret != 0) {
        std::cerr << "hbDNNInfer failed: " << ret << std::endl;
        return;
    }

    // Wait for BPU async inference
    if (task) {
        ret = hbDNNWaitTaskDone(task, 5000);
        if (ret != 0) {
            std::cerr << "hbDNNWaitTaskDone failed: " << ret << std::endl;
            hbDNNReleaseTask(task);
            return;
        }
    }

    // Dequantize outputs
    for (size_t i = 0; i < output_tensors_.size(); i++) {
        hbSysFlushMem(&output_mems_[i], HB_SYS_MEM_CACHE_INVALIDATE);

        auto& prop = output_tensors_[i].properties;
        bool is_int8 = (prop.quantiType == SCALE && prop.scale.scaleData != nullptr);
        float scale = is_int8 ? prop.scale.scaleData[0] : 1.0f;

        int elements = 1;
        for (int d = 0; d < prop.validShape.numDimensions; d++) {
            elements *= prop.validShape.dimensionSize[d];
        }

        if (use_grid_decode_) {
            // Find which GridScale owns this output
            float* dst = nullptr;
            for (auto& gs : grid_scales_) {
                if (gs.bbox_output_idx == (int)i) dst = gs.bbox_data;
                if (gs.cls_output_idx == (int)i) dst = gs.cls_data;
            }
            if (!dst) continue;  // should not happen

            if (is_int8) {
                int8_t* src = (int8_t*)output_mems_[i].virAddr;
                for (int j = 0; j < elements; j++) {
                    dst[j] = (float)src[j] * scale;
                }
            } else {
                float* src = (float*)output_mems_[i].virAddr;
                memcpy(dst, src, elements * sizeof(float));
            }
        } else {
            // Legacy flat output
            if (is_int8) {
                int8_t* src = (int8_t*)output_mems_[0].virAddr;
                for (int j = 0; j < elements; j++) {
                    output_float_[j] = (float)src[j] * scale;
                }
            } else {
                float* src = (float*)output_mems_[0].virAddr;
                memcpy(output_float_, src, elements * sizeof(float));
            }
        }
    }

    if (task) {
        hbDNNReleaseTask(task);
    }
}

// =========================================================================
// Main postprocess entry point
// =========================================================================
void YOLOv11RDK::postprocess_pan(cv::Mat& image,
                                  std::vector<std::vector<Detection_pan>>& outputs) {
    if (use_grid_decode_) {
        decode_grid_postprocess(image, outputs);
    } else 
    {
        decode_postprocess_feilun(image, output_float_, outputs);
    }
}

// =========================================================================
// Grid-based decode for multi-scale detection heads
// bbox format: [left, top, right, bottom] distances from anchor point
// Decode: sigmoid(distance) * stride → pixel offset
// =========================================================================
void YOLOv11RDK::decode_grid_postprocess(
    cv::Mat& image,
    std::vector<std::vector<Detection_pan>>& outputs)
{
    const int XIALIAO_CLASS_NUM = 7;  // match common.h CLASS_NAMES
    outputs.clear();
    outputs.resize(XIALIAO_CLASS_NUM);

    // Temporary storage for all detections before NMS
    std::vector<cv::Rect>   all_boxes_int;
    std::vector<cv::Rect2f> all_boxes;
    std::vector<float>      all_confidences;
    std::vector<int>        all_class_ids;

    for (auto& gs : grid_scales_) {
        int stride = gs.stride;
        int H = gs.grid_h;
        int W = gs.grid_w;
        int nc = output_tensors_[gs.cls_output_idx].properties.validShape.dimensionSize[1];
        float* bbox = gs.bbox_data;  // [1, 4, H, W]  NCHW layout
        float* cls  = gs.cls_data;   // [1, nc, H, W]

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                // Anchor point (grid cell center, in pixel coords)
                float anchor_x = (j + 0.5f) * stride;
                float anchor_y = (i + 0.5f) * stride;

                // Read bbox [4, H, W]: channel at [c * H * W + i * W + j]
                int b_idx_0 = 0 * H * W + i * W + j;  // left
                int b_idx_1 = 1 * H * W + i * W + j;  // top
                int b_idx_2 = 2 * H * W + i * W + j;  // right
                int b_idx_3 = 3 * H * W + i * W + j;  // bottom

                float lt = bbox[b_idx_0];
                float rt = bbox[b_idx_1];
                float rb = bbox[b_idx_2];
                float lb = bbox[b_idx_3];

                // Apply decode: raw (no sigmoid) or sigmoid
                if (decode_raw_) {
                    lt = lt * stride;
                    rt = rt * stride;
                    rb = rb * stride;
                    lb = lb * stride;
                } else {
                    lt = sigmoid(lt) * stride;
                    rt = sigmoid(rt) * stride;
                    rb = sigmoid(rb) * stride;
                    lb = sigmoid(lb) * stride;
                }

                // dist2bbox
                float x1 = anchor_x - lt;
                float y1 = anchor_y - rt;
                float x2 = anchor_x + rb;
                float y2 = anchor_y + lb;

                float bw = x2 - x1;
                float bh = y2 - y1;
                if (bw <= 0 || bh <= 0) continue;

                // Read cls scores: [nc, H, W]
                // Class 0 = back (skip), Class 1 = boss, Class 2 = wheel
                float max_score = 0;
                int best_cls = 0;
                for (int c = 1; c < nc; c++) {  // skip class 0 (back)
                    int cls_idx = c * H * W + i * W + j;
                    float score = sigmoid(cls[cls_idx]);
                    if (score > max_score) {
                        max_score = score;
                        best_cls = c;
                    }
                }

                if (max_score > conf_threshold && bw < input_w * 3 && bh < input_h * 3) {
                    cv::Rect2f box;

                    // Remove letterbox padding
                    box.x = x1 - m_margin_left;
                    box.y = y1 - m_margin_top;
                    box.width  = bw;
                    box.height = bh;

                    // Clip to valid range
                    if (box.x < 0) { box.width += box.x; box.x = 0; }
                    if (box.y < 0) { box.height += box.y; box.y = 0; }
                    float max_x = (float)input_w - 2 * m_margin_left;
                    float max_y = (float)input_h - 2 * m_margin_top;
                    if (box.x + box.width  > max_x) box.width  = max_x - box.x;
                    if (box.y + box.height > max_y) box.height = max_y - box.y;
                    if (box.width <= 0 || box.height <= 0) continue;

                    all_boxes.push_back(box);
                    all_boxes_int.push_back(cv::Rect(box));
                    all_confidences.push_back(max_score);
                    all_class_ids.push_back(best_cls);
                }
            }
        }
    }

    // Per-class NMS (avoids inter-class suppression, e.g. boss inside wheel)
    if (!all_boxes.empty()) {
        std::set<int> unique_classes(all_class_ids.begin(), all_class_ids.end());
        for (int cls : unique_classes) {
            std::vector<cv::Rect> cls_boxes;
            std::vector<float>    cls_scores;
            std::vector<int>      cls_indices;

            for (size_t k = 0; k < all_class_ids.size(); k++) {
                if (all_class_ids[k] == cls) {
                    cls_boxes.push_back(all_boxes_int[k]);
                    cls_scores.push_back(all_confidences[k]);
                    cls_indices.push_back((int)k);
                }
            }

            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(cls_boxes, cls_scores, conf_threshold, nms_threshold, nms_result);

            for (size_t i = 0; i < nms_result.size(); i++) {
                int idx = cls_indices[nms_result[i]];
                Detection_pan det;
                det.class_id = all_class_ids[idx];
                det.conf = all_confidences[idx];
                det.bbox = all_boxes[idx];

                if ("luowen" == m_model_name) {
                    // luowen model: use native class IDs directly (no remapping)
                    outputs[det.class_id].push_back(det);
                } else if (det.class_id == WHEEL) {
                    det.class_id = LUNZI;
                    outputs[LUNZI].push_back(det);
                } else if (det.class_id == BOSS) {
                    det.class_id = HOLE;
                    outputs[HOLE].push_back(det);
                }
            }
        }
    }

#ifdef REL_DBG
    cv::Mat img_dbg_nms;
    cv::resize(image, img_dbg_nms, cv::Size(input_w, input_h));
    for (int i = 0; i < outputs.size(); i++)
    {
        for (auto& o: outputs[i])
        {
            draw_pan(img_dbg_nms, o);
        }
        
    }
#endif // REL_DBG


    // Sort by area descending
    for (auto& o : outputs) {
        std::sort(o.begin(), o.end(),
            [](const Detection_pan& a, const Detection_pan& b) {
                return a.bbox.area() > b.bbox.area();
            });
    }
}

// =========================================================================
// Legacy flat decode for single-output models [1, 7, N]
// =========================================================================
void YOLOv11RDK::decode_postprocess_feilun(
    cv::Mat& image,
    const float* output_data,
    std::vector<std::vector<Detection_pan>>& outputs)
{
    int CLASS_NUM = num_classes;
    int XIALIAO_CLASS_NUM = 7;

    outputs.clear();
    outputs.resize(XIALIAO_CLASS_NUM);

    std::vector<std::vector<cv::Rect2f>> list_boxes(CLASS_NUM);
    std::vector<std::vector<cv::Rect>>   list_boxes_int(CLASS_NUM);
    std::vector<std::vector<float>>      list_confidences(CLASS_NUM);

    for (int d = 0; d < num_detections; d++) {
        float cx    = output_data[0 * num_detections + d];
        float cy    = output_data[1 * num_detections + d];
        float ow    = output_data[2 * num_detections + d];
        float oh    = output_data[3 * num_detections + d];
        float boss  = output_data[5 * num_detections + d];
        float wheel = output_data[6 * num_detections + d];

        float score = std::max(boss, wheel);
        int class_id;
        if (wheel >= boss && wheel > conf_threshold)
            class_id = 2;
        else if (boss > wheel && boss > conf_threshold)
            class_id = 1;
        else
            continue;

        if (score > conf_threshold && ow > 0 && oh > 0 && ow < input_w * 3 && oh < input_h * 3) {
            cv::Rect2f box;
            box.x = cx - 0.5f * ow - m_margin_left;
            box.y = cy - 0.5f * oh - m_margin_top;
            box.width  = ow;
            box.height = oh;

            if (box.x < 0) { box.width += box.x; box.x = 0; }
            if (box.y < 0) { box.height += box.y; box.y = 0; }
            if (box.x + box.width  > input_w - 2 * m_margin_left)
                box.width  = input_w - 2 * m_margin_left - box.x;
            if (box.y + box.height > input_h - 2 * m_margin_top)
                box.height = input_h - 2 * m_margin_top - box.y;
            if (box.width <= 0 || box.height <= 0) continue;

            list_boxes[class_id].push_back(box);
            list_boxes_int[class_id].push_back(cv::Rect(box));
            list_confidences[class_id].push_back(score);
        }
    }

    std::vector<cv::Rect>   combined_boxes_int;
    std::vector<cv::Rect2f> combined_boxes;
    std::vector<float>      combined_confidences;
    std::vector<int>        combined_class_ids;

    for (int cls = 1; cls <= 2; cls++) {
        for (size_t i = 0; i < list_boxes[cls].size(); i++) {
            combined_boxes_int.push_back(list_boxes_int[cls][i]);
            combined_boxes.push_back(list_boxes[cls][i]);
            combined_confidences.push_back(list_confidences[cls][i]);
            combined_class_ids.push_back(cls);
        }
    }

    if (!combined_boxes.empty()) {
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(combined_boxes_int, combined_confidences,
                          conf_threshold, nms_threshold, nms_result);

        for (size_t i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            Detection_pan det;
            det.class_id = combined_class_ids[idx];
            det.conf = combined_confidences[idx];
            det.bbox = combined_boxes[idx];

            if ("luowen" == m_model_name) {
                outputs[det.class_id].push_back(det);
            } else if (det.class_id == WHEEL) {
                det.class_id = LUNZI;
                outputs[LUNZI].push_back(det);
            } else if (det.class_id == BOSS) {
                det.class_id = HOLE;
                outputs[HOLE].push_back(det);
            }
        }
    }

#ifdef REL_DBG
    cv::Mat img_dbg_nms;
    cv::resize(image, img_dbg_nms, cv::Size(input_w, input_h));
    for (int i = 0; i < outputs.size(); i++)
    {
        for (auto& o: outputs[i])
        {
            draw_pan(img_dbg_nms, o);
        }
        
    }
#endif // REL_DBG


    for (auto& o : outputs) {
        std::sort(o.begin(), o.end(),
            [](const Detection_pan& a, const Detection_pan& b) {
                return a.bbox.area() > b.bbox.area();
            });
    }
}

// =========================================================================
// restore_real_xy, draw_pan
// =========================================================================
void YOLOv11RDK::restore_real_xy(cv::Mat& image,
                                  std::vector<Detection_pan>& output) {
    const float inv_scale = 1.0f / m_scale_w;

    for (auto& detection : output) {
        auto& box = detection.bbox;
        box.x = box.x * inv_scale;
        box.y = box.y * inv_scale;
        box.width  = box.width  * inv_scale;
        box.height = box.height * inv_scale;

        for (auto& pt : detection.pts) {
            pt.x = pt.x * inv_scale;
            pt.y = pt.y * inv_scale;
        }
        for (auto& s : detection.sizes) {
            s.width  = s.width  * inv_scale;
            s.height = s.height * inv_scale;
        }
    }
}

void YOLOv11RDK::draw_pan(cv::Mat& image, std::vector<Detection_pan>& output) {
    for (auto& det : output) {
        draw_pan(image, det);
    }
}

void YOLOv11RDK::draw_pan(cv::Mat& image, Detection_pan& detection) {
    cv::Scalar color;
    if (detection.class_id < (int)COLORS.size()) {
        color = cv::Scalar(COLORS[detection.class_id][0],
                           COLORS[detection.class_id][1],
                           COLORS[detection.class_id][2]);
    } else {
        color = cv::Scalar(0, 255, 0);
    }

    cv::Rect box(detection.bbox.x, detection.bbox.y,
                 detection.bbox.width, detection.bbox.height);
    cv::rectangle(image, box, color, 2);

    const auto& names = ("luowen" == m_model_name) ? CLASS_NAMES_LUOWEN : CLASS_NAMES;
    std::string label = cv::format("%s %.2f",
        (detection.class_id < (int)names.size()
            ? names[detection.class_id].c_str() : "?"),
        detection.conf);
    cv::putText(image, label, cv::Point(box.x, std::max(box.y - 5, 15)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}
