#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#else

#endif

#include "yolov11.h"             // Header file for YOLOv11 class
#include "logging.h"             // Logging utilities
#include "cuda_utils.h"          // CUDA utility functions
#include "macros.h"              // Common macros
#include "preprocess.h"          // Preprocessing functions
#include <NvOnnxParser.h>        // NVIDIA ONNX parser for TensorRT
#include "common.h"              // Common definitions and utilities
#include <fstream>               // File stream operations
#include <iostream>              // Input/output stream operations
// #include "Log.h"
#include <filesystem>

#include "cryptohash-linux.h"

namespace fs = std::filesystem;

// Initialize a static logger instance
static Logger logger;
extern int g_save_ret;

// Define whether to use FP16 precision
#define isFP16 true

// Define whether to perform model warmup
#define warmup true


extern "C"  int  ReadMergeMem(char* merge_buf, int merge_len, int idx, char** data, unsigned long long* data_size);

// Constructor for the YOLOv11 class
YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    // Check if the model path does not contain ".onnx"
    if (model_path.find(".onnx") == std::string::npos)
    {
        // Initialize the engine from a serialized engine file
        init(model_path, logger);
    }
    else
    {
        // Build the engine from an ONNX model
        build(model_path, logger);
        // Save the built engine to a file
        saveEngine(model_path);
    }

    // Handle input dimensions based on TensorRT version
#if NV_TENSORRT_MAJOR < 10
    // For TensorRT versions less than 10, get binding dimensions directly
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    // For TensorRT versions 10 and above, use getTensorShape
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}



std::string getFilenameWithoutExtension(const std::string& engine_path) {
    try {
        // 将路径转换为filesystem::path对象
        fs::path path(engine_path);

        // 获取文件名（带扩展名）
        std::string filename = path.filename().string();

        // 查找最后一个点号
        size_t dotPos = filename.find_last_of('.');

        // 如果没有点号，直接返回文件名
        if (dotPos == std::string::npos) {
            return filename;
        }

        // 返回点号之前的部分
        return filename.substr(0, dotPos);
    }
    catch (...) {
        // 异常处理，返回空字符串
        return "";
    }
}

std::string get_model_info(std::string engine_path)
{
    return getFilenameWithoutExtension(engine_path);
}

// Initialize the engine from a serialized engine file
void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    std::string model_info = get_model_info(engine_path);
    if (""!=model_info)
    {
        m_model_name = model_info;
    }

    // Open the engine file in binary mode
    ifstream engineStream(engine_path, ios::binary);
    // Move to the end to determine file size
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    // Move back to the beginning of the file
    engineStream.seekg(0, ios::beg);
    // Allocate memory to read the engine data
    unique_ptr<char[]> engineData(new char[modelSize]);
    // Read the engine data into memory
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // 看看是否加密模型
    char* pData = engineData.get();
    int size_bytes = sizeof(unsigned long long);
    unsigned long long data_size = *((unsigned long long*)pData);
    std::vector<char> data_buf;
    if (modelSize==(data_size + size_bytes))
    {
        // 这是加密模型
        std::vector<std::string> mac_list = get_mac_address_list();
        for (auto& mac: mac_list)
        {
            int ret = decode_buf(engineData.get(), modelSize, mac, m_model_name,data_buf);
            if (0==ret) break;
            
        }
    
        if (data_buf.size()==0)
        {
            std::cerr << "Error read data OR UNAUTHORIZED device" << std::endl;
            std::cerr << "Error read data OR UNAUTHORIZED device" << std::endl;
            std::cerr << "Error read data OR UNAUTHORIZED device" << std::endl;

            return;
        }
        
        pData = data_buf.data();
        data_size = data_buf.size();        
    }
    else
    {
        // 普通模型
        pData = engineData.get();
        data_size = modelSize;
    }

    // Create a TensorRT runtime instance
    runtime = createInferRuntime(logger);
    // Deserialize the CUDA engine from the engine data
    engine = runtime->deserializeCudaEngine(pData, data_size);
    // Create an execution context for the engine
    context = engine->createExecutionContext();

    // Retrieve input dimensions from the engine
    Dims dim_input = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = dim_input.d[2];
    input_w = dim_input.d[3];
    // Retrieve detection attributes and number of detections
    Dims dim_output = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = dim_output.d[1];
    num_detections = dim_output.d[2];
    // Calculate the number of classes based on detection attributes
    num_classes = detection_attribute_size - 4;

    // Allocate CPU memory for output buffer
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    // Allocate GPU memory for input buffer (assuming 3 channels: RGB)
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Allocate GPU memory for output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    // Initialize CUDA preprocessing with maximum image size
    cuda_preprocess_init(MAX_IMAGE_SIZE);

    // Create a CUDA stream for asynchronous operations
    CUDA_CHECK(cudaStreamCreate(&stream));

#if NV_TENSORRT_MAJOR < 10

#else
    // For TensorRT versions 10 and above, use enqueueV3 with the CUDA stream
    context->setTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
#endif

    // Perform model warmup if enabled
    if (warmup) {
        for (int i = 0; i < 1; i++) {
            this->infer(); // Run inference to warm up the model
        }
        printf("model warmup 10 times\n");
    }
}


// Destructor for the YOLOv11 class
YOLOv11::~YOLOv11()
{
    // Synchronize and destroy the CUDA stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    // Free allocated GPU buffers
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    // Free CPU output buffer
    delete[] cpu_output_buffer;

    // Destroy CUDA preprocessing resources
    cuda_preprocess_destroy();
    // Delete TensorRT context, engine, and runtime
    delete context;
    delete engine;
    delete runtime;
}

// Preprocess the input image and transfer it to the GPU buffer
void YOLOv11::preprocess(Mat& image) {
    // Perform CUDA-based preprocessing
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h,        
        resizeKeepAspectRatio, m_scale_h, m_scale_w, m_margin_top, m_margin_left, stream);
    // Synchronize the CUDA stream to ensure preprocessing is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Perform inference using the TensorRT execution context
void YOLOv11::infer()
{
#if NV_TENSORRT_MAJOR < 10
    // For TensorRT versions less than 10, use enqueueV2 with GPU buffers
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    // For TensorRT versions 10 and above, use enqueueV3 with the CUDA stream
    this->context->enqueueV3(this->stream);
#endif
}

// Postprocess the inference output to extract detections
void YOLOv11::postprocess(vector<Detection>& output)
{
#ifdef REL_DBG
    std::vector<float> cpu_input_img(input_w*input_h);
    CUDA_CHECK(cudaMemcpyAsync(cpu_input_img.data(), gpu_buffers[0], input_w * input_h * sizeof(float), cudaMemcpyDeviceToHost, stream));
#endif // REL_DBG

    // Asynchronously copy output from GPU to CPU
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // Synchronize the CUDA stream to ensure copy is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vector<Rect> boxes;          // Bounding boxes
    vector<int> class_ids;       // Class IDs
    vector<float> confidences;   // Confidence scores

#ifdef REL_DBG
    const Mat det_input(input_w, input_h, CV_32F, cpu_input_img.data());
#endif // REL_DBG

    // Create a matrix view of the detection output
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // Iterate over each detection
    for (int i = 0; i < det_output.cols; ++i) {
        // Extract class scores for the current detection
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        // Find the class with the maximum score
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // Check if the confidence score exceeds the threshold
        if (score > conf_threshold) {
            // Extract bounding box coordinates
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            // Calculate top-left corner of the bounding box
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            // Set width and height of the bounding box
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            // Store the bounding box, class ID, and confidence
            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    vector<int> nms_result; // Indices after Non-Maximum Suppression (NMS)
    // Apply NMS to remove overlapping boxes
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    // Iterate over NMS results and populate the output detections
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}
void YOLOv11::postprocess_pan_have_pt(Mat& image, vector<Detection_pan>& output, vector<Detection_pan>& output_fine)
{
    int CLASS_NUM = 3;
    int LAST_CLASS_ID = CLASS_NUM - 1;
    int kpt_begin_idx = 4 + CLASS_NUM;

    // 关键点数量
    int pt_cnt = (detection_attribute_size - (kpt_begin_idx)) / 3;


    vector<Rect2f> boxes, boxes_fine;          // Bounding boxes
    vector<Rect> boxes_int, boxes_int_fine;        // Bounding boxes
    vector<int> class_ids, class_ids_fine;       // Class IDs
    vector<float> confidences, confidences_fine;   // Confidence scores
    vector<int> det_ids, det_ids_fine;   // Confidence scores



    // Create a matrix view of the detection output
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // Iterate over each detection
    for (int i = 0; i < det_output.cols; ++i) {
        // Extract class scores for the current detection
        const Mat classes_scores = det_output.col(i).rowRange(4, kpt_begin_idx);
        Point class_id_point;
        double score;
        // Find the class with the maximum score
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // Check if the confidence score exceeds the threshold
        if (score > conf_threshold && 1 == class_id_point.y) {
            // Extract bounding box coordinates
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect2f box;
            //// Calculate top-left corner of the bounding box
            //box.x = ((cx - 0.5 * ow)*input_w);
            //box.y = ((cy - 0.5 * oh) * input_h);
            //// Set width and height of the bounding box
            //box.width = (ow*input_w);
            //box.height = (oh* input_h);

            // Calculate top-left corner of the bounding box
            box.x = ((cx - 0.5 * ow));
            box.y = ((cy - 0.5 * oh));
            // Set width and height of the bounding box
            box.width = (ow);
            box.height = (oh);

            if (box.x < 0)
            {
                box.width += box.x;
                box.x = 0;
            }

            if (box.x + ow > input_w)
            {
                box.width -= ((box.x + ow) - input_w);
            }

            if (box.y < 0)
            {
                box.height += box.y;
                box.y = 0;
            }

            if (box.y + oh > input_h)
            {
                box.height -= ((box.y + oh) - input_h);
            }


            // Store the bounding box, class ID, and confidence
            boxes.push_back(box);
            boxes_int.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);

            det_ids.push_back(i);
        }

    }

    vector<int> nms_result; // Indices after Non-Maximum Suppression (NMS)
    // Apply NMS to remove overlapping boxes
    dnn::NMSBoxes(boxes_int, confidences, conf_threshold, nms_threshold, nms_result);


    // Iterate over NMS results and populate the output detections
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection_pan result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];

        int real_idx = det_ids[idx];

        const Mat mat_pts = det_output.col(real_idx).rowRange(kpt_begin_idx, kpt_begin_idx + pt_cnt * 3);
        float* ptr = (float*)mat_pts.data;
        for (size_t p = 0; p < pt_cnt; p++)
        {
            const float cx = mat_pts.at<float>(0 + 3 * p, 0);
            const float cy = mat_pts.at<float>(1 + 3 * p, 0);

            cv::Point2f pt;
            pt.x = cx;
            pt.y = cy;

            result.pts.push_back(pt);
            // result.v.push_back(mat_pts.at<float>(2 + 3 * p, 0));
        }

        output.push_back(result);
    }


    // 对output排序，把面积大的放在前面
    std::sort(output.begin(), output.end(),
        [](const Detection_pan& a, const Detection_pan& b) {
        return a.bbox.area() > b.bbox.area();
    });


    // 获取fine 结果
    if (output.size() > 0)
    {
        Rect2f bbox = output[0].bbox;

        // Iterate over each detection
        // 取第一级缩小图的结果 w/8  *  h/8
        int det_cnt = (input_w / 8) * (input_h / 8);
        det_cnt = std::min(det_output.cols, det_cnt);
        for (int i = 0; i < det_cnt; ++i) {
            // Extract class scores for the current detection
            const Mat classes_scores = det_output.col(i).rowRange(4, kpt_begin_idx);
            double score = classes_scores.at<float>(2, 0);

            if (score > conf_threshold_low) {
                // Extract bounding box coordinates
                const float cx = det_output.at<float>(0, i);
                const float cy = det_output.at<float>(1, i);
                const float ow = det_output.at<float>(2, i);
                const float oh = det_output.at<float>(3, i);

                if (!bbox.contains(cv::Point2f(cx, cy))) continue;

                Rect2f box;
                //// Calculate top-left corner of the bounding box
                //box.x = ((cx - 0.5 * ow) * input_w);
                //box.y = ((cy - 0.5 * oh) * input_h);
                //// Set width and height of the bounding box
                //box.width = (ow * input_w);
                //box.height = (oh * input_h);

                // Calculate top-left corner of the bounding box
                box.x = ((cx - 0.5 * ow));
                box.y = ((cy - 0.5 * oh));
                //box.x = ((cx));
                //box.y = ((cy));

                // Set width and height of the bounding box
                box.width = (ow);
                box.height = (oh);

                // Store the bounding box, class ID, and confidence
                boxes_fine.push_back(box);
                boxes_int_fine.push_back(box);
                // class_ids_fine.push_back(LAST_CLASS_ID);

                //Point class_id_point;
                //double score_fine;
                //minMaxLoc(classes_scores, nullptr, &score_fine, nullptr, &class_id_point);
                class_ids_fine.push_back(LAST_CLASS_ID);

                confidences_fine.push_back(score);

                det_ids_fine.push_back(i);
            }
        }

        //vector<float> conf_fine = confidences_fine;   // Confidence scores
        //std::sort(conf_fine.begin(), conf_fine.end(),
        //    [](const float& a, const float& b) {
        //    return a > b;
        //});

        //vector<int> nms_result_fine; // Indices after Non-Maximum Suppression (NMS)

        //// Apply NMS to remove overlapping boxes
        //dnn::NMSBoxes(boxes_int_fine, confidences_fine, conf_threshold_low, nms_threshold, nms_result_fine);

        //for (int i = 0; i < nms_result_fine.size(); i++)
        for (int i = 0; i < class_ids_fine.size(); i++)
        {
            Detection_pan result;
            // int idx = nms_result_fine[i];
            int idx = i;
            result.class_id = class_ids_fine[idx];
            result.conf = confidences_fine[idx];
            result.bbox = boxes_fine[idx];

            output_fine.push_back(result);
        }
    }

}

void nms_det(vector<Detection_pan>& output, 
    const vector<Rect>& boxes_int, const vector<Rect2f>& boxes,
    const vector<float>& confidences, const vector<int>& class_ids,
    float conf_threshold,float nms_threshold)
{
    vector<int> nms_result; // Indices after Non-Maximum Suppression (NMS)
    // Apply NMS to remove overlapping boxes
    dnn::NMSBoxes(boxes_int, confidences, conf_threshold, nms_threshold, nms_result);


    // Iterate over NMS results and populate the output detections
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection_pan result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];

        output.push_back(result);
    }
}

void YOLOv11::postprocess_pan_no_pt(Mat& image, vector<Detection_pan>& output, vector<Detection_pan>& output_fine)
{
    int CLASS_NUM = 3;
    int LAST_CLASS_ID = CLASS_NUM - 1;
    int kpt_begin_idx = 4 + CLASS_NUM;

    // 关键点数量
    int pt_cnt = (detection_attribute_size - (kpt_begin_idx)) / 3;


    vector<Rect2f> boxes, boxes_fine;          // Bounding boxes
    vector<Rect> boxes_int, boxes_int_fine;        // Bounding boxes
    vector<int> class_ids, class_ids_fine;       // Class IDs
    vector<float> confidences, confidences_fine;   // Confidence scores
    vector<int> det_ids, det_ids_fine;   // Confidence scores



    // Create a matrix view of the detection output
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // Iterate over each detection
    for (int i = 0; i < det_output.cols; ++i) {
        // Extract class scores for the current detection
        const Mat classes_scores = det_output.col(i).rowRange(4, kpt_begin_idx);
        Point class_id_point;
        double score;
        // Find the class with the maximum score
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // Check if the confidence score exceeds the threshold

        // class 1 是圆盘   class 2 是圆孔
        if (score > conf_threshold && class_id_point.y>0) {
            // Extract bounding box coordinates
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect2f box;

            if (std::abs(cx) <= 1 && std::abs(cy) <= 1 
                && std::abs(ow) <= 1 && std::abs(oh) <= 1)
            {
                // Calculate top-left corner of the bounding box
                box.x = ((cx - 0.5 * ow) * input_w);
                box.y = ((cy - 0.5 * oh) * input_h);
                // Set width and height of the bounding box
                box.width = (ow * input_w);
                box.height = (oh * input_h);
            }
            else
            {
                // Calculate top-left corner of the bounding box
                box.x = ((cx - 0.5 * ow));
                box.y = ((cy - 0.5 * oh));
                // Set width and height of the bounding box
                box.width = (ow);
                box.height = (oh);
            }


            if (box.x < 0)
            {
                box.width += box.x;
                box.x = 0;
            }

            if (box.x + ow > input_w)
            {
                box.width -= ((box.x + ow) - input_w);
            }

            if (box.y < 0)
            {
                box.height += box.y;
                box.y = 0;
            }

            if (box.y + oh > input_h)
            {
                box.height -= ((box.y + oh) - input_h);
            }

            // Store the bounding box, class ID, and confidence
            if (1==class_id_point.y)
            {
                boxes.push_back(box);
                boxes_int.push_back(box);
                class_ids.push_back(class_id_point.y);
                confidences.push_back(score);

                det_ids.push_back(i);
            }
            else
            {
                boxes_fine.push_back(box);
                boxes_int_fine.push_back(box);
                class_ids_fine.push_back(class_id_point.y);
                confidences_fine.push_back(score);

                det_ids_fine.push_back(i);
            }

        }

    }

#ifdef REL_DBG
    cv::Mat img_dbg_lun;
    cv::resize(image, img_dbg_lun, cv::Size(input_w, input_h));
    draw_boxes(img_dbg_lun, boxes);
#endif // REL_DBG

    nms_det(output, boxes_int, boxes, confidences, class_ids, conf_threshold, nms_threshold);

    // 对output排序，把面积大的放在前面
    std::sort(output.begin(), output.end(),
        [](const Detection_pan& a, const Detection_pan& b) {
        return a.bbox.area() > b.bbox.area();
    });


#ifdef REL_DBG
    cv::Mat img_dbg;
    cv::resize(image, img_dbg,cv::Size(input_w, input_h));
    draw_boxes(img_dbg, boxes_fine);
#endif // REL_DBG


    nms_det(output_fine, boxes_int_fine, boxes_fine, confidences_fine, 
        class_ids_fine, conf_threshold, nms_threshold);

    // 对output排序，先把分数高的放在前面
    std::sort(output_fine.begin(), output_fine.end(),
        [](const Detection_pan& a, const Detection_pan& b) {
        return a.conf > b.conf;
    });

    // 取前4个
    int hole_num = std::min((int)output_fine.size(),4);
    vector<Detection_pan> output_hole = vector<Detection_pan>(output_fine.begin(), output_fine.begin()+ hole_num);

    std::sort(output_hole.begin(), output_hole.end(),
        [](const Detection_pan& a, const Detection_pan& b) {
        return a.bbox.area() > b.bbox.area();
    });

    if (output.size()>0)
    {
        for (size_t i = 0; i < output_hole.size(); i++)
        {
            auto& h = output_hole[i];
            cv::Point2f pt_centor = (h.bbox.tl() + h.bbox.br())/2.0;

            output[0].pts.push_back(pt_centor);
            output[0].sizes.push_back(h.bbox.size());
        }
    }


    output_fine.clear();
}

void YOLOv11::postprocess_xialiao(Mat& image, vector< vector<Detection_pan>>& outputs)
{
    int CLASS_NUM = 7;  // BACK, LUNZI, HOLE, PLATE_TRAY, PLATE_CELL, PLATE_HOLE, PALLET
    int LAST_CLASS_ID = CLASS_NUM - 1;
    int kpt_begin_idx = 4 + CLASS_NUM;

    
    // 关键点数量
    int pt_cnt = (detection_attribute_size - (kpt_begin_idx)) / 3;

    outputs.resize(CLASS_NUM);

    vector< vector<Rect2f>>list_boxes(CLASS_NUM);          // Bounding boxes
    vector < vector<Rect>>list_boxes_int(CLASS_NUM);        // Bounding boxes
    vector < vector<int>>list_class_ids(CLASS_NUM);       // Class IDs
    vector < vector<float>>list_confidences(CLASS_NUM);   // Confidence scores
    vector < vector<int>>list_det_ids(CLASS_NUM);   // Confidence scores

    float x_scale = 1;
    float y_scale = 1;

    // 是否保持比例
    if (resizeKeepAspectRatio > 0)
    {
        x_scale = (float)input_w / ((float)input_w - 2 * m_margin_left);
        y_scale = (float)input_h / ((float)input_h - 2 * m_margin_top);
    }

    // Create a matrix view of the detection output
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // Iterate over each detection
    for (int i = 0; i < det_output.cols; ++i) {
        // Extract class scores for the current detection
        const Mat classes_scores = det_output.col(i).rowRange(4, kpt_begin_idx);
        Point class_id_point;
        double score;
        // Find the class with the maximum score
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // 把大于阈值的bbox提取出来
        //enum CLASS_IDS {
        //    BACK = 0, LUNZI = 1, HOLE = 2,
        //    PLATE_TRAY = 3, PLATE_CELL = 4, PLATE_HOLE = 5
        //};

        if (score > conf_threshold && class_id_point.y>0) {
            // Extract bounding box coordinates
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect2f box;

            if (std::abs(cx) <= 1 && std::abs(cy) <= 1 
                && std::abs(ow) <= 1 && std::abs(oh) <= 1)
            {
                // Calculate top-left corner of the bounding box
                box.x = ((cx - 0.5 * ow) * input_w - m_margin_left) * x_scale;
                box.y = ((cy - 0.5 * oh) * input_h - m_margin_top) * y_scale;
                // Set width and height of the bounding box
                box.width = (ow * input_w)*x_scale;
                box.height = (oh * input_h)*y_scale;
            }
            else
            {
                // Calculate top-left corner of the bounding box
                box.x = ((cx - 0.5 * ow)-m_margin_left) * x_scale;
                box.y = ((cy - 0.5 * oh)-m_margin_top) * y_scale;
                // Set width and height of the bounding box
                box.width = (ow)*x_scale;
                box.height = (oh)*y_scale;
            }


            if (box.x < 0)
            {
                box.width += box.x;
                box.x = 0;
            }

            if (box.x + ow > input_w)
            {
                box.width -= ((box.x + ow) - input_w);
            }

            if (box.y < 0)
            {
                box.height += box.y;
                box.y = 0;
            }

            if (box.y + oh > input_h)
            {
                box.height -= ((box.y + oh) - input_h);
            }

            {
                list_boxes[class_id_point.y].push_back(box);
                list_boxes_int[class_id_point.y].push_back(box);
                list_class_ids[class_id_point.y].push_back(class_id_point.y);
                list_confidences[class_id_point.y].push_back(score);

                list_det_ids[class_id_point.y].push_back(i);
            }

        }

    }

#ifdef REL_DBG
    //cv::Mat img_dbg_lun;
    //cv::resize(image, img_dbg_lun, cv::Size(input_w, input_h));
    //for (size_t i = 0; i < list_boxes.size(); i++)
    //{
    //    draw_boxes(img_dbg_lun, list_boxes[i]);
    //}
    
#endif // REL_DBG

    for (size_t i = 0; i < list_boxes.size(); i++)
    {
        nms_det(outputs[i], list_boxes_int[i], list_boxes[i], list_confidences[i], list_class_ids[i], conf_threshold, nms_threshold);
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

    for (auto& o : outputs)
    {
        // 对output排序，把面积大的放在前面
        std::sort(o.begin(), o.end(),
            [](const Detection_pan& a, const Detection_pan& b) {
            return a.bbox.area() > b.bbox.area();
        });
    }

}

// Postprocess for feilun model: 3 classes (BACK_FEILUN=0, BOSS=1, WHEEL=2)
// BOSS and WHEEL share NMS, then mapped to xialiao CLASS_IDS: WHEEL→LUNZI(1), BOSS→HOLE(2)
void YOLOv11::postprocess_feilun(Mat& image, vector< vector<Detection_pan>>& outputs)
{
    int CLASS_NUM = 3;  // BACK_FEILUN=0, BOSS=1, WHEEL=2
    int kpt_begin_idx = 4 + CLASS_NUM;

    // Map to xialiao CLASS_IDS structure (6 classes)
    // WHEEL(2) → LUNZI(1), BOSS(1) → HOLE(2)
    int XIALIAO_CLASS_NUM = 7;  // 与 xialiao CLASS_IDS 保持一致
    outputs.resize(XIALIAO_CLASS_NUM);

    vector< vector<Rect2f>> list_boxes(CLASS_NUM);
    vector< vector<Rect>> list_boxes_int(CLASS_NUM);
    vector< vector<int>> list_class_ids(CLASS_NUM);
    vector< vector<float>> list_confidences(CLASS_NUM);

    float x_scale = 1;
    float y_scale = 1;

    if (resizeKeepAspectRatio > 0)
    {
        x_scale = (float)input_w / ((float)input_w - 2 * m_margin_left);
        y_scale = (float)input_h / ((float)input_h - 2 * m_margin_top);
    }

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, kpt_begin_idx);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // 跳过背景 (class_id=0)
        if (score > conf_threshold && class_id_point.y > 0) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect2f box;

            if (std::abs(cx) <= 1 && std::abs(cy) <= 1
                && std::abs(ow) <= 1 && std::abs(oh) <= 1)
            {
                box.x = ((cx - 0.5 * ow) * input_w - m_margin_left) * x_scale;
                box.y = ((cy - 0.5 * oh) * input_h - m_margin_top) * y_scale;
                box.width = (ow * input_w) * x_scale;
                box.height = (oh * input_h) * y_scale;
            }
            else
            {
                box.x = ((cx - 0.5 * ow) - m_margin_left) * x_scale;
                box.y = ((cy - 0.5 * oh) - m_margin_top) * y_scale;
                box.width = ow * x_scale;
                box.height = oh * y_scale;
            }

            if (box.x < 0) {
                box.width += box.x;
                box.x = 0;
            }
            if (box.x + box.width > input_w) {
                box.width = input_w - box.x;
            }
            if (box.y < 0) {
                box.height += box.y;
                box.y = 0;
            }
            if (box.y + box.height > input_h) {
                box.height = input_h - box.y;
            }

            int cls = class_id_point.y;
            list_boxes[cls].push_back(box);
            list_boxes_int[cls].push_back(box);
            list_class_ids[cls].push_back(cls);
            list_confidences[cls].push_back(score);
        }
    }

    // NMS: BOSS(1) 和 WHEEL(2) 合并一起做 NMS
    vector<Rect> combined_boxes_int;
    vector<Rect2f> combined_boxes;
    vector<float> combined_confidences;
    vector<int> combined_class_ids;

    for (int cls = 1; cls <= 2; cls++) {
        for (size_t i = 0; i < list_boxes[cls].size(); i++) {
            combined_boxes_int.push_back(list_boxes_int[cls][i]);
            combined_boxes.push_back(list_boxes[cls][i]);
            combined_confidences.push_back(list_confidences[cls][i]);
            combined_class_ids.push_back(cls);
        }
    }

    if (!combined_boxes.empty()) {
        vector<int> nms_result;
        dnn::NMSBoxes(combined_boxes_int, combined_confidences, conf_threshold, nms_threshold, nms_result);

        for (size_t i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            Detection_pan det;
            det.class_id = combined_class_ids[idx];
            det.conf = combined_confidences[idx];
            det.bbox = combined_boxes[idx];

            // 映射到 xialiao CLASS_IDS: WHEEL(2)→LUNZI(1), BOSS(1)→HOLE(2)
            if (det.class_id == WHEEL) {
                det.class_id = LUNZI;       // 修正为 xialiao class_id
                outputs[LUNZI].push_back(det);  // WHEEL → LUNZI
            } else if (det.class_id == BOSS) {
                det.class_id = HOLE;        // 修正为 xialiao class_id
                outputs[HOLE].push_back(det);  // BOSS → HOLE
            }

        }
    }

    // 按面积降序排列
    for (auto& o : outputs) {
        std::sort(o.begin(), o.end(),
            [](const Detection_pan& a, const Detection_pan& b) {
            return a.bbox.area() > b.bbox.area();
        });
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


}


// Postprocess for screen model: 3 classes (BACKGROUND_SCREEN=0, PALLET_SCREEN=1, CIRCLE_SCREEN=2)
// Output to xialiao 7-class structure: PALLET_SCREEN(1) → LUNZI(1), CIRCLE_SCREEN(2) → HOLE(2)
void YOLOv11::postprocess_screen(Mat& image, vector< vector<Detection_pan>>& outputs)
{
    int CLASS_NUM = 3;  // BACKGROUND_SCREEN=0, PALLET_SCREEN=1, CIRCLE_SCREEN=2
    int kpt_begin_idx = 4 + CLASS_NUM;

    int XIALIAO_CLASS_NUM = 7;  // 与 xialiao CLASS_IDS 保持一致
    outputs.resize(XIALIAO_CLASS_NUM);

    vector< vector<Rect2f>> list_boxes(CLASS_NUM);
    vector< vector<Rect>> list_boxes_int(CLASS_NUM);
    vector< vector<int>> list_class_ids(CLASS_NUM);
    vector< vector<float>> list_confidences(CLASS_NUM);

    float x_scale = 1;
    float y_scale = 1;

    if (resizeKeepAspectRatio > 0)
    {
        x_scale = (float)input_w / ((float)input_w - 2 * m_margin_left);
        y_scale = (float)input_h / ((float)input_h - 2 * m_margin_top);
    }

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, kpt_begin_idx);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // skip background (class_id=0)
        if (score > conf_threshold && class_id_point.y > 0) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect2f box;

            if (std::abs(cx) <= 1 && std::abs(cy) <= 1
                && std::abs(ow) <= 1 && std::abs(oh) <= 1)
            {
                box.x = ((cx - 0.5 * ow) * input_w - m_margin_left) * x_scale;
                box.y = ((cy - 0.5 * oh) * input_h - m_margin_top) * y_scale;
                box.width = (ow * input_w) * x_scale;
                box.height = (oh * input_h) * y_scale;
            }
            else
            {
                box.x = ((cx - 0.5 * ow) - m_margin_left) * x_scale;
                box.y = ((cy - 0.5 * oh) - m_margin_top) * y_scale;
                box.width = ow * x_scale;
                box.height = oh * y_scale;
            }

            if (box.x < 0) {
                box.width += box.x;
                box.x = 0;
            }
            if (box.x + box.width > input_w) {
                box.width = input_w - box.x;
            }
            if (box.y < 0) {
                box.height += box.y;
                box.y = 0;
            }
            if (box.y + box.height > input_h) {
                box.height = input_h - box.y;
            }

            int cls = class_id_point.y;
            list_boxes[cls].push_back(box);
            list_boxes_int[cls].push_back(box);
            list_class_ids[cls].push_back(cls);
            list_confidences[cls].push_back(score);
        }
    }

    for (size_t i = 0; i < list_boxes.size(); i++)
    {
        nms_det(outputs[i], list_boxes_int[i], list_boxes[i], list_confidences[i], list_class_ids[i], conf_threshold, nms_threshold);
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


    for (auto& o : outputs)
    {
        std::sort(o.begin(), o.end(),
            [](const Detection_pan& a, const Detection_pan& b) {
            return a.bbox.area() > b.bbox.area();
        });
    }
}

// Postprocess the inference output to extract detections
void YOLOv11::postprocess_pan(Mat& image, vector<vector<Detection_pan>>& outputs)
{
    // Asynchronously copy output from GPU to CPU
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));


#ifdef REL_DBG
    //cv::Mat cuda_img(input_h, input_w, CV_32FC3);
    //CUDA_CHECK(cudaMemcpyAsync(cuda_img.data, gpu_buffers[0], input_w * input_h * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    //Mat cuda_img1(input_h, input_w, CV_32F, cuda_img.data);
#endif // REL_DBG



    // Synchronize the CUDA stream to ensure copy is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if ("shangliao"==m_model_name)
    {
        int CLASS_NUM = 3;
        int LAST_CLASS_ID = CLASS_NUM - 1;
        int kpt_begin_idx = 4 + CLASS_NUM;

        vector<Detection_pan> output, output_fine;

        // 关键点数量
        int pt_cnt = (detection_attribute_size - (kpt_begin_idx)) / 3;

        if (pt_cnt>0)
        {
            postprocess_pan_have_pt(image,output,output_fine);
        }
        else
        {
            postprocess_pan_no_pt(image,output, output_fine);
        }

        outputs.push_back(output);
        outputs.push_back(output_fine);
    }
    else if ("feilun"==m_model_name)
    {
        postprocess_feilun(image, outputs);
    }
    else if ("screen"==m_model_name)
    {
        postprocess_screen(image, outputs);
    }
    else
    {
        postprocess_xialiao(image, outputs);
    }

}


// Build the TensorRT engine from an ONNX model
void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    // Create a TensorRT builder
    auto builder = createInferBuilder(logger);
    // Define network flags for explicit batch dimensions
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // Create a network definition with explicit batch
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // Create builder configuration
    IBuilderConfig* config = builder->createBuilderConfig();
    // Enable FP16 precision if specified
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    // Create an ONNX parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    // Parse the ONNX model file
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    // Build the serialized network plan
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    // Create a TensorRT runtime
    runtime = createInferRuntime(logger);

    // Deserialize the CUDA engine from the serialized plan
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    // Create an execution context for the engine
    context = engine->createExecutionContext();

    // Clean up allocated resources
    delete network;
    delete config;
    delete parser;
    delete plan;
}

// Save the serialized TensorRT engine to a file
bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    // Generate the engine file path by replacing the extension with ".engine"
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false; // Return false if no extension is found
    }

    // Check if the engine is valid
    if (engine)
    {
        // Serialize the engine
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        // Open the engine file in binary write mode
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file " << engine_path << " failed" << std::endl;
            return false;
        }
        // Write the serialized engine data to the file
        file.write((const char*)data->data(), data->size());
        file.close();

        // Free the serialized data memory
        delete data;
    }
    return true;
}

// Draw bounding boxes and labels on the image based on detections
void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    // Calculate the scaling ratios between input and original image dimensions
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    // Iterate over each detection
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        // Assign a color based on the class ID
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Adjust bounding box coordinates based on aspect ratio
        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        // Draw the bounding box on the image
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Prepare the label text with class name and confidence
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        // Calculate the size of the text for background rectangle
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        // Define the background rectangle for the text
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        // Draw the background rectangle
        rectangle(image, text_rect, color, FILLED);
        // Put the text label on the image
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}

void YOLOv11::restore_real_xy_keep(Mat& image, vector<Detection_pan>& output)
{
    // Calculate the scaling ratios between input and original image dimensions
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    // Iterate over each detection
    for (int i = 0; i < output.size(); i++)
    {
        auto& detection = output[i];
        auto& box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;


        // Adjust bounding box coordinates based on aspect ratio
        if (ratio_h >= ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;

            for (auto& pt : detection.pts)
            {
                pt.x = pt.x / ratio_w;
                pt.y = (pt.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            }
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;

            for (auto& pt : detection.pts)
            {
                pt.x = (pt.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
                pt.y = pt.y / ratio_h;
            }
        }
        
    }
}


void YOLOv11::restore_real_xy(Mat& image, vector<Detection_pan>& output)
{
#ifdef REL_DBG
    cv::Mat img_dbg = image.clone();
#endif // REL_DBG

    // Calculate the scaling ratios between input and original image dimensions
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    // Iterate over each detection
    for (int i = 0; i < output.size(); i++)
    {
        auto& detection = output[i];
        auto& box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;


        // Adjust bounding box coordinates based on aspect ratio
        {
            box.x = box.x / ratio_w;
            box.y = box.y / ratio_h;

            box.width = box.width / ratio_w;
            box.height = box.height / ratio_h;

            for (auto& pt : detection.pts)
            {
                pt.x = pt.x / ratio_w;
                pt.y = pt.y / ratio_h;
            }

            for (auto& s : detection.sizes)
            {
                s.width = s.width / ratio_w;
                s.height = s.height / ratio_h;
            }
        }

#ifdef REL_DBG
        draw_pan(img_dbg, detection);
#endif // REL_DBG    

    }
}

void YOLOv11::draw_boxes(Mat& image, vector<Rect2f>& boxes)
{
    cv::Scalar cl_red = Scalar(0, 0, 255);
    int class_id = 0;
    for (auto& box: boxes)
    {
        // Assign a color based on the class ID
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Draw the bounding box on the image
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        class_id++;
        if (class_id >= COLORS.size()) class_id = 0;
    }
}


void YOLOv11::draw_pan(Mat& image, vector<Detection_pan>& output)
{
    // Calculate the scaling ratios between input and original image dimensions

    for (int i = 0; i < output.size(); i++)
    {
        draw_pan(image, output[i]);

        break;
        
    }
}

void YOLOv11::draw_pan(Mat& image, Detection_pan& detection)
{
    // Calculate the scaling ratios between input and original image dimensions

    cv::Scalar cl_red = Scalar(0, 0, 255);
    cv::Scalar cl_g = Scalar(0, 255, 0);

    {
        auto& box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        // Assign a color based on the class ID
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Draw the bounding box on the image
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Prepare the label text with class name and confidence
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);

        // if ("feilun"==m_model_name)
        // {
        //     class_string = CLASS_NAMES_FEILUN[class_id] + ' ' + to_string(conf).substr(0, 4);
        // }
        
        // Calculate the size of the text for background rectangle
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        // Define the background rectangle for the text
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        // Draw the background rectangle
        rectangle(image, text_rect, color, FILLED);
        // Put the text label on the image
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);

        if (g_save_ret > 1)
        {
            if (detection.pts_new.size()== detection.radius.size())
            {
                // draw pt
                int idx = 0;
                for (auto& pt : detection.pts_new)
                {
                    cv::circle(image, pt, 2, cl_red);
                    cv::circle(image, pt, detection.radius[idx], cl_g);

                    putText(image, std::to_string(idx++), pt, FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 1, 0);
                }
            }
            else
            {
                // draw pt
                int idx = 0;
                for (auto& pt : detection.pts)
                {
                    cv::circle(image, pt, 2, cl_red);
                    putText(image, std::to_string(idx++), pt, FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 1, 0);
                }
            }

        }
    }
}