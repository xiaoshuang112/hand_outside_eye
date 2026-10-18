#ifndef AI_CAMERA_HPP
#define AI_CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>
#include "my_base.h"

/**
 * @brief 相机抽象基类
 * 定义所有AI相机的通用接口
 */
class AICamera {
public:
    explicit AICamera(const CONFIG& config) : m_config(config) {}
    virtual ~AICamera() = default;

    // 禁止拷贝
    AICamera(const AICamera&) = delete;
    AICamera& operator=(const AICamera&) = delete;

    // ========== 核心相机操作接口 ==========
    
    /**
     * @brief 打开相机
     * @return true 成功，false 失败
     */
    virtual bool open() = 0;
    
    /**
     * @brief 关闭相机
     */
    virtual void close() = 0;
    
    /**
     * @brief 捕获单帧图像
     * @param timeoutMs 超时时间（毫秒）
     * @return BGR格式的OpenCV Mat，失败返回空Mat
     */
    virtual cv::Mat captureOneFrame(int timeoutMs = 1000) = 0;
    
    /**
     * @brief 捕获彩色图+深度图
     * @param bgrImage 输出BGR彩色图像
     * @param depthFrame 输出深度帧（具体类型由子类定义）
     * @param timeoutMs 超时时间（毫秒）
     * @return 0成功，-1失败
     */
    virtual int captureOneFrameWithDepth(cv::Mat& bgrImage, cv::Mat& depthImage, int timeoutMs = 2000) = 0;
    
    /**
     * @brief 检查相机是否已打开
     */
    virtual bool isOpened() const = 0;
    
    /**
     * @brief 获取相机配置
     */
    const CONFIG& getConfig() const { return m_config; }

    // ========== 曝光控制接口 ==========
    
    /**
     * @brief 设置彩色相机曝光时间
     * @param exposureUs 曝光时间，单位：微秒(μs)
     * @return true 设置成功，false 设置失败
     */
    virtual bool setColorExposureTime(int exposureUs) = 0;
    
    /**
     * @brief 获取当前彩色相机曝光时间
     * @return 曝光时间(微秒)，失败返回 -1
     */
    virtual int getColorExposureTime() = 0;
    
    /**
     * @brief 设置彩色相机自动曝光开关
     * @param enable true: 开启自动曝光, false: 关闭自动曝光
     * @return true 设置成功，false 设置失败
     */
    virtual bool setAutoExposure(bool enable) = 0;
    
    /**
     * @brief 获取彩色相机自动曝光状态
     * @return true: 自动曝光开启, false: 关闭
     */
    virtual bool getAutoExposure() = 0;
    
    /**
     * @brief 获取曝光时间范围
     * @param minUs 输出最小曝光时间(微秒)
     * @param maxUs 输出最大曝光时间(微秒)
     * @return true 获取成功
     */
    virtual bool getExposureRange(int& minUs, int& maxUs) = 0;

    // ========== 增益（曝光度）控制接口 ==========

    /**
     * @brief 设置彩色相机增益
     * @param gain 增益值（单位由设备定义）
     * @return true 设置成功
     */
    virtual bool setColorGain(int gain) = 0;

    /**
     * @brief 获取当前彩色相机增益
     * @return 增益值，失败返回 -1
     */
    virtual int getColorGain() = 0;

    /**
     * @brief 获取增益范围
     * @param min 输出最小增益
     * @param max 输出最大增益
     * @return true 获取成功
     */
    virtual bool getGainRange(int& min, int& max) = 0;

    // ========== 多曝光融合接口 ==========
    
    /**
     * @brief 采集多组不同曝光的原始图像
     * @param exposureValuesUs 曝光时间列表（微秒）
     * @return 采集到的图像序列
     */
    virtual std::vector<cv::Mat> captureMultiExposureImages(const std::vector<int>& exposureValuesUs) = 0;

    // 获取内参矩阵 (3x3矩阵)
    // 如果想得到特定分辨率下的内参，可以传入宽和高，例如 (1280, 720)
    virtual std::vector<std::vector<float>> getCameraIntrinsics(const int imgWidth, const int imgHight)=0;

    /**
     * @brief 设置相机水平和垂直翻转
     * @param flip_h 如果>0 水平翻转
     * @param flip_v 如果>0 垂直翻转
     * @return true 设置成功
     */
    virtual bool setFlip(int flip_v,int flip_h) = 0;

protected:
    CONFIG m_config;
    
    // 缓存值（供子类使用）
    bool m_autoExposureCached = true;
    int m_exposureCached = -1;
    
    // 通用的多曝光采集逻辑
    virtual std::vector<cv::Mat> captureMultiExposureImagesImpl(const std::vector<int>& exposureValuesUs) = 0;


};





class MultiExposureFusion {
public:

    int grid_rows = 5;
    int grid_cols = 5;
    int min_expose_val = 50;
    int max_expose_val = 200;

    // 静态缓存：首次计算后保存各块的图像选择方案，后续调用直接复用
    // 支持 2 套缓存对应不同拍摄位置，cache_slot 指定使用哪套（默认 0）
    int cache_slot = 0;
    static const int kMaxCacheSlots = 2;
    static std::vector<int> s_cached_selection_[kMaxCacheSlots];
    static int s_cached_grid_rows_[kMaxCacheSlots];
    static int s_cached_grid_cols_[kMaxCacheSlots];
    static int s_cached_num_images_[kMaxCacheSlots];
    //int min_expose_val = 30;
    //int max_expose_val = 220;

    // 计算图像块的亮度质量分数
    float computeBlockScore(const cv::Mat& block) {
        cv::Mat gray;
        if (block.channels() == 3) {
            cv::cvtColor(block, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = block.clone();
        }

        // 计算均值和方差
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);

        float mean_val = mean[0];
        float std_val = stddev[0];

        // 理想亮度范围（0-255，中间值128最理想）
        float brightness_score = 1.0f - std::abs(mean_val - 128) / 128.0f;

        // 细节丰富度（方差越大表示细节越丰富）
        float detail_score = std::min(std_val / 64.0f, 1.0f);

        // 综合评分：亮度适中且细节丰富
        return brightness_score * 0.6f + detail_score * 0.4f;
    }

    // 计算图像块的曝光合理性（防止过曝或欠曝）
    // -1 表示欠曝，0 表示非常好，1 表示过曝
    int get_exposed_value(const cv::Mat& block) {
        cv::Mat gray;
        if (block.channels() == 3) {
            cv::cvtColor(block, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = block.clone();
        }

        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        float mean_val = mean[0];

        if (mean_val < min_expose_val)
            return -1;

        if (mean_val > max_expose_val)
            return 1;

        return 0;

        // 较好的曝光范围
        //if (mean_val > min_expose_val && mean_val < max_expose_val)
        //    return 2;
        //// 合理曝光范围：既不太暗也不太亮
        //else if (mean_val > 30 && mean_val < 220)
        //    return 1;
        //else return -1;
    }

    // 主函数：多曝光图像融合
    int detectExposure(const cv::Mat& img,
        std::map<int, std::vector<std::pair<int, int>>>& centor_vals,
        std::map<int, std::vector<std::pair<int, int>>>& edge_vals) {

        // 获取第一张图像的尺寸作为参考
        cv::Size img_size = img.size();
        int height = img_size.height;
        int width = img_size.width;

        if (height < 100 || width < 100) return -1;


        // 计算每个块的尺寸
        int block_height = height / grid_rows;
        int block_width = width / grid_cols;

        // 计算每个网格块的曝光值
        for (int row = 0; row < grid_rows; row++) {
            for (int col = 0; col < grid_cols; col++) {
                // 计算当前块的ROI
                int y_start = row * block_height;
                int x_start = col * block_width;
                int y_end = (row == grid_rows - 1) ? height : (row + 1) * block_height;
                int x_end = (col == grid_cols - 1) ? width : (col + 1) * block_width;

                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);

                // 计算每个网格块的曝光值
                int val = get_exposed_value(img(roi));

                if ((row == 0 || row == grid_rows - 1) ||
                    (col == 0 || col == grid_cols - 1)) {
                    // edge
                    edge_vals[val].push_back({ col,row });
                }
                else {
                    // center
                    centor_vals[val].push_back({ col,row });
                }

            }
        }

        return 0;
    }
    // 主函数：多曝光图像融合
    cv::Mat fuseImages(const std::vector<cv::Mat>& list_img) {
        if (list_img.empty()) {
            throw std::runtime_error("输入图像列表为空");
        }

        // 获取第一张图像的尺寸作为参考
        cv::Size img_size = list_img[0].size();
        int height = img_size.height;
        int width = img_size.width;

        // 计算每个块的尺寸
        int block_height = height / grid_rows;
        int block_width = width / grid_cols;

        // 创建输出图像
        cv::Mat result = cv::Mat::zeros(height, width, list_img[0].type());

        // 为每个网格块选择最佳图像
        for (int row = 0; row < grid_rows; row++) {
            for (int col = 0; col < grid_cols; col++) {
                // 计算当前块的ROI
                int y_start = row * block_height;
                int x_start = col * block_width;
                int y_end = (row == grid_rows - 1) ? height : (row + 1) * block_height;
                int x_end = (col == grid_cols - 1) ? width : (col + 1) * block_width;

                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);

                // 为当前块选择最佳图像（首次计算后缓存，后续复用）
                int best_img_idx = getBestImageForBlock(list_img, roi, row, col);

                // 从最佳图像中提取当前块并拼接到结果中
                cv::Mat best_block = list_img[best_img_idx](roi).clone();
                best_block.copyTo(result(roi));
            }
        }

        return result;
    }

    // 为特定ROI区域选择最佳图像（带静态缓存：首次计算，后续复用）
    // 通过 cache_slot（0 或 1）区分不同拍摄位置，默认 slot=0
    int getBestImageForBlock(const std::vector<cv::Mat>& list_img, const cv::Rect& roi,
                              int row, int col) {
        int slot = (cache_slot >= 0 && cache_slot < kMaxCacheSlots) ? cache_slot : 0;
        int block_idx = row * grid_cols + col;

        // 检查缓存是否命中（相同网格配置、相同图像数量）
        bool cache_hit = (s_cached_grid_rows_[slot] == grid_rows &&
                          s_cached_grid_cols_[slot] == grid_cols &&
                          s_cached_num_images_[slot] == (int)list_img.size() &&
                          block_idx < (int)s_cached_selection_[slot].size());

        if (cache_hit) {
            return s_cached_selection_[slot][block_idx];
        }

        // 首次计算：走完整的亮度评分逻辑
        int best_idx = selectBestImageForBlock(list_img, roi);

        // 保存到缓存（首次调用时分配）
        if (s_cached_selection_[slot].empty()) {
            s_cached_selection_[slot].resize(grid_rows * grid_cols);
            s_cached_grid_rows_[slot] = grid_rows;
            s_cached_grid_cols_[slot] = grid_cols;
            s_cached_num_images_[slot] = (int)list_img.size();
        }
        s_cached_selection_[slot][block_idx] = best_idx;

        return best_idx;
    }

    // 为特定ROI区域选择最佳图像
    int selectBestImageForBlock(const std::vector<cv::Mat>& list_img, const cv::Rect& roi) {
        float best_score = -1.0f;
        int best_idx = 0;

        for (size_t i = 0; i < list_img.size(); i++) {
            // 提取当前图像的ROI区域
            cv::Mat block = list_img[i](roi);

            // 检查曝光是否合理
            if (get_exposed_value(block) != 0) {
                continue;  // 跳过过曝或欠曝的图像块
            }

            // 计算质量分数
            float score = computeBlockScore(block);

            // 更新最佳图像索引
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        // 如果没有找到合理曝光的块，选择评分最高的（即使曝光可能不理想）
        if (best_score < 0) {
            for (size_t i = 0; i < list_img.size(); i++) {
                cv::Mat block = list_img[i](roi);
                float score = computeBlockScore(block);
                if (score > best_score) {
                    best_score = score;
                    best_idx = i;
                }
            }
        }

        std::cout << "选择图像 " << best_idx << " 的块 (ROI: ["
            << roi.x << "," << roi.y << "," << roi.width << "," << roi.height
            << "])，得分: " << best_score << std::endl;

        return best_idx;
    }


    // 增强版本：带平滑过渡的图像融合
    cv::Mat fuseImagesWithBlending(const std::vector<cv::Mat>& list_img,
        int blend_width = 10) {
        if (list_img.empty()) {
            throw std::runtime_error("输入图像列表为空");
        }

        cv::Size img_size = list_img[0].size();
        int height = img_size.height;
        int width = img_size.width;

        // 计算每个块的尺寸
        int block_height = height / grid_rows;
        int block_width = width / grid_cols;

        // 创建权重图和结果图
        cv::Mat result = cv::Mat::zeros(height, width, list_img[0].type());
        cv::Mat weight_sum = cv::Mat::zeros(height, width, CV_32F);

        // 为每张图像创建权重图
        std::vector<cv::Mat> weight_maps(list_img.size());
        for (size_t i = 0; i < list_img.size(); i++) {
            weight_maps[i] = cv::Mat::zeros(height, width, CV_32F);
        }

        // 生成块状权重图
        for (int row = 0; row < grid_rows; row++) {
            for (int col = 0; col < grid_cols; col++) {
                // 计算当前块的ROI
                int y_start = row * block_height;
                int x_start = col * block_width;
                int y_end = (row == grid_rows - 1) ? height : (row + 1) * block_height;
                int x_end = (col == grid_cols - 1) ? width : (col + 1) * block_width;

                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);

                // 选择最佳图像（首次计算后缓存，后续复用）
                int best_img_idx = getBestImageForBlock(list_img, roi, row, col);

                // 为最佳图像设置权重为1，其他为0
                weight_maps[best_img_idx](roi).setTo(1.0);
            }
        }

        // 对权重图进行高斯模糊以实现平滑过渡
        int kernel_size = blend_width * 2 + 1;
        for (size_t i = 0; i < weight_maps.size(); i++) {
            cv::GaussianBlur(weight_maps[i], weight_maps[i],
                cv::Size(kernel_size, kernel_size), blend_width / 2.0);
        }

        // 归一化权重图
        cv::Mat total_weight = cv::Mat::zeros(height, width, CV_32F);
        for (size_t i = 0; i < weight_maps.size(); i++) {
            total_weight += weight_maps[i];
        }

        for (size_t i = 0; i < weight_maps.size(); i++) {
            cv::divide(weight_maps[i], total_weight, weight_maps[i]);
        }

        // 加权融合所有图像
        for (size_t i = 0; i < list_img.size(); i++) {
            cv::Mat weighted_img;
            list_img[i].convertTo(weighted_img, CV_32F);

            // 扩展权重图以匹配图像通道数
            std::vector<cv::Mat> weight_channels;
            for (int c = 0; c < weighted_img.channels(); c++) {
                weight_channels.push_back(weight_maps[i]);
            }

            cv::Mat weight_full;
            cv::merge(weight_channels, weight_full);

            // 应用权重并累加到结果
            cv::multiply(weighted_img, weight_full, weighted_img);
            cv::Mat temp_result;
            result.convertTo(temp_result, CV_32F);
            temp_result += weighted_img;
            temp_result.convertTo(result, list_img[0].type());
        }

        return result;
    }



    // ────────────────────────────────────────────────────────────────────
    // 像素级质量权重 + 拉普拉斯金字塔融合（Mertens 风格）
    //
    // 权重公式：W = 对比度 × (饱和度 + base) + epsilon
    //   - 对比度 C：Laplacian 响应绝对值，纹理丰富区域得高分
    //   - 饱和度 S：BGR 三通道标准差，色彩丰富区域得高分
    //   - base = 0.01：灰度图 fallback，确保饱和度为零时对比度仍生效
    //   - 逐像素归一化后，用拉普拉斯金字塔多尺度融合消除拼接痕迹
    // ────────────────────────────────────────────────────────────────────
    cv::Mat fuseImagesWith_laplace(const std::vector<cv::Mat>& list_img,
        int blend_width = 10) {
        if (list_img.empty()) {
            throw std::runtime_error("输入图像列表为空");
        }

        const int num_images = (int)list_img.size();
        const cv::Size img_size = list_img[0].size();
        const int height = img_size.height;
        const int width = img_size.width;

        // ── 1. 统一转为 float32 [0, 1]，避免后续精度问题 ──
        std::vector<cv::Mat> images_f32(num_images);
        for (int i = 0; i < num_images; i++) {
            list_img[i].convertTo(images_f32[i], CV_32F, 1.0 / 255.0);
        }

        // ── 2. 逐像素计算质量权重：W = C × (S + 0.01) + 1e-6 ──
        //     灰度图(S=0) 退化为纯对比度权重，彩色图(S>0) 对比度×饱和度
        std::vector<cv::Mat> weights(num_images);
        cv::Mat sum_weights = cv::Mat::zeros(height, width, CV_32F);

        for (int i = 0; i < num_images; i++) {
            cv::Mat& img = images_f32[i];

            // (a) 对比度 C：灰度图 Laplacian 响应绝对值
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::Mat laplacian;
            cv::Laplacian(gray, laplacian, CV_32F);
            cv::Mat contrast = cv::abs(laplacian);

            // (b) 饱和度 S：BGR 三通道逐像素标准差（彩色图像区分度高）
            cv::Mat saturation(height, width, CV_32F);
            {
                std::vector<cv::Mat> ch(3);
                cv::split(img, ch);
                cv::Mat mean_c = (ch[0] + ch[1] + ch[2]) / 3.0f;
                cv::Mat var_c = cv::Mat::zeros(height, width, CV_32F);
                for (int c = 0; c < 3; c++) {
                    cv::Mat diff = ch[c] - mean_c;
                    var_c += diff.mul(diff);
                }
                var_c /= 3.0f;
                cv::sqrt(var_c, saturation);
            }

            // W = C × (S + 0.01) + 1e-6
            //    S+0.01 保证灰度图也能靠对比度区分，彩色图饱和度主导
            weights[i] = contrast.mul(saturation + 0.01f) + 1e-6f;
            sum_weights += weights[i];
        }

        // ── 3. 逐像素归一化（每个像素位置各图像权重之和 = 1）──
        for (int i = 0; i < num_images; i++) {
            cv::divide(weights[i], sum_weights, weights[i]);
        }

        // ── 4. 拉普拉斯金字塔多尺度融合 ──
        // 金字塔层数由图像尺寸决定（最多 5 层），pyrDown 自带高斯平滑
        int min_dim = std::min(width, height);
        int max_levels = 0;
        while (min_dim > 1) { min_dim /= 2; max_levels++; }
        int num_levels = std::min(max_levels, 5);

        // 每张图像构建：权重高斯金字塔 + 图像拉普拉斯金字塔
        struct PyramidPair {
            std::vector<cv::Mat> gaussian;   // 权重高斯金字塔 (层 0..N)
            std::vector<cv::Mat> laplacian;  // 图像拉普拉斯金字塔 (层 0..N)
        };
        std::vector<PyramidPair> pyramids(num_images);

        for (int i = 0; i < num_images; i++) {
            PyramidPair& pp = pyramids[i];

            // 权重高斯金字塔：逐层 pyrDown 平滑+降采样
            pp.gaussian.resize(num_levels + 1);
            pp.gaussian[0] = weights[i].clone();
            for (int lv = 0; lv < num_levels; lv++) {
                cv::pyrDown(pp.gaussian[lv], pp.gaussian[lv + 1]);
            }

            // 图像拉普拉斯金字塔：L_i = G_i - pyrUp(pyrDown(G_i))
            pp.laplacian.resize(num_levels + 1);
            cv::Mat current = images_f32[i].clone();
            for (int lv = 0; lv < num_levels; lv++) {
                pp.laplacian[lv] = current.clone();
                cv::Mat down;
                cv::pyrDown(current, down);
                cv::Mat up;
                cv::pyrUp(down, up, current.size());
                pp.laplacian[lv] -= up;          // 高频残差
                current = down;                   // 继续向粗层迭代
            }
            pp.laplacian[num_levels] = current;   // 最粗层即低频基底
        }

        // ── 5. 每层金字塔按权重加权融合 ──
        std::vector<cv::Mat> blended_laplacian(num_levels + 1);
        for (int lv = 0; lv <= num_levels; lv++) {
            blended_laplacian[lv] = cv::Mat::zeros(
                pyramids[0].laplacian[lv].size(), CV_32FC3);

            for (int i = 0; i < num_images; i++) {
                // 单通道权重扩展为三通道
                cv::Mat weight_3ch;
                cv::Mat weight = pyramids[i].gaussian[lv];
                std::vector<cv::Mat> wch(3, weight);
                cv::merge(wch, weight_3ch);

                // 加权累加
                cv::Mat contrib;
                cv::multiply(pyramids[i].laplacian[lv], weight_3ch, contrib);
                blended_laplacian[lv] += contrib;
            }
        }

        // ── 6. 折叠金字塔重建全分辨率图像 ──
        cv::Mat result_f32 = blended_laplacian[num_levels];
        for (int lv = num_levels - 1; lv >= 0; lv--) {
            cv::pyrUp(result_f32, result_f32, blended_laplacian[lv].size());
            result_f32 += blended_laplacian[lv];
        }

        // pyrUp/pyrDown 可能导致尺寸偏差，修正回原始尺寸
        if (result_f32.size() != img_size) {
            cv::resize(result_f32, result_f32, img_size);
        }

        // ── 7. 裁剪 [0, 1] 并转回 uint8 ──
        cv::Mat result;
        result_f32 = cv::max(result_f32, 0.0f);
        result_f32 = cv::min(result_f32, 1.0f);
        result_f32.convertTo(result, CV_8UC3, 255.0);

        return result;
    }

    // 可视化每个块的选择结果
    cv::Mat visualizeSelection(const std::vector<cv::Mat>& list_img) {
        cv::Size img_size = list_img[0].size();
        int height = img_size.height;
        int width = img_size.width;

        // 创建可视化图像（将不同图像的选择用不同颜色标记）
        cv::Mat visualization = cv::Mat::zeros(height, width, CV_8UC3);

        int block_height = height / grid_rows;
        int block_width = width / grid_cols;

        // 定义不同图像索引对应的颜色
        std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0),   // 蓝色 - 图像0
            cv::Scalar(0, 255, 0),   // 绿色 - 图像1
            cv::Scalar(0, 0, 255),   // 红色 - 图像2
            cv::Scalar(255, 255, 0), // 青色 - 图像3
            cv::Scalar(255, 0, 255), // 紫色 - 图像4
            cv::Scalar(0, 255, 255), // 黄色 - 图像5
            cv::Scalar(128, 128, 128) // 灰色 - 其他
        };

        // 分析每个块并标记
        for (int row = 0; row < grid_rows; row++) {
            for (int col = 0; col < grid_cols; col++) {
                int y_start = row * block_height;
                int x_start = col * block_width;
                int y_end = (row == grid_rows - 1) ? height : (row + 1) * block_height;
                int x_end = (col == grid_cols - 1) ? width : (col + 1) * block_width;

                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);
                int best_idx = selectBestImageForBlock(list_img, roi);

                // 获取颜色（如果图像太多，使用灰色）
                cv::Scalar color = (best_idx < (int)colors.size()) ?
                    colors[best_idx] : cv::Scalar(128, 128, 128);

                // 绘制半透明色块
                cv::Mat roi_area = visualization(roi);
                cv::Mat color_mask(roi_area.size(), CV_8UC3, color);
                cv::addWeighted(roi_area, 0.3, color_mask, 0.7, 0, roi_area);

                // 绘制边界和文本
                cv::rectangle(visualization, roi, color, 2);
                std::string text = "Img" + std::to_string(best_idx);
                cv::putText(visualization, text,
                    cv::Point(x_start + 5, y_start + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }

        return visualization;
    }


    // ────────────────────────────────────────────────────────────────────
    // 方法 1: Mertens 融合
    // ────────────────────────────────────────────────────────────────────
    cv::Mat method_mertens(const std::vector<cv::Mat>& images) {
        std::cout << "[方法1] Mertens 多曝光融合 ..." << std::endl;

        cv::Ptr<cv::MergeMertens> merger = cv::createMergeMertens();
        cv::Mat fusion;
        merger->process(images, fusion);  // float32 [0,1]

        cv::Mat result;
        fusion.convertTo(result, CV_8UC3, 255.0);
        return result;
    }



    // ────────────────────────────────────────────────────────────────────
    // 方法 2: Debevec HDR + Reinhard Tone Mapping
    // ────────────────────────────────────────────────────────────────────
    cv::Mat method_debevec(const std::vector<cv::Mat>& images,
                        const std::vector<float>& exposures) {
        std::cout << "[方法2] Debevec CRF + Reinhard Tone Mapping ..." << std::endl;

        // Step 1: 估算相机响应函数 (CRF) 并重建 HDR 辐射图
        cv::Ptr<cv::MergeDebevec> merger = cv::createMergeDebevec();
        cv::Mat hdr;
        merger->process(images, hdr, exposures);

        // Step 2: Tone mapping (Reinhard)
        cv::Ptr<cv::TonemapReinhard> tonemap = cv::createTonemapReinhard(2.2f,  // gamma
                                                                        0.0f,  // intensity
                                                                        0.8f,  // light_adapt
                                                                        0.5f); // color_adapt
        cv::Mat ldr;
        tonemap->process(hdr, ldr);

        cv::Mat result;
        ldr.convertTo(result, CV_8UC3, 255.0);
        return result;
    }


};




#endif // AI_CAMERA_HPP
