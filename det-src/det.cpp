#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <Windows.h>

#else
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#include <cmath>
#include <algorithm>
#include <set>
#include <random>
#include <array>
#include <limits>
#include <numeric>

#include "det.h"
#include <iostream>
#include <fstream>
#include <json.hpp> 
#include <thread>
#include "json.hpp"
#ifdef USE_RDK
#include "yolov11_rdk.h"
#else
#include "yolov11.h"
#endif

#include "INIReader.h"
#include "Log.h"
#include <common.h>

#include "GridFitter.hpp"
#include "PlateTopSelector.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>

#include "cameras/AICamera.hpp"

#ifdef useORB
#include <cameras/OrbbecCamera.hpp>
#endif // REL_DBG

#ifdef useOAK
#include "cameras/OAKCamera.hpp"
#endif // REL_DBG

#ifdef useMVS
#include "cameras/MVSCamera.hpp"
#endif // useMVS

#ifdef useUSB
#include "cameras/USBCamera.hpp"
#endif // useUSB


#include "charuco_detector.hpp"
#include "top_face.hpp"

#define MY_LOG 1

namespace fs = std::filesystem;
using json = nlohmann::json;


//#ifndef M_PI
//const double M_PI = 3.14159265358979323846;
//#endif


const float arc_angle = (2.0 * M_PI) / 360.0;

std::string g_appPath="";
std::string g_paramPath = "";
std::string g_calibrationPath = "";
std::string g_collectPath = "";
std::string g_posPath = "";
std::string LogDir = "";
#ifdef USE_RDK
std::shared_ptr<YOLOv11RDK> g_det_rdk = nullptr;
#define G_DET g_det_rdk
#else
std::shared_ptr<YOLOv11> g_det = nullptr;
#define G_DET g_det
#endif
std::vector<char> g_data;
std::size_t g_data_size = 0;

std::uint32_t g_det_cnt = 0;
int g_save_ret = 1;
int g_save_img_cnt = 10000;
int g_log_num = 10;
int g_flip_h = 0;
int g_flip_v = 0;
int g_flip_soft = 0; // 相机不支持硬件翻转时，回退到软件翻转
int g_save_csv = 0;
int g_min_same_ret = 2;
float g_diff_angle = 0;
int g_anti_clock = 1;
int g_fuse_image = 0;

// mosaic mertens
std::string g_fuse_mode="mertens";
std::string g_hand_eye_mode="";

// 0: 多项式  1: 最近模板比对
int g_adjust_mode = 1;
std::string g_model_name = "xialiao";
std::string g_camera_name = "oak";
int g_resizeKeepAspectRatio = 1;

// 1: 代表正在更新模板，就不要修正最后结果了
int g_update_template = 0;

std::map<int, template_circle> g_map_template;

template_circle g_cur_det_ret;

// std::string g_det_type = "lunzi";
// std::string g_det_type = "plate";
std::string g_det_type = "plate_empty";

CONFIG g_cfg_camera;
std::shared_ptr<AICamera> g_camera = nullptr;
cv::Mat g_bgrImage;
cv::Mat g_depthImage;


cv::Mat g_hand_eye_matrix;
AICameraIntrinsic g_cam_param;
double g_top_face_workpiece_height_mm = std::numeric_limits<double>::quiet_NaN();
cv::Mat g_dist_coeffs;
// 去畸变 后的像素点
vector<Point2f> g_undistortedPoints;

// 检测/显示通路的相机去畸变（cfg.ini [cfg] undistort，1 开启）
int g_undistort_enable = 1;
bool g_undistort_ready = false;      // 内参/畸变系数有效，可以矫正
cv::Mat g_undistort_K;               // 标定文件里的原始内参，不缩放不猜测
cv::Mat g_undistort_D;               // 标定文件里的原始畸变系数
cv::Mat g_undistort_map1;            // CV_16SC2
cv::Mat g_undistort_map2;            // CV_16UC1
cv::Size g_undistort_map_size;       // map 对应的图像尺寸（缓存 key）
// g_bgrImage 当前装的是不是已矫正的图。全局图会被检测、预览、test_mode 读图
// 反复覆盖，谁能用到它（如 estimate_board_distance）必须按这个标志选相机模型。
bool g_bgrImage_undistorted = false;

int g_test_mode = -1;
bool g_clean_mode = false;
bool g_reload_background = false;
vector<int> g_rect;
vector<int> g_rect_lunzi;
int g_confirm = 0;
float g_ref_depth = -1;
// feilun 高度补偿：相机安装倾斜使飞轮在图像左右两侧的等效高度不同，
// 以图像最左/最右为端点做线性补偿，单位 mm，可在 cfg.ini 的 [cfg] 段覆盖
float g_feilun_h_offset_left = -10.0f;
float g_feilun_h_offset_right = 15.0f;
// 多曝光融合参数（xike 模型 screen_centor 检测时使用）
vector<int> g_exposure_list;
int g_exposure_time_det = -1;
int g_cache_slot = 0;
int g_top_capture = 0;
bool g_lunzi_only = false;
float g_lunzi_radius = 0;
int g_exposure_time = -1;
vector<FEILUN_INFO> g_list_feilun;
std::string g_cur_feilun_id;

struct DEPTH_INFO {
    uint32_t  width = 0;
    uint32_t  height = 0;
    float     scale = 0;
    uint32_t  data_len = 0;
    uint16_t* data = nullptr;

    int from_json(json& j)
    {
        try
        {

            if (!j.contains("data_len") ||
                !j.contains("depth_width") ||
                !j.contains("depth_height") ||
                !j.contains("depth_scale")
                ) {
                return -1;
            }

            j["data_len"].get_to(data_len);
            j["depth_width"].get_to(width);
            j["depth_height"].get_to(height);
            j["depth_scale"].get_to(scale);

        }
        catch (...) { return -1; }


        return 0;
    }


    int from_depthMat(const cv::Mat& depthMat)
    {
        if (!depthMat.empty()) {
            width = depthMat.cols;
            height = depthMat.rows;
            scale = 1;
            data_len = width * height;
            data = (uint16_t*)depthMat.data;
            return 0;
        }
        else {
            std::cerr << "Failed to get depth frame (is depth stream enabled?)" << std::endl;

            return -1;
        }

    }

#ifdef useORB
    int from_depthFrame(std::shared_ptr<ob::Frame>& depthFrame)
    {
        if (depthFrame) {
            auto depthVideoFrame = depthFrame->as<const ob::DepthFrame>();
            width = depthVideoFrame->getWidth();
            height = depthVideoFrame->getHeight();
            scale = depthVideoFrame->getValueScale();
            data_len = width * height;
            data = (uint16_t*)depthVideoFrame->getData();
            return 0;
        }
        else {
            std::cerr << "Failed to get depth frame (is depth stream enabled?)" << std::endl;

            return -1;
        }
        
    }


    static void saveDepthFrame(const std::shared_ptr<ob::DepthFrame>& depthFrame, const std::string& depthName) {
        std::vector<int> params;
        params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        params.push_back(0);
        params.push_back(cv::IMWRITE_PNG_STRATEGY);
        params.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);

        cv::Mat depthMat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
        cv::imwrite(depthName, depthMat, params);
        std::cout << "Depth saved:" << depthName << std::endl;
    }



#endif // REL_DBG


    void get_world_xy_by_z_old(int img_x, int img_y,
        float& x, float& y, float z)
    {
        if (z > 0) { // 有效的深度点

            // 10. 将像素坐标 (u,v) 和深度值转换为三维坐标 (x,y,z)
            x = ((float)img_x - g_cam_param.cx) * z / g_cam_param.fx;
            y = ((float)img_y - g_cam_param.cy) * z / g_cam_param.fy;
        }
    }

    void get_world_xy_by_z(int img_x, int img_y,
        float& x, float& y, float z)
    {
        if (z > 0) { // 有效的深度点

            // 10. 将像素坐标 (u,v) 和深度值转换为三维坐标 (x,y,z)
            int pos = g_cam_param.width * img_y + img_x;

            if (pos >= 0 && pos < (int)g_undistortedPoints.size())
            {
                float u = g_undistortedPoints[pos].x;
                float v = g_undistortedPoints[pos].y;

                x = u * z;
                y = v * z;
            }
            else
            {
                // 去畸变表未就绪时回退到原始坐标
                get_world_xy_by_z_old(img_x, img_y, x, y, z);
            }
        }
    }
    

    void get_world_z_by_xy(int img_x, int img_y, float& z)
    {
        z = -1;

        // pixel value multiplied by scale is the actual distance value in millimeters
        size_t depth_pos = width * img_y + img_x;

        // 越界了
        if (depth_pos >= data_len) return;

        z = data[depth_pos] * scale;

        if (z>3000) z = -1;
        
    }

    void get_world_xyz(int img_x,int img_y,
        float& x, float& y, float& z)
    {
        x = y = z = -1;

        get_world_z_by_xy(img_x, img_y, z);

        // attention: if the distance is 0, it means that the depth camera cannot detect the object（may be out of detection range）
        std::cout << "x " << img_x << "y " << img_y << "  dist  " << z << " mm away. " << std::endl;

        if (z > 0) { // 有效的深度点

            // 10. 将像素坐标 (u,v) 和深度值转换为三维坐标 (x,y,z)
            get_world_xy_by_z(img_x, img_y,x,y,z);

            std::cout << "3D Coordinate at (" << img_x << ", " << img_y << "): ";
            std::cout << "X=" << x << " mm, Y=" << y << " mm, Z=" << z << " mm" << std::endl;
        }
    }
};



DEPTH_INFO g_depth_info;

using Vec3 = cv::Vec3d;
using Mat4 = cv::Matx44d;

// 纯计算版本：给定相机坐标系目标点(dlt_cam, z=0表示相机平面)，相机居中于该点
// 相机坐标系下的 xyz 到 机器人 基座标系 下的 xyz。
int compute_base_xyz_ByDstCam(Vec3& dlt_cam,
    const Mat4& T_base_camera,
    Vec3& base_xyz)
{
    cv::Mat cam_pose = cv::Mat::zeros(4, 1, CV_64F);
    for (size_t i = 0; i < 3; i++)
    {
        cam_pose.at<double>(i, 0) = dlt_cam[i]/1000.0; // 换成米
    }
    cam_pose.at<double>(3, 0) = 1.0;

    cv::Mat base_pose = T_base_camera.inv() * cam_pose;

    for (size_t i = 0; i < 3; i++)
    {
        base_xyz[i] = base_pose.at<double>(i, 0) * 1000.0; // 转成毫米
    }    

    return 0;
}

struct Circle {
    cv::Point2f center;
    float radius;
    float rmse; // 均方根误差
};

struct PointWithStats {
    cv::Point2f point;
    float distance_error;  // 到圆心的距离误差
    float angle_error;     // 角度误差
    float residual;        // 残差
    bool is_outlier;       // 是否为异常点
};


class PolynomialFit {
private:
    std::vector<double> coefficients;
    int degree;

public:
    PolynomialFit(int deg) : degree(deg) {}

    // 拟合多项式曲线 y = a0 + a1*x + a2*x^2 + ... + an*x^n
    void fit(std::vector <std::pair<float, float>>& v) {

        std::sort(v.begin(), v.end(),
            [](const std::pair<float, float>& a, 
                const std::pair<float, float>& b) {
            return a.second < b.second;
        });


        int n = v.size();
        int m = degree + 1;

        // 构建范德蒙德矩阵
        std::vector<std::vector<double>> A(m, std::vector<double>(m, 0));
        std::vector<double> B(m, 0);

        // 计算矩阵元素
        for (int k = 0; k < n; k++) {
            double x = v[k].first;
            double y = v[k].second;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A[i][j] += std::pow(x, i + j);
                }
                B[i] += y * std::pow(x, i);
            }
        }

        // 高斯消元法求解线性方程组
        coefficients = gaussElimination(A, B);
    }

    // 预测x对应的y值
    double predict(double x) {
        if (coefficients.size() == 0) return x;

        double result = 0;
        for (int i = 0; i <= degree; i++) {
            result += coefficients[i] * std::pow(x, i);
        }
        return result;
    }

    // 获取拟合系数
    std::vector<double> getCoefficients() const {
        return coefficients;
    }

private:
    std::vector<double> gaussElimination(std::vector<std::vector<double>> A,
        std::vector<double> B) {
        int n = A.size();

        // 前向消元
        for (int i = 0; i < n; i++) {
            // 主元归一化
            double pivot = A[i][i];
            for (int j = i; j < n; j++) {
                A[i][j] /= pivot;
            }
            B[i] /= pivot;

            // 消去下方元素
            for (int k = i + 1; k < n; k++) {
                double factor = A[k][i];
                for (int j = i; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                B[k] -= factor * B[i];
            }
        }

        // 回代求解
        std::vector<double> x(n, 0);
        for (int i = n - 1; i >= 0; i--) {
            x[i] = B[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= A[i][j] * x[j];
            }
        }

        return x;
    }
};


// 选择多项式阶数（通常2-3阶足够）
PolynomialFit fitter(3);

// 计算两个三维点之间的欧几里得距离
float calculateDistance(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/*
std::string GetLocalDate() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d");
    std::string time_str = ss.str();
    return time_str;
}


std::time_t StringToDatetime(std::string str)
{
    char* cha = (char*)str.data();             // 将string转换成char*。
    tm tm_;                                    // 定义tm结构体。
    int year, month, day, hour, minute, second;// 定义时间的各个int临时变量。
    sscanf(cha, "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second);// 将string存储的日期时间，转换为int临时变量。
    tm_.tm_year = year - 1900;                 // 年，由于tm结构体存储的是从1900年开始的时间，所以tm_year为int临时变量减去1900。
    tm_.tm_mon = month - 1;                    // 月，由于tm结构体的月份存储范围为0-11，所以tm_mon为int临时变量减去1。
    tm_.tm_mday = day;                         // 日。
    tm_.tm_hour = hour;                        // 时。
    tm_.tm_min = minute;                       // 分。
    tm_.tm_sec = second;                       // 秒。
    tm_.tm_isdst = 0;                          // 非夏令时。
    std::time_t t_ = mktime(&tm_);                  // 将tm结构体转换成time_t格式。
    return t_;                                 // 返回值。

}
*/
std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

class CircleFitterWithOutlierRemoval {
public:

    // 方法1: 使用RANSAC算法拟合圆并检测异常点
    static Circle fitCircleRANSAC(const std::vector<cv::Point2f>& points,
        std::vector<bool>& inliers,
        int max_iterations = 100,
        float distance_threshold = 3.0f) {
        Circle best_circle;
        int best_inliers_count = 0;
        std::vector<bool> best_inlier_mask(points.size(), false);

        if (points.size() < 3) {
            std::cerr << "Error: Need at least 3 points to fit a circle" << std::endl;
            return best_circle;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points.size() - 1);

        for (int iter = 0; iter < max_iterations; ++iter) {
            // 随机选择3个点
            std::set<int> selected_indices;
            std::vector<cv::Point2f> sample_points;

            while (sample_points.size() < 3) {
                int idx = dis(gen);
                if (selected_indices.find(idx) == selected_indices.end()) {
                    selected_indices.insert(idx);
                    sample_points.push_back(points[idx]);
                }
            }

            // 用3个点拟合圆
            Circle circle = fitCircleFrom3Points(sample_points);
            if (circle.radius <= 0) continue;

            // 计算内点数量
            std::vector<bool> current_inlier_mask(points.size(), false);
            int current_inliers = 0;

            for (size_t i = 0; i < points.size(); ++i) {
                float dx = points[i].x - circle.center.x;
                float dy = points[i].y - circle.center.y;
                float distance = std::sqrt(dx * dx + dy * dy);
                float error = std::abs(distance - circle.radius);

                if (error < distance_threshold) {
                    current_inlier_mask[i] = true;
                    current_inliers++;
                }
            }

            // 更新最佳拟合
            if (current_inliers > best_inliers_count) {
                best_inliers_count = current_inliers;
                best_circle = circle;
                best_inlier_mask = current_inlier_mask;
            }
        }

        // 使用所有内点重新拟合
        if (best_inliers_count > 0) {
            std::vector<cv::Point2f> inlier_points;
            for (size_t i = 0; i < points.size(); ++i) {
                if (best_inlier_mask[i]) {
                    inlier_points.push_back(points[i]);
                }
            }
            best_circle = fitCircleLeastSquares(inlier_points);
        }

        inliers = best_inlier_mask;
        return best_circle;
    }

    // 方法2: 基于统计的异常点检测
    static Circle fitCircleWithStatisticalOutlierRemoval(const std::vector<cv::Point2f>& points,
        std::vector<PointWithStats>& point_stats,
        float outlier_threshold = 2.0f) {
        // 首先拟合圆
        Circle circle = fitCircleLeastSquares(points);

        // 计算每个点的统计信息
        point_stats.clear();
        std::vector<float> residuals;

        for (const auto& point : points) {
            PointWithStats stat;
            stat.point = point;

            // 计算到圆心的距离
            float dx = point.x - circle.center.x;
            float dy = point.y - circle.center.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            stat.distance_error = std::abs(distance - circle.radius);
            stat.residual = stat.distance_error;
            residuals.push_back(stat.residual);

            point_stats.push_back(stat);
        }

        // 计算残差的统计信息
        float mean_residual = 0;
        for (float residual : residuals) {
            mean_residual += residual;
        }
        mean_residual /= residuals.size();

        float std_residual = 0;
        for (float residual : residuals) {
            std_residual += (residual - mean_residual) * (residual - mean_residual);
        }
        std_residual = std::sqrt(std_residual / residuals.size());

        // 标记异常点
        for (auto& stat : point_stats) {
            stat.is_outlier = (std::abs(stat.residual - mean_residual) > outlier_threshold * std_residual);
        }

        // 使用非异常点重新拟合
        std::vector<cv::Point2f> inlier_points;
        for (const auto& stat : point_stats) {
            if (!stat.is_outlier) {
                inlier_points.push_back(stat.point);
            }
        }

        if (inlier_points.size() >= 3) {
            circle = fitCircleLeastSquares(inlier_points);
        }

        return circle;
    }

    // 方法3: 基于角度分布的异常点检测（适用于等间距点）
    static Circle fitCircleWithAngularOutlierRemoval(const std::vector<cv::Point2f>& points,
        std::vector<PointWithStats>& point_stats,
        cv::Point2f expected_center = cv::Point2f(-1, -1)) {
        // 如果没有提供预期中心点，先用所有点拟合一个初始圆
        Circle initial_circle;
        if (expected_center.x < 0) {
            initial_circle = fitCircleLeastSquares(points);
            expected_center = initial_circle.center;
        }

        // 计算每个点的角度
        std::vector<std::pair<float, int>> angles_with_index;
        for (size_t i = 0; i < points.size(); ++i) {
            float angle = std::atan2(points[i].y - expected_center.y,
                points[i].x - expected_center.x);
            if (angle < 0) angle += 2 * M_PI;
            angles_with_index.push_back({ angle, static_cast<int>(i) });
        }

        // 按角度排序
        std::sort(angles_with_index.begin(), angles_with_index.end());

        // 计算统计信息
        point_stats.clear();
        point_stats.resize(points.size());

        // 预期角度间隔（36度 = π/5弧度）
        float expected_angle_step = 2 * M_PI / 10;  // 假设10个点

        for (size_t i = 0; i < angles_with_index.size(); ++i) {
            int idx = angles_with_index[i].second;
            float current_angle = angles_with_index[i].first;

            PointWithStats stat;
            stat.point = points[idx];

            // 计算理论角度
            float expected_angle = i * expected_angle_step;

            // 计算角度误差
            float angle_error = std::abs(current_angle - expected_angle);
            if (angle_error > M_PI) {
                angle_error = 2 * M_PI - angle_error;
            }
            stat.angle_error = angle_error;

            // 计算距离误差
            float dx = points[idx].x - expected_center.x;
            float dy = points[idx].y - expected_center.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            float radius = initial_circle.radius > 0 ? initial_circle.radius : distance;
            stat.distance_error = std::abs(distance - radius);

            // 综合残差
            stat.residual = stat.angle_error + stat.distance_error * 0.1f; // 调整权重
            stat.is_outlier = false; // 先标记为非异常点

            point_stats[idx] = stat;
        }

        // 使用所有点拟合最终圆
        Circle final_circle = fitCircleLeastSquares(points);
        final_circle.rmse = calculateRMSE(points, final_circle);

        return final_circle;
    }

    // 方法4: 综合异常点检测
    static Circle fitCircleComprehensive(const std::vector<cv::Point2f>& points,
        std::vector<PointWithStats>& point_stats,
        float distance_threshold = 2.0f,
        float angle_threshold = 0.2f) {
        // 步骤1: 使用RANSAC初步拟合
        std::vector<bool> ransac_inliers;
        Circle circle = fitCircleRANSAC(points, ransac_inliers, 100, distance_threshold);

        // 步骤2: 计算详细的统计信息
        point_stats.clear();
        point_stats.resize(points.size());

        float expected_angle_step = 2 * M_PI / points.size();

        for (size_t i = 0; i < points.size(); ++i) {
            PointWithStats stat;
            stat.point = points[i];

            // 距离误差
            float dx = points[i].x - circle.center.x;
            float dy = points[i].y - circle.center.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            stat.distance_error = std::abs(distance - circle.radius);

            // 角度误差
            float angle = std::atan2(dy, dx);
            if (angle < 0) angle += 2 * M_PI;

            // 找到最近的理论角度
            int nearest_index = static_cast<int>(std::round(angle / expected_angle_step));
            float expected_angle = nearest_index * expected_angle_step;
            float angle_error = std::abs(angle - expected_angle);
            if (angle_error > M_PI) {
                angle_error = 2 * M_PI - angle_error;
            }
            stat.angle_error = angle_error;

            // 综合残差
            stat.residual = stat.distance_error + stat.angle_error * 10.0f;

            // 异常点判断
            stat.is_outlier = (stat.distance_error > distance_threshold) ||
                (stat.angle_error > angle_threshold);

            point_stats[i] = stat;
        }

        // 步骤3: 使用非异常点重新拟合
        std::vector<cv::Point2f> inlier_points;
        for (size_t i = 0; i < point_stats.size(); ++i) {
            if (!point_stats[i].is_outlier) {
                inlier_points.push_back(points[i]);
            }
        }

        if (inlier_points.size() >= 3) {
            circle = fitCircleLeastSquares(inlier_points);
            circle.rmse = calculateRMSE(inlier_points, circle);
        }

        return circle;
    }

private:
    // 从3个点拟合圆
    static Circle fitCircleFrom3Points(const std::vector<cv::Point2f>& points) {
        Circle result;
        if (points.size() < 3) return result;

        cv::Point2f p1 = points[0];
        cv::Point2f p2 = points[1];
        cv::Point2f p3 = points[2];

        float x1 = p1.x, y1 = p1.y;
        float x2 = p2.x, y2 = p2.y;
        float x3 = p3.x, y3 = p3.y;

        float a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;
        if (std::abs(a) < 1e-10) return result; // 三点共线

        float b = (x1 * x1 + y1 * y1) * (y3 - y2) +
            (x2 * x2 + y2 * y2) * (y1 - y3) +
            (x3 * x3 + y3 * y3) * (y2 - y1);

        float c = (x1 * x1 + y1 * y1) * (x2 - x3) +
            (x2 * x2 + y2 * y2) * (x3 - x1) +
            (x3 * x3 + y3 * y3) * (x1 - x2);

        float d = (x1 * x1 + y1 * y1) * (x3 * y2 - x2 * y3) +
            (x2 * x2 + y2 * y2) * (x1 * y3 - x3 * y1) +
            (x3 * x3 + y3 * y3) * (x2 * y1 - x1 * y2);

        result.center.x = -b / (2 * a);
        result.center.y = -c / (2 * a);
        result.radius = std::sqrt(b * b + c * c - 4 * a * d) / (2 * std::abs(a));

        return result;
    }

    // 最小二乘法拟合圆
    static Circle fitCircleLeastSquares(const std::vector<cv::Point2f>& points) {
        Circle result;
        if (points.size() < 3) return result;

        // 构建线性方程组 Ax = b
        cv::Mat A(points.size(), 3, CV_32F);
        cv::Mat b(points.size(), 1, CV_32F);

        for (size_t i = 0; i < points.size(); ++i) {
            float x = points[i].x;
            float y = points[i].y;
            A.at<float>(i, 0) = 2 * x;
            A.at<float>(i, 1) = 2 * y;
            A.at<float>(i, 2) = -1;
            b.at<float>(i, 0) = x * x + y * y;
        }

        // 求解线性方程组 A^T * A * x = A^T * b
        cv::Mat AtA = A.t() * A;
        cv::Mat Atb = A.t() * b;

        cv::Mat solution;
        cv::solve(AtA, Atb, solution, cv::DECOMP_SVD);

        // 提取圆心和半径
        float cx = solution.at<float>(0, 0);
        float cy = solution.at<float>(1, 0);
        float c = solution.at<float>(2, 0);

        result.center = cv::Point2f(cx, cy);
        result.radius = std::sqrt(cx * cx + cy * cy - c);

        return result;
    }

    // 计算均方根误差
    static float calculateRMSE(const std::vector<cv::Point2f>& points, const Circle& circle) {
        float error_sum = 0;
        for (const auto& point : points) {
            float dx = point.x - circle.center.x;
            float dy = point.y - circle.center.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            float error = distance - circle.radius;
            error_sum += error * error;
        }
        return std::sqrt(error_sum / points.size());
    }

public:
    // 可视化函数
    static void visualizeResults(const std::vector<cv::Point2f>& points,
        const std::vector<PointWithStats>& point_stats,
        const Circle& circle,
        const std::string& window_name = "Circle Fitting") {
        // 创建图像
        cv::Mat img(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

        if (points.empty()) return;

        // 计算边界框
        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;

        for (const auto& pt : points) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }

        // 添加边距
        float margin = 50;
        min_x -= margin; max_x += margin;
        min_y -= margin; max_y += margin;

        // 缩放因子
        float scale_x = (img.cols - 100) / (max_x - min_x);
        float scale_y = (img.rows - 100) / (max_y - min_y);
        float scale = std::min(scale_x, scale_y);

        // 坐标转换函数
        auto transform_point = [&](const cv::Point2f& pt) -> cv::Point {
            int x = 50 + static_cast<int>((pt.x - min_x) * scale);
            int y = 50 + static_cast<int>((pt.y - min_y) * scale);
            return cv::Point(x, y);
        };

        // 绘制圆
        if (circle.radius > 0) {
            cv::Point center = transform_point(circle.center);
            int radius = static_cast<int>(circle.radius * scale);
            if (radius > 0 && radius < 10000) {
                cv::circle(img, center, radius, cv::Scalar(0, 255, 0), 2);
                cv::circle(img, center, 3, cv::Scalar(255, 0, 0), -1);
            }
        }

        // 绘制点
        for (size_t i = 0; i < points.size(); ++i) {
            cv::Point pt_img = transform_point(points[i]);
            cv::Scalar color = (i < point_stats.size() && point_stats[i].is_outlier) ?
                cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 0);
            int radius = (i < point_stats.size() && point_stats[i].is_outlier) ? 5 : 3;
            cv::circle(img, pt_img, radius, color, -1);

            // 添加编号
            std::string label = std::to_string(i);
            cv::putText(img, label, cv::Point(pt_img.x + 5, pt_img.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }

        // 显示信息
        std::string info = cv::format("Center: (%.2f, %.2f), Radius: %.2f",
            circle.center.x, circle.center.y, circle.radius);
        cv::putText(img, info, cv::Point(10, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        std::string rmse_info = cv::format("RMSE: %.2f", circle.rmse);
        cv::putText(img, rmse_info, cv::Point(10, 40),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        cv::imshow(window_name, img);
        cv::waitKey(0);
    }
};


class PointCorrector {
public:

    // 方法1: 基于相邻点的线性插值修正
    static std::vector<cv::Point2f> correctOutlierPointsLinear(
        const std::vector<cv::Point2f>& points,
        const std::vector<bool>& is_outlier,
        const Circle& circle) {

        std::vector<cv::Point2f> corrected_points = points;

        for (size_t i = 0; i < points.size(); ++i) {
            if (is_outlier[i]) {
                // 找到相邻的正常点
                int prev_idx = findPrevValidPoint(i, is_outlier);
                int next_idx = findNextValidPoint(i, is_outlier, points.size());

                if (prev_idx >= 0 && next_idx >= 0) {
                    // 线性插值修正
                    corrected_points[i] = interpolateBetweenPoints(
                        points[prev_idx], points[next_idx],
                        static_cast<float>(i - prev_idx) / (next_idx - prev_idx));
                }
                else {
                    // 如果找不到相邻点，使用圆周约束修正
                    corrected_points[i] = projectPointToCircle(points[i], circle);
                }
            }
        }

        return corrected_points;
    }

    // 方法2: 基于圆周和角度的修正
    static std::vector<cv::Point2f> correctOutlierPointsCircular(
        const std::vector<cv::Point2f>& points,
        const std::vector<bool>& is_outlier,
        const Circle& circle) {

        std::vector<cv::Point2f> corrected_points = points;

        // 计算所有正常点的角度
        std::vector<std::pair<float, int>> valid_angles;
        for (size_t i = 0; i < points.size(); ++i) {
            if (!is_outlier[i]) {
                float angle = calculateAngle(points[i], circle.center);
                valid_angles.push_back({ angle, static_cast<int>(i) });
            }
        }

        // 按角度排序
        std::sort(valid_angles.begin(), valid_angles.end());

        for (size_t i = 0; i < points.size(); ++i) {
            if (is_outlier[i]) {
                // 计算理论角度（等间距分布）
                float theoretical_angle = 2 * M_PI * i / points.size();

                // 在正常点中找到最近的角度
                int nearest_idx = findNearestValidPointByAngle(
                    theoretical_angle, valid_angles, points.size());

                if (nearest_idx >= 0) {
                    // 使用相邻正常点的角度差来估计正确位置
                    corrected_points[i] = estimatePointPosition(
                        points, is_outlier, i, circle, theoretical_angle);
                }
                else {
                    // 直接投影到圆周上
                    corrected_points[i] = projectPointToCircle(points[i], circle);
                }
            }
        }

        return corrected_points;
    }

    // 方法3: 基于多项式拟合的修正
    static std::vector<cv::Point2f> correctOutlierPointsPolynomial(
        const std::vector<cv::Point2f>& points,
        const std::vector<bool>& is_outlier,
        const Circle& circle) {

        std::vector<cv::Point2f> corrected_points = points;

        for (size_t i = 0; i < points.size(); ++i) {
            if (is_outlier[i]) {
                // 获取前后几个正常点
                std::vector<cv::Point2f> neighbor_points;
                std::vector<int> neighbor_indices;

                // 向前查找
                for (int j = 1; j <= 3; ++j) {
                    int idx = (i - j + points.size()) % points.size();
                    if (!is_outlier[idx]) {
                        neighbor_points.push_back(points[idx]);
                        neighbor_indices.push_back(idx);
                    }
                }

                // 向后查找
                for (int j = 1; j <= 3; ++j) {
                    int idx = (i + j) % points.size();
                    if (!is_outlier[idx]) {
                        neighbor_points.push_back(points[idx]);
                        neighbor_indices.push_back(idx);
                    }
                }

                if (neighbor_points.size() >= 3) {
                    // 使用多项式拟合修正
                    corrected_points[i] = polynomialInterpolation(
                        neighbor_points, neighbor_indices, i, circle);
                }
                else {
                    // 使用圆周投影
                    corrected_points[i] = projectPointToCircle(points[i], circle);
                }
            }
        }

        return corrected_points;
    }

    // 方法4: 综合修正方法
    static std::vector<cv::Point2f> correctOutlierPointsComprehensive(
        const std::vector<cv::Point2f>& points,
        const std::vector<bool>& is_outlier,
        const Circle& circle) {

        std::vector<cv::Point2f> corrected_points = points;

        for (size_t i = 0; i < points.size(); ++i) {
            if (is_outlier[i]) {
                cv::Point2f corrected_point;

                // 尝试多种修正方法
                std::vector<cv::Point2f> candidates;

                // 方法1: 相邻点线性插值
                int prev_idx = findPrevValidPoint(i, is_outlier);
                int next_idx = findNextValidPoint(i, is_outlier, points.size());
                if (prev_idx >= 0 && next_idx >= 0) {
                    cv::Point2f linear_interp = interpolateBetweenPoints(
                        points[prev_idx], points[next_idx],
                        static_cast<float>(i - prev_idx) / (next_idx - prev_idx));
                    candidates.push_back(projectPointToCircle(linear_interp, circle));
                }

                // 方法2: 理论角度位置
                float theoretical_angle = 2 * M_PI * i / points.size();
                cv::Point2f angle_point = pointOnCircle(circle, theoretical_angle);
                candidates.push_back(angle_point);

                // 方法3: 基于相邻正常点的平均位置
                cv::Point2f avg_point = calculateAverageNeighborPosition(
                    points, is_outlier, i, circle);
                if (isValidPoint(avg_point)) {
                    candidates.push_back(projectPointToCircle(avg_point, circle));
                }

                // 选择最佳候选点（最接近理论位置的）
                corrected_point = selectBestCandidate(candidates, theoretical_angle, circle);
                corrected_points[i] = corrected_point;
            }
        }

        return corrected_points;
    }

public:
    // 找到前一个有效的点
    static int findPrevValidPoint(int current_idx, const std::vector<bool>& is_outlier) {
        for (int i = current_idx - 1; i >= 0; --i) {
            if (!is_outlier[i]) {
                return i;
            }
        }
        // 循环查找
        for (int i = is_outlier.size() - 1; i > current_idx; --i) {
            if (!is_outlier[i]) {
                return i;
            }
        }
        return -1;
    }

    // 找到后一个有效的点
    static int findNextValidPoint(int current_idx, const std::vector<bool>& is_outlier, int size) {
        for (int i = current_idx + 1; i < size; ++i) {
            if (!is_outlier[i]) {
                return i;
            }
        }
        // 循环查找
        for (int i = 0; i < current_idx; ++i) {
            if (!is_outlier[i]) {
                return i;
            }
        }
        return -1;
    }

    // 线性插值
    static cv::Point2f interpolateBetweenPoints(const cv::Point2f& p1, const cv::Point2f& p2, float ratio) {
        return cv::Point2f(
            p1.x + ratio * (p2.x - p1.x),
            p1.y + ratio * (p2.y - p1.y)
        );
    }

    // 计算点的角度
    static float calculateAngle(const cv::Point2f& point, const cv::Point2f& center) {
        float angle = std::atan2(point.y - center.y, point.x - center.x);
        if (angle < 0) angle += 2 * M_PI;
        return angle;
    }

    // 将点投影到圆周上
    static cv::Point2f projectPointToCircle(const cv::Point2f& point, const Circle& circle) {
        float dx = point.x - circle.center.x;
        float dy = point.y - circle.center.y;
        float distance = std::sqrt(dx * dx + dy * dy);

        if (distance < 1e-10) {
            // 如果点与圆心重合，选择一个默认方向
            return cv::Point2f(circle.center.x + circle.radius, circle.center.y);
        }

        float scale = circle.radius / distance;
        return cv::Point2f(
            circle.center.x + dx * scale,
            circle.center.y + dy * scale
        );
    }

    // 在圆周上指定角度的点
    static cv::Point2f pointOnCircle(const Circle& circle, float angle_arc) {
        return cv::Point2f(
            circle.center.x + circle.radius * std::cos(angle_arc),
            circle.center.y + circle.radius * std::sin(angle_arc)
        );
    }

    // 在圆周上指定角度的点
    static cv::Point2f pointOnCircle_angle(const Circle& circle, float angle) {
        // 把角度转成弧度
        double arc = arc_angle * angle;
        return cv::Point2f(
            circle.center.x + circle.radius * std::cos(arc),
            circle.center.y + circle.radius * std::sin(arc)
        );
    }

    // 找到最近的有效点（按角度）
    static int findNearestValidPointByAngle(float target_angle,
        const std::vector<std::pair<float, int>>& valid_angles,
        int total_points) {
        if (valid_angles.empty()) return -1;

        float min_diff = 2 * M_PI;
        int nearest_idx = -1;

        for (const auto& angle_idx : valid_angles) {
            float diff = std::abs(angle_idx.first - target_angle);
            if (diff > M_PI) diff = 2 * M_PI - diff;  // 考虑角度循环
            if (diff < min_diff) {
                min_diff = diff;
                nearest_idx = angle_idx.second;
            }
        }

        return nearest_idx;
    }

    // 估计点的位置
    static cv::Point2f estimatePointPosition(const std::vector<cv::Point2f>& points,
        const std::vector<bool>& is_outlier,
        int outlier_idx,
        const Circle& circle,
        float theoretical_angle) {
        // 使用相邻正常点来估计
        int prev_idx = findPrevValidPoint(outlier_idx, is_outlier);
        int next_idx = findNextValidPoint(outlier_idx, is_outlier, points.size());

        if (prev_idx >= 0 && next_idx >= 0) {
            // 计算角度差
            float prev_angle = calculateAngle(points[prev_idx], circle.center);
            float next_angle = calculateAngle(points[next_idx], circle.center);

            // 处理角度跨越0的情况
            if (next_angle < prev_angle) next_angle += 2 * M_PI;
            if (theoretical_angle < prev_angle) theoretical_angle += 2 * M_PI;

            float ratio = (theoretical_angle - prev_angle) / (next_angle - prev_angle);
            cv::Point2f interpolated = interpolateBetweenPoints(points[prev_idx], points[next_idx], ratio);
            return projectPointToCircle(interpolated, circle);
        }
        else {
            return pointOnCircle(circle, theoretical_angle);
        }
    }

    // 多项式插值
    static cv::Point2f polynomialInterpolation(const std::vector<cv::Point2f>& neighbor_points,
        const std::vector<int>& neighbor_indices,
        int target_idx,
        const Circle& circle) {
        // 简化的多项式插值：使用最近的3个点
        if (neighbor_points.size() >= 3) {
            // 计算质心
            cv::Point2f centroid(0, 0);
            for (const auto& pt : neighbor_points) {
                centroid.x += pt.x;
                centroid.y += pt.y;
            }
            centroid.x /= neighbor_points.size();
            centroid.y /= neighbor_points.size();

            return projectPointToCircle(centroid, circle);
        }
        return pointOnCircle(circle, 2 * M_PI * target_idx / neighbor_indices.size());
    }

    // 计算相邻正常点的平均位置
    static cv::Point2f calculateAverageNeighborPosition(const std::vector<cv::Point2f>& points,
        const std::vector<bool>& is_outlier,
        int outlier_idx,
        const Circle& circle) {
        cv::Point2f sum(0, 0);
        int count = 0;

        // 查找相邻的正常点
        for (int i = 1; i <= 2; ++i) {
            int prev_idx = (outlier_idx - i + points.size()) % points.size();
            int next_idx = (outlier_idx + i) % points.size();

            if (!is_outlier[prev_idx]) {
                sum.x += points[prev_idx].x;
                sum.y += points[prev_idx].y;
                count++;
            }

            if (!is_outlier[next_idx]) {
                sum.x += points[next_idx].x;
                sum.y += points[next_idx].y;
                count++;
            }
        }

        if (count > 0) {
            sum.x /= count;
            sum.y /= count;
            return sum;
        }

        return cv::Point2f(-1, -1);  // 无效点
    }

    // 检查点是否有效
    static bool isValidPoint(const cv::Point2f& point) {
        return point.x >= 0 && point.y >= 0;
    }

    // 选择最佳候选点
    static cv::Point2f selectBestCandidate(const std::vector<cv::Point2f>& candidates,
        float theoretical_angle,
        const Circle& circle) {
        if (candidates.empty()) {
            return pointOnCircle(circle, theoretical_angle);
        }

        if (candidates.size() == 1) {
            return projectPointToCircle(candidates[0], circle);
        }

        // 选择最接近理论角度的点
        float best_diff = 2 * M_PI;
        cv::Point2f best_candidate = candidates[0];

        for (const auto& candidate : candidates) {
            cv::Point2f projected = projectPointToCircle(candidate, circle);
            float angle = calculateAngle(projected, circle.center);
            float diff = std::abs(angle - theoretical_angle);
            if (diff > M_PI) diff = 2 * M_PI - diff;

            if (diff < best_diff) {
                best_diff = diff;
                best_candidate = projected;
            }
        }

        return best_candidate;
    }

public:
    // 可视化修正结果
    static void visualizeCorrection(const std::vector<cv::Point2f>& original_points,
        const std::vector<cv::Point2f>& corrected_points,
        const std::vector<bool>& is_outlier,
        const Circle& circle,
        const std::string& window_name = "Point Correction") {
        // 创建图像
        cv::Mat img(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

        if (original_points.empty()) return;

        // 计算边界框
        float min_x = original_points[0].x, max_x = original_points[0].x;
        float min_y = original_points[0].y, max_y = original_points[0].y;

        for (const auto& pt : original_points) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }

        for (const auto& pt : corrected_points) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }

        // 添加边距
        float margin = 50;
        min_x -= margin; max_x += margin;
        min_y -= margin; max_y += margin;

        // 缩放因子
        float scale_x = (img.cols - 100) / (max_x - min_x);
        float scale_y = (img.rows - 100) / (max_y - min_y);
        float scale = std::min(scale_x, scale_y);

        // 坐标转换函数
        auto transform_point = [&](const cv::Point2f& pt) -> cv::Point {
            int x = 50 + static_cast<int>((pt.x - min_x) * scale);
            int y = 50 + static_cast<int>((pt.y - min_y) * scale);
            return cv::Point(x, y);
        };

        // 绘制圆
        if (circle.radius > 0) {
            cv::Point center = transform_point(circle.center);
            int radius = static_cast<int>(circle.radius * scale);
            if (radius > 0 && radius < 10000) {
                cv::circle(img, center, radius, cv::Scalar(0, 255, 0), 2);
                cv::circle(img, center, 3, cv::Scalar(255, 0, 0), -1);
            }
        }

        // 绘制原始点和修正点
        for (size_t i = 0; i < original_points.size(); ++i) {
            cv::Point original_img = transform_point(original_points[i]);
            cv::Point corrected_img = transform_point(corrected_points[i]);

            // 原始点（红色）
            cv::circle(img, original_img, 4, cv::Scalar(0, 0, 255), -1);

            if (is_outlier[i]) {
                // 修正点（蓝色）
                cv::circle(img, corrected_img, 4, cv::Scalar(255, 0, 0), -1);

                // 绘制修正箭头
                cv::arrowedLine(img, original_img, corrected_img,
                    cv::Scalar(255, 100, 0), 1, cv::LINE_AA, 0, 0.1);

                // 添加编号
                std::string label = std::to_string(i);
                cv::putText(img, label, cv::Point(original_img.x + 5, original_img.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
            }
            else {
                // 正常点（绿色）
                cv::circle(img, corrected_img, 4, cv::Scalar(0, 255, 0), -1);
            }
        }

        // 显示图例
        cv::putText(img, "Red: Original Points", cv::Point(10, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        cv::putText(img, "Green: Normal Points", cv::Point(10, 40),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(img, "Blue: Corrected Points", cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

        cv::imshow(window_name, img);
        cv::waitKey(0);
    }
};


template <typename T> T calculateDistance_core(T x1, T y1, T x2, T y2) {
    return (T)std::hypot(x2 - x1, y2 - y1);
}



// 静态成员定义（2 套缓存对应不同拍摄位置）
std::vector<int> MultiExposureFusion::s_cached_selection_[2];
int MultiExposureFusion::s_cached_grid_rows_[2] = {0, 0};
int MultiExposureFusion::s_cached_grid_cols_[2] = {0, 0};
int MultiExposureFusion::s_cached_num_images_[2] = {0, 0};

int get_new_camera_image_by_adjust_exposure(std::vector<cv::Mat>& list_img, int dark, int bright)
{
    if (0== dark&&0== bright) return -1;

    int minUs=50; int maxUs = 1999;

    g_camera->getExposureRange(minUs, maxUs);
    int exposureUs = g_camera->getColorExposureTime();

    std::vector<int> exposureValuesUs;
    if (dark>0)
    {
        int new_us = (minUs + exposureUs) / 2;
        exposureValuesUs.push_back(new_us);
    }
    if (bright>0)
    {
        int new_us = (maxUs + exposureUs) / 2;
        exposureValuesUs.push_back(new_us);
    }

    list_img = g_camera->captureMultiExposureImages(exposureValuesUs);

    return 0;
}
// 分析块的曝光情况，获取最佳图像
int get_good_camera_image_by_fusion(cv::Mat& bgrImage)
{
    // 创建融合器
    MultiExposureFusion fusion;

    fusion.grid_rows = 5;
    fusion.grid_cols = 5;

    // 检测图中各个块的曝光情况
    std::map<int, std::vector<std::pair<int, int>>> centor_vals;
    std::map<int, std::vector<std::pair<int, int>>> edge_vals;

    fusion.detectExposure(bgrImage, centor_vals, edge_vals);

    int dark = 0;
    int bright = 0;
    if (centor_vals[1].size() > 0)
    {
        dark = 1;
    }
    else if (centor_vals[-1].size() > 0)
    {
        bright = 1;
    }

    if (edge_vals[1].size() > 2)
    {
        dark = 1;
    }
    else if (edge_vals[-1].size() > 2)
    {
        bright = 1;
    }

    // 无需调节
    if (0==dark && 0==bright) return 0;

    std::vector<cv::Mat> list_img;

    get_new_camera_image_by_adjust_exposure(list_img,dark,bright);

    if (list_img.size() == 0) return 0;

    list_img.push_back(bgrImage);

    bgrImage = fusion.fuseImages(list_img);

    return 0;
}


// 分析块的曝光情况，获取最佳图像
int get_good_camera_image(cv::Mat& bgrImage, cv::Mat& depthImage, int fusion=1)
{
    int ret = g_camera->captureOneFrameWithDepth(bgrImage, depthImage);

    if (bgrImage.empty() || 
        ((g_cfg_camera.enableDepth)&&depthImage.empty()) )
    {
        return -1;
    }

    if (fusion==0)
    {
        // 无需 fusion
        return 0;
    }
    

    return get_good_camera_image_by_fusion(bgrImage);

}



bool GetBasePathInfo() {

    char szPath[2048] = { 0 };
    char* p = szPath;
#if WIN32
    GetModuleFileNameA(NULL, szPath, sizeof(szPath));
    char split = '\\';
#else
    readlink("/proc/self/exe", szPath, sizeof(szPath));
    char split = '/';
#endif

    while (strchr(p, split)) //找路径分隔符号
    {
        p = strchr(p, split);
        p++;
    }
    char t = *p;
    *p = 0; // 路径 放在 szPath里了
    g_appPath = szPath;

#if WIN32

#else
    if ("" == g_appPath)
    {
        g_appPath = "./";
    }
#endif

    LogDir = g_appPath + "ret_img/det_logs/";

    return true;
}

bool endWith_str(const std::string& str, const std::string& tail) {
    if (str.size() == 0 || tail.size() == 0)
        return false;

    if (str.size() < tail.size())
        return false;

    return str.compare(str.size() - tail.size(), tail.size(), tail) == 0;
}

bool startWith_str(const std::string& str, const std::string& head) {
    if (str.size() == 0 || head.size() == 0)
        return false;

    if (str.size() < head.size())
        return false;

    return str.compare(0, head.size(), head) == 0;
}


static std::string string_format(const char* format, ...) {
#if 1 // 最大长度限制：1024 - 1
    char buff[4096] = { 0 };
    va_list args;
    va_start(args, format);
    vsprintf(buff, format, args);
    va_end(args);
    std::string str(buff);
    return str;
#else // 无长度限制
    va_list args;
    va_start(args, format);
    int count = vsnprintf(NULL, 0, format, args); // 使用vsnprintf：warning C4996; 使用vsnprintf_s：无法自动计算长度
    va_end(args);
    va_start(args, format);
    char* buff = (char*)malloc(count * sizeof(wchar_t));
    vsnprintf(buff, count, format, args);
    va_end(args);
    std::string str(buff, count);
    free(buff);
    return str;
#endif
}


bool overwriteFile(const std::string& filename, const std::string& content) {
    try {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "can not open file: " << filename << "\n";
            return false;
        }

        // 写入数据
        file << content;

        return true;  // 成功写入
    }
    catch (const std::exception& e) {
        std::cerr << "error file i/o  " << e.what() << std::endl;
        return false;
    }

}

bool overwriteFile(const std::string& filename, const char* data, const int& data_len) {
    try {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "can not open file: " << filename << "\n";
            return false;
        }

        // 写入数据
        file.write(data, data_len);

        return true;  // 成功写入
    }
    catch (const std::exception& e) {
        std::cerr << "error file i/o  " << e.what() << std::endl;
        return false;
    }

}

std::string readFirstLine(const std::string& filePath) {
    std::ifstream file(filePath);

    std::string firstLine = "";
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << "\n";
        return firstLine;
    }

    std::getline(file, firstLine);

    return firstLine;
}


// 把当前序号原子写回 ret_img/cnt.txt：先写临时文件并 fsync，再 rename 覆盖。
// 直接截断原文件写的话，掉电/被杀会留下空文件或半截数字，而空文件会让下次
// 启动的序号从 0 重新开始，把已有图片覆盖掉。
// g_appPath 为空时直接返回，避免未 init 的进程把 0 写到当前工作目录。
bool write_det_cnt_file()
{
    if (g_appPath.empty()) return false;

    const std::string ret_img_dir = g_appPath + "ret_img/";
    if (!fs::exists(ret_img_dir)) fs::create_directory(ret_img_dir);

    const std::string cnt_file = ret_img_dir + "cnt.txt";
    const std::string tmp_file = cnt_file + ".tmp";

    {
        std::ofstream ofs(tmp_file, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            std::cerr << "can not open file: " << tmp_file << "\n";
            return false;
        }
        ofs << g_det_cnt;
        ofs.flush();
        if (!ofs) {
            std::cerr << "write file failed: " << tmp_file << "\n";
            return false;
        }
#ifndef _WIN32
        const int fd = ::open(tmp_file.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
#endif
    }

    std::error_code ec;
    fs::rename(tmp_file, cnt_file, ec);
    if (ec) {
        std::cerr << "rename failed: " << tmp_file << " -> " << cnt_file
                  << ": " << ec.message() << "\n";
        return false;
    }
    return true;
}


// 序号递增 + 回绕，并立即落盘，保证磁盘上的序号跟已保存的图片同步
void advance_det_cnt()
{
    g_det_cnt++;
    if (g_det_cnt > static_cast<std::uint32_t>(g_save_img_cnt)) g_det_cnt = 0;
    write_det_cnt_file();
}


/**
 * @brief Setting up Tensorrt logger
*/
#ifndef USE_RDK
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
        {
#ifdef MY_LOG
#else
            std::cerr<<  "[NVIDIA ERROR]\n" << msg << std::endl;
#endif
        }
    }
}logger;
#endif


void save_csv(std::string csv_file, json& j)
{
    // {"angle":-37.39291000366211, "filename" : "00003.jpg", "ret-code" : 0, "ret-msg" : "ok"}
    if (j.contains("angle")&& j.contains("filename"))
    {
        float angle; std::string filename;
        j["angle"].get_to(angle);
        j["filename"].get_to(filename);

        std::ofstream file(csv_file, std::ios::app);

        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << csv_file << std::endl;
            return;
        }

        std::string cur_time = getCurrentTime();
        std::string csv_info = string_format("%f,%s,%s\n", angle, filename.c_str(), cur_time.c_str());

        // 写入数据行
        
        file << csv_info;

        file.close();
    }
}


void make_ret_json(std::string str, char** ret_json, int* json_len)
{
    if (g_data.size() == 0) g_data.resize(2048);

    if (str.length()> g_data.size())
    {
        std::size_t max_size = std::min(str.length()*2, (std::size_t)1024*100);
        g_data.resize(max_size);
    }

    char* pData = g_data.data();
#ifdef _WIN32
    memcpy_s(pData, g_data.size(), str.c_str(), str.size());

#else
    memcpy(pData, str.c_str(), str.size());
#endif    

    if (g_data.size() > str.size()) g_data[str.size()] = 0;

    *ret_json = pData;
    *json_len = str.size();

}

void make_ret_json(json& j, char** ret_json, int* json_len)
{
    if (g_data.size() == 0) g_data.resize(2048);

    std::string str = j.dump();

    make_ret_json(str, ret_json, json_len);

    // LOG
    int err_code = -1;
    if (j.contains("ret-code"))
    {
        j["ret-code"].get_to(err_code);
    }

#ifdef MY_LOG 
    if (0 == err_code)
        util::Log::log("[DET RET]\n  %s", str.c_str());
    else
        util::Log::error("[DET ERROR]\n  %s", str.c_str());

#else
    if (0 == err_code)
        std::cout<<"[DET RET]\n"<< str << std::endl;
    else
        std::cerr << "[DET ERROR]\n"<< str << std::endl;

#endif    


}


bool LoadConfig() {
    bool ret = false;

    // app config
    std::string str_cfg_file = g_appPath + "cfg.ini";

    if (fs::exists(str_cfg_file))
    {
        // load an ini file
        INIReader myIni(str_cfg_file);
        g_model_name = myIni.GetString("cfg", "model_name", "xialiao");
        g_camera_name = myIni.GetString("cfg", "camera_name", "orb");
        g_resizeKeepAspectRatio = myIni.GetInteger("cfg", "resizeKeepAspectRatio", 1);
        g_fuse_image = myIni.GetInteger("cfg", "fuse_image", 0);

        // feilun 高度补偿端点值 (mm)：图像最左侧 / 最右侧
        g_feilun_h_offset_left = (float)myIni.GetReal("cfg", "feilun_h_offset_left", -10.0);
        g_feilun_h_offset_right = (float)myIni.GetReal("cfg", "feilun_h_offset_right", 15.0);

        // mosaic mertens
        g_fuse_mode = myIni.GetString("cfg", "fuse_mode", "mertens");


        // ; 手眼标定模式
        // hand_eye_mode=eye_in_hand       
        g_hand_eye_mode = myIni.GetString("cfg", "hand_eye_mode", "eye_in_hand");
        
        
        g_save_ret = myIni.GetInteger("cfg", "save_ret", 1);
        g_save_img_cnt = myIni.GetInteger("cfg", "save_img_cnt", 10000);        
        g_log_num = myIni.GetInteger("cfg", "log_num", 10);
        g_anti_clock = myIni.GetInteger("cfg", "anti_clock", 1);
        g_exposure_time = myIni.GetInteger("cfg", "exposure_time", -1);
        if (1 != g_anti_clock) g_anti_clock = -1;

        g_adjust_mode = myIni.GetInteger("cfg", "adjust_mode", 1);

        g_flip_h = myIni.GetInteger("cfg", "flip_h", 0);
        g_flip_v = myIni.GetInteger("cfg", "flip_v", 0);

        // 检测/显示通路是否做相机畸变矫正（1 开启）
        g_undistort_enable = myIni.GetInteger("cfg", "undistort", 1);

        g_save_csv = myIni.GetInteger("cfg", "save_csv", 0);
        g_min_same_ret = myIni.GetInteger("cfg", "min_same_ret", 2);
        g_diff_angle = myIni.GetReal("cfg", "diff_angle", 1.5293112);


//# [608.212,   0,      648.738]
//# [0,      608.21,   392.18 ]
//# [0,        0,        1    ]
        // g_cam_param
        ///< Focal length in x direction
        g_cam_param.fx = myIni.GetReal("cam_param", "fx", 608.212);
        ///< Focal length in y direction        
        g_cam_param.fy = myIni.GetReal("cam_param", "fy", 608.21);
        ///< Optical center abscissa        
        g_cam_param.cx = myIni.GetReal("cam_param", "cx", 648.738);
        ///< Optical center ordinate
        g_cam_param.cy = myIni.GetReal("cam_param", "cy", 392.18);
        
        g_cam_param.width = myIni.GetInteger("cam_param", "width", 2448);
        g_cam_param.height = myIni.GetInteger("cam_param", "height", 2048);
    }

    // robot_config.json
    str_cfg_file = g_appPath + "robot_config.json";

    if (fs::exists(str_cfg_file))
    {
        try
        {
            // 1. 创建文件输入流
            std::ifstream file(str_cfg_file);

            // 2. 检查文件是否成功打开（非常重要！）
            if (!file.is_open()) {
                std::cerr << "无法打开文件!" << std::endl;
                return true;
            }

            // 3. 解析文件内容到 json 对象（两种方法任选其一）
            // 方法一：使用 parse 函数（推荐，出错信息更详细）
            json js = json::parse(file);

            if (js.contains("cameraIp")) {
                js["cameraIp"].get_to(g_cfg_camera.cam_ip);
            }

            if (js.contains("imageWidth")) {
                js["imageWidth"].get_to(g_cfg_camera.cam_width);
            }

            if (js.contains("imageHeight")) {
                js["imageHeight"].get_to(g_cfg_camera.cam_height);
            }

        }
        catch(...)
        {

        }


    }

    
    // feilun_types.json
    str_cfg_file = g_appPath + "feilun_types.json";

    if (fs::exists(str_cfg_file))
    {
        try
        {
            // 1. 创建文件输入流
            std::ifstream file(str_cfg_file);

            // 2. 检查文件是否成功打开（非常重要！）
            if (!file.is_open()) {
                std::cerr << "无法打开文件!" << std::endl;
                return true;
            }

            // 3. 解析文件内容到 json 对象（两种方法任选其一）
            // 方法一：使用 parse 函数（推荐，出错信息更详细）
            json js = json::parse(file);


            if (js.contains("feilun_types") && js["feilun_types"].is_array()) {
                g_list_feilun.clear();
                for (auto& item : js["feilun_types"]) {
                    FEILUN_INFO info;
                    if (item.contains("id"))            item["id"].get_to(info.id);
                    if (item.contains("diameter"))      item["diameter"].get_to(info.diameter);
                    if (item.contains("height"))        item["height"].get_to(info.height);
                    if (item.contains("boss_diameter")) item["boss_diameter"].get_to(info.boss_diameter);
                    if (item.contains("boss_height"))   item["boss_height"].get_to(info.boss_height);
                    g_list_feilun.push_back(info);
                }
            }            
            
        }
        catch(...)
        {

        }
        

    }




    // hand_eye_pose_config.json
    str_cfg_file = g_appPath + "hand_eye_pose_config.json";

    if (fs::exists(str_cfg_file))
    {
        try
        {
            // 1. 创建文件输入流
            std::ifstream file(str_cfg_file);

            // 2. 检查文件是否成功打开（非常重要！）
            if (!file.is_open()) {
                std::cerr << "无法打开文件!" << std::endl;
                return true;
            }

            // 3. 解析文件内容到 json 对象（两种方法任选其一）
            // 方法一：使用 parse 函数（推荐，出错信息更详细）
            json js = json::parse(file);

            if (js.contains("wheel_rect")) {
                js["wheel_rect"].get_to(g_rect_lunzi);
            }
            
        }
        catch(...)
        {

        }
        

    }


    return true;
}



void CheckLogsSum(int max) {
    int numFiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(LogDir)) {
        if (std::filesystem::is_regular_file(entry)) {
            numFiles++;
        }
    }

    if (numFiles > max) {
        int numFilesToDelete = numFiles - max;

        // 遍历日志目录
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(LogDir)) {
            if (std::filesystem::is_regular_file(entry)) {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) { return a.filename().string() < b.filename().string(); });

        // 按名称排序后，删除过期的
        for (int i = 0; i < numFilesToDelete; i++) {
            std::filesystem::remove(files[i]);
        }
    }
}

void InitLog(std::string logpath) {

    if (!std::filesystem::exists(LogDir)) {
        std::filesystem::create_directories(LogDir);
    }
    else {
        CheckLogsSum(g_log_num);
    }
    util::Log::InitSpdLog(logpath, g_log_num);

}



// 最小二乘法拟合圆
static Circle fitCircleLeastSquares(const std::vector<cv::Point2f>& points) {
    Circle result;
    if (points.size() < 3) {
        std::cerr << "Error: Need at least 3 points to fit a circle" << std::endl;
        return result;
    }

    // 构建线性方程组 Ax = b
    cv::Mat A(points.size(), 3, CV_32F);
    cv::Mat b(points.size(), 1, CV_32F);

    for (size_t i = 0; i < points.size(); ++i) {
        float x = points[i].x;
        float y = points[i].y;
        A.at<float>(i, 0) = 2 * x;
        A.at<float>(i, 1) = 2 * y;
        A.at<float>(i, 2) = -1;
        b.at<float>(i, 0) = x * x + y * y;
    }

    // 求解线性方程组 A^T * A * x = A^T * b
    cv::Mat AtA = A.t() * A;
    cv::Mat Atb = A.t() * b;

    cv::Mat solution;
    cv::solve(AtA, Atb, solution, cv::DECOMP_SVD);

    // 提取圆心和半径
    float cx = solution.at<float>(0, 0);
    float cy = solution.at<float>(1, 0);
    float c = solution.at<float>(2, 0);

    result.center = cv::Point2f(cx, cy);
    result.radius = std::sqrt(cx * cx + cy * cy - c);

    // 计算拟合误差
    float error_sum = 0;
    for (const auto& point : points) {
        float dx = point.x - cx;
        float dy = point.y - cy;
        float distance = std::sqrt(dx * dx + dy * dy);
        float error = distance - result.radius;
        error_sum += error * error;
    }
    result.rmse = std::sqrt(error_sum / points.size());

    return result;
}

int update_PolynomialFit()
{
    std::vector <std::pair<float, float>> list_values;

    for (auto& d:g_map_template)
    {
        std::pair<float, float> p = std::make_pair(d.second.angle_det,d.second.angle);
        list_values.push_back(p);
    }

    // 拟合曲线
    fitter.fit(list_values);

    return 0;
}


int build_undistortedPoints()
{
    // 图像已做去畸变时，坐标换算必须走矫正图对应的针孔模型（newK），
    // 这张按畸变模型预计算的表再参与进来就会二次矫正，因此保持为空，
    // get_world_xy_by_z 会自动回退到针孔分支。
    if (g_undistort_ready)
    {
        g_undistortedPoints.clear();
        return 0;
    }

    // ===================== 1. 输入你的相机参数 =====================
    Mat K = (Mat_<float>(3, 3) <<
        g_cam_param.fx,    0,    g_cam_param.cx,    // fx,  0, cx
        0,    g_cam_param.fy,    g_cam_param.cy,    //  0, fy, cy
        0,         0,      1.0     //  0,  0,  1
    );

    // 畸变系数 k1,k2,p1,p2,k3
    // g_dist_coeffs

    // ===================== 3. 构建所有像素点 (u, v) =====================
    vector<Point2f> pixelPoints;
    for (int v = 0; v < g_cam_param.height; v++) {
        for (int u = 0; u < g_cam_param.width; u++) {
            pixelPoints.emplace_back((float)u, (float)v);
        }
    }

    // ===================== 4. 去畸变（关键！） =====================
    g_undistortedPoints.clear();
    undistortPoints(pixelPoints, g_undistortedPoints, K, g_dist_coeffs);

    return 0;
}


// 初始化图像去畸变。只做参数校验与日志，内参/畸变系数原样使用标定文件里的值，
// 不缩放也不猜测。标定文件缺失或开关关闭时保持原图，行为与改动前一致。
static void init_undistort()
{
    g_undistort_ready = false;
    g_undistort_map1.release();
    g_undistort_map2.release();
    g_undistort_map_size = Size();

    if (!g_undistort_enable)
    {
        std::cout << "[undistort] disabled by cfg.ini (undistort=0)" << std::endl;
        return;
    }

    if (g_cam_param.fx <= 0 || g_cam_param.fy <= 0 || g_dist_coeffs.empty())
    {
        std::cout << "[undistort] camera_matrix/distortion_coefficients missing, keep raw image" << std::endl;
        return;
    }

    g_undistort_K = (Mat_<double>(3, 3) <<
        g_cam_param.fx,    0,    g_cam_param.cx,
        0,    g_cam_param.fy,    g_cam_param.cy,
        0,         0,      1.0
    );
    g_undistort_D = g_dist_coeffs.clone();
    g_undistort_ready = true;

    std::cout << "[undistort] enabled, alpha=1.0" << std::endl;
    std::cout << "[undistort] K=\n" << g_undistort_K << std::endl;
    std::cout << "[undistort] D=" << g_undistort_D << std::endl;
    std::cout << "[undistort] cfg size=" << g_cam_param.width << "x" << g_cam_param.height << std::endl;
}


// 按实际图像尺寸建立映射表，尺寸没变就复用（RDK X3 上不能每帧重建）
static bool ensure_undistort_map(const Size& size)
{
    if (!g_undistort_ready || size.width <= 0 || size.height <= 0) return false;
    if (!g_undistort_map1.empty() && g_undistort_map_size == size) return true;

    // alpha=1.0：保留全部有效像素，边缘留黑边，输出尺寸不变
    Rect valid_roi;
    Mat newK = getOptimalNewCameraMatrix(g_undistort_K, g_undistort_D, size, 1.0, size, &valid_roi);
    initUndistortRectifyMap(g_undistort_K, g_undistort_D, noArray(), newK, size, CV_16SC2,
                            g_undistort_map1, g_undistort_map2);
    g_undistort_map_size = size;

    // 图已按 newK 矫正，坐标换算用的内参必须同步成 newK，
    // 否则会拿畸变坐标去反算，等于二次矫正
    g_cam_param.fx = (float)newK.at<double>(0, 0);
    g_cam_param.fy = (float)newK.at<double>(1, 1);
    g_cam_param.cx = (float)newK.at<double>(0, 2);
    g_cam_param.cy = (float)newK.at<double>(1, 2);

    std::cout << "[undistort] map built: size=" << size.width << "x" << size.height
              << " valid_roi=[" << valid_roi.x << "," << valid_roi.y << ","
              << valid_roi.width << "," << valid_roi.height << "]"
              << " newK fx=" << g_cam_param.fx << " fy=" << g_cam_param.fy
              << " cx=" << g_cam_param.cx << " cy=" << g_cam_param.cy << std::endl;

    return true;
}


// 去畸变入口（原地）。彩色与深度用同一对映射表，保证两者像素仍然一一对应。
static bool undistort_frame(Mat& bgrImage, Mat* depthImage = nullptr)
{
    if (!g_undistort_enable || !g_undistort_ready || bgrImage.empty()) return false;

    Size size = bgrImage.size();
    if (!ensure_undistort_map(size)) return false;

    Mat corrected;
    remap(bgrImage, corrected, g_undistort_map1, g_undistort_map2,
          INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    bgrImage = corrected;

    if (depthImage != nullptr && !depthImage->empty())
    {
        if (depthImage->size() == size)
        {
            Mat depth_corrected;
            remap(*depthImage, depth_corrected, g_undistort_map1, g_undistort_map2,
                  INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
            *depthImage = depth_corrected;
        }
        else
        {
            std::cerr << "[undistort] depth size " << depthImage->cols << "x" << depthImage->rows
                      << " != color size " << size.width << "x" << size.height
                      << ", depth left raw" << std::endl;
        }
    }

    return true;
}


ALVA_DLL_API int __cdecl 
AI_det_init(const char* appDir)
{
    if (nullptr!=g_camera && nullptr!=G_DET)
    {
        return 0;
    }
    
    std::string g_test_mode = "";
    g_appPath = "";
    if (nullptr!=appDir)
    {
        g_test_mode = appDir;

        if ("test"==g_test_mode || "clean"==g_test_mode)
        {
            
        }
        else
        {
            g_test_mode = "";
            g_appPath = appDir;

            // 检查并添加路径分隔符
            if (!g_appPath.empty()) {
                char lastChar = g_appPath.back();
                if (lastChar != '/' && lastChar != '\\') {
                    g_appPath += '/';
                }
            }  

        }

    }

    if ( nullptr==appDir || "test"!=g_test_mode || !fs::exists(appDir))
    {
        GetBasePathInfo();
    }


    // LoadConfig
    LoadConfig();

    std::cout << "LoadConfig" << std::endl;

    std::string ret_img_dir = g_appPath + "ret_img/";
    if (!fs::exists(ret_img_dir)) fs::create_directory(ret_img_dir);

    // 尽早读取上次保存的序号。init 后面还有相机/模型初始化，那些步骤失败会提前
    // return，如果那时还没读到文件，AI_det_close() 就会把内存里的 0 写回 cnt.txt，
    // 下次启动序号从 0 开始，把已有图片覆盖掉。
    {
        const std::string cnt_path = ret_img_dir + "cnt.txt";
        if (fs::exists(cnt_path))
        {
            const std::string cnt_str = readFirstLine(cnt_path);
            bool valid = !cnt_str.empty();
            for (size_t i = 0; valid && i < cnt_str.size(); i++)
            {
                if (cnt_str[i] < '0' || cnt_str[i] > '9') valid = false;
            }

            if (!valid)
            {
                std::cerr << "[det_cnt] cnt.txt 内容非法(\"" << cnt_str
                          << "\")，本进程序号从 " << g_det_cnt << " 开始: "
                          << cnt_path << std::endl;
            }
            else
            {
                g_det_cnt = static_cast<std::uint32_t>(std::stoul(cnt_str));
            }
        }
    }

    std::string logpath = ret_img_dir + "det_logs/";
    if (!fs::exists(logpath)) fs::create_directory(logpath);

    LogDir = g_appPath + "ret_img/det_logs/";

    logpath = LogDir + "det-log.txt";


    g_paramPath = g_appPath + "param/";
    g_calibrationPath = g_paramPath + "calibration";
    g_posPath = g_paramPath + "pos/";

    if (!fs::exists(g_paramPath))
        fs::create_directory(g_paramPath); 

    if (!fs::exists(g_calibrationPath))
        fs::create_directory(g_calibrationPath);

    if (!fs::exists(g_posPath))
        fs::create_directory(g_posPath);        


#ifdef MY_LOG 
    InitLog(logpath);
#endif

if("test"!=g_test_mode && "clean"!=g_test_mode)
{


#ifdef useORB

    // if ("orb"== g_camera_name)
    {
        g_camera = std::make_shared<OrbbecCameraV2>(g_cfg_camera);

        // 打开相机
        if (!g_camera->open()) {
            std::cerr << "init camera failed, please check camera" << std::endl;
            return -1;
        }        
    }

#else


#ifdef useOAK
    // 使用默认配置
    // if ("oak"== g_camera_name)
    {
        g_camera = std::make_shared<OAKCamera>(g_cfg_camera);

        // 打开相机
        if (!g_camera->open()) {
            std::cerr << "init camera failed, please check camera" << std::endl;
            return -1;
        }

    }
    
#else

#ifdef useMVS
    // 使用默认配置
    // if ("oak"== g_camera_name)
    {
        g_camera = std::make_shared<MVSCamera>(g_cfg_camera);

        // 打开相机
        if (!g_camera->open()) {
            std::cerr << "init camera failed, please check camera" << std::endl;
            return -1;
        }

    }

#else

#ifdef useUSB
    // USB 摄像头 (OpenCV VideoCapture)
    {
        g_camera = std::make_shared<USBCamera>(g_cfg_camera);

        if (!g_camera->open()) {
            std::cerr << "init camera failed, please check camera" << std::endl;
            return -1;
        }
    }

#else

#endif // useUSB

#endif // useMVS

#endif // useOAK


#endif // useORB

    // 应用曝光时间配置
    if (g_exposure_time > 0 && g_camera && g_camera->isOpened())
    {
        g_camera->setAutoExposure(false);
        g_camera->setColorExposureTime(g_exposure_time);
        std::cout << "ExposureTime set to " << g_exposure_time << "us, AutoExposure disabled" << std::endl;
    }
    else if (g_exposure_time == -1)
    {
        std::cout << "ExposureTime unchanged (g_exposure_time=-1)" << std::endl;
    }

    // 应用翻转配置（相机硬件翻转，不支持时回退到软件翻转）
    if (g_camera && g_camera->isOpened() && (g_flip_v > 0 || g_flip_h > 0))
    {
        if (!g_camera->setFlip(g_flip_v, g_flip_h))
        {
            std::cerr << "camera setFlip not supported, fallback to software flip" << std::endl;
            g_flip_soft = 1;
        }
    }

}
else
{

}

    g_cam_param.width = g_cfg_camera.cam_width;
    g_cam_param.height = g_cfg_camera.cam_height;


    // 从文件读取内参
    cv::Mat camera_matrix, dist_coeffs;

    // 尝试加载已有相机内参，否则从图像标定
    // 手眼矩阵：根据 hand_eye_mode 选择文件
    std::string cam_file = g_paramPath + "hand_eye_result.yaml";
    if ("eye_in_hand"!=g_hand_eye_mode)
    {
        cam_file = g_paramPath + "eye_to_hand_result.yaml";
    }

    cv::FileStorage fs_cam(cam_file, cv::FileStorage::READ);
    if (fs_cam.isOpened()) {

        // 优先加载 T_base_camera（眼在手外）
        if (!fs_cam["T_base_camera"].empty()) {
            fs_cam["T_base_camera"] >> g_hand_eye_matrix;
            std::cout << "T_base_camera loaded from " << cam_file << std::endl;
        }
        // T_base_camera 不存在时才加载 T_flange_camera（眼在手内）
        else if (!fs_cam["T_flange_camera"].empty()) {
            fs_cam["T_flange_camera"] >> g_hand_eye_matrix;
            std::cout << "T_flange_camera loaded from " << cam_file << std::endl;
        } else {
            std::cout << "No hand-eye matrix found in " << cam_file << std::endl;
        }

        // 相机内参始终从 hand_eye_result.yaml 加载（不受 hand_eye_mode 影响）
        fs_cam["camera_matrix"] >> camera_matrix;
        fs_cam["distortion_coefficients"] >> dist_coeffs;


        g_cam_param.fx = camera_matrix.at<double>(0, 0);
        g_cam_param.fy = camera_matrix.at<double>(1, 1);

        g_cam_param.cx = camera_matrix.at<double>(0, 2);
        g_cam_param.cy = camera_matrix.at<double>(1, 2);


        g_dist_coeffs = dist_coeffs.clone();

        // 必须在 build_undistortedPoints 之前：启用图像去畸变后，
        // 坐标表要让位给矫正图对应的针孔模型
        init_undistort();

        build_undistortedPoints();

        fs_cam.release();
    }




#ifdef USE_RDK
    // std::string modelPath = g_appPath + "models/" + g_model_name + ".bin";
    std::string modelPath = g_appPath + "models/" + g_model_name + "_rdk.bin";
    

    try
    {
        g_det_rdk = std::make_shared<YOLOv11RDK>(modelPath);
        g_det_rdk->resizeKeepAspectRatio = g_resizeKeepAspectRatio;
        g_det_rdk->m_model_name = g_model_name;
    }
    catch (...)
    {
        std::cerr << "init rdk det engine failed, please check models dir" << std::endl;
        return -1;
    }
#else
    std::string enginePath = g_appPath + "models/" + g_model_name + ".m";

    try
    {
        g_det = std::make_shared<YOLOv11>(enginePath, logger);
        G_DET->resizeKeepAspectRatio = g_resizeKeepAspectRatio;
    }
    catch (...)
    {
        std::cerr << "init det engine failed, please check cuda and models dir" << std::endl;
        return -1;
    }
#endif

    return 0;    
}

int crop_image(const char* input_json, cv::Mat& image, json& j)
{
    int ret = -1;

    //=====================================
    // 截取图像
    cv::Rect crop(-100, -100, 0, 0);
    try
    {
        // 在这里把json倒一手
        json j_input = json::parse(input_json);


        if (j_input.contains("rect") &&
            j_input["rect"].is_array()) {

            std::vector<int> pts;
            j_input["rect"].get_to(pts);

            if (4 == pts.size())
            {
                if (pts[0] >= 0 && pts[0] < image.cols &&
                    pts[1] >= 0 && pts[1] < image.rows &&
                    (pts[0] + pts[2] < image.cols) &&
                    (pts[1] + pts[3] < image.rows)
                    )
                {
                    crop = cv::Rect(pts[0], pts[1], pts[2], pts[3]);
                }

            }
        }

    }
    catch (...)
    {

    }


    return 0;
}

int read_image(const char* img, int img_w, int img_h, int img_cnt,std::vector<cv::Mat>&list_images, json& j)
{
    int ret = -1;
    
    //=====================================
    // 读取图像
    j["ret-msg"] = "error: load image";

    std::size_t img_size = img_w* img_h*3;

    try {
        if (img_w == 0 && img_h == 0)
        {
            // load from file
            if (!fs::exists(img))
            {
                j["ret-msg"] = "error: load image file";
                return ret;
            }

            cv::Mat image = cv::imread(img);
            list_images.push_back(image);
        }
        else if (img_w > 0 && img_h == 0)
        {
            // load from jpeg buf
            std::vector<char> jpeg_data(img_w);
            memcpy(jpeg_data.data(), img, img_w);

            cv::Mat image = cv::imdecode(cv::Mat(jpeg_data), cv::IMREAD_COLOR);
            jpeg_data.clear();

            list_images.push_back(image);
        }
        else if (img_w > 10 && img_h > 10)
        {

            if (g_save_ret > 0 && img_cnt>1)
            {
                advance_det_cnt();
            }

            // load from bgr buf
            char* pImgData = (char*)img;
            for (size_t i = 0; i < img_cnt; i++)
            {
                cv::Mat image = cv::Mat(img_h, img_w, CV_8UC3, (char*)pImgData);

                pImgData += img_size;

                list_images.push_back(image);

                // save file
                if (g_save_ret > 0 && img_cnt>1)
                {
                    std::string ret_img_dir = g_appPath + "input_img/";
                    if (!fs::exists(ret_img_dir)) fs::create_directory(ret_img_dir);

                    ret_img_dir = string_format("%s%05d/", ret_img_dir.c_str(), g_det_cnt);
                    if (!fs::exists(ret_img_dir)) fs::create_directory(ret_img_dir);

                    std::string ret_img = string_format("%s%d.jpg", ret_img_dir.c_str(), i);

                    cv::imwrite(ret_img, image);
                }

            }

        }


        if (list_images.size()==0 || list_images[0].cols < 10 || list_images[0].rows < 10)
        {
            j["ret-msg"] = "error: image data is null";
            return ret;
        }

    }
    catch (...) {
        return ret;
    }

    return 0;
}

int merge_det(cv::Mat& image, Detection_pan& detections_pan, vector<Detection_pan>& output_fine, json& j)
{
    // 根据detections_pan的点拟合一个圆
    std::vector<cv::Point2f> pts = detections_pan.pts;

    if (pts.size()>10)
    {
        pts.erase(pts.begin());
        pts.erase(pts.end() - 1);
    }


    Circle c = fitCircleLeastSquares(pts);

#ifdef REL_DBG
    cv::Mat img_dbg = image.clone();
    G_DET->draw_pan(img_dbg, detections_pan);
#endif // REL_DBG


    // 算出两点间弧线距离
    double dist = 2 * c.radius * M_PI/ pts.size();

    // 以detections_pan的检测点为中心，弧线长度的一半为宽，按分数筛选 fine 结果
    std::vector<int> list_change(detections_pan.pts.size());
    dist *= 0.33;
    int idx = 0;
    float up_threshold = 0.38;
    float low_threshold = 0.2;
    float up_low = up_threshold - low_threshold + 0.01;
    for (auto& p: detections_pan.pts)
    {
        list_change[idx] = 0;
        float x = p.x - dist;
        float y = p.y - dist;

        cv::Rect2f bbox(x,y, dist*2, dist*2);

        float max_score = 0;
        Detection_pan* pFine = NULL;
        cv::Point2f r_centor_max;
        for (auto& r : output_fine)
        {    
            cv::Point2f r_centor(r.bbox.x+r.bbox.width/2, r.bbox.y + r.bbox.height / 2);
            if (bbox.contains(r_centor) && r.conf> max_score )
            {
                max_score = r.conf;
                r_centor_max = r_centor;

                pFine = &r;
            }

        }

        if (max_score> up_threshold)
        {
            p = r_centor_max;

            list_change[idx] = 1;
        }
        else if (max_score >= low_threshold && max_score <= up_threshold)
        {
            // 做一个柔和处理
            float low_ratio = std::abs(max_score- low_threshold)/up_low;

            p = p*(1-low_ratio)+ r_centor_max*low_ratio;

            list_change[idx] = 2;
        }

        idx++;
    }

#ifdef REL_DBG
    cv::Mat img_dbg_new = image.clone();
    G_DET->draw_pan(img_dbg_new, detections_pan);
#endif // REL_DBG

    return 0;
}

/**
 * 精确定位螺丝孔圆心
 * @param image 输入图像
 * @param yolo_bbox YOLO检测框 (x1, y1, x2, y2)
 * @param refine_method 优化方法: 0-轮廓矩心(默认), 1-最小二乘拟合
 * @param visualize 是否可视化过程
 * @return 精修后的圆心坐标 (在原始图像中的位置)
 */
int refineScrewHoleCenter(const Mat& image, Point2f& new_c,float& r,
    int refine_method = 0, bool visualize = false) {

    Mat roi = image.clone();

    // 可视化ROI
    Mat process_visual;
    if (visualize) {
        roi.copyTo(process_visual);
    }

    // 2. 图像预处理
    Mat gray, binary, blurred;

    // 转换为灰度图
    if (roi.channels() == 3) {
        cvtColor(roi, gray, COLOR_BGR2GRAY);
    }
    else {
        roi.copyTo(gray);
    }

    // =================================================
    // 将图像中心  为圆心，半径为r的区域 用黑色填充
    // 计算图像中心点
    Point center(image.cols / 2, image.rows / 2);

    // 定义半径（例如图像宽度的1/3）
    double max_radius = std::min(roi.cols, roi.rows) / 2.0;
    int radius_fill = max_radius*0.75;
    // 或者直接指定：int radius = 100;

    // 创建一个全黑图像（与原始图像大小相同）
    Mat mask = Mat::zeros(image.size(), CV_8UC1);

    // 在mask上画一个白色填充圆
    circle(mask, center, radius_fill, Scalar(255), -1); // -1表示填充

    // 将原始图像中圆形区域设为黑色
    gray.setTo(Scalar(39), mask);

    // =================================================
    // 高斯模糊去噪 (可选，根据图像噪声情况调整)
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);

    // =================================================
    // 自适应二值化 - 对光照不均更鲁棒
    adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 11, 2);

    // =================================================
    // 形态学操作 (可选，用于连接断裂边缘)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    if (visualize) {
        Mat binary_display;
        cvtColor(binary, binary_display, COLOR_GRAY2BGR);
        hconcat(process_visual, binary_display, process_visual);
    }

    // =================================================
    // 3. 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if (contours.empty()) {
        // 未找到轮廓，返回YOLO框中心
        cerr << "警告: 未检测到轮廓，使用YOLO框中心" << endl;
        return -1;
    }

    // =================================================
    // 4. 选择目标轮廓 (按面积筛选，假设螺丝孔是最大的)
    vector<Point> target_contour;
    double max_area = 0;
    int target_idx = -1;

    // 可选: 根据已知直径(180像素)估算面积范围过滤
    double expected_radius = max_radius * 0.9;
    double expected_area = CV_PI * expected_radius * expected_radius;  // π * r²
    
    double min_area = CV_PI * radius_fill * radius_fill;   // 允许的最小面积比例

    max_radius = max_radius * 0.985;
    double max_area_thresh = CV_PI * max_radius * max_radius; // 允许的最大面积比例

    std::vector<double>list_cand_area;
    std::vector<double>list_cand_idx;
    double max_area_contours = 0;
    int max_idx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        if (area> max_area_contours)
        {
            max_area_contours = area;
            max_idx = i;
        }

        if (area > max_area && area > min_area && area < max_area_thresh) {
            list_cand_area.push_back(area);
            list_cand_idx.push_back(i);
        }
    }

    if (list_cand_area.size()==0) {
        // 没有符合面积要求的轮廓
        return -1;
    }

    if (list_cand_area.size()>1)
    {
        target_contour = contours[list_cand_idx[0]];
    }
    else
    {
        target_contour = contours[list_cand_idx[0]];
    }  

    // =================================================
    // 5. 圆心精确定位 (根据选择的方法)
    Point2d refined_center;
    float radius;
    if (refine_method == 0) {
        // 方法1: 轮廓矩心法 (快速)
        Moments m = moments(target_contour, true);
        if (m.m00 == 0) {
            refined_center = Point2d(roi.cols / 2.0, roi.rows / 2.0);
        }
        else {
            refined_center = Point2d(m.m10 / m.m00, m.m01 / m.m00);
        }
    }
    else {
        // 方法2: 最小二乘圆拟合 (精度更高)
        // 收集轮廓点
        vector<Point2f> points;
        for (const auto& pt : target_contour) {
            points.push_back(Point2f(pt.x, pt.y));
        }

        if (points.size() < 3) {
            // 点太少，使用矩心法
            Moments m = moments(target_contour, true);
            refined_center = Point2d(m.m10 / m.m00, m.m01 / m.m00);
        }
        else {
            // 最小二乘圆拟合
            Point2f center;
            minEnclosingCircle(points, center, radius);

            // 使用迭代优化 (可选，精度更高但更慢)
            // 这里使用简单的代数拟合
            vector<float> x_vals, y_vals;
            for (const auto& p : points) {
                x_vals.push_back(p.x);
                y_vals.push_back(p.y);
            }

            // 简单的圆拟合算法
            int n = points.size();
            float sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_x3 = 0, sum_y3 = 0;
            float sum_xy = 0, sum_x1y2 = 0, sum_x2y1 = 0;

            for (const auto& p : points) {
                float x = p.x;
                float y = p.y;
                float x2 = x * x;
                float y2 = y * y;

                sum_x += x;
                sum_y += y;
                sum_x2 += x2;
                sum_y2 += y2;
                sum_x3 += x2 * x;
                sum_y3 += y2 * y;
                sum_xy += x * y;
                sum_x1y2 += x * y2;
                sum_x2y1 += x2 * y;
            }

            float C = n * sum_x2 - sum_x * sum_x;
            float D = n * sum_xy - sum_x * sum_y;
            float E = n * sum_x3 + n * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
            float G = n * sum_y2 - sum_y * sum_y;
            float H = n * sum_x2y1 + n * sum_y3 - (sum_x2 + sum_y2) * sum_y;

            float denominator = C * G - D * D;
            if (fabs(denominator) < 1e-10) {
                // 退化解，使用最小外接圆
                refined_center = center;
            }
            else {
                float a = (E * G - H * D) / (denominator * 2.0f);
                float b = (H * C - E * D) / (denominator * 2.0f);
                refined_center = Point2d(a, b);

                // 计算半径（平均距离法）
                double sum_r = 0.0;
                for (const auto& p : points) {
                    double dx = p.x - a;
                    double dy = p.y - b;
                    sum_r += sqrt(dx * dx + dy * dy);
                }
                radius = sum_r / n;
            }

        }
    }


    // 6. 坐标转换回原始图像
    Point2d global_center = refined_center;

    // 7. 可视化 (可选)
    if (visualize) {
        // 绘制轮廓
        Mat contour_visual = Mat::zeros(roi.size(), CV_8UC3);
        drawContours(contour_visual, vector<vector<Point>>{target_contour},
            -1, Scalar(0, 255, 0), 1);

        // 绘制精修后的圆心
        circle(contour_visual, refined_center, 3, Scalar(0, 0, 255), -1);
        circle(contour_visual, refined_center, radius, Scalar(255, 0, 0), 1); // 绘制估计的圆

        // 绘制原始YOLO框中心
        Point yolo_center_in_roi(image.cols / 2, image.rows / 2);
        circle(contour_visual, yolo_center_in_roi, 3, Scalar(255, 255, 0), -1);

        hconcat(process_visual, contour_visual, process_visual);

        // 显示各步骤结果
        resize(process_visual, process_visual, Size(), 0.8, 0.8);
        //imshow("处理过程: 原ROI | 二值化 | 轮廓+圆心", process_visual);
        //waitKey(0);

        // 在原始图像上显示最终结果
        Mat result_image = image.clone();
        circle(result_image, global_center, 5, Scalar(0, 0, 255), -1); // 精修圆心
        circle(result_image, global_center, radius, Scalar(0, 255, 0), 2); // 拟合圆

        // YOLO原始框和中心
        Point yolo_global_center = yolo_center_in_roi;
        circle(result_image, yolo_global_center, 3, Scalar(255, 255, 0), -1);

        // 连接线显示偏移
        line(result_image, yolo_global_center, global_center, Scalar(255, 0, 255), 1);

        //imshow("最终定位结果", result_image);
        //waitKey(0);
    }

    new_c = global_center;
    r = radius;
    return 0;
}

// 计算矩形的内切圆
void getInscribedCircleFromRect(const Rect& rect, Point2f& center, float& radius) {
    // 圆心是矩形中心
    center.x = rect.x + rect.width / 2.0f;
    center.y = rect.y + rect.height / 2.0f;

    // 半径是宽高中较小值的一半
    radius = min(rect.width, rect.height) / 2.0f;
}

int refine_hole(cv::Mat& image, Detection_pan& detections_pan)
{
    if (detections_pan.pts.size()!= detections_pan.sizes.size())
    {
        return -1;
    }

    // 从第一个开始，第0个是中间大圆，暂时不处理了
    for (size_t i = 1; i < detections_pan.pts.size(); i++)
    {
        auto& p = detections_pan.pts[i];
        auto& s = detections_pan.sizes[i];

        cv::Rect r(p.x - s.width / 2, p.y - s.height / 2, s.width, s.height);

        // 1. 准备ROI区域 (适当扩大边界)
        int margin = 2;  // 扩大边界，确保包含完整圆
        int x1 = std::max(0, r.x - margin);
        int y1 = std::max(0, r.y - margin);
        int x2 = std::min(image.cols, r.x + r.width + margin);
        int y2 = std::min(image.rows, r.y + r.height + margin);

        r = cv::Rect(x1, y1, x2 - x1, y2 - y1);

        cv::Mat crop = image(r).clone();

        Point2f new_c; 
        float radius;

        bool visualize = false;
#ifdef REL_DBG
        visualize = true;
#endif // REL_DBG

        int ret = refineScrewHoleCenter(crop, new_c, radius, 1, visualize);

        if (0==ret)
        {
            new_c.x += r.x;
            new_c.y += r.y;

            float dist = calculateDistance_core(new_c.x, new_c.y, p.x, p.y);

            if (dist<=5)
            {
                // 求一个平均值
                new_c = (p + new_c) / 2;
            }
            else
            {
                // 距离过远，还是用 yolo 检测的值
                getInscribedCircleFromRect(r, new_c, radius);
            }

        }
        else
        {
            getInscribedCircleFromRect(r, new_c, radius);
        }

        detections_pan.pts_new.push_back(new_c);
        detections_pan.radius.push_back(radius);
    }

    return 0;
}

int det_image(cv::Mat& image, std::vector< std::vector<Detection_pan>>& detections, json& j)
{
    std::string err = "error: det catch error";
    j["ret-msg"] = err;

    try {
        // Preprocess the image
        //cv::Mat img_rgb; 
        //cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);

        G_DET->preprocess(image);

        // Perform inference
        G_DET->infer();

        //auto start = std::chrono::high_resolution_clock::now();
        //int loop = 1000;

        //for (size_t i = 0; i < loop; i++)
        //{
        //    G_DET->infer();
        //}

        //auto end = std::chrono::high_resolution_clock::now();

        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "Inference "<< loop << "times,  completed in " << elapsed.count() / loop << " seconds." << std::endl;

        // Postprocess to get detections
        // std::vector<Detection> detections;
        // yolov11.postprocess(detections);
        if ("shangliao"==g_model_name)
        {
            G_DET->postprocess_pan(image, detections);

            vector<Detection_pan>& detections_pan = detections[0];
            vector<Detection_pan>& output_fine = detections[1];

            G_DET->restore_real_xy(image, detections_pan);
            G_DET->restore_real_xy(image, output_fine);

            // 根据output_fine的结果，优化detections_pan的结果
            if (detections_pan.size() > 0 && output_fine.size() > 0)
            {
                merge_det(image, detections_pan[0], output_fine, j);
            }
        }
        else
        {
            G_DET->postprocess_pan(image, detections);

            for (size_t i = 0; i < detections.size(); i++)
            {
                G_DET->restore_real_xy(image, detections[i]);
            }
        }


    }
    catch (const std::exception& e) {
        err = string_format("Error during inference: %s", e.what());
        j["ret-msg"] = err;
        std::cerr << err << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << err << std::endl;
        return -1;
    }

    return 0;
}


// 使用向量计算角度
float calculateAngleBetweenLines(const cv::Point2f& C,  // 顶点
    const cv::Point2f& A,  // 第一个点
    const cv::Point2f& B) { // 第二个点
// 计算向量
    cv::Point2f vectorCA = A - C;
    cv::Point2f vectorCB = B - C;

    // 计算向量的模长
    float lengthCA = std::sqrt(vectorCA.x * vectorCA.x + vectorCA.y * vectorCA.y);
    float lengthCB = std::sqrt(vectorCB.x * vectorCB.x + vectorCB.y * vectorCB.y);

    // 防止除零错误
    if (lengthCA < 1e-10 || lengthCB < 1e-10) {
        return 0.0f;
    }

    // 计算点积
    float dotProduct = vectorCA.x * vectorCB.x + vectorCA.y * vectorCB.y;

    // 计算余弦值
    float cosTheta = dotProduct / (lengthCA * lengthCB);

    // 限制余弦值范围以防止数值误差
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));

    // 计算角度（弧度）
    float angle = std::acos(cosTheta);

    return angle;
}


// 基本旋转计算
cv::Point2f getRotatedPointOnCircle(const cv::Point2f& center,    // 圆心
    float radius,                 // 半径
    const cv::Point2f& p1,        // 圆周上原始点
    float angle) {                // 旋转弧度

// 计算原始点相对于圆心的角度
    float originalAngle = std::atan2(p1.y - center.y, p1.x - center.x);

    // 计算旋转后的角度
    float newAngle = originalAngle + angle;

    // 计算新的交点坐标
    cv::Point2f newPoint;
    newPoint.x = center.x + radius * std::cos(newAngle);
    newPoint.y = center.y + radius * std::sin(newAngle);

    return newPoint;
}

int handle_ret_pt10(Mat& image, Detection_pan& det_ret, json& j)
{
    int ret = -1;
    j["ret-msg"] = "error: handle_ret";

    cv::Scalar cl_red = Scalar(255, 0, 255);

    std::vector<cv::Point2f> pts = det_ret.pts;

    int zero_idx = 0;
    int zero_idx_new = 0;
    if (pts.size() > 10)
    {
        pts.erase(pts.begin());
        pts.erase(pts.end() - 1);
    }

    // 找到异常点
    std::vector<PointWithStats> point_stats2;
    Circle c = CircleFitterWithOutlierRemoval::fitCircleWithStatisticalOutlierRemoval(
        pts, point_stats2);

    // draw pt
    cv::circle(image, c.center, 2, cl_red);

    cv::circle(image, c.center, c.radius, cl_red);


#ifdef REL_DBG
    cv::Mat img_dbg = image.clone();

    // draw pt
    cv::circle(img_dbg, c.center, 2, cl_red);

    cv::circle(img_dbg, c.center, c.radius, cl_red);

    G_DET->draw_pan(img_dbg, det_ret);

#endif // REL_DBG

    // 如果 0 1 2 都是异常点 就要校正一下
    if (point_stats2[0].is_outlier &&
        point_stats2[1].is_outlier &&
        point_stats2[2].is_outlier)
    {

        // 标记异常点
        std::vector<bool> is_outlier(pts.size(), false);
        for (size_t i = 0; i < point_stats2.size(); i++)
        {
            if (point_stats2[i].is_outlier) is_outlier[i] = true;
        }

        // 方法1: 线性插值修正
        std::cout << "\n=== Method 1: Linear Interpolation Correction ===" << std::endl;
        auto corrected_points1 = PointCorrector::correctOutlierPointsPolynomial(pts, is_outlier, c);

#ifdef REL_DBG
        // 可视化结果（可选）
        // PointCorrector::visualizeCorrection(pts, corrected_points1, is_outlier, c);
#endif // REL_DBG

        // 修改为修正后的点
        det_ret.pts = corrected_points1;
    }
    else
    {
        det_ret.pts = pts;

        for (size_t i = 0; i < 3; i++)
        {
            if (point_stats2[i].is_outlier)
            {
                zero_idx_new++;
            }
            else break;
        }
    }

    // Draw detections on the image
    G_DET->draw_pan(image, det_ret);


    float gap_angle = (zero_idx_new - zero_idx) * 36;

    // 正常是77.6  垂直翻转是 6.4
    float zero_pt_to_top_hole_angle = 77.6 - gap_angle;

    // 使用没有检测误差的点
    zero_idx = zero_idx_new;

    float cur_hole_angle = (zero_pt_to_top_hole_angle - g_diff_angle) * arc_angle;
    float p1_angle_dst = (90.0 - zero_pt_to_top_hole_angle) * arc_angle;
    float cur_diff_angle = g_diff_angle * arc_angle;

    cv::Point2f hole1 = getRotatedPointOnCircle(c.center, c.radius, det_ret.pts[zero_idx], -cur_hole_angle);

    // 根据p1的坐标，求出三个圆孔的角度
    std::vector<cv::Point2f> pt_hole(3);
    pt_hole[0] = hole1;
    pt_hole[1] = getRotatedPointOnCircle(c.center, c.radius, hole1, -120 * arc_angle);
    pt_hole[2] = getRotatedPointOnCircle(c.center, c.radius, hole1, -240 * arc_angle);

    // 按 y 坐标从小到大排序
    std::sort(pt_hole.begin(), pt_hole.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });

    cv::Point2f pt_hole_up = cv::Point2f(c.center.x, c.center.y - c.radius);

    // up line
    cv::line(image, c.center, pt_hole_up, Scalar(0, 0, 255));

    cv::line(image, c.center, pt_hole[0], Scalar(0, 255, 255), 2);
    cv::line(image, c.center, pt_hole[1], Scalar(255, 0, 0));
    cv::line(image, c.center, pt_hole[2], Scalar(255, 0, 0));


    // cv::line(image, c.center, pt_hole_up, Scalar(0, 255, 255));
    float delta_angle = calculateAngleBetweenLines(c.center, pt_hole_up, pt_hole[0]);

    // 转成角度显示
    if (pt_hole[0].x > pt_hole_up.x)
    {
        delta_angle = -delta_angle;
    }

    // 转成角度显示
    delta_angle = delta_angle / arc_angle;

    putText(image, string_format("%.2f", delta_angle), c.center, FONT_HERSHEY_DUPLEX, 0.6, Scalar(0, 0, 255));

    ret = 0;
    j["angle"] = delta_angle;
    j["ret-msg"] = "ok";

    det_ret.angle = delta_angle;

    return ret;
}


std::vector<size_t> sortIndexes(const std::vector<float>& nums) {
    std::vector<size_t> idx(nums.size());

    // 用 0, 1, 2, ... 填充索引数组
    std::iota(idx.begin(), idx.end(), 0);

    // 根据 nums 中的值对索引进行排序
    std::sort(idx.begin(), idx.end(),
        [&nums](size_t i1, size_t i2) {
        return std::abs(nums[i1]) < std::abs(nums[i2]);
    });

    return idx;
}

std::string formatFloat(double value, int precision = 2) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

/**
 * 归一化函数：将圆周上的点映射到单位圆上
 * @param circle 输入的圆
 * @return 映射到单位圆上的点坐标
 */
cv::Point2d normalizeToUnitCircle(const template_circle& circle) {
    // 计算圆周上的点相对于圆心的向量
    cv::Point2d relative = circle.pts[0] - cv::Point2f(circle.centor.x, circle.centor.y);

    // 归一化到单位圆（除以半径）
    return relative / circle.radius;
}

/**
 * 计算两个归一化点的夹角（弧度）
 * @param p1 单位圆上的第一个点
 * @param p2 单位圆上的第二个点
 * @return 夹角（弧度），范围[0, π]
 */
float calculateAngle(const cv::Point2d& p1, const cv::Point2d& p2) {
    // 使用点积公式计算夹角：cosθ = (p1·p2) / (|p1||p2|)
    // 由于已经是单位圆上的点，|p1| = |p2| = 1
    float dot = p1.x * p2.x + p1.y * p2.y;

    // 防止浮点数误差导致acos参数超出[-1, 1]
    dot = std::max(-1.0f, std::min(1.0f, dot));

    return std::acos(dot);
}

/**
 * 计算两个归一化点的有符号夹角（弧度）
 * @param p1 单位圆上的第一个点
 * @param p2 单位圆上的第二个点
 * @return 有符号夹角（弧度），范围(-π, π]，正数表示逆时针方向
 */
double calculateSignedAngle(const cv::Point2d& p1, const cv::Point2d& p2) {
    // 使用atan2计算每个点的角度，然后相减
    double angle1 = std::atan2(p1.y, p1.x);
    double angle2 = std::atan2(p2.y, p2.x);

    // 计算差值并归一化到(-π, π]范围
    double diff = angle2 - angle1;
    if (diff > CV_PI) diff -= 2 * CV_PI;
    if (diff <= -CV_PI) diff += 2 * CV_PI;

    return diff;
}

float adjust_angle_by_template(float angle)
{
    // 从模板中找到 角度相差最小的
    template_circle* p = NULL;
    float min_diff = 999;
    for (auto& m : g_map_template)
    {
        float diff = std::abs((float)m.first- angle);
        if (diff< min_diff)
        {
            min_diff = diff;
            p = &(m.second);
        }
    }

    if (NULL!=p)
    {
        // 计算夹角
        cv::Point2d p_template = normalizeToUnitCircle(*p);
        cv::Point2d p_det = normalizeToUnitCircle(g_cur_det_ret);

        // 步骤2：计算夹角
        double angle_rad = calculateSignedAngle(p_det, p_template);
        double angle_deg = std::abs(angle_rad * 180.0 / CV_PI);

        // 根据夹角更新 angle
        if (p->angle<angle) angle = p->angle + angle_deg;
        else angle = p->angle - angle_deg;
    }

    // 返回修正后夹角
    return angle;
}

int handle_ret_pt4(Mat& image, Detection_pan& det_ret, json& j)
{

#ifdef REL_DBG
    cv::Mat img_dbg_pre = image.clone();

    G_DET->draw_pan(img_dbg_pre, det_ret);

#endif // REL_DBG

    int ret = -1;
    j["ret-msg"] = "error: handle_ret";

    cv::Scalar cl_red = Scalar(255, 0, 255);

    std::vector<cv::Point2f> pts = det_ret.pts;
    cv::Point2f pt_centor(-100, -100);

    if (det_ret.pts.size() >= 4)
    {
        pt_centor = det_ret.pts[0];

        if (det_ret.pts.size() == det_ret.sizes.size())
        {
            pts = det_ret.pts_new;
        }
        else
        {
            pts.erase(pts.begin());
        }
        
    }
    else
    {
        // 不能少于4个点
        j["ret-msg"] = "error: det points is less than 4";
        return ret;
    }

    // 找到异常点
    std::vector<PointWithStats> point_stats2;
    Circle c = CircleFitterWithOutlierRemoval::fitCircleWithStatisticalOutlierRemoval(
        pts, point_stats2);

    // draw pt
    cv::circle(image, c.center, 2, cl_red);

    cv::circle(image, c.center, c.radius, cl_red);


#ifdef REL_DBG
    cv::Mat img_dbg = image.clone();

    // draw pt
    cv::circle(img_dbg, c.center, 2, cl_red);

    cv::circle(img_dbg, c.center, c.radius, cl_red);

    G_DET->draw_pan(img_dbg, det_ret);

#endif // REL_DBG

    // 如果 0 1 2 都是异常点 就要校正一下


    // Draw detections on the image
    G_DET->draw_pan(image, det_ret);


    // 根据p1的坐标，求出三个圆孔的角度
    cv::Point2f pt_hole_up = cv::Point2f(c.center.x + c.radius, c.center.y);

    // 矫正角度  
    if (std::abs(g_diff_angle)-0.01>0)
    {
        float cur_diff_angle = g_diff_angle * arc_angle;
        cv::Point2f hole1 = getRotatedPointOnCircle(c.center, c.radius, pt_hole_up, cur_diff_angle);
        pt_hole_up = hole1;
    }


    // 求出每一个hole和 pt_hole_up 的夹角
    std::vector<float> list_delta_angle;
    for (auto &p : pts)
    {
        float delta_angle = calculateAngleBetweenLines(c.center, pt_hole_up, p);
        list_delta_angle.push_back(delta_angle);
    }


    std::vector<size_t> list_idx = sortIndexes(list_delta_angle);

    std::vector<cv::Point2f> pt_hole;
    for (size_t i = 0; i < list_idx.size(); i++)
    {
        pt_hole.push_back(pts[list_idx[i]]);
    }

    // up line
    cv::line(image, c.center, pt_hole_up, Scalar(0, 0, 255));

    // cv::line(image, c.center, pt_hole[0], Scalar(255, 0, 0));
    cv::line(image, c.center, pt_hole[1], Scalar(255, 0, 0));
    cv::line(image, c.center, pt_hole[2], Scalar(255, 0, 0));

    cv::line(image, c.center, pt_hole[0], Scalar(0, 255, 255), 2);


    // 计算最近大孔和最上面 需要旋转的角度
    // float delta_angle = calculateAngleBetweenLines(c.center, pt_hole_up, pt_hole[0]);
    float delta_angle = list_delta_angle[list_idx[0]];

    // 按配置输出 正负号 默认 g_anti_clock==1 就是 逆 时针是负方向
    // 转成角度显示
    if (pt_hole[0].y > pt_hole_up.y)
    {
        // 目前大孔在 最上面的 右边，需要 逆 时针旋转
        delta_angle = std::abs(delta_angle) * (g_anti_clock);
    }
    else
    {
        // 目前大孔在 最上面的 左边，需要 顺 时针旋转
        delta_angle = std::abs(delta_angle) * (-g_anti_clock);
    }

    // 转成角度显示
    delta_angle = delta_angle / arc_angle;

    // ==========================================
    // 全局缓存
    g_cur_det_ret.centor = cv::Point3f(c.center.x, c.center.y,0);
    g_cur_det_ret.radius = c.radius;

    g_cur_det_ret.pts = pt_hole;
    g_cur_det_ret.angle_det = delta_angle;

    // ==========================================
    // 在这里校正一下
    if (0 == g_update_template)
    {
        if (0 == g_adjust_mode)
        {
            // 多项式方式
            delta_angle = fitter.predict(delta_angle);
        }
        else
        {
            // 最近模板比对
            delta_angle = adjust_angle_by_template(delta_angle);
        }

    }


    putText(image, string_format("%.2f", delta_angle), c.center, FONT_HERSHEY_DUPLEX, 0.6, Scalar(0, 0, 255));

    ret = 0;
    j["angle"] = delta_angle;
    j["ret-msg"] = "ok";

    // 矫正角度  
    if (std::abs(g_diff_angle) - 0.01 > 0)
    {
        j["zero_diff"] = formatFloat(g_diff_angle);
    }

    det_ret.angle = delta_angle;


    return ret;
}

bool bbox_contains(Rect2f& big_box, Rect2f& small_box,float ratio = 0.8)
{
    if (small_box.area()<=0)
    {
        return false;
    }

    cv::Rect2f intersect = big_box & small_box;

    return intersect.area()/small_box.area()>=ratio;    
    
}

static float boxOverlapWithTray(const cv::Rect2f& tray, const cv::Rect2f& candidate)
{
    if (candidate.area() <= 0.0f)
        return 0.0f;
    cv::Rect2f intersection = tray & candidate;
    return intersection.area() / candidate.area();
}

static void draw_plate_empty_final_dets(Mat& image, vector< vector<Detection_pan>>& list_dets)
{
    for (auto& d : list_dets[PLATE_CELL])
    {
        if (d.class_id >= 0)
            G_DET->draw_pan(image, d);
    }
    for (auto& d : list_dets[PLATE_HOLE])
    {
        if (d.class_id >= 0)
            G_DET->draw_pan(image, d);
    }
    for (auto& d : list_dets[LUNZI])
    {
        if (d.class_id >= 0)
            G_DET->draw_pan(image, d);
    }
    for (auto& d : list_dets[HOLE])
    {
        if (d.class_id >= 0)
            G_DET->draw_pan(image, d);
    }
}

enum class TraySlotState {
    Unknown,
    Empty,
    Wheel
};

enum class TrayCandidateRole {
    Primary,
    Supplemental,
    Rejected
};

enum class TrayCandidateReason {
    None,
    OutsideSelectionRect,
    SelectionRectBoundary,
    OutsideTray,
    BoundaryLowOverlap,
    InvalidBbox,
    PrimaryGridUnavailable,
    SupplementalNoOpenSlot,
    SupplementalTooFar,
    SupplementalConflict,
    DuplicateSlot
};

static const char* trayCandidateRoleName(TrayCandidateRole role)
{
    switch (role)
    {
    case TrayCandidateRole::Primary: return "primary";
    case TrayCandidateRole::Supplemental: return "supplemental";
    case TrayCandidateRole::Rejected: return "rejected";
    }
    return "rejected";
}

static const char* trayCandidateReasonName(TrayCandidateReason reason)
{
    switch (reason)
    {
    case TrayCandidateReason::None: return "none";
    case TrayCandidateReason::OutsideSelectionRect:
        return "outside_selection_rect";
    case TrayCandidateReason::SelectionRectBoundary:
        return "selection_rect_boundary";
    case TrayCandidateReason::OutsideTray: return "outside_tray";
    case TrayCandidateReason::BoundaryLowOverlap:
        return "boundary_low_overlap";
    case TrayCandidateReason::InvalidBbox: return "invalid_bbox";
    case TrayCandidateReason::PrimaryGridUnavailable:
        return "primary_grid_unavailable";
    case TrayCandidateReason::SupplementalNoOpenSlot:
        return "supplemental_no_open_slot";
    case TrayCandidateReason::SupplementalTooFar:
        return "supplemental_too_far";
    case TrayCandidateReason::SupplementalConflict:
        return "supplemental_conflict";
    case TrayCandidateReason::DuplicateSlot: return "duplicate_slot";
    }
    return "none";
}

struct TrayCandidate {
    TraySlotState state = TraySlotState::Unknown;
    int class_id = -1;
    float confidence = 0.0f;
    cv::Rect2f bbox;
    cv::Point2f center;
    cv::Point2f object_center;
    cv::Point2f hole_center;
    int source_index = -1;
    bool has_center_hole = false;
    bool size_outlier = false;
    bool acceptable_size = false;
    bool fallback_primary = false;
    bool used_for_grid = false;

    float tray_overlap_ratio = 0.0f;
    float tray_edge_distance = -1.0f;
    float width_ratio = 0.0f;
    float height_ratio = 0.0f;
    bool object_center_in_tray = false;
    bool object_center_in_safe_tray = false;
    bool hole_center_in_tray = false;
    bool hole_center_in_safe_tray = false;

    // 业务 rect 是粗拍目标料盘的第一层约束。候选不会因为靠近边界而
    // 立即消失：边界候选进入 Supplemental，供后续缺格恢复使用。
    bool selection_rect_enabled = false;
    bool object_center_in_selection_rect = false;
    bool object_center_in_safe_selection_rect = false;
    float selection_overlap_ratio = 0.0f;
    float selection_edge_distance = -1.0f;

    // 同一格位可能同时被模型报成 plate_cell 和 lunzi。合并后保留两类
    // 证据，几何拟合只消费一个位置，状态在归位后单独裁决。
    bool has_empty_evidence = false;
    bool has_wheel_evidence = false;
    float empty_confidence = 0.0f;
    float wheel_confidence = 0.0f;
    int merged_evidence_count = 1;
    bool state_recovered = false;
    bool position_reconstructed = false;
    bool local_depth_valid = false;
    float local_depth = 0.0f;
    float depth_delta_to_plane = 0.0f;

    TrayCandidateRole role = TrayCandidateRole::Rejected;
    TrayCandidateReason reason = TrayCandidateReason::None;
};

struct TraySlot {
    int index = -1;
    cv::Point2f grid_center;
    TraySlotState state = TraySlotState::Unknown;
    int candidate_index = -1;
    float match_distance = FLT_MAX;
};

struct TrayPostprocessResult {
    std::array<TraySlot, 8> slots;
    std::vector<TrayCandidate> candidates;
    int empty_count = 0;
    int wheel_count = 0;
    int unknown_count = 8;
    bool tray_complete = false;
    std::string tray_source;
    cv::Rect2f tray_roi;
    cv::Rect2f safe_tray_roi;
};

struct TrayPostprocessConfig {
    float min_tray_overlap_ratio = 0.60f;
    float safe_edge_margin_ratio = 0.08f;
    float safe_edge_margin_min_px = 8.0f;
    float safe_edge_margin_max_px = 24.0f;

    float primary_min_width_ratio = 0.75f;
    float primary_max_width_ratio = 1.30f;
    float primary_min_height_ratio = 0.75f;
    float primary_max_height_ratio = 1.30f;

    float acceptable_min_width_ratio = 0.65f;
    float acceptable_max_width_ratio = 1.40f;
    float acceptable_min_height_ratio = 0.65f;
    float acceptable_max_height_ratio = 1.40f;

    float grid_match_ratio = 0.40f;
    float supplemental_conflict_margin = 0.15f;
    float selection_min_overlap_ratio = 0.55f;
    float selection_recovery_overlap_ratio = 0.20f;
    float selection_safe_edge_margin_ratio = 0.25f;
    float selection_safe_edge_margin_min_px = 36.0f;
    float selection_safe_edge_margin_max_px = 80.0f;
    float same_slot_merge_ratio = 0.32f;
    float wheel_state_min_margin = 0.12f;
    float empty_dominant_min_ratio = 0.60f;
    int empty_dominant_min_slots = 4;
    float depth_plane_tolerance_mm = 14.0f;
    float depth_wheel_min_height_mm = 18.0f;
    float depth_annulus_inner_ratio = 0.15f;
    float depth_annulus_outer_ratio = 0.34f;
    float depth_min_valid_ratio = 0.25f;
};

static float medianValue(std::vector<float> vals)
{
    if (vals.empty())
        return 0.0f;
    std::sort(vals.begin(), vals.end());
    const size_t mid = vals.size() / 2;
    if ((vals.size() % 2) != 0)
        return vals[mid];
    return (vals[mid - 1] + vals[mid]) * 0.5f;
}

static float dist2d(const cv::Point2f& a, const cv::Point2f& b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static cv::Rect2f insetRect(const cv::Rect2f& rect, float margin)
{
    if (rect.width <= 0.0f || rect.height <= 0.0f)
        return cv::Rect2f();
    float max_margin = std::max(
        0.0f, (std::min(rect.width, rect.height) - 1.0f) * 0.5f);
    margin = std::max(0.0f, std::min(margin, max_margin));
    return cv::Rect2f(
        rect.x + margin, rect.y + margin,
        rect.width - 2.0f * margin, rect.height - 2.0f * margin);
}

static bool hasValidTraySelectionRect()
{
    if (g_rect.size() != 8)
        return false;

    std::vector<cv::Point2f> polygon;
    polygon.reserve(4);
    for (int i = 0; i < 4; ++i)
        polygon.emplace_back(g_rect[i * 2], g_rect[i * 2 + 1]);
    return std::abs(cv::contourArea(polygon)) > 1.0;
}

static std::vector<cv::Point2f> currentTraySelectionPolygon()
{
    std::vector<cv::Point2f> polygon;
    if (!hasValidTraySelectionRect())
        return polygon;
    polygon.reserve(4);
    for (int i = 0; i < 4; ++i)
        polygon.emplace_back(g_rect[i * 2], g_rect[i * 2 + 1]);
    return polygon;
}

static float boxOverlapWithConvexPolygon(
    const cv::Rect2f& bbox,
    const std::vector<cv::Point2f>& polygon)
{
    if (bbox.area() <= 0.0f || polygon.size() < 3)
        return 0.0f;
    std::vector<cv::Point2f> box = {
        bbox.tl(),
        cv::Point2f(bbox.x + bbox.width, bbox.y),
        bbox.br(),
        cv::Point2f(bbox.x, bbox.y + bbox.height)
    };
    std::vector<cv::Point2f> intersection;
    const float area = cv::intersectConvexConvex(box, polygon, intersection);
    return area > 0.0f ? area / bbox.area() : 0.0f;
}

static float signedDistanceToSelectionPolygon(
    const std::vector<cv::Point2f>& polygon,
    const cv::Point2f& point)
{
    if (polygon.size() < 3)
        return -1.0f;
    return static_cast<float>(cv::pointPolygonTest(polygon, point, true));
}

static float distanceToTrayEdge(
    const cv::Rect2f& tray,
    const cv::Point2f& center)
{
    if (!tray.contains(center))
        return -1.0f;
    return std::min({
        center.x - tray.x,
        tray.x + tray.width - center.x,
        center.y - tray.y,
        tray.y + tray.height - center.y });
}

struct TrayReferenceSize {
    float width = 0.0f;
    float height = 0.0f;
    float side = 0.0f;
    cv::Rect2f safe_tray;
    int rect_reference_candidate_count = 0;
    int tray_reference_candidate_count = 0;
    int reference_pool_count = 0;
    int excluded_from_reference_count = 0;
    bool tray_filter_triggered = false;
    bool tray_filter_applied = false;
    bool valid = false;
};

static TrayReferenceSize computeCandidateEvidence(
    std::vector<TrayCandidate>& cands,
    const cv::Rect2f& tray,
    const std::vector<cv::Point2f>* selection_polygon,
    const std::string& tray_source,
    const TrayPostprocessConfig& cfg,
    bool use_tray_reference_pool)
{
    std::vector<float> all_w;
    std::vector<float> all_h;
    std::vector<float> empty_w;
    std::vector<float> empty_h;
    std::vector<float> wheel_w;
    std::vector<float> wheel_h;

    // 先计算与候选尺寸无关的空间证据。任务20的有效 Rect 候选超过8个
    // 且模型检出了真实 tray 时，只允许同时属于 Rect 和 tray 的候选
    // 建立尺寸参考，避免相邻托盘把中位尺寸拉偏。完整候选数组不裁剪，
    // 后续分类、同格合并和 Supplemental 缺格恢复仍沿用原流程。
    for (auto& c : cands)
    {
        c.tray_overlap_ratio = boxOverlapWithTray(tray, c.bbox);
        c.object_center_in_tray = tray.contains(c.object_center);
        c.tray_edge_distance = distanceToTrayEdge(tray, c.object_center);
        c.hole_center_in_tray =
            c.has_center_hole && tray.contains(c.hole_center);
        c.selection_rect_enabled =
            selection_polygon != nullptr && selection_polygon->size() >= 3;
        if (c.selection_rect_enabled)
        {
            c.selection_overlap_ratio = boxOverlapWithConvexPolygon(
                c.bbox, *selection_polygon);
            c.selection_edge_distance = signedDistanceToSelectionPolygon(
                *selection_polygon, c.object_center);
            c.object_center_in_selection_rect =
                c.selection_edge_distance >= 0.0f;
        }
    }

    auto belongs_to_rect_reference = [&](const TrayCandidate& c) {
        return c.selection_rect_enabled &&
            c.object_center_in_selection_rect &&
            c.selection_overlap_ratio >=
                cfg.selection_recovery_overlap_ratio;
    };
    auto belongs_to_detected_tray_reference =
        [&](const TrayCandidate& c) {
            // Rect remains the coarse business boundary.  The detected tray
            // only disambiguates candidates that already belong to the Rect.
            return belongs_to_rect_reference(c) &&
                c.object_center_in_tray &&
                c.tray_overlap_ratio >= cfg.min_tray_overlap_ratio;
        };

    TrayReferenceSize reference;
    for (const auto& c : cands)
    {
        if (c.bbox.width <= 0.0f || c.bbox.height <= 0.0f)
            continue;
        if (belongs_to_rect_reference(c))
            ++reference.rect_reference_candidate_count;
        if (belongs_to_detected_tray_reference(c))
            ++reference.tray_reference_candidate_count;
    }

    reference.tray_filter_triggered =
        use_tray_reference_pool &&
        tray_source == "detected" &&
        reference.rect_reference_candidate_count > 8;
    // 至少保留现有网格拟合入口所要求的4个候选；证据不足时保持旧的
    // Rect参考池，不为plate_tray增强逻辑新增失败条件。
    reference.tray_filter_applied =
        reference.tray_filter_triggered &&
        reference.tray_reference_candidate_count >= 4;

    for (const auto& c : cands)
    {
        if (c.bbox.width <= 0.0f || c.bbox.height <= 0.0f)
            continue;

        const bool belongs_to_primary_region =
            reference.tray_filter_applied ?
                belongs_to_detected_tray_reference(c) :
                (c.selection_rect_enabled ?
                    belongs_to_rect_reference(c) :
                    (c.object_center_in_tray &&
                     c.tray_overlap_ratio >=
                        cfg.min_tray_overlap_ratio));
        if (use_tray_reference_pool && !belongs_to_primary_region)
        {
            ++reference.excluded_from_reference_count;
            continue;
        }

        ++reference.reference_pool_count;
        all_w.push_back(c.bbox.width);
        all_h.push_back(c.bbox.height);
        if (c.state == TraySlotState::Empty)
        {
            empty_w.push_back(c.bbox.width);
            empty_h.push_back(c.bbox.height);
        }
        else if (c.state == TraySlotState::Wheel)
        {
            wheel_w.push_back(c.bbox.width);
            wheel_h.push_back(c.bbox.height);
        }
    }

    if (all_w.empty() || all_h.empty())
        return reference;

    reference.width = medianValue(all_w);
    reference.height = medianValue(all_h);
    reference.side = 0.5f * (reference.width + reference.height);
    reference.valid =
        reference.width > 0.0f && reference.height > 0.0f &&
        std::isfinite(reference.width) && std::isfinite(reference.height);
    if (!reference.valid)
        return reference;

    float edge_margin = std::max(
        cfg.safe_edge_margin_min_px,
        std::min(reference.side * cfg.safe_edge_margin_ratio,
                 cfg.safe_edge_margin_max_px));
    reference.safe_tray = insetRect(tray, edge_margin);

    const float empty_mw = empty_w.size() >= 2 ?
        medianValue(empty_w) : reference.width;
    const float empty_mh = empty_h.size() >= 2 ?
        medianValue(empty_h) : reference.height;
    const float wheel_mw = wheel_w.size() >= 2 ?
        medianValue(wheel_w) : reference.width;
    const float wheel_mh = wheel_h.size() >= 2 ?
        medianValue(wheel_h) : reference.height;

    for (auto& c : cands)
    {
        const float ref_w = (c.state == TraySlotState::Empty) ?
            empty_mw : wheel_mw;
        const float ref_h = (c.state == TraySlotState::Empty) ?
            empty_mh : wheel_mh;
        if (ref_w <= 0.0f || ref_h <= 0.0f)
            continue;

        c.width_ratio = c.bbox.width / ref_w;
        c.height_ratio = c.bbox.height / ref_h;
        const bool primary_size =
            c.width_ratio >= cfg.primary_min_width_ratio &&
            c.width_ratio <= cfg.primary_max_width_ratio &&
            c.height_ratio >= cfg.primary_min_height_ratio &&
            c.height_ratio <= cfg.primary_max_height_ratio;
        c.acceptable_size =
            c.width_ratio >= cfg.acceptable_min_width_ratio &&
            c.width_ratio <= cfg.acceptable_max_width_ratio &&
            c.height_ratio >= cfg.acceptable_min_height_ratio &&
            c.height_ratio <= cfg.acceptable_max_height_ratio;
        c.size_outlier = !primary_size;

        c.object_center_in_safe_tray =
            reference.safe_tray.contains(c.object_center);
        c.hole_center_in_safe_tray =
            c.has_center_hole && reference.safe_tray.contains(c.hole_center);
        if (c.selection_rect_enabled)
        {
            const float selection_edge_margin = std::max(
                cfg.selection_safe_edge_margin_min_px,
                std::min(reference.side *
                    cfg.selection_safe_edge_margin_ratio,
                    cfg.selection_safe_edge_margin_max_px));
            c.object_center_in_safe_selection_rect =
                c.selection_edge_distance >= selection_edge_margin;
        }

        // fitted tray 由候选自身生成，只记录边界证据，不能用来循环自证。
        (void)tray_source;
    }

#ifdef REL_DBG
    std::cout << "tray reference filter triggered="
              << (reference.tray_filter_triggered ? 1 : 0)
              << " applied=" << (reference.tray_filter_applied ? 1 : 0)
              << " rect_count="
              << reference.rect_reference_candidate_count
              << " tray_count="
              << reference.tray_reference_candidate_count
              << " min_tray_count=4"
              << std::endl;
    std::cout << "tray reference pool enabled="
              << (use_tray_reference_pool ? 1 : 0)
              << " count=" << reference.reference_pool_count
              << " excluded=" << reference.excluded_from_reference_count
              << " size=(" << reference.width << ','
              << reference.height << ')' << std::endl;
#endif
    return reference;
}

static void classifyCandidateRoles(
    std::vector<TrayCandidate>& cands,
    const std::string& tray_source,
    const TrayPostprocessConfig& cfg)
{
    const bool fitted_tray = tray_source == "fitted";
    for (auto& c : cands)
    {
        c.fallback_primary = false;
        c.used_for_grid = false;
        c.reason = TrayCandidateReason::None;

        if (c.bbox.width <= 0.0f || c.bbox.height <= 0.0f ||
            !std::isfinite(c.bbox.width) || !std::isfinite(c.bbox.height))
        {
            c.role = TrayCandidateRole::Rejected;
            c.reason = TrayCandidateReason::InvalidBbox;
            continue;
        }

        const bool primary_size = !c.size_outlier;

        // rect 是业务明确指定的目标区域，必须先于 plate_tray 生效。
        // 靠边候选不直接删除，而是隔离到 Supplemental，后续仅能补 Unknown。
        if (c.selection_rect_enabled)
        {
            const bool recoverable_by_overlap =
                c.selection_overlap_ratio >=
                    cfg.selection_recovery_overlap_ratio;
            if (!c.object_center_in_selection_rect &&
                !recoverable_by_overlap)
            {
                c.role = TrayCandidateRole::Rejected;
                c.reason = TrayCandidateReason::OutsideSelectionRect;
                continue;
            }

            if (!c.object_center_in_safe_selection_rect ||
                c.selection_overlap_ratio <
                    cfg.selection_min_overlap_ratio)
            {
                if (c.acceptable_size || recoverable_by_overlap)
                {
                    c.role = TrayCandidateRole::Supplemental;
                    c.reason = TrayCandidateReason::SelectionRectBoundary;
                }
                else
                {
                    c.role = TrayCandidateRole::Rejected;
                    c.reason = TrayCandidateReason::InvalidBbox;
                }
                continue;
            }
        }

        if (fitted_tray)
        {
            if (primary_size)
                c.role = TrayCandidateRole::Primary;
            else if (c.acceptable_size)
                c.role = TrayCandidateRole::Supplemental;
            else
            {
                c.role = TrayCandidateRole::Rejected;
                c.reason = TrayCandidateReason::InvalidBbox;
            }
            continue;
        }

        const bool compatible_edge_keep =
            c.tray_overlap_ratio >= cfg.min_tray_overlap_ratio &&
            c.hole_center_in_tray;
        if (!c.object_center_in_tray && !compatible_edge_keep)
        {
            c.role = TrayCandidateRole::Rejected;
            c.reason = TrayCandidateReason::OutsideTray;
            continue;
        }

        if (c.object_center_in_safe_tray && primary_size &&
            c.tray_overlap_ratio >= cfg.min_tray_overlap_ratio)
        {
            c.role = TrayCandidateRole::Primary;
            continue;
        }

        if (!c.acceptable_size)
        {
            c.role = TrayCandidateRole::Rejected;
            c.reason = TrayCandidateReason::InvalidBbox;
            continue;
        }

        c.role = TrayCandidateRole::Supplemental;
        if (!c.object_center_in_safe_tray ||
            c.tray_overlap_ratio < cfg.min_tray_overlap_ratio)
            c.reason = TrayCandidateReason::BoundaryLowOverlap;
    }

#ifdef REL_DBG
    int primary_count = 0;
    int supplemental_count = 0;
    int rejected_count = 0;
    for (size_t i = 0; i < cands.size(); ++i)
    {
        const auto& c = cands[i];
        if (c.role == TrayCandidateRole::Primary) ++primary_count;
        else if (c.role == TrayCandidateRole::Supplemental)
            ++supplemental_count;
        else ++rejected_count;
        std::cout << "candidate idx=" << i
                  << " class=" << c.class_id
                  << " center=(" << c.center.x << ',' << c.center.y << ')'
                  << " overlap=" << c.tray_overlap_ratio
                  << " rect_overlap=" << c.selection_overlap_ratio
                  << " rect_edge=" << c.selection_edge_distance
                  << " edge_distance=" << c.tray_edge_distance
                  << " size_ratio=(" << c.width_ratio << ','
                  << c.height_ratio << ')'
                  << " has_hole=" << c.has_center_hole
                  << " confidence=" << c.confidence
                  << " role=" << trayCandidateRoleName(c.role)
                  << " reason=" << trayCandidateReasonName(c.reason)
                  << std::endl;
    }
    std::cout << "primary_count=" << primary_count
              << " supplemental_count=" << supplemental_count
              << " rejected_count=" << rejected_count << std::endl;
#endif
}

static std::vector<TrayCandidate> buildTrayCandidates(
    const vector< vector<Detection_pan>>& list_dets,
    const cv::Rect2f& tray)
{
    std::vector<TrayCandidate> cands;

    auto append_class_candidates = [&](int object_class, int hole_class,
                                       TraySlotState state) {
        const auto& objects = list_dets[object_class];
        const auto& holes = list_dets[hole_class];
        std::vector<bool> hole_used(holes.size(), false);

        for (size_t i = 0; i < objects.size(); ++i)
        {
            const auto& obj = objects[i];
            if (obj.class_id < 0)
                continue;

            TrayCandidate c;
            c.state = state;
            c.class_id = object_class;
            c.confidence = obj.conf;
            c.bbox = obj.bbox;
            c.source_index = static_cast<int>(i);
            c.object_center = (obj.bbox.tl() + obj.bbox.br()) * 0.5f;
            c.center = c.object_center;
            c.has_empty_evidence = state == TraySlotState::Empty;
            c.has_wheel_evidence = state == TraySlotState::Wheel;
            c.empty_confidence = c.has_empty_evidence ? obj.conf : 0.0f;
            c.wheel_confidence = c.has_wheel_evidence ? obj.conf : 0.0f;

            // 模型输出数组顺序不是接口保证。孔按空间关系做一对一匹配：
            // 必须落在外框内，并优先选择离外框几何中心最近的孔。
            int best_hole = -1;
            float best_cost = std::numeric_limits<float>::max();
            const float max_center_distance = 0.45f *
                std::max(obj.bbox.width, obj.bbox.height);
            for (size_t hi = 0; hi < holes.size(); ++hi)
            {
                if (hole_used[hi] || holes[hi].class_id < 0 ||
                    holes[hi].bbox.area() <= 0.0f)
                    continue;
                const cv::Point2f hc =
                    (holes[hi].bbox.tl() + holes[hi].bbox.br()) * 0.5f;
                const float d = dist2d(c.object_center, hc);
                if (!obj.bbox.contains(hc) || d > max_center_distance)
                    continue;
                const float cost = d - holes[hi].conf * 2.0f;
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_hole = static_cast<int>(hi);
                }
            }

            if (best_hole >= 0)
            {
                hole_used[best_hole] = true;
                const auto& hole = holes[best_hole];
                c.has_center_hole = true;
                c.hole_center =
                    (hole.bbox.tl() + hole.bbox.br()) * 0.5f;
                c.center = c.hole_center;
            }
            cands.push_back(c);
        }
    };

    append_class_candidates(PLATE_CELL, PLATE_HOLE, TraySlotState::Empty);
    append_class_candidates(LUNZI, HOLE, TraySlotState::Wheel);

    (void)tray;
    return cands;
}

static bool resolveTrayRoi(
    const vector< vector<Detection_pan>>& list_dets,
    const std::vector<cv::Point2f>& candidate_centers,
    float cell_size,
    cv::Rect2f& tray_roi,
    std::string& tray_source)
{
    if (!list_dets[PLATE_TRAY].empty())
    {
        tray_roi = list_dets[PLATE_TRAY][0].bbox;
        tray_source = "detected";
        return true;
    }

    if (candidate_centers.size() < 4)
        return false;

    cv::RotatedRect rc = cv::minAreaRect(candidate_centers);
    tray_roi = rc.boundingRect();
    float pad = cell_size * 0.5f;
    tray_roi.x -= pad;
    tray_roi.y -= pad;
    tray_roi.width += 2.0f * pad;
    tray_roi.height += 2.0f * pad;
    tray_source = "fitted";
    return true;
}

static int trayCandidateRoleRank(TrayCandidateRole role)
{
    if (role == TrayCandidateRole::Primary) return 2;
    if (role == TrayCandidateRole::Supplemental) return 1;
    return 0;
}

static std::vector<TrayCandidate> consolidateSameSlotCandidates(
    const std::vector<TrayCandidate>& input,
    float reference_side,
    const TrayPostprocessConfig& cfg)
{
    if (input.empty() || reference_side <= 0.0f)
        return input;

    const float merge_distance =
        std::max(12.0f, reference_side * cfg.same_slot_merge_ratio);
    std::vector<bool> consumed(input.size(), false);
    std::vector<TrayCandidate> output;
    output.reserve(input.size());

    for (size_t seed = 0; seed < input.size(); ++seed)
    {
        if (consumed[seed])
            continue;

        std::vector<int> group;
        group.push_back(static_cast<int>(seed));
        consumed[seed] = true;
        for (size_t cursor = 0; cursor < group.size(); ++cursor)
        {
            const int base = group[cursor];
            for (size_t j = seed + 1; j < input.size(); ++j)
            {
                if (consumed[j])
                    continue;
                if (dist2d(input[base].center, input[j].center) <=
                    merge_distance)
                {
                    consumed[j] = true;
                    group.push_back(static_cast<int>(j));
                }
            }
        }

        float empty_confidence = 0.0f;
        float wheel_confidence = 0.0f;
        bool has_empty = false;
        bool has_wheel = false;
        int best_role_rank = -1;
        for (int index : group)
        {
            const auto& c = input[index];
            has_empty = has_empty || c.has_empty_evidence ||
                c.state == TraySlotState::Empty;
            has_wheel = has_wheel || c.has_wheel_evidence ||
                c.state == TraySlotState::Wheel;
            empty_confidence = std::max(empty_confidence,
                c.has_empty_evidence || c.state == TraySlotState::Empty ?
                    std::max(c.empty_confidence, c.confidence) : 0.0f);
            wheel_confidence = std::max(wheel_confidence,
                c.has_wheel_evidence || c.state == TraySlotState::Wheel ?
                    std::max(c.wheel_confidence, c.confidence) : 0.0f);
            best_role_rank = std::max(
                best_role_rank, trayCandidateRoleRank(c.role));
        }

        TraySlotState resolved_state = TraySlotState::Unknown;
        if (has_empty && has_wheel)
        {
            resolved_state =
                wheel_confidence - empty_confidence >=
                    cfg.wheel_state_min_margin ?
                TraySlotState::Wheel : TraySlotState::Empty;
        }
        else if (has_empty)
            resolved_state = TraySlotState::Empty;
        else if (has_wheel)
            resolved_state = TraySlotState::Wheel;

        int representative = group.front();
        float best_score = -std::numeric_limits<float>::max();
        for (int index : group)
        {
            const auto& c = input[index];
            float score = trayCandidateRoleRank(c.role) * 10.0f;
            if (c.state == resolved_state) score += 3.0f;
            if (c.has_center_hole) score += 1.0f;
            if (!c.size_outlier) score += 0.5f;
            score += c.confidence;
            if (score > best_score)
            {
                best_score = score;
                representative = index;
            }
        }

        TrayCandidate merged = input[representative];
        merged.state = resolved_state;
        merged.class_id = resolved_state == TraySlotState::Wheel ?
            LUNZI : PLATE_CELL;
        merged.confidence = resolved_state == TraySlotState::Wheel ?
            wheel_confidence : empty_confidence;
        merged.has_empty_evidence = has_empty;
        merged.has_wheel_evidence = has_wheel;
        merged.empty_confidence = empty_confidence;
        merged.wheel_confidence = wheel_confidence;
        merged.merged_evidence_count = static_cast<int>(group.size());
        merged.state_recovered = has_empty && has_wheel;
        if (best_role_rank == 2)
        {
            merged.role = TrayCandidateRole::Primary;
            merged.reason = TrayCandidateReason::None;
        }
        else if (best_role_rank == 1)
        {
            merged.role = TrayCandidateRole::Supplemental;
            if (merged.reason == TrayCandidateReason::None)
                merged.reason = TrayCandidateReason::BoundaryLowOverlap;
        }

#ifdef REL_DBG
        if (group.size() > 1)
        {
            std::cout << "merge same slot evidence_count=" << group.size()
                      << " center=(" << merged.center.x << ','
                      << merged.center.y << ") empty_conf="
                      << empty_confidence << " wheel_conf="
                      << wheel_confidence << " resolved="
                      << (resolved_state == TraySlotState::Empty ?
                          "empty" : "wheel") << std::endl;
        }
#endif
        output.push_back(merged);
    }

    return output;
}

static float referenceSideForCandidates(
    const std::vector<TrayCandidate>& candidates,
    const std::vector<int>& indices)
{
    std::vector<float> widths;
    std::vector<float> heights;
    widths.reserve(indices.size());
    heights.reserve(indices.size());
    for (int index : indices)
    {
        if (index < 0 || index >= static_cast<int>(candidates.size()))
            continue;
        const auto& c = candidates[index];
        if (c.bbox.width > 0.0f && c.bbox.height > 0.0f)
        {
            widths.push_back(c.bbox.width);
            heights.push_back(c.bbox.height);
        }
    }
    return 0.5f * (medianValue(widths) + medianValue(heights));
}

static bool buildPrimaryGrid(
    std::vector<TrayCandidate>& candidates,
    const cv::Rect2f& tray_roi,
    const std::vector<cv::Point2f>* selection_polygon,
    const cv::Mat* edge_image,
    cv::Mat* p_img_dbg,
    TrayFitting& tray_fitter,
    std::array<cv::Point2f, 8>& ordered_grid,
    RowPairSelectionResult& selection_result,
    float& reference_side,
    std::string& failure_reason)
{
    std::vector<int> primary_indices;
    std::vector<int> fallback_indices;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        candidates[i].used_for_grid = false;
        candidates[i].fallback_primary = false;
        if (candidates[i].role == TrayCandidateRole::Primary)
        {
            primary_indices.push_back(static_cast<int>(i));
            fallback_indices.push_back(static_cast<int>(i));
        }
    }

    // 一次确定性回退：只整体加入尺寸尚可且不是低交叠边缘带的候选。
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const auto& c = candidates[i];
        if (c.role == TrayCandidateRole::Supplemental &&
            c.acceptable_size &&
            c.reason != TrayCandidateReason::BoundaryLowOverlap &&
            c.reason != TrayCandidateReason::SelectionRectBoundary)
            fallback_indices.push_back(static_cast<int>(i));
    }

    auto try_fit = [&](const std::vector<int>& indices,
                       std::string& attempt_reason) {
        if (indices.size() < 4)
        {
            attempt_reason = "too few primary tray candidates";
            return false;
        }

        std::vector<cv::Point2f> centers;
        std::vector<DetectionPoint> detections;
        centers.reserve(indices.size());
        detections.reserve(indices.size());
        for (int index : indices)
        {
            const auto& c = candidates[index];
            centers.push_back(c.center);
            detections.push_back(
                { c.class_id, c.confidence, c.center.x, c.center.y });
        }

        reference_side = referenceSideForCandidates(candidates, indices);
        if (reference_side <= 0.0f)
        {
            attempt_reason = "invalid primary reference size";
            return false;
        }

        cv::RotatedRect rc = cv::minAreaRect(centers);
        float degrees = rc.size.height > rc.size.width ?
            rc.angle - 90.0f : -rc.angle;
        tray_fitter.all_pts = detections;
        tray_fitter.bbox = tray_roi;
        tray_fitter.cell_size = static_cast<size_t>(
            std::max(1.0f, std::round(reference_side)));
        tray_fitter.center_pt = rc.center;
        tray_fitter.radians = degrees * static_cast<float>(M_PI) / 180.0f;
        tray_fitter.p_img_dbg = p_img_dbg;
        selection_result = RowPairSelectionResult{};

#ifdef REL_DBG
        std::cout << "grid_input_count=" << indices.size()
                  << " grid_angle=" << degrees
                  << " reference_side=" << reference_side << std::endl;
#endif
        return tray_fitter.fitOrderedGrid2x4(
            &ordered_grid, selection_polygon, &attempt_reason,
            edge_image, &selection_result);
    };

    std::string primary_reason;
    if (try_fit(primary_indices, primary_reason))
    {
        for (int index : primary_indices)
            candidates[index].used_for_grid = true;
        failure_reason.clear();
        return true;
    }

    if (fallback_indices.size() > primary_indices.size())
    {
        std::string fallback_reason;
        if (try_fit(fallback_indices, fallback_reason))
        {
            for (int index : fallback_indices)
            {
                candidates[index].used_for_grid = true;
                if (candidates[index].role == TrayCandidateRole::Supplemental)
                    candidates[index].fallback_primary = true;
            }
            failure_reason.clear();
            return true;
        }
        failure_reason = "primary grid unavailable: " + primary_reason +
            "; fallback failed: " + fallback_reason;
    }
    else
        failure_reason = "primary grid unavailable: " + primary_reason;

    for (auto& c : candidates)
    {
        if (c.role != TrayCandidateRole::Rejected &&
            c.reason == TrayCandidateReason::None)
            c.reason = TrayCandidateReason::PrimaryGridUnavailable;
    }
    return false;
}

static float estimateGridSpacing(const std::array<cv::Point2f, 8>& grid)
{
    float col_sum = 0.0f;
    int col_cnt = 0;
    for (int row = 0; row < 2; ++row)
    {
        int base = row * 4;
        for (int c = 0; c < 3; ++c)
        {
            col_sum += dist2d(grid[base + c], grid[base + c + 1]);
            col_cnt++;
        }
    }

    float row_sum = 0.0f;
    int row_cnt = 0;
    for (int c = 0; c < 4; ++c)
    {
        row_sum += dist2d(grid[c], grid[4 + c]);
        row_cnt++;
    }

    float col_spacing = col_cnt > 0 ? col_sum / col_cnt : 0.0f;
    float row_spacing = row_cnt > 0 ? row_sum / row_cnt : 0.0f;
    if (col_spacing <= 0.0f && row_spacing <= 0.0f)
        return 0.0f;
    if (col_spacing <= 0.0f)
        return row_spacing;
    if (row_spacing <= 0.0f)
        return col_spacing;
    return std::min(col_spacing, row_spacing);
}

struct TrayMatchEdge {
    int cand_idx;
    int slot_idx;
    float cost;
    float distance;
};

static bool robustDepthInAnnulus(
    const cv::Point2f& center,
    float inner_radius,
    float outer_radius,
    float min_valid_ratio,
    float& depth)
{
    depth = 0.0f;
    if (inner_radius < 0.0f || outer_radius <= inner_radius ||
        g_depth_info.width <= 0 || g_depth_info.height <= 0)
        return false;

    const int x0 = std::max(0,
        static_cast<int>(std::floor(center.x - outer_radius)));
    const int y0 = std::max(0,
        static_cast<int>(std::floor(center.y - outer_radius)));
    const int x1 = std::min(static_cast<int>(g_depth_info.width) - 1,
        static_cast<int>(std::ceil(center.x + outer_radius)));
    const int y1 = std::min(static_cast<int>(g_depth_info.height) - 1,
        static_cast<int>(std::ceil(center.y + outer_radius)));
    if (x0 >= x1 || y0 >= y1)
        return false;

    const float inner2 = inner_radius * inner_radius;
    const float outer2 = outer_radius * outer_radius;
    constexpr int step = 3;
    int sampled = 0;
    std::vector<float> depths;
    for (int y = y0; y <= y1; y += step)
    {
        for (int x = x0; x <= x1; x += step)
        {
            const float dx = x - center.x;
            const float dy = y - center.y;
            const float radius2 = dx * dx + dy * dy;
            if (radius2 < inner2 || radius2 > outer2)
                continue;
            ++sampled;
            float z = 0.0f;
            g_depth_info.get_world_z_by_xy(x, y, z);
            if (z > 0.0f && z < 3000.0f)
                depths.push_back(z);
        }
    }
    if (sampled <= 0 || depths.size() < 20 ||
        static_cast<float>(depths.size()) / sampled < min_valid_ratio)
        return false;

    depth = medianValue(depths);
    return std::isfinite(depth) && depth > 0.0f;
}

// 透明空料盘在 RGB 上可能被模型误报为 lunzi，但其深度仍与料盘平面
// 基本一致；真实金属轮盘则明显高出料盘平面。只有在当前料盘已有足够
// 多个 Empty 证据形成共识时，才允许深度把单一 wheel 证据降级为空位。
// 深度缺失或处于中间带时保持模型结论，不做猜测。
static void applyDepthAssistedTrayStateRecovery(
    std::vector<TrayCandidate>& candidates,
    const std::array<cv::Point2f, 8>& grid,
    float grid_reference_side,
    float tray_plane_depth,
    const TrayPostprocessConfig& cfg)
{
    const float spacing = estimateGridSpacing(grid);
    if (spacing <= 0.0f || tray_plane_depth <= 0.0f)
        return;

    const float match_threshold = spacing * cfg.grid_match_ratio;
    std::array<int, 8> evidence_candidate;
    evidence_candidate.fill(-1);
    for (size_t ci = 0; ci < candidates.size(); ++ci)
    {
        const auto& c = candidates[ci];
        if (c.role == TrayCandidateRole::Rejected &&
            c.reason != TrayCandidateReason::InvalidBbox)
            continue;
        int best_slot = -1;
        float best_distance = std::numeric_limits<float>::max();
        for (int si = 0; si < 8; ++si)
        {
            const float d = dist2d(c.center, grid[si]);
            if (d < best_distance)
            {
                best_distance = d;
                best_slot = si;
            }
        }
        if (best_slot < 0 || best_distance > match_threshold)
            continue;
        const int old = evidence_candidate[best_slot];
        if (old < 0 || candidates[old].confidence < c.confidence)
            evidence_candidate[best_slot] = static_cast<int>(ci);
    }

    int empty_evidence_slots = 0;
    int occupied_evidence_slots = 0;
    for (int ci : evidence_candidate)
    {
        if (ci < 0)
            continue;
        ++occupied_evidence_slots;
        if (candidates[ci].has_empty_evidence ||
            candidates[ci].state == TraySlotState::Empty)
            ++empty_evidence_slots;
    }
    const float empty_ratio = occupied_evidence_slots > 0 ?
        static_cast<float>(empty_evidence_slots) / occupied_evidence_slots :
        0.0f;
    if (empty_evidence_slots < cfg.empty_dominant_min_slots ||
        empty_ratio < cfg.empty_dominant_min_ratio)
        return;

    const float depth_radius_base = spacing > 0.0f ?
        spacing : std::max(1.0f, grid_reference_side);
    const float inner_radius = depth_radius_base *
        cfg.depth_annulus_inner_ratio;
    const float outer_radius = depth_radius_base *
        cfg.depth_annulus_outer_ratio;

    std::array<float, 8> slot_depth{};
    std::array<bool, 8> slot_depth_valid{};
    for (int si = 0; si < 8; ++si)
    {
        slot_depth_valid[si] = robustDepthInAnnulus(
            grid[si], inner_radius, outer_radius,
            cfg.depth_min_valid_ratio, slot_depth[si]);
    }

    for (size_t ci = 0; ci < candidates.size(); ++ci)
    {
        auto& c = candidates[ci];
        if (c.state != TraySlotState::Wheel || c.has_empty_evidence)
            continue;
        if (c.selection_rect_enabled &&
            !c.object_center_in_safe_selection_rect)
            continue;
        if (!c.object_center_in_tray ||
            (!c.object_center_in_safe_tray &&
             c.tray_overlap_ratio < cfg.min_tray_overlap_ratio))
            continue;

        int best_slot = -1;
        float best_distance = std::numeric_limits<float>::max();
        for (int si = 0; si < 8; ++si)
        {
            const float d = dist2d(c.center, grid[si]);
            if (d < best_distance)
            {
                best_distance = d;
                best_slot = si;
            }
        }
        if (best_slot < 0 || best_distance > match_threshold ||
            !slot_depth_valid[best_slot])
            continue;

        c.local_depth_valid = true;
        c.local_depth = slot_depth[best_slot];
        c.depth_delta_to_plane = tray_plane_depth - c.local_depth;
        if (std::abs(c.depth_delta_to_plane) >
            cfg.depth_plane_tolerance_mm)
            continue;

        c.state = TraySlotState::Empty;
        c.class_id = PLATE_CELL;
        c.state_recovered = true;
        if (c.role == TrayCandidateRole::Rejected &&
            c.reason == TrayCandidateReason::InvalidBbox)
        {
            c.role = TrayCandidateRole::Supplemental;
            c.reason = TrayCandidateReason::None;
        }
#ifdef REL_DBG
        std::cout << "depth state recovery candidate=" << ci
                  << " slot=" << best_slot
                  << " local_depth=" << c.local_depth
                  << " tray_depth=" << tray_plane_depth
                  << " delta=" << c.depth_delta_to_plane
                  << std::endl;
#endif
    }

    // 若某格完全漏检，只有“空位占多数 + 该格没有候选冲突 + 局部深度
    // 与料盘平面一致”三项同时满足时，才用标准网格重建一个空位。
    // 这一级不会把已有 wheel 覆盖掉，也不会在无深度时凭空补状态。
    for (int si = 0; si < 8; ++si)
    {
        if (evidence_candidate[si] >= 0 || !slot_depth_valid[si])
            continue;
        const float delta = tray_plane_depth - slot_depth[si];
        if (std::abs(delta) > cfg.depth_plane_tolerance_mm)
            continue;

        TrayCandidate reconstructed;
        reconstructed.state = TraySlotState::Empty;
        reconstructed.class_id = PLATE_CELL;
        reconstructed.confidence = 0.0f;
        reconstructed.center = grid[si];
        reconstructed.object_center = grid[si];
        const float side = std::max(1.0f, grid_reference_side);
        reconstructed.bbox = cv::Rect2f(
            grid[si].x - side * 0.5f,
            grid[si].y - side * 0.5f,
            side, side);
        reconstructed.acceptable_size = true;
        reconstructed.role = TrayCandidateRole::Supplemental;
        reconstructed.reason = TrayCandidateReason::None;
        reconstructed.state_recovered = true;
        reconstructed.position_reconstructed = true;
        reconstructed.local_depth_valid = true;
        reconstructed.local_depth = slot_depth[si];
        reconstructed.depth_delta_to_plane = delta;
        reconstructed.object_center_in_tray = true;
        reconstructed.object_center_in_safe_tray = true;
        reconstructed.tray_overlap_ratio = 1.0f;
        reconstructed.selection_rect_enabled = !candidates.empty() &&
            candidates.front().selection_rect_enabled;
        reconstructed.object_center_in_selection_rect = true;
        reconstructed.object_center_in_safe_selection_rect = true;
        reconstructed.selection_overlap_ratio = 1.0f;
        candidates.push_back(reconstructed);
#ifdef REL_DBG
        std::cout << "depth grid reconstruction slot=" << si
                  << " local_depth=" << slot_depth[si]
                  << " tray_depth=" << tray_plane_depth
                  << " delta=" << delta << std::endl;
#endif
    }
}

static TrayPostprocessResult assignTraySlotsTwoPass(
    const std::vector<TrayCandidate>& candidates,
    const std::array<cv::Point2f, 8>& grid,
    float cell_size,
    const std::string& tray_source,
    const cv::Rect2f& tray_roi,
    const cv::Rect2f& safe_tray_roi)
{
    TrayPostprocessConfig cfg;
    TrayPostprocessResult result;
    result.tray_source = tray_source;
    result.candidates = candidates;
    result.tray_roi = tray_roi;
    result.safe_tray_roi = safe_tray_roi;

    float spacing = estimateGridSpacing(grid);
    float match_threshold = spacing > 0.0f ?
        spacing * cfg.grid_match_ratio : cell_size * 0.5f;

    for (int i = 0; i < 8; ++i)
    {
        result.slots[i].index = i;
        result.slots[i].grid_center = grid[i];
        result.slots[i].state = TraySlotState::Unknown;
        result.slots[i].candidate_index = -1;
    }

    auto edge_less = [&](const TrayMatchEdge& a, const TrayMatchEdge& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        if (a.distance != b.distance) return a.distance < b.distance;
        if (a.slot_idx != b.slot_idx) return a.slot_idx < b.slot_idx;
        const auto& ca = result.candidates[a.cand_idx];
        const auto& cb = result.candidates[b.cand_idx];
        if (ca.center.y != cb.center.y) return ca.center.y < cb.center.y;
        if (ca.center.x != cb.center.x) return ca.center.x < cb.center.x;
        if (ca.class_id != cb.class_id) return ca.class_id < cb.class_id;
        return ca.confidence > cb.confidence;
    };

    std::vector<TrayMatchEdge> primary_edges;
    for (size_t ci = 0; ci < result.candidates.size(); ++ci)
    {
        const auto& c = result.candidates[ci];
        if (c.role != TrayCandidateRole::Primary && !c.fallback_primary)
            continue;
        for (int si = 0; si < 8; ++si)
        {
            float d = dist2d(c.center, grid[si]);
            if (d > match_threshold)
                continue;

            float norm_d = match_threshold > 0.0f ? d / match_threshold : d;
            float cost = norm_d;
            if (c.has_center_hole) cost -= 0.01f;
            cost -= c.confidence * 0.002f;
            primary_edges.push_back(
                { static_cast<int>(ci), si, cost, d });
        }
    }

    std::sort(primary_edges.begin(), primary_edges.end(), edge_less);

    std::vector<bool> cand_used(result.candidates.size(), false);
    std::vector<int> slot_cand(8, -1);
    std::array<bool, 8> primary_conflict{};

    for (const auto& e : primary_edges)
    {
        if (cand_used[e.cand_idx] || slot_cand[e.slot_idx] >= 0)
            continue;
        cand_used[e.cand_idx] = true;
        slot_cand[e.slot_idx] = e.cand_idx;
    }

    for (int si = 0; si < 8; ++si)
    {
        int ci = slot_cand[si];
        if (ci < 0)
            continue;

        float d_assigned = dist2d(result.candidates[ci].center, grid[si]);
        bool ambiguous = false;
        for (size_t oj = 0; oj < result.candidates.size(); ++oj)
        {
            const auto& other = result.candidates[oj];
            if (static_cast<int>(oj) == ci ||
                (other.role != TrayCandidateRole::Primary &&
                 !other.fallback_primary) ||
                other.state == result.candidates[ci].state)
                continue;

            float d_other = dist2d(other.center, grid[si]);
            if (d_other <= match_threshold &&
                std::abs(d_assigned - d_other) <
                    match_threshold * cfg.supplemental_conflict_margin)
            {
                ambiguous = true;
                result.candidates[ci].reason =
                    TrayCandidateReason::DuplicateSlot;
                result.candidates[oj].reason =
                    TrayCandidateReason::DuplicateSlot;
                break;
            }
        }

        if (ambiguous)
        {
            primary_conflict[si] = true;
            continue;
        }

        result.slots[si].candidate_index = ci;
        result.slots[si].state = result.candidates[ci].state;
        result.slots[si].match_distance = d_assigned;
    }

    std::vector<int> open_slots;
    for (int si = 0; si < 8; ++si)
    {
        if (result.slots[si].state == TraySlotState::Unknown &&
            !primary_conflict[si])
            open_slots.push_back(si);
    }

    std::vector<int> supplemental_indices;
    for (size_t ci = 0; ci < result.candidates.size(); ++ci)
    {
        const auto& c = result.candidates[ci];
        if (c.role == TrayCandidateRole::Supplemental &&
            !c.fallback_primary)
            supplemental_indices.push_back(static_cast<int>(ci));
    }

    if (open_slots.empty())
    {
        for (int ci : supplemental_indices)
            result.candidates[ci].reason =
                TrayCandidateReason::SupplementalNoOpenSlot;
    }
    else
    {
        std::vector<TrayMatchEdge> supplemental_edges;
        std::vector<bool> supplemental_has_edge(
            result.candidates.size(), false);
        for (int ci : supplemental_indices)
        {
            const auto& c = result.candidates[ci];
            for (int si : open_slots)
            {
                float d = dist2d(c.center, grid[si]);
                if (d > match_threshold)
                    continue;
                supplemental_has_edge[ci] = true;
                float norm_d = match_threshold > 0.0f ?
                    d / match_threshold : d;
                supplemental_edges.push_back({ ci, si, norm_d, d });
            }
            if (!supplemental_has_edge[ci])
                result.candidates[ci].reason =
                    TrayCandidateReason::SupplementalTooFar;
        }

        std::vector<TrayMatchEdge> slot_winners;
        std::vector<bool> conflict_candidate(
            result.candidates.size(), false);
        for (int si : open_slots)
        {
            std::vector<TrayMatchEdge> contenders;
            for (const auto& edge : supplemental_edges)
            {
                if (edge.slot_idx == si)
                    contenders.push_back(edge);
            }
            if (contenders.empty())
                continue;
            std::sort(contenders.begin(), contenders.end(), edge_less);

            TrayMatchEdge winner = contenders.front();
            bool resolved = true;
            if (contenders.size() > 1)
            {
                const TrayMatchEdge nearest = contenders[0];
                const TrayMatchEdge& second = contenders[1];
                const float norm_margin = match_threshold > 0.0f ?
                    (second.distance - nearest.distance) / match_threshold :
                    second.distance - nearest.distance;
                if (norm_margin < cfg.supplemental_conflict_margin)
                {
                    std::vector<TrayMatchEdge> close_contenders;
                    for (const auto& contender : contenders)
                    {
                        float contender_margin = match_threshold > 0.0f ?
                            (contender.distance - nearest.distance) /
                                match_threshold :
                            contender.distance - nearest.distance;
                        if (contender_margin <
                            cfg.supplemental_conflict_margin)
                            close_contenders.push_back(contender);
                    }

                    const bool any_hole = std::any_of(
                        close_contenders.begin(), close_contenders.end(),
                        [&](const TrayMatchEdge& edge) {
                            return result.candidates[edge.cand_idx]
                                .has_center_hole;
                        });
                    if (any_hole)
                    {
                        close_contenders.erase(std::remove_if(
                            close_contenders.begin(), close_contenders.end(),
                            [&](const TrayMatchEdge& edge) {
                                return !result.candidates[edge.cand_idx]
                                    .has_center_hole;
                            }), close_contenders.end());
                    }

                    auto size_error = [&](const TrayMatchEdge& edge) {
                        const auto& c = result.candidates[edge.cand_idx];
                        return std::abs(c.width_ratio - 1.0f) +
                            std::abs(c.height_ratio - 1.0f);
                    };
                    float best_size_error = std::numeric_limits<float>::max();
                    for (const auto& contender : close_contenders)
                        best_size_error = std::min(
                            best_size_error, size_error(contender));
                    close_contenders.erase(std::remove_if(
                        close_contenders.begin(), close_contenders.end(),
                        [&](const TrayMatchEdge& edge) {
                            return size_error(edge) > best_size_error + 0.05f;
                        }), close_contenders.end());

                    float best_confidence = -1.0f;
                    for (const auto& contender : close_contenders)
                        best_confidence = std::max(
                            best_confidence,
                            result.candidates[contender.cand_idx].confidence);
                    close_contenders.erase(std::remove_if(
                        close_contenders.begin(), close_contenders.end(),
                        [&](const TrayMatchEdge& edge) {
                            return result.candidates[edge.cand_idx].confidence <
                                best_confidence - 0.05f;
                        }), close_contenders.end());

                    if (close_contenders.size() == 1)
                        winner = close_contenders.front();
                    else
                        resolved = false;
                }
            }

            if (resolved)
                slot_winners.push_back(winner);
            else
            {
                for (const auto& contender : contenders)
                    conflict_candidate[contender.cand_idx] = true;
            }
        }

        std::sort(slot_winners.begin(), slot_winners.end(), edge_less);
        for (const auto& winner : slot_winners)
        {
            if (cand_used[winner.cand_idx] ||
                result.slots[winner.slot_idx].state != TraySlotState::Unknown)
                continue;
            cand_used[winner.cand_idx] = true;
            auto& slot = result.slots[winner.slot_idx];
            slot.candidate_index = winner.cand_idx;
            slot.state = result.candidates[winner.cand_idx].state;
            slot.match_distance = winner.distance;
            result.candidates[winner.cand_idx].reason =
                TrayCandidateReason::None;
        }

        for (int ci : supplemental_indices)
        {
            if (cand_used[ci])
                continue;
            if (conflict_candidate[ci])
                result.candidates[ci].reason =
                    TrayCandidateReason::SupplementalConflict;
            else if (supplemental_has_edge[ci])
                result.candidates[ci].reason =
                    TrayCandidateReason::DuplicateSlot;
        }
    }

    result.empty_count = 0;
    result.wheel_count = 0;
    result.unknown_count = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (result.slots[i].state == TraySlotState::Empty)
            result.empty_count++;
        else if (result.slots[i].state == TraySlotState::Wheel)
            result.wheel_count++;
        else
            result.unknown_count++;
    }
    result.tray_complete = (result.empty_count + result.wheel_count == 8 &&
                            result.unknown_count == 0);

#ifdef REL_DBG
    for (int ci : supplemental_indices)
    {
        std::cout << "supplemental idx=" << ci
                  << " result="
                  << trayCandidateReasonName(result.candidates[ci].reason)
                  << std::endl;
    }
#endif
    return result;
}

static void fillPlateEmptyJson(
    json& j,
    const TrayPostprocessResult& result,
    const std::vector<cv::Point2f>& grab_pts,
    float avg_depth)
{
    json slots_json = json::array();
    for (int i = 0; i < 8; ++i)
    {
        json slot;
        slot["index"] = i;
        if (result.slots[i].state == TraySlotState::Empty)
            slot["state"] = "empty";
        else if (result.slots[i].state == TraySlotState::Wheel)
            slot["state"] = "wheel";
        else
            slot["state"] = "unknown";

        const int candidate_index = result.slots[i].candidate_index;
        if (candidate_index < 0)
        {
            slot["evidence_source"] = "none";
            slot["position_reconstructed"] = false;
        }
        else
        {
            const auto& c = result.candidates[candidate_index];
            slot["position_reconstructed"] = c.position_reconstructed;
            if (c.position_reconstructed)
                slot["evidence_source"] = "grid_depth";
            else if (c.state_recovered &&
                     c.has_empty_evidence && c.has_wheel_evidence)
                slot["evidence_source"] = "merged_empty_wheel";
            else if (c.state_recovered)
                slot["evidence_source"] = "depth_reclassified";
            else if (c.state == TraySlotState::Empty)
                slot["evidence_source"] = c.has_center_hole ?
                    "plate_cell_hole" : "plate_cell";
            else if (c.state == TraySlotState::Wheel)
                slot["evidence_source"] = c.has_center_hole ?
                    "lunzi_hole" : "lunzi";
            else
                slot["evidence_source"] = "unknown";
        }
        slots_json.push_back(slot);
    }

    j["slots"] = slots_json;
    j["tray_source"] = result.tray_source;
    j["tray_complete"] = result.tray_complete;
    j["empty_count"] = result.empty_count;
    j["wheel_count"] = result.wheel_count;
    j["unknown_count"] = result.unknown_count;

    vector< vector<float>> list_pts_out;
    for (const auto& pt : grab_pts)
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = avg_depth;
        g_depth_info.get_world_xy_by_z(pt.x, pt.y, x, y, z);
        list_pts_out.push_back({ x, y, z, 10.0f });
    }
    j["pts"] = list_pts_out;

    vector< vector<float>> list_cell_out;
    // 与 pts_cell 一一对应。每项为左上、右上、右下、左下四个
    // 世界坐标角点 [[x,y,z], ...]，与 pts_cell 使用同一坐标系。
    vector< vector< vector<float>>> list_cell_rect_out;
    vector< vector<float>> list_lunzi_out;
    for (int i = 0; i < 8; ++i)
    {
        const auto& slot = result.slots[i];
        if (slot.candidate_index < 0)
            continue;

        const auto& c = result.candidates[slot.candidate_index];
        float x = 0.0f;
        float y = 0.0f;
        float z = avg_depth;
        g_depth_info.get_world_xy_by_z(c.center.x, c.center.y, x, y, z);
        vector<float> pt_world = { x, y, z, 10.0f };
        if (slot.state == TraySlotState::Empty)
        {
            list_cell_out.push_back(pt_world);
            std::array<cv::Point2f, 4> corners = {
                c.bbox.tl(),
                cv::Point2f(c.bbox.x + c.bbox.width, c.bbox.y),
                c.bbox.br(),
                cv::Point2f(c.bbox.x, c.bbox.y + c.bbox.height)
            };
            vector<vector<float>> rect_world;
            for (const auto& corner : corners)
            {
                float corner_x = 0.0f;
                float corner_y = 0.0f;
                float corner_z = avg_depth;
                g_depth_info.get_world_xy_by_z(
                    corner.x, corner.y, corner_x, corner_y, corner_z);
                rect_world.push_back({ corner_x, corner_y, corner_z });
            }
            list_cell_rect_out.push_back(rect_world);
        }
        else if (slot.state == TraySlotState::Wheel)
            list_lunzi_out.push_back(pt_world);
    }
    j["pts_cell"] = list_cell_out;
    j["pts_cell_rect"] = list_cell_rect_out;
    j["pts_lunzi"] = list_lunzi_out;

    if (result.tray_complete)
    {
        j["ret-code"] = 0;
        j["ret-msg"] = "ok";
    }
    else
    {
        j["ret-code"] = 0;
        int resolved = result.empty_count + result.wheel_count;
        j["ret-msg"] = "warning: incomplete tray, resolved " +
            std::to_string(resolved) + "/8 positions";
    }
}

static void drawTrayPostprocessOverlay(
    Mat& image,
    const TrayPostprocessResult& result)
{
    if (result.tray_roi.area() > 0.0f)
        cv::rectangle(image, result.tray_roi, cv::Scalar(255, 0, 255), 2);
    if (result.safe_tray_roi.area() > 0.0f)
        cv::rectangle(image, result.safe_tray_roi, cv::Scalar(255, 255, 0), 1);

    for (size_t ci = 0; ci < result.candidates.size(); ++ci)
    {
        const auto& c = result.candidates[ci];
        if (c.bbox.area() <= 0.0f)
            continue;
        cv::Scalar color;
        std::string label;
        if (c.role == TrayCandidateRole::Primary)
        {
            color = cv::Scalar(0, 180, 0);
            label = "primary";
        }
        else if (c.role == TrayCandidateRole::Supplemental)
        {
            color = cv::Scalar(0, 215, 255);
            label = c.fallback_primary ? "fallback" : "supplemental";
        }
        else
        {
            color = cv::Scalar(128, 128, 128);
            label = "rejected";
        }
        if (c.reason != TrayCandidateReason::None)
            label += std::string(":") + trayCandidateReasonName(c.reason);
        cv::rectangle(image, c.bbox, color, 2);
        cv::putText(image, label,
            cv::Point(static_cast<int>(c.bbox.x),
                      std::max(12, static_cast<int>(c.bbox.y) - 4)),
            cv::FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv::LINE_AA);
    }

    for (int i = 0; i < 8; ++i)
    {
        const auto& slot = result.slots[i];
        cv::Scalar color;
        std::string label = "P" + std::to_string(i) + " ";
        if (slot.state == TraySlotState::Empty)
        {
            color = Scalar(0, 255, 0);
            label += "empty";
        }
        else if (slot.state == TraySlotState::Wheel)
        {
            color = Scalar(255, 0, 0);
            label += "wheel";
        }
        else
        {
            color = Scalar(0, 255, 255);
            label += "unknown";
        }

        cv::circle(image, slot.grid_center, 8, color, 2);
        if (slot.candidate_index >= 0)
        {
            const auto& c = result.candidates[slot.candidate_index];
            cv::rectangle(image, c.bbox, color, 3);
        }

        cv::putText(image, label,
            cv::Point(static_cast<int>(slot.grid_center.x + 5),
                      static_cast<int>(slot.grid_center.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

static void fillIncompletePlateEmptyJson(json& j, const std::string& msg)
{
    j["ret-code"] = 0;
    j["ret-msg"] = msg;
    j["tray_complete"] = false;
    j["empty_count"] = 0;
    j["wheel_count"] = 0;
    j["unknown_count"] = 8;
    j["pts"] = json::array();
    j["pts_cell"] = json::array();
    j["pts_cell_rect"] = json::array();
    j["pts_lunzi"] = json::array();

    json slots_json = json::array();
    for (int i = 0; i < 8; ++i)
    {
        json slot;
        slot["index"] = i;
        slot["state"] = "unknown";
        slots_json.push_back(slot);
    }
    j["slots"] = slots_json;
}


// 把轮盘和圆孔对应上
// 列表 det_lunzi 和 det_hole 是一一对应的
void align_lunzi_hole(vector<Detection_pan>& det_lunzi,
    vector<Detection_pan>& det_hole,
    bool keep_without_hole = false)
{
    if (g_lunzi_only)
    {
        det_hole.clear();
        return;
    }

    if (det_hole.size() == 0 && det_lunzi.size() > 0)
    {
        if (keep_without_hole || "feilun" == g_model_name)
        {
            // 没有 hole 时，为每个 lunzi 生成占位 hole
            for (size_t i = 0; i < det_lunzi.size(); i++)
            {
                Detection_pan pan;
                pan.class_id = -1;                        // 标记为占位
                pan.bbox = det_lunzi[i].bbox;             // 用 lunzi 的 bbox
                det_hole.push_back(pan);
            }
        }
        return;
    }

    vector<Detection_pan> det_hole_bak = det_hole;
    det_hole.clear();

    int idx = 0;
    while (idx < det_lunzi.size())
    {
        Detection_pan& d = det_lunzi[idx];

        for (size_t i = 0; i < det_hole_bak.size(); i++)
        {
            if (bbox_contains(d.bbox, det_hole_bak[i].bbox))
            {
                det_hole.push_back(det_hole_bak[i]);
                det_hole_bak.erase(det_hole_bak.begin() + i);
                break;
            }
        }

        // 如果没找到匹配的hole
        if (det_hole.size() <= idx)
        {
            if (keep_without_hole || "feilun" == g_model_name)
            {
                // 无 hole 时生成占位项，保持 lunzi/hole 索引一一对应
                Detection_pan pan;
                pan.class_id = -1;                        // 标记为占位
                pan.bbox = d.bbox;                        // 用当前 lunzi 的 bbox
                det_hole.push_back(pan);
            }
            else
            {
                // xialiao: 删除无效轮子
                det_lunzi.erase(det_lunzi.begin() + idx);
                continue;
            }
        }

        idx++;
    }

    if (keep_without_hole || "feilun"==g_model_name)
    {
        // 轮子模式下孤立 hole 不能代表一个轮子，直接丢弃。
    }
    else
    {
        // 处理多余出来的小圆
        for (auto& d: det_hole_bak)
        {
            Detection_pan empty_lunzi;

            // 他们是一一对应的
            det_lunzi.push_back(empty_lunzi);
            det_hole.push_back(d);
        }        
    }
    

}


// 根据 PLATE_TRAY 的矩形，把不是空托盘内的 检测结果过滤掉
void filter_hole_by_plate_bbox(Detection_pan& det_plate, vector<Detection_pan>& det_lunzi, vector<Detection_pan>& det_hole)
{
    if (det_hole.size() == 0) return;

    vector<Detection_pan> det_pan_bak = det_lunzi;
    vector<Detection_pan> det_hole_bak = det_hole;

    det_lunzi.clear();
    det_hole.clear();

    Detection_pan max_lunzi;
    double min_dist = 9999;
    int min_idx = -1;

    int idx = 0;
    while (idx < det_hole_bak.size())
    {
        Detection_pan& d = det_hole_bak[idx]; 
        
        if (bbox_contains(det_plate.bbox, d.bbox))
        {
            det_lunzi.push_back(det_pan_bak[idx]);
            det_hole.push_back(det_hole_bak[idx]);

            if (det_lunzi.back().bbox.area()>max_lunzi.bbox.area())
            {
                max_lunzi = det_lunzi.back();
            }      

        }


        idx++;
    }

    if (max_lunzi.bbox.area()>100&&
        det_lunzi.size()>0&&det_lunzi[0].class_id==PLATE_CELL)
    {
        float min_area = max_lunzi.bbox.area() * 0.8;

        vector<float> list_area;
        for (auto& d : det_lunzi)
        {
            list_area.push_back(d.bbox.area());
        }

        idx = 0;
        while (idx < det_lunzi.size())
        {
            Detection_pan& d = det_lunzi[idx];

            // 如果盘子面积不及最大轮子 80% 就不是合格的轮子
            if (d.bbox.area() < min_area)
            {
                det_lunzi.erase(det_lunzi.begin() + idx);
                det_hole.erase(det_hole.begin() + idx);
                continue;
            }

            idx++;
        }  


        idx = 0;
        while (idx < det_lunzi.size())
        {
            Detection_pan& d = det_lunzi[idx];

            // 如果盘子面积不及最大轮子 80% 就不是合格的轮子
            if (d.bbox.area() < min_area)
            {
                det_lunzi.erase(det_lunzi.begin() + idx);
                det_hole.erase(det_hole.begin() + idx);
                continue;
            }

            idx++;
        }  
    }
    

}





void det_to_json(Detection_pan& det, json& j)
{
    vector<float> list_pts = { det.bbox.tl().x ,det.bbox.tl().y ,det.bbox.br().x ,det.bbox.br().y };
    vector<float> list_v = { det.c.centor.x, det.c.centor.y, det.c.centor.z };
    float radius = det.c.radius;

    j["bbox"] = list_pts;
    j["centor"] = list_v;
    j["radius"] = radius;

}

void splitByMedian(const std::vector<float>& list_depth,
    std::vector<float>& group_small,
    std::vector<float>& group_big) {
    if (list_depth.empty()) return;

    // 复制并排序以找到中位数
    std::vector<float> sorted = list_depth;
    std::sort(sorted.begin(), sorted.end());

    float median;
    if (sorted.size() % 2 == 0) {
        median = (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0f;
    }
    else {
        median = sorted[sorted.size() / 2];
    }

    // 按中位数分组
    for (float depth : list_depth) {
        if (depth < median) {
            group_small.push_back(depth);
        } else if (depth > median){
            group_big.push_back(depth);
        }
    }
}

float calculateAverage(const std::vector<float>& list_depth) {
    if (list_depth.empty()) {
        return 0.0f;
    }

    // 使用 accumulate 计算总和
    float sum = std::accumulate(list_depth.begin(), list_depth.end(), 0.0f);

    return sum / list_depth.size();
}

bool is_depth_validate(const float& depth, const std::string& det_type, const float& ref_depth)
{
    if (depth<100) return false;

    float max_diff = 100;

    if ("lunzi"==det_type)
    {
        max_diff = 100;
    }
    else if ("plate"==det_type)
    {
        max_diff = 100;
    }
    else if ("plate_empty"==det_type)
    {
        max_diff = 100;
    }    
    

    if (ref_depth>10 && std::abs(depth-ref_depth)>max_diff)
    {
        return false;
    }
    

    return true;
}

cv::Point2f get_center_pt(const std::vector<int>& rect) {
    if (rect.size() != 8) {
        return cv::Point2f(0, 0);
    }
    
    // 四个顶点的平均值即为中心点
    float sum_x = rect[0] + rect[2] + rect[4] + rect[6];
    float sum_y = rect[1] + rect[3] + rect[5] + rect[7];
    
    return cv::Point2f(sum_x / 4.0f, sum_y / 4.0f);
}

// 判断pt是否在 rect围成的区域内
// rect 的分布为 x0,y0,x1,y1,x2,y2,x3,y3
// 分别对应 tl tr br bl
// 就是 上左 上右 下右 下左
bool isPtInArea(const std::vector<int>& rect, cv::Point2f pt) {
    if (rect.size() == 0) {
        return true;  // 必须是4个点，每个点x、y两个坐标
    }
    
    if (rect.size() != 8) {
        return false;  // 必须是4个点，每个点x、y两个坐标
    }
    
    // 构建多边形的四个顶点 (tl, tr, br, bl)
    std::vector<cv::Point2f> polygon;
    polygon.push_back(cv::Point2f(rect[0], rect[1]));  // tl
    polygon.push_back(cv::Point2f(rect[2], rect[3]));  // tr
    polygon.push_back(cv::Point2f(rect[4], rect[5]));  // br
    polygon.push_back(cv::Point2f(rect[6], rect[7]));  // bl
    
    // 射线法判断点是否在多边形内
    bool inside = false;
    int n = polygon.size();
    
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const cv::Point2f& p1 = polygon[i];
        const cv::Point2f& p2 = polygon[j];
        
        // 判断射线是否穿过边
        bool intersect = ((p1.y > pt.y) != (p2.y > pt.y)) &&
                         (pt.x < (p2.x - p1.x) * (pt.y - p1.y) / (p2.y - p1.y) + p1.x);
        
        if (intersect) {
            inside = !inside;
        }
    }
    
    return inside;
}


void get_avg_depth_by_radius(float img_x, float img_y, 
    vector<float>list_r, // 计算这个几个半径上的
    int step,   // 多少度采样一个点
    int mode,   // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
    float& avg_depth,float* ref_depth=nullptr,float* max_diff=nullptr, float minZ=0)
{
    Circle c;
    c.center.x = img_x;
    c.center.y = img_y;

    vector<float> list_depth;
    for (auto& r: list_r)
    {
        c.radius = r;
        for (size_t i = 0; i < 360; i+=step)
        {
            cv::Point2f pt = PointCorrector::pointOnCircle_angle(c,i);

            if (!isPtInArea(g_rect,pt))
            {
                continue;
            }
            

            float z = -1;
            g_depth_info.get_world_z_by_xy(pt.x, pt.y, z);
            if (z > minZ &&z<3000)
            {
                if (nullptr==ref_depth || nullptr==max_diff || *ref_depth<10 )
                {
                    list_depth.push_back(z);
                }
                else if(std::abs(*ref_depth-z)<*max_diff)
                {
                    list_depth.push_back(z);
                }
                else
                {
                    std::cout<< "invalidate deepth value"<< std::endl;
                }          
            } 
                
        }
    }

    // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
    if (0==mode)
    {
        avg_depth = calculateAverage(list_depth);
    }
    else
    {
        // std::vector<float> group_small;
        // std::vector<float> group_big;

        // splitByMedian(list_depth, group_small, group_big);

        // if (1 == mode)
        // {
        //     // 1: 最大的那些平均
        //     avg_depth = calculateAverage(group_big);
        // }
        // else if (2 == mode)
        // {
        //     // 2: 最 小 的那些平均  
        //     avg_depth = calculateAverage(group_small);
        // }

        std::sort(list_depth.begin(),list_depth.end());

        // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均
        int cnt = 0;  
        std::size_t total = 0;
        std::size_t max_cnt = 10;
        if(max_cnt>list_depth.size())max_cnt = list_depth.size();
        if (2==mode)
        {
            /* 最小的那些平均 */
            int idx = 0;
            while (idx<max_cnt && idx<list_depth.size())
            {
                total+=list_depth[idx++];
            }
            cnt = idx;
        }
        else
        {
            /* 最大的那些平均 */
            int idx = list_depth.size()-1;
            while (cnt<max_cnt && idx>=0)
            {
                total+=list_depth[idx--];
                cnt++;
            }
        }
        
        if (cnt>0)
        {
            avg_depth = ((double)total)/cnt;
        }


    }
    
}

void randomSampleInRect(const cv::Rect2f& bbox, int sample_cnt, std::vector<cv::Point>& list_pts) {
    // 清空输出容器
    list_pts.clear();
    
    if (sample_cnt <= 0) return;
    
    // 使用随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // 计算矩形边界（整数化处理）
    int x_min = static_cast<int>(std::floor(bbox.x));
    int x_max = static_cast<int>(std::floor(bbox.x + bbox.width));
    int y_min = static_cast<int>(std::floor(bbox.y));
    int y_max = static_cast<int>(std::floor(bbox.y + bbox.height));
    
    // 确保矩形有效
    if (x_min >= x_max || y_min >= y_max) return;
    
    // 定义均匀分布
    std::uniform_int_distribution<int> dist_x(x_min, x_max - 1);
    std::uniform_int_distribution<int> dist_y(y_min, y_max - 1);
    
    // 随机采样
    list_pts.reserve(sample_cnt);
    for (int i = 0; i < sample_cnt; ++i) {
        int x = dist_x(gen);
        int y = dist_y(gen);
        list_pts.emplace_back(x, y);
    }
}

void get_avg_depth_by_bbox(Rect2f bbox, int sample_cnt, 
    int mode,   // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
    float& avg_depth,float* ref_depth=nullptr,float* max_diff=nullptr)
{
    std::vector<cv::Point> list_pts;
    randomSampleInRect(bbox,sample_cnt,list_pts);

    vector<float> list_depth;
    for (auto& pt: list_pts)
    {
        float z = 0;
        g_depth_info.get_world_z_by_xy(pt.x, pt.y, z);
        if (z > 0&&z<3000)
        {
            if (nullptr==ref_depth || nullptr==max_diff || *ref_depth<10 )
            {
                list_depth.push_back(z);
            }
            else if(std::abs(*ref_depth-z)<*max_diff)
            {
                list_depth.push_back(z);
            }
            else
            {
                std::cout<< "get_avg_depth_by_bbox  invalidate deepth value"<< std::endl;
            }               
        }
    }

    avg_depth = 0;

    // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
    if (0==mode)
    {
        avg_depth = calculateAverage(list_depth);
    }
    else
    {
        // std::vector<float> group_small;
        // std::vector<float> group_big;

        // splitByMedian(list_depth, group_small, group_big);

        // if (1 == mode)
        // {
        //     // 1: 最大的那些平均
        //     avg_depth = calculateAverage(group_big);
        // }
        // else if (2 == mode)
        // {
        //     // 2: 最 小 的那些平均  
        //     avg_depth = calculateAverage(group_small);
        // }

        std::sort(list_depth.begin(),list_depth.end());

        // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均
        int cnt = 0;  
        std::size_t total = 0;
        std::size_t max_cnt = 10;
        if(max_cnt>list_depth.size())max_cnt = list_depth.size();
        if (2==mode)
        {
            /* 最小的那些平均 */
            int idx = 0;
            while (idx<max_cnt && idx<list_depth.size())
            {
                total+=list_depth[idx++];
            }
            cnt = idx;
        }
        else
        {
            /* 最大的那些平均 */
            int idx = list_depth.size()-1;
            while (cnt<max_cnt && idx>=0)
            {
                total+=list_depth[idx--];
                cnt++;
            }
        }
        
        if (cnt>0)
        {
            avg_depth = ((double)total)/cnt;
        }
        
    }
    
}

// 在指定区域内按固定步长采样，先用中位数/MAD剔除异常深度，再对内点求平均。
// 与随机抽样相比，同一帧输入会得到稳定一致的结果；MAD 对透明膜反光、孔洞和
// 背景深度更不敏感，适合料盘平面深度估计。
bool get_robust_depth_by_bbox(const Rect2f& bbox, float& robust_depth,
    float* ref_depth = nullptr, float* max_diff = nullptr)
{
    robust_depth = 0.0f;

    const int x_begin = std::max(0, static_cast<int>(std::ceil(bbox.x)));
    const int y_begin = std::max(0, static_cast<int>(std::ceil(bbox.y)));
    const int x_end = std::min(static_cast<int>(g_cam_param.width),
        static_cast<int>(std::floor(bbox.x + bbox.width)));
    const int y_end = std::min(static_cast<int>(g_cam_param.height),
        static_cast<int>(std::floor(bbox.y + bbox.height)));
    if (x_begin >= x_end || y_begin >= y_end)
        return false;

    constexpr int sample_step = 3;
    std::vector<float> depths;
    depths.reserve(((x_end - x_begin) / sample_step + 1) *
                   ((y_end - y_begin) / sample_step + 1));
    for (int y = y_begin; y < y_end; y += sample_step)
    {
        for (int x = x_begin; x < x_end; x += sample_step)
        {
            float z = 0.0f;
            g_depth_info.get_world_z_by_xy(x, y, z);
            if (z <= 0.0f || z >= 3000.0f)
                continue;
            if (ref_depth != nullptr && max_diff != nullptr && *ref_depth >= 10.0f &&
                std::abs(z - *ref_depth) >= *max_diff)
                continue;
            depths.push_back(z);
        }
    }

    constexpr size_t min_valid_samples = 20;
    if (depths.size() < min_valid_samples)
        return false;

    auto median = [](std::vector<float> values) {
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        const float upper = values[mid];
        if ((values.size() & 1U) != 0U)
            return upper;
        std::nth_element(values.begin(), values.begin() + mid - 1,
                         values.begin() + mid);
        return (values[mid - 1] + upper) * 0.5f;
    };

    const float depth_median = median(depths);
    std::vector<float> deviations;
    deviations.reserve(depths.size());
    for (float z : depths)
        deviations.push_back(std::abs(z - depth_median));
    const float mad = median(deviations);

    // 1.4826 将 MAD 换算到近似标准差；至少保留 +/-3 mm，避免量化噪声
    // 导致内点过少，同时把阈值限制在 20 mm 内，防止背景形成长尾。
    const float inlier_threshold = std::max(3.0f,
        std::min(20.0f, 3.0f * 1.4826f * mad));
    double sum = 0.0;
    size_t inlier_count = 0;
    for (float z : depths)
    {
        if (std::abs(z - depth_median) <= inlier_threshold)
        {
            sum += z;
            ++inlier_count;
        }
    }
    if (inlier_count < min_valid_samples)
        return false;

    robust_depth = static_cast<float>(sum / inlier_count);
    return true;
}

void get_centor_depth_by_area_and_cur_depthinfo
(const vector<int>& area, 
    int mode, // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均 
    float& centor_dist)
{
    centor_dist = -1;    
    if (area.size()!=8) return;

    cv::Point2f cp = get_center_pt(area);
    // 在原半径上 放大两圈，确保取的点在 盘面上
    vector<float>list_r = { 20,50 };
    int step = 3; // 采样步长
    // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
    // int mode = 0;        
    get_avg_depth_by_radius(cp.x, cp.y, 
        list_r, step, mode, centor_dist);            
}

// 基于中位数的方法（推荐）：对异常值更鲁棒，适合 3-5 个数的场景
std::vector<float> removeOutliers(const std::vector<float>& list_depth, float max_diff) {
    if (list_depth.size() <= 2) {
        return list_depth;  // 数量太少，不做剔除
    }
    
    std::vector<float> sorted = list_depth;
    std::sort(sorted.begin(), sorted.end());
    
    // 计算中位数（对异常值更鲁棒）
    float median;
    if (sorted.size() % 2 == 0) {
        median = (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0f;
    } else {
        median = sorted[sorted.size()/2];
    }
    
    // 找出所有在 [median - max_diff, median + max_diff] 范围内的值
    std::vector<float> result;
    for (float val : list_depth) {
        if (std::abs(val - median) <= max_diff) {
            result.push_back(val);
        }
    }
    
    // 如果剔除后数量太少，返回原始数据或者保留最接近中位数的值
    if (result.empty()) {
        // 返回最接近中位数的值
        auto it = std::min_element(list_depth.begin(), list_depth.end(),
            [median](float a, float b) {
                return std::abs(a - median) < std::abs(b - median);
            });
        return std::vector<float>{*it};
    }
    
    return result;
}



void printRect(const std::vector<int>& rect) {
    if (rect.size() >= 8) {
        std::cout << "rect[tl(" << rect[0] << "," << rect[1] 
                  << "), tr(" << rect[2] << "," << rect[3] 
                  << "), br(" << rect[4] << "," << rect[5] 
                  << "), bl(" << rect[6] << "," << rect[7] << ")]" 
                  << std::endl;
    } else {
        std::cout << "Invalid rect (size=" << rect.size() << ")" << std::endl;
    }
}


// 通过连续采集4次深度，获取一个比较合理的深度值
void get_centor_depth_by_area_and_cap_depth_4times
(const vector<int>& area, float& centor_dist)
{
    centor_dist = -1;
    vector<float> list_depth;
    if (area.size()==8)
    {
        std::cout<< "get_ref_depth_by_area in rect " ;
        printRect(area);

        // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
        int mode = 0; 

        for (size_t i = 0; i < 4; i++)
        {
            cv::Mat bgrImage, depthImage;

            for (size_t ll = 0; ll < 3; ll++)
            {
                int ret = g_camera->captureOneFrameWithDepth(bgrImage, depthImage);

                if (!bgrImage.empty() && !depthImage.empty())
                {
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }

            if (!bgrImage.empty() && !depthImage.empty())
            {
                g_depth_info.from_depthMat(depthImage);

                float depth = -1;
                get_centor_depth_by_area_and_cur_depthinfo(area, mode, depth);

                if (depth > 0) list_depth.push_back(depth);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        }
        
    }   
    
    // 计算一个平均值
    vector<float> list_new = removeOutliers(list_depth,30);

    if (list_new.size()>1)
    {
        centor_dist = calculateAverage(list_new);
    }
    
}



void det_to_json(vector<Detection_pan>& dets, json& j_list)
{
    for (auto& d: dets)
    {
        json j;
        det_to_json(d,j);
        j_list.push_back(j);
    }

}


// 基于K-means聚类的布局检测
std::pair<int, int> detectLayoutByClustering(const std::vector<cv::Point2f>& points) {
    // 尝试检测行数和列数

    // 收集所有x和y坐标
    std::vector<float> x_coords, y_coords;
    for (const auto& p : points) {
        x_coords.push_back(p.x);
        y_coords.push_back(p.y);
    }

    // 对坐标排序并去重（考虑容差）
    auto deduplicateWithTolerance = [](std::vector<float>& coords, float tolerance = 40.0f) {
        std::sort(coords.begin(), coords.end());
        std::vector<float> unique_coords;
        for (float coord : coords) {
            if (unique_coords.empty() || std::abs(coord - unique_coords.back()) > tolerance) {
                unique_coords.push_back(coord);
            }
        }
        return unique_coords;
    };

    std::vector<float> unique_x = deduplicateWithTolerance(x_coords);
    std::vector<float> unique_y = deduplicateWithTolerance(y_coords);

    // 可能的布局组合：4x2 或 2x4
    if (unique_y.size() == 4 && unique_x.size() == 2) {
        return { 4, 2 };  // 4行2列
    }
    else if (unique_y.size() == 2 && unique_x.size() == 4) {
        return { 2, 4 };  // 2行4列
    }
    else if (unique_y.size() > unique_x.size()) {
        // 如果不同的y值更多，可能是4x2
        return { 4, 2 };
    }
    else {
        // 否则可能是2x4
        return { 2, 4 };
    }
}

// 鲁棒的网格排列
std::vector<std::vector<cv::Point2f>> robustArrangeToGrid(
    const std::vector<cv::Point2f>& points,
    int rows, int cols) {

    if (points.size() != rows * cols) {
        std::cerr << "Error: Expected " << rows * cols << " points, got " << points.size() << std::endl;
        return {};
    }

    // 使用K-means聚类分别对行和列进行聚类

    // 1. 对y坐标聚类（确定行）
    std::vector<float> y_values;
    for (const auto& p : points) {
        y_values.push_back(p.y);
    }

    // 简单的K-means聚类（rows个中心）
    std::vector<float> y_centers(rows);
    std::sort(y_values.begin(), y_values.end());

    // 初始化聚类中心（使用分位数）
    for (int i = 0; i < rows; ++i) {
        int idx = (y_values.size() * (2 * i + 1)) / (2 * rows);
        y_centers[i] = y_values[idx];
    }

    // 分配点到行聚类
    std::vector<std::vector<cv::Point2f>> row_clusters(rows);
    bool changed = true;
    int max_iter = 10;

    for (int iter = 0; iter < max_iter && changed; ++iter) {
        // 清空聚类
        for (auto& cluster : row_clusters) cluster.clear();

        // 分配点到最近的y中心
        for (const auto& p : points) {
            int best_cluster = 0;
            float min_dist = std::abs(p.y - y_centers[0]);

            for (int i = 1; i < rows; ++i) {
                float dist = std::abs(p.y - y_centers[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = i;
                }
            }
            row_clusters[best_cluster].push_back(p);
        }

        // 更新中心
        changed = false;
        for (int i = 0; i < rows; ++i) {
            if (!row_clusters[i].empty()) {
                float new_center = 0;
                for (const auto& p : row_clusters[i]) {
                    new_center += p.y;
                }
                new_center /= row_clusters[i].size();

                if (std::abs(new_center - y_centers[i]) > 0.001f) {
                    changed = true;
                    y_centers[i] = new_center;
                }
            }
        }
    }

    // 对行聚类按y坐标排序
    std::vector<int> row_order(rows);
    for (int i = 0; i < rows; ++i) row_order[i] = i;
    std::sort(row_order.begin(), row_order.end(),
        [&y_centers](int a, int b) {
        return y_centers[a] < y_centers[b];
    });

    // 2. 对每行的点进行x坐标聚类（确定列）
    std::vector<std::vector<cv::Point2f>> grid(rows, std::vector<cv::Point2f>(cols));

    for (int i = 0; i < rows; ++i) {
        int actual_row = row_order[i];
        auto& row_points = row_clusters[actual_row];

        if (row_points.size() != cols) {
            std::cerr << "Warning: Row " << i << " has " << row_points.size() << " points, expected " << cols << std::endl;
            // 使用简单排序作为后备
            std::sort(row_points.begin(), row_points.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.x < b.x;
            });
            grid[i] = row_points;
            continue;
        }

        // 对当前行的点按x坐标排序
        std::sort(row_points.begin(), row_points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        grid[i] = row_points;
    }

    return grid;
}

// 完整解决方案
std::vector<std::vector<cv::Point2f>> autoDetectAndArrange(const Detection_pan& det_plate,const std::vector<cv::Point2f>& points) {
    if (points.size() != 8) {
        std::cerr << "Error: Expected 8 points, got " << points.size() << std::endl;
        return {};
    }

    // 检测布局
    //auto [rows, cols] = detectLayoutByClustering(points);

    //std::cout << "Detected layout: " << rows << " rows × " << cols << " columns" << std::endl;

    int rows = 4;
    int cols = 2;
    if (det_plate.bbox.width> det_plate.bbox.height)
    {
        rows = 2;
        cols = 4;
    }

    // 使用鲁棒的网格排列
    return robustArrangeToGrid(points, rows, cols);
}



std::vector<std::vector<cv::Point2f>> arrangeTo3x3Grid(const std::vector<cv::Point2f>& list_cells) {
    if (list_cells.size() != 9) {
        std::cerr << "Error: Expected 9 points, but got " << list_cells.size() << std::endl;
        return {};
    }

    std::vector<std::vector<cv::Point2f>> grid(3, std::vector<cv::Point2f>(3));

    // 复制点集以便排序
    std::vector<cv::Point2f> sorted_points = list_cells;

    // 方法1：按y坐标排序（从上到下），然后每3个点按x坐标排序（从左到右）
    std::sort(sorted_points.begin(), sorted_points.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;  // 按y坐标升序排列（从顶部到底部）
    });

    // 现在每3个点是一行，对每行按x坐标排序
    for (int row = 0; row < 3; ++row) {
        // 获取当前行的3个点
        auto row_begin = sorted_points.begin() + row * 3;
        auto row_end = sorted_points.begin() + (row + 1) * 3;

        // 对当前行的点按x坐标排序（从左到右）
        std::sort(row_begin, row_end,
            [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        // 将排序后的点存入grid
        std::copy(row_begin, row_end, grid[row].begin());
    }

    return grid;
}

std::vector<std::vector<cv::Point2f>> arrangeTo3x3GridRobust(const Detection_pan& det_plate, const std::vector<cv::Point2f>& list_cells) {
    if (list_cells.size() != 9) {
        std::cerr << "Error: Expected 9 points, but got " << list_cells.size() << std::endl;
        return {};
    }

    // 提取x坐标和y坐标
    std::vector<float> x_coords, y_coords;
    for (const auto& point : list_cells) {
        x_coords.push_back(point.x);
        y_coords.push_back(point.y);
    }

    // 对y坐标排序，找出分位点来划分三行
    std::vector<float> sorted_y = y_coords;
    std::sort(sorted_y.begin(), sorted_y.end());

    // 使用分位数确定行边界（假设点的垂直分布大致均匀）
    float y_threshold1 = (sorted_y[2] + sorted_y[3]) / 2.0f;  // 第1/3分界线
    float y_threshold2 = (sorted_y[5] + sorted_y[6]) / 2.0f;  // 第2/3分界线

    // 将点分成三行
    std::vector<std::vector<cv::Point2f>> rows(3);
    for (const auto& point : list_cells) {
        if (point.y < y_threshold1) {
            rows[0].push_back(point);  // 第一行（最上面）
        }
        else if (point.y < y_threshold2) {
            rows[1].push_back(point);  // 第二行
        }
        else {
            rows[2].push_back(point);  // 第三行（最下面）
        }
    }

    // 确保每行都有3个点
    if (rows[0].size() != 3 || rows[1].size() != 3 || rows[2].size() != 3) {
        std::cerr << "Warning: Rows not evenly distributed. Using fallback method." << std::endl;
        // 如果分布不均，退回到简单排序方法
        return arrangeTo3x3Grid(list_cells);
    }

    // 对每行按x坐标排序（从左到右）
    std::vector<std::vector<cv::Point2f>> grid(3, std::vector<cv::Point2f>(3));
    for (int i = 0; i < 3; ++i) {
        std::sort(rows[i].begin(), rows[i].end(),
            [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });
        grid[i] = rows[i];
    }

    return grid;
}

// 将角度转换为弧度
float degreesToRadians(float degrees) {
    return degrees * M_PI / 180.0f;
}


/**
 * @brief 计算使矩形充满图像所需的最佳距离
 * @param g_cam_param 相机内参
 * @param img 当前图像
 * @param bbox 图像中的矩形区域
 * @param current_z 当前摄像头到矩形的实际距离（毫米）
 * @param target_margin 目标边缘像素（默认100）
 * @return 调整后的距离z（毫米）
 */
float calculateOptimalDistance(const AICameraIntrinsic& g_cam_param, 
                                const cv::Rect& bbox,
                                float current_z,
                                int target_margin = 100) {
    // 计算当前矩形在图像中的尺寸
    float current_rect_width_px = bbox.width;
    float current_rect_height_px = bbox.height;
    
    // 计算可用的图像区域（减去margin）
    float available_width_px = g_cam_param.width - 2 * target_margin;
    float available_height_px = g_cam_param.height - 2 * target_margin;
    
    // 矩形实际物理尺寸（根据当前距离计算）
    // 相机投影公式: pixel_size = physical_size * focal_length / distance
    // 所以: physical_size = pixel_size * distance / focal_length
    float rect_physical_width = current_rect_width_px * current_z / g_cam_param.fx;
    float rect_physical_height = current_rect_height_px * current_z / g_cam_param.fy;
    
    // 计算使矩形宽度充满可用区域所需距离
    // 目标宽度像素 = rect_physical_width * focal_length / target_z
    // target_z = rect_physical_width * focal_length / 目标宽度像素
    float target_z_by_width = rect_physical_width * g_cam_param.fx / available_width_px;
    
    // 计算使矩形高度充满可用区域所需距离
    float target_z_by_height = rect_physical_height * g_cam_param.fy / available_height_px;
    
    // 取两者中的较大值，确保两个方向都满足margin要求
    // 距离越大，图像中的矩形越小，因此选择较大的距离以保证矩形不超出边缘
    float optimal_z = std::max(target_z_by_width, target_z_by_height);
    
    return optimal_z;
}

enum class FitStrategy {
    BALANCED,   // 平衡策略，让矩形完全在margin内
    PRIORITY_WIDTH,  // 优先满足宽度方向
    PRIORITY_HEIGHT  // 优先满足高度方向
};

float calculateOptimalDistanceEx(const AICameraIntrinsic& g_cam_param,
                                  const cv::Rect& bbox,
                                  float current_z,
                                  int target_margin = 100,
                                  FitStrategy strategy = FitStrategy::BALANCED) {
    // 计算当前矩形在图像中的尺寸
    float current_rect_width_px = bbox.width;
    float current_rect_height_px = bbox.height;
    
    // 计算可用的图像区域（减去margin）
    float available_width_px = g_cam_param.width - 2 * target_margin;
    float available_height_px = g_cam_param.height - 2 * target_margin;
    
    // 矩形实际物理尺寸
    float rect_physical_width = current_rect_width_px * current_z / g_cam_param.fx;
    float rect_physical_height = current_rect_height_px * current_z / g_cam_param.fy;
    
    // 按宽度和高度分别计算目标距离
    float target_z_by_width = rect_physical_width * g_cam_param.fx / available_width_px;
    float target_z_by_height = rect_physical_height * g_cam_param.fy / available_height_px;
    
    float optimal_z;
    switch (strategy) {
        case FitStrategy::PRIORITY_WIDTH:
            optimal_z = target_z_by_width;
            break;
        case FitStrategy::PRIORITY_HEIGHT:
            optimal_z = target_z_by_height;
            break;
        case FitStrategy::BALANCED:
        default:
            // 选择较大的距离，确保两个方向都不超出margin
            optimal_z = std::max(target_z_by_width, target_z_by_height);
            break;
    }
    
    return optimal_z;
}

/**
 * @brief 调整距离使矩形充满图像（带margin）
 * @param g_cam_param 相机内参
 * @param img 当前图像
 * @param bbox 图像中的矩形区域
 * @param current_z 当前摄像头到矩形的实际距离（毫米）
 * @param target_margin 目标边缘像素（默认100）
 * @return 调整后的新距离（毫米）
 * 
 * @note 假设摄像头和矩形平面保持平行
 */
float adjustDistanceToFillImage(const AICameraIntrinsic& g_cam_param,
                                 const cv::Mat& img,
                                 const cv::Rect& bbox,
                                 float current_z,
                                 int target_margin = 100) {
    // 参数验证
    if (g_cam_param.width <= 2 * target_margin || g_cam_param.height <= 2 * target_margin) {
        // margin过大，无法满足要求
        return current_z;
    }
    
    if (bbox.width <= 0 || bbox.height <= 0) {
        // 无效的矩形
        return current_z;
    }
    
    // 计算最佳距离
    float optimal_z = calculateOptimalDistanceEx(g_cam_param, bbox, 
                                                current_z, target_margin);
    
    // 可选：添加一些保护措施，避免距离过近或过远
    float min_z = 100.0f;  // 最小距离10厘米
    float max_z = 2000.0f; // 最大距离2米
    
    optimal_z = std::max(min_z, std::min(optimal_z, max_z));
    
    return optimal_z;
}




// calibration 空料盘
void calibration_plate_empty(Mat& image, Mat* p_img_dbg,
    vector< vector<Detection_pan>>& list_dets, json& j,
    const Mat* p_img_edge = nullptr)
{
    //enum CLASS_IDS {
    //    BACK = 0, LUNZI = 1, HOLE = 2,
    //    PLATE_TRAY = 3, PLATE_CELL = 4, PLATE_HOLE = 5
    //};    
    
    auto& plates = list_dets[PLATE_TRAY];
    auto fail_calibration = [&](const std::string& msg) {
        j["ret-msg"] = msg;
        j["ret-code"] = -101;
        j["tray_complete"] = false;
        j["pts"] = json::array();
    };

    if (plates.empty())
    {
        fail_calibration("det no plate");
        return;
    }

    // 示教只使用完整落在已检测料盘框内的孔位。边界外相邻料盘即使槽位中心
    // 落入框内，也不能参与当前料盘的 2x4 拟合。
    filter_hole_by_plate_bbox(
        plates[0], list_dets[LUNZI], list_dets[HOLE]);
    filter_hole_by_plate_bbox(
        plates[0], list_dets[PLATE_CELL], list_dets[PLATE_HOLE]);

    std::vector<float> candidate_widths;
    std::vector<float> candidate_heights;
    std::vector<cv::Point2f> candidate_centers;
    for (size_t i = 0; i < list_dets[PLATE_CELL].size(); ++i)
    {
        const auto& d = list_dets[PLATE_CELL][i];
        if (d.class_id < 0) continue;
        candidate_widths.push_back(d.bbox.width);
        candidate_heights.push_back(d.bbox.height);
        cv::Point2f cp = (d.bbox.tl() + d.bbox.br()) * 0.5f;
        if (i < list_dets[PLATE_HOLE].size() && list_dets[PLATE_HOLE][i].class_id >= 0)
            cp = (list_dets[PLATE_HOLE][i].bbox.tl() +
                  list_dets[PLATE_HOLE][i].bbox.br()) * 0.5f;
        candidate_centers.push_back(cp);
    }
    for (size_t i = 0; i < list_dets[LUNZI].size(); ++i)
    {
        const auto& d = list_dets[LUNZI][i];
        if (d.class_id < 0) continue;
        candidate_widths.push_back(d.bbox.width);
        candidate_heights.push_back(d.bbox.height);
        cv::Point2f cp = (d.bbox.tl() + d.bbox.br()) * 0.5f;
        if (i < list_dets[HOLE].size() && list_dets[HOLE][i].class_id >= 0)
            cp = (list_dets[HOLE][i].bbox.tl() +
                  list_dets[HOLE][i].bbox.br()) * 0.5f;
        candidate_centers.push_back(cp);
    }
    if (candidate_widths.empty())
    {
        fail_calibration("det no tray candidates");
        return;
    }
    float avg_side = 0.5f * (
        medianValue(candidate_widths) + medianValue(candidate_heights));

    cv::Rect2f tray_roi;
    std::string tray_source;
    if (!resolveTrayRoi(list_dets, candidate_centers, avg_side, tray_roi, tray_source))
    {
        fail_calibration("det no plate roi");
        return;
    }
    std::vector<TrayCandidate> candidates = buildTrayCandidates(list_dets, tray_roi);
    if (candidates.empty())
    {
        fail_calibration("det no tray candidates in roi");
        return;
    }
    std::vector<cv::Point2f> selection_polygon =
        currentTraySelectionPolygon();
    TrayPostprocessConfig postprocess_cfg;
    TrayReferenceSize tray_reference = computeCandidateEvidence(
        candidates, tray_roi,
        selection_polygon.empty() ? nullptr : &selection_polygon,
        tray_source, postprocess_cfg, false);
    if (!tray_reference.valid)
    {
        fail_calibration("det no valid tray reference candidates");
        return;
    }
    classifyCandidateRoles(candidates, tray_source, postprocess_cfg);
    candidates = consolidateSameSlotCandidates(
        candidates, tray_reference.side, postprocess_cfg);

    std::array<cv::Point2f, 8> ordered_grid;
    std::string grid_failure_reason;
    RowPairSelectionResult row_pair_selection;
    float grid_reference_side = tray_reference.side;
    TrayFitting trayFitter;
    if (!buildPrimaryGrid(
            candidates, tray_roi,
            selection_polygon.empty() ? nullptr : &selection_polygon,
            p_img_edge, p_img_dbg, trayFitter, ordered_grid,
            row_pair_selection, grid_reference_side,
            grid_failure_reason))
    {
        std::string msg = "tray grid fit failed";
        if (!grid_failure_reason.empty())
            msg += ": " + grid_failure_reason;
        fail_calibration(msg);
        return;
    }

    TrayPostprocessResult pp_result = assignTraySlotsTwoPass(
        candidates, ordered_grid, grid_reference_side, tray_source,
        tray_roi, tray_reference.safe_tray);
    if (!pp_result.tray_complete)
    {
        fail_calibration("tray grid is incomplete");
        return;
    }

    std::vector<cv::Point2f> selected_grid(ordered_grid.begin(), ordered_grid.end());
    cv::RotatedRect rc = cv::minAreaRect(selected_grid);

    // 需要旋转的角度
    float degrees = std::abs(rc.angle);
    if (degrees<45) {
        degrees = -degrees;
    }
    else
    {
        degrees = 90-degrees;        
    }

    float radians = degreesToRadians(degrees);

    cv::Scalar cl_red = Scalar(0, 0, 255);

//     trayFitter.center_pt = rc.center;
//     trayFitter.radians = radians;
//     trayFitter.p_img_dbg = p_img_dbg;

//     // 执行拟合
//     trayFitter.fitCenterPoints();

// #ifdef REL_DBG
//     if (NULL!= p_img_dbg)
//     {
//         cv::Scalar cl_p = Scalar(255, 0, 255);
//         for (auto& c : trayFitter.clusters)
//         {
//             for (const auto& dp : c.pts_ret) {
//                 cv::circle(*p_img_dbg, cv::Point(dp.x, dp.y), 20,
//                     cv::Scalar(0, 0, 255), -1);
//             }
//         }

//         for (const auto& dp : trayFitter.list_pts) {
//             cv::circle(*p_img_dbg, cv::Point(dp.x, dp.y), 20,
//                 cl_p, -1);
//         }
//     }


// #endif // 

//     std::vector<cv::Point2f> list_pts = trayFitter.list_pts;
//     if (list_pts.size()==0)
//     {
//         j["ret-msg"] = "det no plate";
//         j["ret-code"] = -101;
//         return;
//     }

//     // 求个平均
//     cv::Point2f pt_avg(0,0);
//     for (auto& pt : list_pts)
//     {   
//         pt_avg+=pt;
//     }

//     pt_avg = pt_avg/list_pts.size();
    
    // cv::Point2f pt_avg = (plates[0].bbox.tl()+plates[0].bbox.br())/2;

    cv::Point2f pt_avg(0.0f, 0.0f);
    for (const auto& pt : ordered_grid)
        pt_avg += pt;
    pt_avg *= (1.0f / ordered_grid.size());

    // 画出来看看
    cv::circle(image, pt_avg, 10, cl_red, 8);

    float x, y, z;
    float max_diff = 100.0f;
    if (!get_robust_depth_by_bbox(
            rc.boundingRect2f(), z, &g_ref_depth, &max_diff) ||
        !is_depth_validate(z, "plate_empty", g_ref_depth))
    {
        fail_calibration("plate depth is invalidate");
        return;
    }

    // 根据 z 值 获取对应的 x y
    g_depth_info.get_world_xy_by_z(pt_avg.x, pt_avg.y, x, y, z);

    // 构建 最终输出的点
    vector<float> pt_world = { x,y, 0, -degrees,z};

    // 如果角度很小 而且 x y 也很小，那就只调整 z 
    if (std::abs(degrees)<3 
    // && std::abs(x)<30&& std::abs(y)<30
    )
    {
        // 计算最佳距离
        float optimal_z = adjustDistanceToFillImage(g_cam_param,image,plates[0].bbox,z);

        float ratio = 0.8f;
        pt_world = { x,y, (z-optimal_z)*ratio, 0,z};
    }
    

    j["pts"] = pt_world;     
    j["tray_complete"] = true;
    j["calibration_depth_method"] = "median_mad_mean";
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;         
}

// calibration 轮盘
void calibration_lunzi(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    cv::Scalar cl_red = Scalar(0, 0, 255);
    cv::Scalar cl_g = Scalar(0, 255, 0);

    // 检测传送带上的 轮盘 分布
    vector<Detection_pan>& dets = list_dets[LUNZI];
    
    vector<float> list_pts_avg = {0,0,0};
    cv::Point2f pt_avg(0,0);
    int cnt = 0;
    for (auto& det: dets)
    {
        cv::circle(image, cv::Point(det.c.centor.x, det.c.centor.y), 10, cl_g, 8);

        pt_avg += cv::Point2f(det.c.centor.x, det.c.centor.y);

        // 在原半径上 放大两圈，确保取的点在 盘面上
        vector<float>list_r = { det.c.radius+4,det.c.radius+8 };

        float x, y, z;
        // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
        int mode = 0;
        int step = 3; // 采样步长
        get_avg_depth_by_radius(det.c.centor.x, det.c.centor.y, 
            list_r, step, mode, z);

        if (z<10||z>3000) continue;

        // 根据 z 值 获取对应的 x y
        g_depth_info.get_world_xy_by_z(det.c.centor.x, det.c.centor.y, x, y, z);

        // 构建 最终输出的点
        list_pts_avg[0]+= x;
        list_pts_avg[1]+= y;
        list_pts_avg[2]+= z;

        cnt++;
    }


    if (dets.size()==0 || 0==cnt)
    {
        j["ret-msg"] = "det no lunzi";
        j["ret-code"] = -100;
        return;
    }

    cv::Point2f pt_c((list_pts_avg[0]/cnt),(list_pts_avg[1]/cnt));
    float avg_z = list_pts_avg[2]/cnt;
    vector< float > list_pts = {pt_c.x,pt_c.y,0,0,avg_z};
 
    pt_avg /= cnt;
    cv::circle(image, pt_avg, 10, cl_red, 8);    

    j["pts"] = list_pts;     
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;

}

// feilun 向下标定：在 g_rect 区域内计算最大深度的平均值，返回 ref_depth
void calibration_feilun_down(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    if (g_rect.size() != 8)
    {
        j["ret-msg"] = "rect is invalid";
        j["ret-code"] = -1;
        return;
    }

    // g_rect → Rect2f
    float x_min = std::min({g_rect[0], g_rect[2], g_rect[4], g_rect[6]});
    float y_min = std::min({g_rect[1], g_rect[3], g_rect[5], g_rect[7]});
    float x_max = std::max({g_rect[0], g_rect[2], g_rect[4], g_rect[6]});
    float y_max = std::max({g_rect[1], g_rect[3], g_rect[5], g_rect[7]});
    cv::Rect2f bbox(x_min, y_min, x_max - x_min, y_max - y_min);

    int sample_cnt = 1000;  // 采样点数
    int mode = 1;           // 最大的那些平均
    float ref_depth = -1;
    get_avg_depth_by_bbox(bbox, sample_cnt, mode, ref_depth);

    if (ref_depth < 10 || ref_depth > 3000)
    {
        j["ret-msg"] = "ref_depth is invalid";
        j["ret-code"] = -1;
        return;
    }

    j["ref_depth"] = ref_depth;
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;
}

// 采集一张图片并计算摄像头到标定板的距离（毫米），失败返回 -1
float estimate_board_distance(double& h, json* j_diag = nullptr)
{
    h = -1;
    CharucoBoardConfig board_cfg;
    CharucoDetector detector(board_cfg);

    // 搜索 hand_eye_result*.yaml（支持 -orb, -oak 等后缀）
    bool params_loaded = false;
    std::string loaded_params_file;
    if (fs::exists(g_paramPath))
    {
        // 手眼矩阵：根据 hand_eye_mode 选择文件
        loaded_params_file = g_paramPath + "hand_eye_result.yaml";
        if ("eye_in_hand"!=g_hand_eye_mode)
        {
            loaded_params_file = g_paramPath + "eye_to_hand_result.yaml";
        }

        cv::FileStorage fs_yml(loaded_params_file, cv::FileStorage::READ);
        if (fs_yml.isOpened())
        {
            cv::Mat K, D;
            fs_yml["camera_matrix"] >> K;
            fs_yml["distortion_coefficients"] >> D;
            if (!K.empty() && !D.empty())
            {
                detector.setCameraParams(K, D);
                params_loaded = true;
            }
            fs_yml.release();
        }

    }

    // 如果 YAML 加载失败，用 g_cam_param 构造内参
    if (!params_loaded && g_cam_param.fx > 0)
    {
        cv::Mat K = (cv::Mat_<double>(3, 3) <<
            g_cam_param.fx, 0, g_cam_param.cx,
            0, g_cam_param.fy, g_cam_param.cy,
            0, 0, 1);
        detector.setCameraParams(K, g_dist_coeffs.empty() ? cv::Mat::zeros(5, 1, CV_64F) : g_dist_coeffs);
        loaded_params_file = "from cfg.ini";
        params_loaded = true;
    }

    if (!params_loaded)
    {
        if (j_diag) { (*j_diag)["error"] = "camera_params"; (*j_diag)["g_paramPath"] = g_paramPath; }
        std::cerr << "[estimate_board_distance] camera params not loaded" << std::endl;
        return -1;
    }

    cv::Mat image;
    const char* image_source = "";
    bool image_undistorted = false;
    if (1 == g_test_mode && !g_bgrImage.empty())
    {
        image = g_bgrImage;
        image_source = "test_mode";
        // 只有在全局图确实被矫正过时才换模型；test_mode 刚读进来的原图不算
        image_undistorted = g_bgrImage_undistorted;
    }
    else if (g_camera && g_camera->isOpened())
    {
        // 现场是实时抓帧，拿到的是原图，与标定文件的 K/D 天然配对
        image = g_camera->captureOneFrame();
        image_source = "camera";
    }
    else if (!g_bgrImage.empty())
    {
        image = g_bgrImage;
        image_source = "bgrImage";
        image_undistorted = g_bgrImage_undistorted;
    }

    // 去畸变后图像的内参已经是 newK，位姿解算必须换成 newK + 零畸变，
    // 否则等于拿畸变坐标去解算，测距会有系统误差
    if (image_undistorted)
    {
        cv::Mat K_new = (cv::Mat_<double>(3, 3) <<
            g_cam_param.fx, 0, g_cam_param.cx,
            0, g_cam_param.fy, g_cam_param.cy,
            0, 0, 1);
        detector.setCameraParams(K_new, cv::Mat::zeros(5, 1, CV_64F));
        loaded_params_file += " (undistorted, use newK)";
    }

    if (image.empty())
    {
        if (j_diag)
        {
            (*j_diag)["error"] = "no_image";
            (*j_diag)["camera_opened"] = (g_camera && g_camera->isOpened());
            (*j_diag)["bgrImage_empty"] = g_bgrImage.empty();
        }
        std::cerr << "[estimate_board_distance] no image available" << std::endl;
        return -1;
    }

    if (j_diag)
    {
        (*j_diag)["image_source"] = image_source;
        (*j_diag)["image_size"] = std::to_string(image.cols) + "x" + std::to_string(image.rows);
        (*j_diag)["params_file"] = loaded_params_file;
        (*j_diag)["board_cfg"] = std::to_string(board_cfg.squares_x) + "x" + std::to_string(board_cfg.squares_y)
            + " sq=" + std::to_string(board_cfg.square_length) + " mk=" + std::to_string(board_cfg.marker_length)
            + " dict=" + std::to_string(board_cfg.dictionary_id);
    }

    float dist = detector.estimateDistance(image, h);
    if (dist < 0)
    {
        if (j_diag) { (*j_diag)["error"] = "detect_failed"; }
        std::cerr << "[estimate_board_distance] board detection FAILED" << std::endl;

        std::string dbg_img = g_appPath + "board_detect_fail.jpg";
        cv::imwrite(dbg_img, image);
        std::cout << "[estimate_board_distance] debug image saved to " << dbg_img << std::endl;
        return -1;
    }

    return dist;
}

// feilun 向下放置：在 g_rect 内找下一个可放置飞轮的空位
int det_to_pts_feilun_down(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    j["pts_cell"] = json::array();

    if (g_rect.size() != 8)
    {
        j["ret-msg"] = "invalid g_rect";
        j["ret-code"] = -1;
        return -1;
    }

    if (g_ref_depth < 100)
    {
        j["ret-msg"] = "ref_depth is invalid";
        j["ret-code"] = -1;
        return -1;
    }

    // 用 g_cur_feilun_id 在 g_list_feilun 中按 id 字段匹配
    int feilun_idx = -1;
    for (size_t k = 0; k < g_list_feilun.size(); k++)
    {
        if (g_list_feilun[k].id == g_cur_feilun_id)
        {
            feilun_idx = (int)k;
            break;
        }
    }

    if (feilun_idx < 0)
    {
        j["ret-msg"] = "g_cur_feilun_id not found in feilun_types";
        j["ret-code"] = -1;
        return -1;
    }

    float plane_depth = g_ref_depth;

    FEILUN_INFO& feilun = g_list_feilun[feilun_idx];
    vector<Detection_pan>& dets = list_dets[LUNZI];

    float feilun_diameter = feilun.diameter;
    float feilun_radius = feilun_diameter / 2;
    float cell_size = feilun_diameter + 10;  // 飞轮直径 + 10mm 间隔

    // g_rect 的 4 个角点到世界坐标
    float corners_img[4][2] = {
        {(float)g_rect[0], (float)g_rect[1]},
        {(float)g_rect[2], (float)g_rect[3]},
        {(float)g_rect[4], (float)g_rect[5]},
        {(float)g_rect[6], (float)g_rect[7]}
    };

    float wx_min = std::numeric_limits<float>::max(), wy_min = std::numeric_limits<float>::max();
    float wx_max = std::numeric_limits<float>::lowest(), wy_max = std::numeric_limits<float>::lowest();
    for (int i = 0; i < 4; i++)
    {
        float wx, wy;
        g_depth_info.get_world_xy_by_z(corners_img[i][0], corners_img[i][1], wx, wy, plane_depth);
        if (wx < wx_min) wx_min = wx;
        if (wx > wx_max) wx_max = wx;
        if (wy < wy_min) wy_min = wy;
        if (wy > wy_max) wy_max = wy;
    }

    // 已有飞轮的世界位置（WHEEL → LUNZI）
    vector<cv::Point3f> occupied;
    for (auto& det : dets)
    {
        if (det.class_id < 0) continue;
        cv::Point2f cp = (det.bbox.tl() + det.bbox.br()) / 2;
        float wx, wy;
        g_depth_info.get_world_xy_by_z(cp.x, cp.y, wx, wy, plane_depth);
        occupied.push_back(cv::Point3f(wx, wy, plane_depth));
    }

    // z = g_ref_depth - feilun.eight
    float z = plane_depth - feilun.height-5;

    // 从左下角 (wx_min, wy_min) 开始，逐行扫描
    for (float wy = wy_min; wy + feilun_diameter <= wy_max; wy += cell_size)
    {
        for (float wx = wx_min; wx + feilun_diameter <= wx_max; wx += cell_size)
        {
            float cx = wx + feilun_radius;
            float cy = wy + feilun_radius;

            // 检查是否与已有飞轮重叠
            bool free = true;
            for (auto& oc : occupied)
            {
                float dx = cx - oc.x;
                float dy = cy - oc.y;
                if (std::sqrt(dx * dx + dy * dy) < cell_size)
                {
                    free = false;
                    break;
                }
            }

            if (free)
            {
                j["pts_cell"].push_back({cx, cy, z, 0});
                j["ret-msg"] = "ok";
                j["ret-code"] = 0;
                return 0;
            }
        }
    }

    // 全部填满
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;
    return 0;
}


/*
根据 trayFitter.list_pts_sort 的顺序，为list_det排序
排序规则为：
d是 list_det 中的元素
cv::Point2f cp = (d.bbox.tl()+ d.bbox.br())/2;
cp是检测结果的中心点
dist是 cp和 trayFitter.list_pts_sort 中每个元素 pt 的距离，如果距离最小的，
而且距离小于 trayFitter.cell_size/2 的 可以认为 d 排序应该和 pt 在 trayFitter.list_pts_sort中顺序一致
最终输出的 list_det 就按 d 应该在trayFitter.list_pts_sort中顺序来排列

*/
void sort_cells(TrayFitting& trayFitter, vector<Detection_pan>& list_det)
{
    if (list_det.empty() || trayFitter.list_pts_sort.empty())
        return;

    int n = (int)list_det.size();
    int m = (int)trayFitter.list_pts_sort.size();
    float threshold = trayFitter.cell_size / 2.0f;

    // 为每个检测结果找到在 list_pts_sort 中最匹配的点
    vector<int> match_idx(n, -1);
    for (int i = 0; i < n; i++)
    {
        cv::Point2f cp = (list_det[i].bbox.tl() + list_det[i].bbox.br()) / 2;
        float best_dist = std::numeric_limits<float>::max();
        int best_j = -1;
        for (int j = 0; j < m; j++)
        {
            cv::Point2f pt = trayFitter.list_pts_sort[j];
            float dx = cp.x - pt.x;
            float dy = cp.y - pt.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_j = j;
            }
        }
        if (best_dist < threshold)
            match_idx[i] = best_j;
    }

    // 按 match_idx 排序，未匹配的放在末尾
    vector<int> indices(n);
    for (int i = 0; i < n; i++)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        int ma = match_idx[a] >= 0 ? match_idx[a] : m;
        int mb = match_idx[b] >= 0 ? match_idx[b] : m;
        return ma < mb;
    });

    vector<Detection_pan> sorted(n);
    for (int i = 0; i < n; i++)
        sorted[i] = list_det[indices[i]];
    list_det = std::move(sorted);
}




double getDistanceToImageCenter(const cv::Mat& image, const cv::Point& pt) {
    // 计算图像中心点坐标
    cv::Point2d center(image.cols / 2.0, image.rows / 2.0);
    
    // 计算点到中心的欧氏距离
    double dx = pt.x - center.x;
    double dy = pt.y - center.y;
    
    return std::sqrt(dx * dx + dy * dy);
}

double getMinDistanceToEdge(const cv::Mat& image, const cv::Point& pt) {
    // 获取图像尺寸
    int rows = image.rows;
    int cols = image.cols;
    
    // 计算到四条边的距离
    int dist_top = pt.y;                    // 到上边距离
    int dist_bottom = rows - 1 - pt.y;      // 到下边距离
    int dist_left = pt.x;                   // 到左边距离
    int dist_right = cols - 1 - pt.x;       // 到右边距离
    
    // 返回最小距离
    return std::min({dist_top, dist_bottom, dist_left, dist_right});
}



int det_to_pts_plate_top(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    // 精拍时机器人已经移动到粗拍候选位上方。只接受图像中心附近的
    // 本格位证据，禁止用相邻格位的孔或轮子代替当前格位。
    j["pts_cell_rect"] = json::array();

    if (1 != g_top_capture)
    {
        j["ret-msg"] = "invalid plate top capture mode";
        j["ret-code"] = -1;
        return -1;
    }

    vector< vector<float>> list_lunzi_out;
    vector< vector<float>> list_cell_out;
    vector< vector< vector<float>>> list_cell_rect_out;

    const cv::Point2f image_center(
        image.cols * 0.5f, image.rows * 0.5f);
    auto make_candidates = [](
        const vector<Detection_pan>& detections) {
        vector<PlateTopCandidate> candidates;
        candidates.reserve(detections.size());
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto& det = detections[i];
            if (det.class_id < 0 || det.bbox.width <= 0.0f ||
                det.bbox.height <= 0.0f)
                continue;
            candidates.push_back(
                { static_cast<int>(i), det.bbox });
        }
        return candidates;
    };

    std::cout << "plate_top raw_counts cell="
              << list_dets[PLATE_CELL].size()
              << " plate_hole=" << list_dets[PLATE_HOLE].size()
              << " lunzi=" << list_dets[LUNZI].size()
              << " hole=" << list_dets[HOLE].size()
              << std::endl;

    PlateTopSelection selection = selectPlateTopTarget(
        image.size(),
        make_candidates(list_dets[PLATE_CELL]),
        make_candidates(list_dets[PLATE_HOLE]),
        make_candidates(list_dets[LUNZI]));

    if (selection.kind == PlateTopSelectionKind::None)
    {
        selection.target_px = image_center;
        selection.source = "image_center_fallback";
    }

    const cv::Point2f target_px = selection.target_px;
    const cv::Rect2f selected_cell_bbox = selection.cell_bbox;
    const char* target_source = selection.source;

    cv::circle(image, target_px, 10, Scalar(0, 255, 0), 8);
    if (p_img_dbg != nullptr && !p_img_dbg->empty())
        cv::circle(*p_img_dbg, target_px, 10, Scalar(0, 255, 0), 8);

    std::cout << "plate_top selected=" << target_source
              << " target_px=(" << target_px.x << "," << target_px.y << ")"
              << " image_center=(" << image_center.x << "," << image_center.y
              << ") distance=" << cv::norm(target_px - image_center)
              << std::endl;

    cv::Rect2f rect_all(1,1,image.cols-2,image.rows-2);


    // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
    int mode = 2;
    float avg_depth = 0;
    float max_diff = 100;
    get_avg_depth_by_bbox(rect_all,1000,mode,avg_depth,&g_ref_depth,&max_diff);

    // mode = 1;
    // float avg_depth_max = 0;
    // get_avg_depth_by_bbox(rect_all,1000,mode,avg_depth_max,&g_ref_depth,&max_diff);  

    // // 检查深度合法性
    if( 
        // !is_depth_validate(avg_depth_max, "plate_empty", g_ref_depth) || 
        !is_depth_validate(avg_depth, "plate_empty", g_ref_depth) )
    {
        j["ret-msg"] = "plate depth is invalidate";
        j["ret-code"] = -1;   
        return -1;
    }
    
    // avg_depth_max = (avg_depth_max+avg_depth)/2;            

    float x = 0.0f;
    float y = 0.0f;
    const float z = avg_depth;
    g_depth_info.get_world_xy_by_z(
        target_px.x, target_px.y, x, y, avg_depth);
    const float radius = 10.0f;
    vector<float> pt_world = { x, y, z, radius };

    if (selection.kind == PlateTopSelectionKind::Cell ||
        selection.kind == PlateTopSelectionKind::Hole ||
        selection.kind == PlateTopSelectionKind::None)
    {
        list_cell_out.push_back(pt_world);

        if (selection.kind == PlateTopSelectionKind::Cell)
        {
            std::array<cv::Point2f, 4> corners = {
                selected_cell_bbox.tl(),
                cv::Point2f(
                    selected_cell_bbox.x + selected_cell_bbox.width,
                    selected_cell_bbox.y),
                selected_cell_bbox.br(),
                cv::Point2f(
                    selected_cell_bbox.x,
                    selected_cell_bbox.y + selected_cell_bbox.height)
            };
            vector<vector<float>> rect_world;
            for (const auto& corner : corners)
            {
                float corner_x = 0.0f;
                float corner_y = 0.0f;
                float corner_z = avg_depth;
                g_depth_info.get_world_xy_by_z(
                    corner.x, corner.y, corner_x, corner_y, corner_z);
                rect_world.push_back({ corner_x, corner_y, corner_z });
            }
            list_cell_rect_out.push_back(rect_world);
        }
    }
    else
    {
        list_lunzi_out.push_back(pt_world);
    }

    j["pts"] = list_lunzi_out;
    j["pts_cell"] = list_cell_out;
    j["pts_cell_rect"] = list_cell_rect_out;
    j["pts_lunzi"] = list_lunzi_out;   
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;   
    
    return 0;
}

// 处理空料盘
int det_to_pts_plate_empty(Mat& image, Mat* p_img_dbg,
    vector< vector<Detection_pan>>& list_dets, json& j,
    const Mat* p_img_edge = nullptr)
{
    if (1 == g_top_capture)
    {
        return det_to_pts_plate_top(image, p_img_dbg, list_dets, j);
    }

    std::vector<float> candidate_widths;
    std::vector<float> candidate_heights;
    for (auto& d : list_dets[PLATE_CELL])
    {
        if (d.class_id < 0) continue;
        candidate_widths.push_back(d.bbox.width);
        candidate_heights.push_back(d.bbox.height);
    }
    for (auto& d : list_dets[LUNZI])
    {
        if (d.class_id < 0) continue;
        candidate_widths.push_back(d.bbox.width);
        candidate_heights.push_back(d.bbox.height);
    }
    if (candidate_widths.empty())
        return -1;
    float avg_side = 0.5f * (
        medianValue(candidate_widths) + medianValue(candidate_heights));

    std::vector<cv::Point2f> candidate_centers;
    for (size_t i = 0; i < list_dets[PLATE_CELL].size(); ++i)
    {
        auto& d = list_dets[PLATE_CELL][i];
        if (d.class_id < 0) continue;

        cv::Point2f cp = (d.bbox.tl() + d.bbox.br()) / 2;
        if (i < list_dets[PLATE_HOLE].size() && list_dets[PLATE_HOLE][i].class_id >= 0)
        {
            auto& hole = list_dets[PLATE_HOLE][i];
            cp = (hole.bbox.tl() + hole.bbox.br()) / 2;
        }
        candidate_centers.push_back(cp);
    }
    for (size_t i = 0; i < list_dets[LUNZI].size(); ++i)
    {
        auto& d = list_dets[LUNZI][i];
        if (d.class_id < 0) continue;

        cv::Point2f cp = (d.bbox.tl() + d.bbox.br()) / 2;
        if (i < list_dets[HOLE].size() && list_dets[HOLE][i].class_id >= 0)
        {
            auto& hole = list_dets[HOLE][i];
            cp = (hole.bbox.tl() + hole.bbox.br()) / 2;
        }
        candidate_centers.push_back(cp);
    }

    cv::Rect2f tray_roi;
    std::string tray_source;
    if (!resolveTrayRoi(list_dets, candidate_centers, avg_side, tray_roi, tray_source))
    {
        fillIncompletePlateEmptyJson(j, "warning: tray_not_found");
        j["tray_source"] = "none";
        return 0;
    }

    std::vector<TrayCandidate> candidates = buildTrayCandidates(list_dets, tray_roi);
    if (candidates.empty())
    {
        fillIncompletePlateEmptyJson(j, "warning: no tray candidates");
        j["tray_source"] = tray_source;
        return 0;
    }
    std::vector<cv::Point2f> selection_polygon =
        currentTraySelectionPolygon();
    TrayPostprocessConfig postprocess_cfg;
    const bool use_tray_reference_pool =
        hasValidTraySelectionRect() && tray_source == "detected" &&
        g_top_capture != 1;
    TrayReferenceSize tray_reference = computeCandidateEvidence(
        candidates, tray_roi,
        selection_polygon.empty() ? nullptr : &selection_polygon,
        tray_source, postprocess_cfg, use_tray_reference_pool);
    if (!tray_reference.valid)
    {
        draw_plate_empty_final_dets(image, list_dets);
        fillIncompletePlateEmptyJson(
            j, "warning: no valid tray reference candidates");
        j["tray_source"] = tray_source;
        return 0;
    }
    classifyCandidateRoles(candidates, tray_source, postprocess_cfg);
    candidates = consolidateSameSlotCandidates(
        candidates, tray_reference.side, postprocess_cfg);

    std::array<cv::Point2f, 8> ordered_grid;
    std::string grid_failure_reason;
    RowPairSelectionResult row_pair_selection;
    float grid_reference_side = tray_reference.side;
    TrayFitting trayFitter;
    if (!buildPrimaryGrid(
            candidates, tray_roi,
            selection_polygon.empty() ? nullptr : &selection_polygon,
            p_img_edge, p_img_dbg, trayFitter, ordered_grid,
            row_pair_selection, grid_reference_side,
            grid_failure_reason))
    {
#ifdef REL_DBG
        std::cout << "grid selection failed source="
                  << TrayFitting::rowPairSelectionSourceName(
                         row_pair_selection.source)
                  << " reason=" << row_pair_selection.failure_reason
                  << std::endl;
#endif
        draw_plate_empty_final_dets(image, list_dets);
        std::string msg = "warning: grid fit failed";
        if (!grid_failure_reason.empty())
            msg += ": " + grid_failure_reason;
        fillIncompletePlateEmptyJson(j, msg);
        j["tray_source"] = tray_source;
        return 0;
    }

#ifdef REL_DBG
    if (NULL != p_img_dbg)
    {
        cv::Scalar cl_p = Scalar(255, 0, 255);
        for (const auto& pt : trayFitter.list_pts_sort)
            cv::circle(*p_img_dbg, pt, 12, cl_p, 2);
        for (const auto& pt : trayFitter.list_pts)
            cv::circle(*p_img_dbg, pt, 20, cv::Scalar(0, 0, 255), -1);
    }
#endif

    cv::Scalar cl_red = Scalar(0, 0, 255);
    for (const auto& pt : trayFitter.list_pts)
        cv::circle(image, pt, 10, cl_red, 8);

    float avg_depth = 0.0f;
    float max_diff = 100.0f;
    std::vector<cv::Point2f> depth_region_points(
        ordered_grid.begin(), ordered_grid.end());
    cv::RotatedRect rc = cv::minAreaRect(depth_region_points);
    if (g_rect.size() == 8)
    {
        std::vector<cv::Point> vertices;
        for (int i = 0; i < 4; ++i)
            vertices.push_back(cv::Point(g_rect[i * 2], g_rect[i * 2 + 1]));
        rc = cv::minAreaRect(vertices);
        cv::Point2f ct = rc.center;
        rc.size.width = rc.size.width * 0.7f;
        rc.size.height = rc.size.height * 0.7f;
        rc.center = ct;
    }

    if (!get_robust_depth_by_bbox(
            rc.boundingRect2f(), avg_depth, &g_ref_depth, &max_diff) ||
        !is_depth_validate(avg_depth, "plate_empty", g_ref_depth))
    {
        j["ret-msg"] = "plate depth is invalidate";
        j["ret-code"] = -1;
        return -1;
    }

    applyDepthAssistedTrayStateRecovery(
        candidates, ordered_grid, grid_reference_side, avg_depth,
        postprocess_cfg);
    TrayPostprocessResult pp_result = assignTraySlotsTwoPass(
        candidates, ordered_grid, grid_reference_side, tray_source,
        tray_roi, tray_reference.safe_tray);

    drawTrayPostprocessOverlay(image, pp_result);
#ifdef REL_DBG
    if (p_img_dbg != nullptr && !p_img_dbg->empty())
        drawTrayPostprocessOverlay(*p_img_dbg, pp_result);
#endif

    fillPlateEmptyJson(j, pp_result, trayFitter.list_pts, avg_depth);
    return 0;
}
// 处理轮盘
int det_to_pts_lunzi(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    cv::Scalar cl_red = Scalar(0, 0, 255);
    // if (g_rect.size()!=8) g_rect = g_rect_lunzi;


    // if (g_rect.size()==8)
    // {
    //     for (size_t i = 0; i < 3; i++)
    //     {
    //         cv::line(image, cv::Point(g_rect[i*2+0],g_rect[i*2+1]), cv::Point(g_rect[i*2+2],g_rect[i*2+3]), cl_red);
    //     }  
        
    //     cv::line(image, cv::Point(g_rect[0],g_rect[1]), cv::Point(g_rect[6],g_rect[7]), cl_red);
    // }
        

    float centor_dist = -1;
    float max_diff = 100;
    if (g_ref_depth>100)
    {
        centor_dist = g_ref_depth;
    }
    else
    {
        // if (g_rect.size()==8)
        // {
        //     std::cout<< "det lunzi lunzi lunzi in rect " ;
        //     printRect(g_rect);


        //     // g_rect → Rect2f
        //     float x_min = std::min({g_rect[0], g_rect[2], g_rect[4], g_rect[6]});
        //     float y_min = std::min({g_rect[1], g_rect[3], g_rect[5], g_rect[7]});
        //     float x_max = std::max({g_rect[0], g_rect[2], g_rect[4], g_rect[6]});
        //     float y_max = std::max({g_rect[1], g_rect[3], g_rect[5], g_rect[7]});
        //     cv::Rect2f bbox(x_min, y_min, x_max - x_min, y_max - y_min);            

        //     int sample_cnt = 1000;  // 采样点数
        //     int mode = 1;           // 最大的那些平均
        //     get_avg_depth_by_bbox(bbox, sample_cnt, mode,centor_dist);
        // }    
                
    }


    // 检测传送带上的 轮盘 分布
    vector<Detection_pan>& dets = list_dets[LUNZI];
    vector<Detection_pan> dets_holes = list_dets[HOLE];
    vector< vector<float> > list_pts;

    // 记录有效轮子的中心和半径，用于计算轮子间隙处的传送带深度
    struct WheelInfo { cv::Point2f center; float radius; float z; };
    std::vector<WheelInfo> valid_wheels;

    Detection_pan max_lunzi;
    double min_dist = 9999;
    int min_idx = -1;
    for (size_t i = 0; i < dets.size(); i++)
    {
        auto& det = dets[i];
        if (det.class_id < 0)
            continue;

        if (det.bbox.area()>max_lunzi.bbox.area())
        {
            max_lunzi = det;
        }

        double dist = getDistanceToImageCenter(image, cv::Point2f(det.c.centor.x, det.c.centor.y));
        if (dist<min_dist)
        {
            min_dist = dist;
            min_idx = i;
        }      
    }
  
    float min_area = max_lunzi.bbox.area() * 0.8;
    for (size_t i = 0; i < dets.size(); i++)
    {
        if (1==g_top_capture)
        {
            if (i!=min_idx)
            {
                continue;
            }

            std::cout<< "top_capture  top_capture  lunzi lunzi lunzi" << std::endl;
        }
        

        auto& det = dets[i];
        if (det.class_id < 0)
            continue;

        float cx = det.c.centor.x;
        float cy = det.c.centor.y;
        if (g_lunzi_only)
        {
            cx = (det.bbox.tl().x + det.bbox.br().x) * 0.5f;
            cy = (det.bbox.tl().y + det.bbox.br().y) * 0.5f;
        }

        // 如果盘子面积不及最大轮子 80% 而且 盘子中心 距离图像边缘 不足最大轮子 宽度的一半，就不是合格的轮子
        int dist = getMinDistanceToEdge(image, cv::Point2f(cx, cy));
        if (std::abs(dist)<max_lunzi.bbox.width/2 && det.bbox.area()<min_area)
        {
            continue;
        }
        
        if (g_rect.size()>0)
        {
            if(!isPtInArea(g_rect, cv::Point2f(cx, cy)))
                continue;
        }

        cv::circle(image, cv::Point(cx, cy), 10, Scalar(255, 0, 0), 8);

        float x, y, z;
        int step = 3; // 采样步长

        float max_d=0,min_d=0;
        // 分类方式 0: 全平均  1: 最大的那些平均  2: 最 小 的那些平均  
        int mode = 2;
        float mid_radius = 0;
        bool has_valid_hole =
            i < dets_holes.size() &&
            dets_holes[i].class_id >= 0 &&
            dets_holes[i].bbox.area() > 0;

        if (g_lunzi_only || !has_valid_hole)
        {
            // 不依赖中心孔，在轮子内外圆环上采样深度
            if (!g_lunzi_only)
            {
                std::cout << "lunzi fallback: hole missing, use wheel ring depth, index="
                          << i << std::endl;
            }
            float inner_r = det.c.radius * 0.35f;
            mid_radius = det.c.radius * 0.75f;
            vector<float> list_r_inner = { inner_r };
            get_avg_depth_by_radius(cx, cy,
                list_r_inner, step, mode, min_d, &centor_dist, &max_diff);
        }
        else
        {
            auto& det_hole = dets_holes[i];
            get_avg_depth_by_bbox(det_hole.bbox,1000,mode,min_d,&centor_dist,&max_diff);

            // 在原半径上 放大两圈，确保取的点在 盘面上
            mid_radius = (det.c.radius + det_hole.c.radius)/2;

            if ("feilun"==g_model_name)
            {
                if (det.c.radius>det_hole.c.radius)
                {
                    mid_radius = det_hole.c.radius + (det.c.radius - det_hole.c.radius)/3;
                }
                else
                {
                    mid_radius = det_hole.c.radius*2;
                }
            }
        }

        vector<float>list_r = { mid_radius+2,mid_radius+4 };

        mode = 0;        
        get_avg_depth_by_radius(cx, cy, 
            list_r, step, mode, max_d,&centor_dist,&max_diff, min_d+5);    


        // 检查深度合法性
        if( !is_depth_validate(max_d, "lunzi", g_ref_depth) || 
            !is_depth_validate(min_d, "lunzi", g_ref_depth) )
        {
            std::cout<< "lunzi depth is invalidate"<< std::endl;
            // j["ret-msg"] = "lunzi depth is invalidate";
            // j["ret-code"] = -1;   
            continue;
        }
        
        cv::circle(image, cv::Point(cx, cy), 10, cl_red, 8);

        z = (max_d + min_d)/2;

        // 如果 z 比 min_d 大20毫米 就不对
        if (z>(min_d+20)) z = min_d+20;

        if ("feilun"==g_model_name)
        {
            if (z>(min_d+5)) z = min_d+5;
        }
        

        // 根据 z 值 获取对应的 x y
        g_depth_info.get_world_xy_by_z(cx, cy, x, y, z);
        

        // 把半径外扩 step 个像素，以便抓取
        float x_outer = cx;
        float y_outer = cy;
        step = 3;
        x_outer = cx + det.c.radius + step;

        cv::Point2f pt_outer(x_outer, y_outer);
        g_depth_info.get_world_xy_by_z(pt_outer.x, pt_outer.y, x_outer, y_outer, z);

        // 计算3d 半径
        float radius = calculateDistance(x,y,z, x_outer, y_outer, z);

        // 构建 最终输出的点
        vector<float> pt_world = {x,y,z,radius};
        list_pts.push_back(pt_world);

        // 记录有效轮子信息，用于计算传送带深度
        valid_wheels.push_back({cv::Point2f(cx, cy),
                                det.c.radius, z});
    }

    // ---- 计算传送带深度（非轮子区域，深度值众数） ----------------------------
    float worktop_depth = -1;
    if (!valid_wheels.empty() && !g_depthImage.empty()) {
        // 以所有有效轮子深度的中位数作为轮面参考深度
        std::vector<float> wheel_zs;
        for (auto& w : valid_wheels) wheel_zs.push_back(w.z);
        std::sort(wheel_zs.begin(), wheel_zs.end());
        float wheel_z_ref = wheel_zs[wheel_zs.size() / 2];

        int dw = g_depthImage.cols, dh = g_depthImage.rows;

        // 用直方图统计非轮子区域中 [lo, hi] 范围内的深度值，取众数
        float lo = wheel_z_ref + 5.0f;
        float hi = wheel_z_ref + 80.0f;
        int bin_count = 80;  // 1mm 一个 bin
        std::vector<int> hist(bin_count, 0);

        int step = 3;  // 采样步长，3 像素一次
        for (int y = 0; y < dh; y += step) {
            const uint16_t* row = g_depthImage.ptr<uint16_t>(y);
            for (int x = 0; x < dw; x += step) {
                uint16_t val = row[x];
                if (val == 0) continue;
                float d = (float)val;
                if (d < lo || d > hi) continue;

                // 跳过轮子区域（圆周内）
                bool on_wheel = false;
                for (auto& w : valid_wheels) {
                    float dx2 = (x - w.center.x) * (x - w.center.x);
                    float dy2 = (y - w.center.y) * (y - w.center.y);
                    if (dx2 + dy2 < w.radius * w.radius) {
                        on_wheel = true;
                        break;
                    }
                }
                if (on_wheel) continue;

                int bin = (int)(d - lo);
                if (bin >= 0 && bin < bin_count) hist[bin]++;
            }
        }

        // 找众数（直方图最大值对应的深度）
        int max_count = 0;
        int best_bin = -1;
        for (int b = 0; b < bin_count; b++) {
            if (hist[b] > max_count) {
                max_count = hist[b];
                best_bin = b;
            }
        }

        if (best_bin >= 0) {
            worktop_depth = lo + (float)best_bin + 0.5f;  // bin 中心
        }
    }
    j["worktop_depth"] = worktop_depth;

    j["pts_lunzi"] = list_pts;
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;

    return 0;
}

// 处理 screen_centor：通过 7 个 circle 拟合圆心 + 定位孔
// PALLET_SCREEN(1) → LUNZI(1), CIRCLE_SCREEN(2) → HOLE(2)
// 圆周上 7 个 circle：6 个均匀分布(60°间隔) + 1 个定位孔(在其中2个中间)
int det_to_pts_screen_centor(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    cv::Scalar cl_red = Scalar(0, 0, 255);
    cv::Scalar cl_green = Scalar(0, 255, 0);
    cv::Scalar cl_blue = Scalar(255, 0, 0);
    cv::Scalar cl_yellow = Scalar(0, 255, 255);

    vector<Detection_pan>& dets_circle = list_dets[CIRCLE_SCREEN];  // index 2 == HOLE
    vector< vector<double> > list_pts;

    if (dets_circle.size() < 5)
    {
        j["ret-msg"] = "not enough circles detected, need > 4";
        j["ret-code"] = -1;
        return -1;
    }

    // 计算每个 circle 的中心（内切圆圆心）
    struct CircleInfo {
        int idx;
        cv::Point2f center;
        float radius;
        float angle;
    };


    // 圆孔中心
    vector<cv::Point2f> pts;    
    vector<CircleInfo> circles;
    for (size_t i = 0; i < dets_circle.size(); i++)
    {
        cv::Point2f c;
        float r;
        getInscribedCircleFromRect(dets_circle[i].bbox, c, r);
        circles.push_back({ (int)i, c, r, 0.0f });

        pts.push_back(c);
    }



    // 找到异常点
    std::vector<PointWithStats> point_stats2;
    Circle c = CircleFitterWithOutlierRemoval::fitCircleWithStatisticalOutlierRemoval(
        pts, point_stats2);

    // draw pt
    cv::circle(image, c.center, 2, cl_red);

    cv::circle(image, c.center, c.radius, cl_green);

    int n = circles.size();

    // 计算每个 circle 相对拟合圆心的角度
    for (auto& ci : circles)
    {
        float dx = ci.center.x - c.center.x;
        float dy = ci.center.y - c.center.y;
        ci.angle = std::atan2(dy, dx) * 180.0 / CV_PI;
        if (ci.angle < 0) ci.angle += 360.0;
    }

    // 按角度排序
    std::sort(circles.begin(), circles.end(),
        [](const CircleInfo& a, const CircleInfo& b) { return a.angle < b.angle; });

    // 计算相邻角度差
    vector<float> gaps(n);
    for (int i = 0; i < n - 1; i++)
        gaps[i] = circles[i + 1].angle - circles[i].angle;
    gaps[n - 1] = circles[0].angle + 360.0 - circles[n - 1].angle;

    // 找到两个最小 gap → 定位孔在这两个 gap 之间
    // 6 个均匀孔 gap≈60°，2 个小 gap 分别在定位孔两侧
    vector<int> gap_idx(n);
    for (int i = 0; i < n; i++) gap_idx[i] = i;
    std::sort(gap_idx.begin(), gap_idx.end(),
        [&](int a, int b) { return gaps[a] < gaps[b]; });

    // 最小的两个 gap 的公共端点就是定位孔
    int small1 = gap_idx[0];
    int small2 = gap_idx[1];
    int position_hole_idx = (small1 + 1) % n;
    if (position_hole_idx != small2)
    {
        position_hole_idx = small1;
        if ((small1 - 1 + n) % n != small2)
        {
            j["ret-msg"] = "cannot identify position hole: two smallest gaps not adjacent";
            j["ret-code"] = -1;
            return -1;
        }
    }

    // 验证定位孔确实存在：最小 gap 应明显小于中位 gap
    // 如果定位孔缺失，所有圆均匀分布，gap 都接近（如 6 个圆每个 ~60°），
    // 则最小 gap 与中位 gap 接近（比值 ≈ 1.0），无法区分定位孔
    {
        float median_gap = gaps[gap_idx[n / 2]];
        float ratio = gaps[small1] / std::max(median_gap, 1e-6f);
        if (ratio > 0.65f)
        {
            j["ret-msg"] = "cannot identify position hole: gaps too uniform, min/median ratio="
                          + std::to_string(ratio);
            j["ret-code"] = -1;
            return -1;
        }
    }

    CircleInfo& pos_hole = circles[position_hole_idx];


    //=========================================
    // 根据p1的坐标，求出三个圆孔的角度
    cv::Point2f pt_hole_right = cv::Point2f(c.center.x + c.radius, c.center.y);
    

    // 求出 pos_hole 和 pt_hole_right 的夹角
    float delta_angle = calculateAngleBetweenLines(c.center, pt_hole_right, pos_hole.center);

    
    // 转成角度显示
    delta_angle = delta_angle / arc_angle;    
    if (pos_hole.center.y > pt_hole_right.y)
    {
        // 目前 pos_hole 在 零位 下面，需要 逆 时针旋转
        delta_angle = -std::abs(delta_angle);
    }
    else
    {
        // 目前 pos_hole 在 零位 上面，需要 顺 时针旋转
        delta_angle = std::abs(delta_angle);
    }


    //============================================
    // MVS 相机无深度信息，直接使用 ref_depth
    float z = g_ref_depth;

    // 拟合圆心转世界坐标
    float cx, cy;
    g_depth_info.get_world_xy_by_z(c.center.x, c.center.y, cx, cy, z);

    // 圆盘半径转世界坐标
    int step = 3;
    float x_outer = c.center.x + c.radius + step;
    cv::Point2f pt_outer(x_outer, c.center.y);
    float y_outer_tmp;
    g_depth_info.get_world_xy_by_z(pt_outer.x, pt_outer.y, x_outer, y_outer_tmp, z);
    float pallet_radius = calculateDistance(cx, cy, z, x_outer, y_outer_tmp, z);

    // 定位孔世界坐标
    float px, py;
    g_depth_info.get_world_xy_by_z(pos_hole.center.x, pos_hole.center.y, px, py, z);
    float pos_hole_radius = 0;

    // 绘制拟合圆
    cv::circle(image, cv::Point(c.center.x, c.center.y), 2, cl_red, -1);         // 圆心实心点
    cv::circle(image, cv::Point(c.center.x, c.center.y), 8, cl_red, 3);          // 圆心标记环
    cv::circle(image, cv::Point(c.center.x, c.center.y), c.radius, cl_green, 3); // 拟合圆周

    // 绘制所有 circles，定位孔用特殊颜色
    for (size_t i = 0; i < circles.size(); i++)
    {
        auto& ci = circles[i];
        cv::Scalar color = ((int)i == position_hole_idx) ? cl_blue : cl_yellow;
        cv::circle(image, cv::Point(ci.center.x, ci.center.y), 5, color, 3);
    }

    // pallet 拟合圆心
    // Vec3 dlt_cam = {cx, cy, z},base_xyz;
    // compute_base_xyz_ByDstCam(dlt_cam,g_hand_eye_matrix,base_xyz);

    vector<double> pt_center = { cx, cy, z, delta_angle };
    // vector<double> pt_center = { base_xyz[0], base_xyz[1], base_xyz[2], delta_angle };    
    list_pts.push_back(pt_center);

    // 定位孔
    // dlt_cam = {px, py, z};
    // compute_base_xyz_ByDstCam(dlt_cam,g_hand_eye_matrix,base_xyz);
    vector<double> pt_pos_hole = { px, py, z, pos_hole_radius };
    //vector<double> pt_pos_hole = { base_xyz[0], base_xyz[1], base_xyz[2], pos_hole_radius };
    list_pts.push_back(pt_pos_hole);

    j["pts_lunzi"] = list_pts;
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;

    return 0;
}

static void draw_top_face_debug(Mat& image, const json& det_json)
{
    if (g_rect.size() == 8)
    {
        cv::Scalar cl_rect = Scalar(0, 0, 255);
        for (size_t i = 0; i < 3; i++)
        {
            cv::line(image,
                cv::Point(g_rect[i * 2 + 0], g_rect[i * 2 + 1]),
                cv::Point(g_rect[i * 2 + 2], g_rect[i * 2 + 3]),
                cl_rect, 2);
        }
        cv::line(image,
            cv::Point(g_rect[0], g_rect[1]),
            cv::Point(g_rect[6], g_rect[7]),
            cl_rect, 2);
    }

    if (!det_json.contains("objects") || !det_json["objects"].is_array())
        return;

    for (const auto& obj : det_json["objects"])
    {
        if (!obj.contains("status") || obj["status"] != "ok")
            continue;
        if (obj.contains("corners_px") && obj["corners_px"].is_array() &&
            obj["corners_px"].size() == 4)
        {
            std::vector<cv::Point> pts;
            for (const auto& p : obj["corners_px"])
            {
                if (p.is_array() && p.size() >= 2)
                    pts.emplace_back(cv::Point(cvRound(p[0].get<double>()),
                        cvRound(p[1].get<double>())));
            }
            if (pts.size() == 4)
            {
                for (int i = 0; i < 4; i++)
                    cv::line(image, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }
        if (obj.contains("center_px") && obj["center_px"].is_array() &&
            obj["center_px"].size() >= 2)
        {
            cv::Point c(cvRound(obj["center_px"][0].get<double>()),
                cvRound(obj["center_px"][1].get<double>()));
            cv::circle(image, c, 5, Scalar(0, 0, 255), -1);
            if (obj.contains("angle_deg"))
            {
                double ang = obj["angle_deg"].get<double>() * CV_PI / 180.0;
                cv::Point e(cvRound(c.x + 40 * std::cos(ang)),
                    cvRound(c.y - 40 * std::sin(ang)));
                cv::line(image, c, e, Scalar(255, 0, 0), 2);
            }
        }
    }
}

static void copy_top_face_diag(const json& det_json, json& j)
{
    static const char* keys[] = {
        "background_generation", "feature_backend", "elapsed_ms",
        "alignment_mode", "background_updated", "background_image",
        "foreground_object_count", "stage_ms"
    };
    for (const char* k : keys)
    {
        if (det_json.contains(k))
            j[k] = det_json[k];
    }
}

int det_to_pts_top_face(Mat& image, const json& det_json, json& j)
{
    j["pts_lunzi"] = json::array();
    j["list_pts_base"] = json::array();
    j["angle_deg"] = json::array();
    j["pts_top_face"] = json::array();
    j["DET-TYPE"] = g_det_type;
    if(std::isfinite(g_top_face_workpiece_height_mm))j["workpiece_height_mm"]=g_top_face_workpiece_height_mm;
    copy_top_face_diag(det_json, j);
    if (det_json.contains("top_face_count"))
        j["top_face_count"] = det_json["top_face_count"];
    if (det_json.contains("objects"))
        j["objects"] = det_json["objects"];

    draw_top_face_debug(image, det_json);

    if (g_clean_mode)
    {
        int count = det_json.value("top_face_count", 0);
        if (count <= 0)
        {
            j["ret-msg"] = "no top_face detected";
            j["ret-code"] = -1;
            return -1;
        }
        j["ret-msg"] = "ok";
        j["ret-code"] = 0;
        return 0;
    }

    if (g_ref_depth < 100)
    {
        j["ret-msg"] = "ref_depth is not set";
        j["ret-code"] = -1;
        return -1;
    }

    if (!det_json.contains("objects") || !det_json["objects"].is_array())
    {
        j["ret-msg"] = "no top_face detected";
        j["ret-code"] = -1;
        return -1;
    }

    struct Hit {
        double cx, cy, angle, length_px;
        double dist;
    };
    std::vector<Hit> hits;
    for (const auto& obj : det_json["objects"])
    {
        if (!obj.contains("status") || obj["status"] != "ok")
            continue;
        if (!obj.contains("center_px") || !obj["center_px"].is_array() ||
            obj["center_px"].size() < 2)
            continue;
        Hit h{};
        h.cx = obj["center_px"][0].get<double>();
        h.cy = obj["center_px"][1].get<double>();
        h.angle = obj.value("angle_deg", 0.0);
        h.length_px = obj.value("apparent_length_px", 0.0);
        h.dist = getDistanceToImageCenter(image, cv::Point2f((float)h.cx, (float)h.cy));
        hits.push_back(h);
    }

    if (hits.empty())
    {
        if (det_json.value("ret-code", -1) == 0 &&
            det_json.value("top_face_count", 0) == 0)
        {
            j["ret-msg"] = "ok";
            j["ret-code"] = 0;
            return 0;
        }
        j["ret-msg"] = "no top_face detected";
        j["ret-code"] = -1;
        return -1;
    }

    std::sort(hits.begin(), hits.end(),
        [](const Hit& a, const Hit& b) { return a.dist < b.dist; });

    float plane_depth = g_ref_depth;
    vector<vector<float>> list_pts;
    vector<vector<double>> list_pts_base;
    vector<double> list_angle;

    for (size_t i = 0; i < hits.size(); i++)
    {
        if (1 == g_top_capture && i != 0)
            continue;
        const Hit& h = hits[i];
        int ix = cvRound(h.cx);
        int iy = cvRound(h.cy);
        float wx = 0, wy = 0;
        g_depth_info.get_world_xy_by_z(ix, iy, wx, wy, plane_depth);

        // pts_lunzi[3] is the top-face long edge in millimetres (not lunzi radius).
        // Project both ends along the detected long axis at ref_depth.
        float length_mm = 0;
        if (h.length_px > 0)
        {
            // angle_deg is counter-clockwise-positive; negate so the projected endpoints are unchanged.
            const double rad = -h.angle * CV_PI / 180.0;
            const double hx = 0.5 * h.length_px * std::cos(rad);
            const double hy = 0.5 * h.length_px * std::sin(rad);
            float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            g_depth_info.get_world_xy_by_z(
                cvRound(h.cx + hx), cvRound(h.cy + hy), x1, y1, plane_depth);
            g_depth_info.get_world_xy_by_z(
                cvRound(h.cx - hx), cvRound(h.cy - hy), x2, y2, plane_depth);
            length_mm = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }

        list_pts.push_back({wx, wy, plane_depth, length_mm});
        Vec3 dlt_cam = {wx, wy, plane_depth}, base_xyz;
        if (!g_hand_eye_matrix.empty())
            compute_base_xyz_ByDstCam(dlt_cam, g_hand_eye_matrix, base_xyz);
        else
            base_xyz = {wx, wy, plane_depth};
        // Only calibrated TCP XY changes; camera coordinates, Z and length stay legacy.
        double base_x=base_xyz[0], base_y=base_xyz[1];
        try {
            const cv::Matx33d imageCamera(g_cam_param.fx,0,g_cam_param.cx,
                                         0,g_cam_param.fy,g_cam_param.cy,0,0,1);
            top_face_tcp_xy(h.cx,h.cy,image.size(),base_x,base_y,
                g_top_face_workpiece_height_mm,imageCamera,g_bgrImage_undistorted);
        } catch (const std::exception& e) {
            j["ret-msg"]=e.what();
            j["ret-code"]=-1;
            return -1;
        }
        list_pts_base.push_back({base_x, base_y, base_xyz[2], length_mm});
        double a = h.angle;
        while (a > 90.0)
            a -= 180.0;
        while (a <= -90.0)
            a += 180.0;
        list_angle.push_back(a);
        j["pts_top_face"].push_back({h.cx, h.cy, a});
    }

    j["pts_lunzi"] = list_pts;
    j["list_pts_base"] = list_pts_base;
    j["angle_deg"] = list_angle;
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;
    return 0;
}

// 处理 feilun_up：2D 摄像头，平面假设，匹配 feilun 型号后计算 hole 中心 xyz
int det_to_pts_feilun_up(Mat& image, Mat* p_img_dbg, vector< vector<Detection_pan>>& list_dets, json& j)
{
    j["pts_lunzi"] = json::array();

    // clean 模式：只看推理后有没有检测到 feilun，无需匹配和计算
    if (g_clean_mode)
    {
        vector<Detection_pan>& dets = list_dets[LUNZI];
        if (dets.empty())
        {
            j["ret-msg"] = "no feilun detected";
            j["ret-code"] = -1;
            return -1;
        }
        j["ret-msg"] = "ok";
        j["ret-code"] = 0;
        return 0;
    }

    if (g_ref_depth < 100)
    {
        j["ret-msg"] = "ref_depth is not set";
        j["ret-code"] = -1;
        return -1;
    }
    if (g_list_feilun.empty())
    {
        j["ret-msg"] = "g_list_feilun is empty";
        j["ret-code"] = -1;
        return -1;
    }

    // 如果调用者指定了 feilun_id，则按 ID 查找飞轮
    FEILUN_INFO* p_fixed_feilun = nullptr;
    if (!g_cur_feilun_id.empty())
    {
        for (auto& fl : g_list_feilun)
        {
            if (fl.id == g_cur_feilun_id)
            {
                p_fixed_feilun = &fl;
                break;
            }
        }
        if (!p_fixed_feilun)
        {
            j["ret-msg"] = "g_cur_feilun_id not found in feilun_types";
            j["ret-code"] = -1;
            return -1;
        }
    }

    float plane_depth = g_ref_depth;  // 摄像头到平面的距离


    // WHEEL → LUNZI(1), BOSS → HOLE(2)
    vector<Detection_pan>& dets = list_dets[LUNZI];
    vector<Detection_pan>& dets_holes = list_dets[HOLE];

    if (dets.empty())
    {
        j["ret-msg"] = "no wheel detected";
        j["ret-code"] = -1;
        return -1;
    }

    // 按到图像中心的距离排序（近的在前）
    vector<size_t> sorted_idx(dets.size());
    vector<double> dist_to_center(dets.size());
    for (size_t i = 0; i < dets.size(); i++)
    {
        sorted_idx[i] = i;
        cv::Point2f cp = (dets[i].bbox.tl() + dets[i].bbox.br()) / 2;
        dist_to_center[i] = getDistanceToImageCenter(image, cp);
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
        [&](size_t a, size_t b) { return dist_to_center[a] < dist_to_center[b]; });

    vector< vector<float> > list_pts;
    vector< vector<double> > list_pts_base;
    vector<std::string> list_feilun_id;

    
    for (size_t idx : sorted_idx)
    {
        auto& det = dets[idx];
        if (det.class_id < 0) continue;

        // top_capture 只保留距离图像中心最近的
        if (1 == g_top_capture && idx != sorted_idx[0]) continue;

        // WHEEL 直径（世界 mm）：从 bbox 内切圆计算
        cv::Point2f wc;
        float wr;
        getInscribedCircleFromRect(det.bbox, wc, wr);

        float wcx, wcy;
        g_depth_info.get_world_xy_by_z(wc.x, wc.y, wcx, wcy, plane_depth);
        float wex, wey;
        g_depth_info.get_world_xy_by_z(wc.x + wr, wc.y, wex, wey, plane_depth);
        float wheel_diameter = 2 * std::sqrt((wex - wcx) * (wex - wcx) + (wey - wcy) * (wey - wcy));

        // BOSS 直径（世界 mm）：c.radius 已由 calculate_centor_info 根据 hole bbox 设置
        float boss_diameter = 0;
        if (det.c.radius > 0)
        {
            float bcx, bcy;
            g_depth_info.get_world_xy_by_z(det.c.centor.x, det.c.centor.y, bcx, bcy, plane_depth);
            float bex, bey;
            g_depth_info.get_world_xy_by_z(det.c.centor.x + det.c.radius, det.c.centor.y, bex, bey, plane_depth);
            boss_diameter = 2 * std::sqrt((bex - bcx) * (bex - bcx) + (bey - bcy) * (bey - bcy));
        }

        FEILUN_INFO* p_feilun = nullptr;

        if (p_fixed_feilun)
        {
            // 调用者指定了 feilun_id，直接使用，不做自动匹配
            p_feilun = p_fixed_feilun;
        }
        else
        {
            // 在 g_list_feilun 中找最佳匹配：综合 wheel 直径 + boss 直径差异最小
            int best_idx = -1;
            float best_score = std::numeric_limits<float>::max();
            for (size_t k = 0; k < g_list_feilun.size(); k++)
            {
                float d_diff = std::abs(wheel_diameter - g_list_feilun[k].diameter);
                float h_diff = (boss_diameter > 0) ? std::abs(boss_diameter - g_list_feilun[k].boss_diameter) : 0;
                float score = d_diff + h_diff;
                if (score < best_score)
                {
                    best_score = score;
                    best_idx = (int)k;
                }
            }

            if (best_idx < 0) continue;

            p_feilun = &g_list_feilun[best_idx];
        }

        // 检查 wheel 直径差异是否超过 10%
        float diameter_diff_ratio = std::abs(wheel_diameter - p_feilun->diameter) / p_feilun->diameter;
        if (diameter_diff_ratio >= 0.1f)
        {
            if (1 == g_top_capture)
            {
                j["ret-msg"] = "feilun diameter mismatch";
                j["ret-code"] = -1;
                return -1;
            }
            continue;
        }

        FEILUN_INFO& feilun = *p_feilun;

        // WHEEL 中心 → 世界坐标（z = ref_depth，即飞轮底面所在平面）
        float wx, wy;
        g_depth_info.get_world_xy_by_z(det.c.centor.x, det.c.centor.y, wx, wy, plane_depth);

        // 相机安装倾斜补偿：飞轮在图像左右两侧的等效高度不同，
        // 按飞轮中心 x 的归一化位置线性插值（t=0 图像最左，t=1 图像最右）
        float h_offset_t = 0.0f;
        if (image.cols > 1)
        {
            h_offset_t = det.c.centor.x / (float)(image.cols - 1);
            h_offset_t = std::min(std::max(h_offset_t, 0.0f), 1.0f);
        }
        float height_compensated = feilun.height + g_feilun_h_offset_left
            + (g_feilun_h_offset_right - g_feilun_h_offset_left) * h_offset_t;

        // BOSS 中心 → 世界坐标（z = ref_depth - height_compensated，BOSS 更靠近摄像头）
        float boss_z = plane_depth - height_compensated;
        float bx = wx, by = wy;  // 默认回退到 WHEEL 坐标
        if (idx < dets_holes.size() && dets_holes[idx].c.radius > 0)
        {
            auto& hole = dets_holes[idx];
            g_depth_info.get_world_xy_by_z(hole.c.centor.x, hole.c.centor.y, bx, by, boss_z);
        }

        // 平均 WHEEL 和 BOSS 的世界坐标，减小图像边缘的透视误差
        float avg_x = (wx + bx) / 2.0f;
        float avg_y = (wy + by) / 2.0f;

        // 输出半径用 boss 直径的一半
        float radius = feilun.boss_diameter / 2;

        // 这里的z 修改为建议抓取位置
        float z = boss_z + feilun.boss_height - 1;
        list_pts.push_back({avg_x, avg_y, z, radius});

        Vec3 dlt_cam = {avg_x, avg_y, z},base_xyz;
        compute_base_xyz_ByDstCam(dlt_cam,g_hand_eye_matrix,base_xyz);
        list_pts_base.push_back({base_xyz[0], base_xyz[1], base_xyz[2], radius, feilun.height});

        list_feilun_id.push_back(feilun.id);
    }

    j["pts_lunzi"] = list_pts;
    j["list_pts_base"] = list_pts_base;
    j["feilun_id"] = list_feilun_id;
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;
    return 0;
}

// 处理码垛
int det_to_pts_plate(Mat& image, Mat* p_img_dbg,
    vector< vector<Detection_pan>>& list_dets, json& j,
    const Mat* p_img_edge = nullptr)
{
    return det_to_pts_plate_empty(image, p_img_dbg, list_dets, j, p_img_edge);
}


// 结合中间孔的信息，计算出中心圆的信息
void calculate_centor_info(Detection_pan* det, Detection_pan* hole, CLASS_IDS obj_type)
{
    Point2f c(0,0);
    float radius = 0;

    if (LUNZI==obj_type)
    {
        if (nullptr!=det)
        {
            getInscribedCircleFromRect(det->bbox, c, radius);
            det->c.centor = Point3f(c.x, c.y, 0);
            det->c.radius = radius;
        }

        if (nullptr!=hole)
        {
            getInscribedCircleFromRect(hole->bbox, c, radius);
            hole->c.centor = Point3f(c.x,c.y,0);
            hole->c.radius = radius;
        }
    }
    else
    {
        if (nullptr!=hole)
        {
            getInscribedCircleFromRect(hole->bbox, c, radius);
            det->c.centor = Point3f(c.x,c.y,0);
            det->c.radius = radius;
        }
        else
        {
            getInscribedCircleFromRect(det->bbox, c, radius);
            det->c.centor = Point3f(c.x, c.y, 0);
            det->c.radius = radius/2;
        }
    }


}


void calculate_centor_info(vector<Detection_pan>& det_outs, vector<Detection_pan>& list_hole, CLASS_IDS obj_type)
{
    if(det_outs.size()==0)return;

    bool hasHole = (list_hole.size() == det_outs.size());
    for (size_t i = 0; i < det_outs.size(); i++)
    {
        if (hasHole)
            calculate_centor_info(&det_outs[i], &list_hole[i], obj_type);
        else
            calculate_centor_info(&det_outs[i], nullptr, obj_type);
    }
}

// 整合一下检测结果，把轮盘和圆孔对应上
int merge_ret_xialiao(Mat& image, vector< vector<Detection_pan>>& det_outs, json& j)
{
    // 整合一下检测结果，把轮盘和圆孔对应上
    int ret = 0;

    //enum CLASS_IDS {
    //    BACK = 0, LUNZI = 1, HOLE = 2,
    //    PLATE_TRAY = 3, PLATE_CELL = 4, PLATE_HOLE = 5
    //};    
    // ==================================================================
    // // 结合中间孔的信息，计算出中心圆的信息
    if ("screen" == g_model_name)
    {
        // screen: 保留所有 circle，不通过 align_lunzi_hole 过滤
        // pallet=1 个，circle=7 个（6个均匀分布+1个定位孔）
        calculate_centor_info(det_outs[LUNZI], det_outs[HOLE], LUNZI);
    }
    else
    {
        align_lunzi_hole(det_outs[LUNZI], det_outs[HOLE], true);
        if (g_lunzi_only)
        {
            vector<Detection_pan> empty_holes;
            calculate_centor_info(det_outs[LUNZI], empty_holes, LUNZI);
        }
        else
            calculate_centor_info(det_outs[LUNZI], det_outs[HOLE], LUNZI);
    }
    align_lunzi_hole(det_outs[PLATE_CELL], det_outs[PLATE_HOLE], false);

    // ==================================================================
    // // 结合中间孔的信息，计算出中心圆的信息

    // PLATE_TRAY = 3, 
    if(det_outs[PLATE_TRAY].size()>0)
        calculate_centor_info(det_outs[PLATE_TRAY].data(), nullptr, PLATE_TRAY);

    // PLATE_CELL = 4, 
    // PLATE_HOLE = 5   
    calculate_centor_info(det_outs[PLATE_CELL], det_outs[PLATE_HOLE], PLATE_CELL);

    return ret;
}



int handle_ret_xialiao(Mat& image, vector< vector<Detection_pan>>& det_outs,
    json& j, const Mat* tray_edge_source = nullptr)
{
    int ret = -1;
    j["ret-msg"] = "error: handle_ret_xialiao";

    if (det_outs.size() <= PALLET) return ret;

    Mat* p_img_dbg = NULL;
#ifdef REL_DBG
    Mat img_dbg = image.clone();
    p_img_dbg = &img_dbg;
#endif // REL_DBG

    const cv::Mat* p_img_edge =
        tray_edge_source != nullptr && !tray_edge_source->empty() &&
        tray_edge_source->size() == image.size()
            ? tray_edge_source
            : nullptr;

    for (size_t i = 1; i < det_outs.size(); i++)
    {
        if (g_lunzi_only && i == HOLE)
            continue;
        if ("plate_empty" == g_det_type && 1 != g_top_capture &&
            (i == PLATE_CELL || i == PLATE_HOLE || i == LUNZI || i == HOLE))
            continue;
        vector<Detection_pan> det_ret = det_outs[i];
        for (auto& o : det_ret)
        {
            G_DET->draw_pan(image, o);
        }
    }

    // 绘制检测区域 g_rect，便于看出检测范围在哪里
    if (g_rect.size()==8 &&
        (startWith_str(g_det_type,"feilun")||
                    "feilun" == g_model_name )
        )
    {
        cv::Scalar cl_rect = Scalar(0, 0, 255);
        for (size_t i = 0; i < 3; i++)
        {
            cv::line(image, cv::Point(g_rect[i*2+0],g_rect[i*2+1]), cv::Point(g_rect[i*2+2],g_rect[i*2+3]), cl_rect, 2);
        }
        cv::line(image, cv::Point(g_rect[0],g_rect[1]), cv::Point(g_rect[6],g_rect[7]), cl_rect, 2);
    }


    // 整合一下检测结果，把轮盘和圆孔对应上
    merge_ret_xialiao(image, det_outs,j);

    // =========================================
    // 接下来输出 json结果
    j["DET-TYPE"] = g_det_type;

    if(endWith_str(g_det_type,"_zero"))
    {
        // 校准流程
        if ("plate_empty_zero" == g_det_type || 
            "plate_zero" == g_det_type)
        {
            // 检测 空料盘的 抓取位置
            // PLATE_TRAY = 3, 
            calibration_plate_empty(
                image, p_img_dbg, det_outs, j, p_img_edge);

            json js;
            det_to_pts_plate_empty(
                image, p_img_dbg, det_outs, js, p_img_edge);

            if (js.contains("pts_cell"))
            {
                j["pts_cell"] = js["pts_cell"];      
            }
            if (js.contains("pts_lunzi"))
            {
                j["pts_lunzi"] = js["pts_lunzi"];      
            }       
                   
        }
        else if ("lunzi_zero" == g_det_type)
        {
            // 轮盘在 流水线
            // LUNZI = 1
            // HOLE = 2,
            calibration_lunzi(image, p_img_dbg, det_outs, j);
        }
        else if ("feilun_down_zero" == g_det_type ||
            "feilun_up_zero" == g_det_type)
        {
            // feilun在 下料 流水线
            // calibration_feilun_down(image, p_img_dbg, det_outs, j);
        }
        
        ret = 0;
    }
    else
    {
        if ("plate_empty" == g_det_type)
        {
            // 检测 空料盘的 抓取位置
            // PLATE_TRAY = 3, 
            ret = det_to_pts_plate_empty(
                image, p_img_dbg, det_outs, j, p_img_edge);
        }
        else if ("lunzi" == g_det_type)
        {
            // 轮盘在 流水线
            // LUNZI = 1
            // HOLE = 2,
            ret = det_to_pts_lunzi(image, p_img_dbg, det_outs, j);
        }
        else if ("feilun_up" == g_det_type)
        {
            // 轮盘在 流水线
            // LUNZI = 1
            // HOLE = 2,
            ret = det_to_pts_feilun_up(image, p_img_dbg, det_outs, j);
        }        
        else if ("plate" == g_det_type)
        {
            // 检测码垛上料盘的 空位
            // PLATE_CELL = 4, 
            // PLATE_HOLE = 5
            ret = det_to_pts_plate(
                image, p_img_dbg, det_outs, j, p_img_edge);
        }
        else if ("feilun_down" == g_det_type)
        {
            // 当前feilun的半径为 g_lunzi_radius 毫米
            // 在g_rect 区域内，寻找一个可以放置 当前飞轮的位置
            // 从g_rect 的左下角开始放置，每个 feilun 间隔 10 毫米
            // 建议位置 以 "pts_cell":[[x,y,z,r]] 方式返回，
            // xyz 单位是毫米 ，r是建议选择角度，这里设置为0就行
            // 一次返回一个放置位置，如果已经填满，则返回 "pts_cell":[]
            ret = det_to_pts_feilun_down(image, p_img_dbg, det_outs, j);
        }
        else if ("screen_centor" == g_det_type)
        {
            // 找到 pallet 中心位置
            // PALLET_SCREEN(1) 框出圆盘区域
            // CIRCLE_SCREEN(2) 是圆盘边缘同一半径上的小孔
            // 通过 pallet 和 circle 计算出中心位置
            ret = det_to_pts_screen_centor(image, p_img_dbg, det_outs, j);
        }


        if (0==ret)
        {
            if (!j.contains("ret-msg") ||
                j["ret-msg"].get<std::string>().find("warning:") == std::string::npos)
            {
                j["ret-msg"] = "ok";
            }
        }
               
        j["ret-code"] = ret;
    }

 
#ifdef REL_DBG
    std::string str_json = j.dump();
#endif // REL_DBG

    return ret;
}




int handle_ret(Mat& image, Detection_pan& det_ret, json& j)
{
    int ret = -1;
    
    //if (det_ret.pts.size()== det_ret.sizes.size())
    if (det_ret.pts.size()<6)
    {
        ret = handle_ret_pt4(image, det_ret,j);
    }
    else
    {
        ret = handle_ret_pt10(image, det_ret,j);
    }

    return ret;
}



struct GridCandidate {
    int img_index;      // 输入图像索引
    double gray_value;  // 网格平均灰度值
};

bool compareByDistanceTo128(const GridCandidate& a, const GridCandidate& b) {
    return std::abs(a.gray_value - 128) < std::abs(b.gray_value - 128);
}



/*
* get_merge_images
根据输入的src_imgs(一般是5张图)，输出3~4张用于检测的图
图像输出到output_imgs
* @return: 0: 成功，其他：失败
*/

int get_merge_images(std::vector<cv::Mat>& src_imgs, std::vector<cv::Mat>& output_imgs) {
    // 检查输入有效性
    if (src_imgs.empty()) {
        return -1; // 输入图像为空
    }

    int n = src_imgs.size();
    if (n < 2) {
        return -1; // 至少需要2张输入图像
    }

    // 检查所有图像尺寸是否相同
    cv::Size first_size = src_imgs[0].size();
    for (int i = 1; i < n; i++) {
        if (src_imgs[i].size() != first_size) {
            return -1; // 图像尺寸不一致
        }
    }

    int total_width = first_size.width;
    int total_height = first_size.height;

    // 将输入图像转换为灰度图
    std::vector<cv::Mat> gray_imgs;
    for (int i = 0; i < n; i++) {
        cv::Mat gray_img;
        if (src_imgs[i].channels() == 3) {
            cv::cvtColor(src_imgs[i], gray_img, cv::COLOR_BGR2GRAY);
        }
        else if (src_imgs[i].channels() == 1) {
            gray_img = src_imgs[i].clone();
        }
        else {
            return -1; // 不支持的通道数
        }
        gray_imgs.push_back(gray_img);
    }

    // 计算网格尺寸（处理不能整除的情况）
    int grid_cols = 4;
    int grid_rows = 4;

    std::vector<int> grid_widths(grid_cols, total_width / grid_cols);
    std::vector<int> grid_heights(grid_rows, total_height / grid_rows);

    // 处理宽度余数
    int width_remainder = total_width % grid_cols;
    for (int i = 0; i < width_remainder; i++) {
        grid_widths[i]++;
    }

    // 处理高度余数
    int height_remainder = total_height % grid_rows;
    for (int i = 0; i < height_remainder; i++) {
        grid_heights[i]++;
    }

    // 为每个网格位置存储候选列表
    std::vector<std::vector<GridCandidate>> grid_candidates(grid_rows * grid_cols);

    // 计算每个网格的灰度值并排序
    for (int grid_idx = 0; grid_idx < grid_rows * grid_cols; grid_idx++) {
        int grid_row = grid_idx / grid_cols;
        int grid_col = grid_idx % grid_cols;

        // 计算当前网格的坐标范围
        int start_x = 0;
        for (int i = 0; i < grid_col; i++) {
            start_x += grid_widths[i];
        }
        int end_x = start_x + grid_widths[grid_col];

        int start_y = 0;
        for (int i = 0; i < grid_row; i++) {
            start_y += grid_heights[i];
        }
        int end_y = start_y + grid_heights[grid_row];

        // 对每张输入图像，计算当前网格的平均灰度值
        for (int img_idx = 0; img_idx < n; img_idx++) {
            cv::Mat grid_roi = gray_imgs[img_idx](cv::Range(start_y, end_y), cv::Range(start_x, end_x));
            cv::Scalar mean_value = cv::mean(grid_roi);
            grid_candidates[grid_idx].push_back({ img_idx, mean_value[0] });
        }

        // 按与128的距离排序候选
        std::sort(grid_candidates[grid_idx].begin(), grid_candidates[grid_idx].end(), compareByDistanceTo128);
    }

    // 清空输出向量并创建4张输出图像
    output_imgs.clear();
    for (int i = 0; i < 4; i++) {
        output_imgs.push_back(cv::Mat::zeros(total_height, total_width, CV_8UC3));
    }

    // 填充输出图像
    for (int grid_idx = 0; grid_idx < grid_rows * grid_cols; grid_idx++) {
        int grid_row = grid_idx / grid_cols;
        int grid_col = grid_idx % grid_cols;

        // 计算当前网格的坐标范围
        int start_x = 0;
        for (int i = 0; i < grid_col; i++) {
            start_x += grid_widths[i];
        }
        int end_x = start_x + grid_widths[grid_col];

        int start_y = 0;
        for (int i = 0; i < grid_row; i++) {
            start_y += grid_heights[i];
        }
        int end_y = start_y + grid_heights[grid_row];

        cv::Rect grid_roi(start_x, start_y, end_x - start_x, end_y - start_y);

        // 为每张输出图像选择对应的网格
        for (int output_idx = 0; output_idx < 4; output_idx++) {
            int candidate_rank = 0; // 默认首选

            if (output_idx == 0) {
                // 第一张图：全取首选
                candidate_rank = 0;
            }
            else if (output_idx == 1) {
                // 第二张图：全取二选
                candidate_rank = 1;
            }
            else if (output_idx == 2) {
                // 第三张图：复杂规则
                if (grid_row == 0 || grid_row == 2) { // 第一、三行
                    candidate_rank = (grid_col % 2 == 1) ? 0 : 1; // 奇数列取首选，偶数列取二选
                }
                else { // 第二、四行
                    candidate_rank = (grid_col % 2 == 1) ? 1 : 0; // 奇数列取二选，偶数列取首选
                }
            }
            else if (output_idx == 3) {
                // 第四张图：复杂规则
                if (grid_row == 0 || grid_row == 2) { // 第一、三行
                    candidate_rank = (grid_col % 2 == 1) ? 1 : 0; // 奇数列取二选，偶数列取首选
                }
                else { // 第二、四行
                    candidate_rank = (grid_col % 2 == 1) ? 0 : 1; // 奇数列取首选，偶数列取二选
                }
            }

            // 检查候选排名是否有效
            if (candidate_rank >= n) {
                candidate_rank = n - 1; // 如果排名超出范围，取最后一个候选
            }

            int selected_img_idx = grid_candidates[grid_idx][candidate_rank].img_index;
            cv::Mat selected_grid = src_imgs[selected_img_idx](grid_roi);
            selected_grid.copyTo(output_imgs[output_idx](grid_roi));
        }
    }

    return 0; // 成功
}

#ifndef USE_RDK
extern  bool detectCenterRect(const cv::Mat& color,
                      const cv::Mat& depth,
                      std::vector<cv::Point>& corners_out,
                      const CenterRectConfig& cfg = CenterRectConfig(),
                      cv::Mat* out_image = nullptr,double* c_depth_avg = nullptr);

extern  bool detectPalletByEdgesInROI(const cv::Mat& color,
                      const cv::Mat& depth,
                      const cv::Rect& yolo_roi,
                      std::vector<cv::Point>& corners_out,
                      const CenterRectConfig& cfg);
#endif

int ai_det_tray(json& j, char** ret_json, int* json_len)
{
    cv::Mat& image = g_bgrImage;

    int ret = -1;
    //=====================================
    // 处理 空托盘 检测
    //   先用 YOLO 检测 pallet 大体区域，再用边缘检测精化边界

    std::vector<cv::Point> corners_out;
    CenterRectConfig cfg;

    cfg.cam_fx = g_cam_param.fx;
    cfg.cam_fy = g_cam_param.fy;
    cfg.cam_cx = g_cam_param.cx;
    cfg.cam_cy = g_cam_param.cy;


    cv::Mat image_out;
    double c_depth_avg = 0;
    j["ret-msg"] = "error: det no result";

    // ---- 第一步：YOLO 检测 pallet 大体区域 -------------------------------
    bool yolo_found = false;
    cv::Rect yolo_roi;
    {
        std::vector<std::vector<Detection_pan>> detections;
        int yolo_ret = det_image(image, detections, j);

        if (yolo_ret == 0 && detections.size() > PALLET) {
            std::vector<Detection_pan>& pallet_dets = detections[PALLET];
            if (!pallet_dets.empty()) {
                // 取置信度最高或面积最大的 pallet 检测
                Detection_pan& best = pallet_dets[0]; // 已按面积降序排列
                yolo_roi = cv::Rect(
                    std::max(0, (int)best.bbox.x),
                    std::max(0, (int)best.bbox.y),
                    std::min(image.cols - std::max(0, (int)best.bbox.x), (int)best.bbox.width),
                    std::min(image.rows - std::max(0, (int)best.bbox.y), (int)best.bbox.height)
                );
                yolo_found = (yolo_roi.width > 20 && yolo_roi.height > 20);
#ifdef MY_LOG
                std::cout << "[ai_det_tray] YOLO found pallet at "
                          << yolo_roi.x << "," << yolo_roi.y
                          << " " << yolo_roi.width << "x" << yolo_roi.height
                          << " conf=" << best.conf << std::endl;
#endif
            }
        }
    }

    // ---- 第二步：传统方法精化边界 ------------------------------------------
    bool det_ret = false;

#ifndef USE_RDK
    if (yolo_found) {
        // 用 YOLO ROI 引导的边缘检测精化
        det_ret = detectPalletByEdgesInROI(image, g_depthImage,
                                            yolo_roi, corners_out, cfg);
#ifdef MY_LOG
        if (det_ret)
            std::cout << "[ai_det_tray] ROI edge detection succeeded" << std::endl;
        else
            std::cout << "[ai_det_tray] ROI edge detection failed, fallback to detectCenterRect" << std::endl;
#endif
    }

    // 回退：YOLO 未找到或 ROI 边缘检测失败时，用原始方法
    if (!det_ret) {
        det_ret = detectCenterRect(image, g_depthImage,
                                    corners_out, cfg, &image_out, &c_depth_avg);
    }
#endif

    // 二层回退：无深度图且 detectCenterRect 也失败时，直接用 YOLO bbox 作为近似边界
    if (!det_ret && yolo_found) {
        corners_out.clear();
        corners_out.push_back(cv::Point(yolo_roi.x, yolo_roi.y));                      // TL
        corners_out.push_back(cv::Point(yolo_roi.x + yolo_roi.width, yolo_roi.y));     // TR
        corners_out.push_back(cv::Point(yolo_roi.x + yolo_roi.width, yolo_roi.y + yolo_roi.height)); // BR
        corners_out.push_back(cv::Point(yolo_roi.x, yolo_roi.y + yolo_roi.height));    // BL
        c_depth_avg = 1000.0;
        det_ret = true;
#ifdef MY_LOG
        std::cout << "[ai_det_tray] using YOLO bbox directly as fallback" << std::endl;
#endif
    }

    if (det_ret) {
        // 估算中心深度（仅当深度图可用时使用真实深度）
        if (!g_depthImage.empty()) {
            double sum_depth = 0; int cnt = 0;
            cv::Rect br = cv::boundingRect(corners_out);
            cv::Rect sample_rect(
                br.x + br.width / 4, br.y + br.height / 4,
                br.width / 2, br.height / 2);
            sample_rect &= cv::Rect(0, 0, g_depthImage.cols, g_depthImage.rows);
            for (int r = sample_rect.y; r < sample_rect.y + sample_rect.height; r++) {
                const uint16_t* row = g_depthImage.ptr<uint16_t>(r);
                for (int c = sample_rect.x; c < sample_rect.x + sample_rect.width; c++) {
                    if (row[c] > 0) { sum_depth += row[c]; cnt++; }
                }
            }
            c_depth_avg = (cnt > 0) ? (sum_depth / cnt) : c_depth_avg;
        } else {
            // 无深度图时使用默认值（世界坐标将为近似值）
            if (c_depth_avg <= 0) c_depth_avg = 1000.0;
        }
    }

    if (!det_ret)
    {
        make_ret_json(j, ret_json, json_len);
        return ret;
    }


    cv::RotatedRect rc = cv::minAreaRect(corners_out);

    // 需要旋转的角度
    float degrees = std::abs(rc.angle);
    if (degrees<45) {
        degrees = -degrees;
    }
    else
    {
        degrees = 90-degrees;        
    }

    float radians = degreesToRadians(degrees);

    //=====================================
    // 分析检测结果
    cv::Point2f pt_avg = rc.center;

    // 画出来看看
    cv::Scalar cl_red = Scalar(0, 0, 255);
    cv::circle(image, pt_avg, 10, cl_red, 8);

    // 世界坐标
    float x, y, z;
    z = c_depth_avg;

    // 根据 z 值 获取对应的 x y
    g_depth_info.get_world_xy_by_z(pt_avg.x, pt_avg.y, x, y, z);

    // 构建 最终输出的点
    vector<float> pt_world = { x,y, 0, -degrees,z};

    // 如果角度很小 而且 x y 也很小，那就只调整 z 
    if (std::abs(degrees)<3 
    // && std::abs(x)<30&& std::abs(y)<30
    )
    {
        // 计算最佳距离
        float optimal_z = adjustDistanceToFillImage(g_cam_param,image,rc.boundingRect(),z,200);

        pt_world = { x,y, optimal_z-z, 0,z};
    }
    

    // 这是托盘的四个顶点
    cv::Scalar cl_y = Scalar(0, 255, 255);
    cv::Scalar cl_g = Scalar(0, 255, 0);
    std::vector<int> rect_pts;
    for (size_t i = 0; i < corners_out.size(); i++)
    {
        rect_pts.push_back(corners_out[i].x);
        rect_pts.push_back(corners_out[i].y);
        
        cv::circle(image, corners_out[i], 5, cl_y, 5);
    }

    

    // 计算摆放空盘的位置
    cv::Point top_mid = (corners_out[0] + corners_out[1])/2;
    cv::Point bottom_mid = (corners_out[2] + corners_out[3])/2;    

    // left plate
    cv::Point2f top_mid_left = (corners_out[0] + top_mid)/2;
    cv::Point2f bottom_mid_left = (bottom_mid + corners_out[3])/2;
    
    cv::Point2f centor_left = (top_mid_left + bottom_mid_left)/2;
    cv::Point2f tray_left_pt1 = (top_mid_left + centor_left)/2;
    cv::Point2f tray_left_pt2 = (centor_left + bottom_mid_left)/2;

    cv::circle(image, tray_left_pt1, 10, cl_g, 8);
    cv::circle(image, tray_left_pt2, 10, cl_g, 8);

    // right plate
    cv::Point2f top_mid_right = (corners_out[1] + top_mid)/2;
    cv::Point2f bottom_mid_right = (bottom_mid + corners_out[2])/2;
    
    cv::Point2f centor_right = (top_mid_right + bottom_mid_right)/2;
    cv::Point2f tray_right_pt1 = (top_mid_right + centor_right)/2;
    cv::Point2f tray_right_pt2 = (centor_right + bottom_mid_right)/2;

    cv::circle(image, tray_right_pt1, 10, cl_g, 8);
    cv::circle(image, tray_right_pt2, 10, cl_g, 8);    

    // 不计算了，假设半径就是10毫米
    float radius = 10;       
    std::vector<std::vector<std::vector<float>>> pts_plate;
    std::vector<std::vector<float>> pts_left;
    std::vector<std::vector<float>> pts_right;

    // left
    {
        // 根据 z 值 获取对应的 x y
        g_depth_info.get_world_xy_by_z(tray_left_pt1.x, tray_left_pt1.y, x, y, z);

        // 构建 最终输出的点
        vector<float> pt_world = { x,y,z,radius };
        pts_left.push_back(pt_world);

        // 根据 z 值 获取对应的 x y
        g_depth_info.get_world_xy_by_z(tray_left_pt2.x, tray_left_pt2.y, x, y, z);

        // 构建 最终输出的点
        pt_world = { x,y,z,radius };
        pts_left.push_back(pt_world);        
    }    

    // right
    {
        // 根据 z 值 获取对应的 x y
        g_depth_info.get_world_xy_by_z(tray_right_pt1.x, tray_right_pt1.y, x, y, z);

        // 构建 最终输出的点
        vector<float> pt_world = { x,y,z,radius };
        pts_right.push_back(pt_world);

        // 根据 z 值 获取对应的 x y
        g_depth_info.get_world_xy_by_z(tray_right_pt2.x, tray_right_pt2.y, x, y, z);

        // 构建 最终输出的点
        pt_world = { x,y,z,radius };
        pts_right.push_back(pt_world);        
    }  

    pts_plate.push_back(pts_left);
    pts_plate.push_back(pts_right);
    

/*

"pts":[x,y,z,r,dist],
"rect":[x0,y0,x1,y1,x2,y2,x3,y3],
"pts_plate":[
[[x,y,z,r],[x,y,z,r]],
[[x,y,z,r],[x,y,z,r]]
]

*/    

    j["pts"] = pt_world;
    j["rect"] = rect_pts;
    j["pts_plate"] = pts_plate;
    if (yolo_found) {
        j["pallet"] = {yolo_roi.x, yolo_roi.y, yolo_roi.width, yolo_roi.height};
    }
    j["ret-msg"] = "ok";
    j["ret-code"] = 0;

    make_ret_json(j, ret_json, json_len);


    return ret;
}


static bool needsTrayRowEdgeSelection(const std::string& det_type)
{
    return det_type == "plate" ||
        det_type == "plate_empty" ||
        det_type == "plate_zero" ||
        det_type == "plate_empty_zero";
}

static void applyConfiguredFlip(cv::Mat& image)
{
    // 相机不支持硬件翻转时，才回退到软件翻转
    if (1 != g_flip_soft)
        return;
    if (image.empty())
        return;
    if (1 == g_flip_v && 1 == g_flip_h)
        cv::flip(image, image, -1);
    else if (1 == g_flip_v)
        cv::flip(image, image, 0);
    else if (1 == g_flip_h)
        cv::flip(image, image, 1);
}

int ai_det_one_image(std::vector<Detection_pan>& detections_pan, json& j,
    char** ret_json, int* json_len, cv::Mat* tray_edge_source = nullptr)
{
    cv::Mat& image = g_bgrImage;

    int ret = -1;
    //=====================================
    // 看看是否需要翻转    
    applyConfiguredFlip(image);
    if (tray_edge_source != nullptr && !tray_edge_source->empty())
        applyConfiguredFlip(*tray_edge_source);

    //=====================================
    // 保留一张 原图
    // cv::Mat image_bak = image.clone();
    if (!g_clean_mode && g_save_ret > 0)
    {
        // save file
        {
            advance_det_cnt();

            std::string ret_img_dir = g_appPath + "src_img/";
            if (!fs::exists(ret_img_dir)) fs::create_directory(ret_img_dir);
            std::string ret_img = string_format("%s%05d.jpg", ret_img_dir.c_str(), g_det_cnt);

            cv::imwrite(ret_img, image);

            if (g_cfg_camera.enableDepth && !g_depthImage.empty())
            {
                std::string depth_img = string_format("%s%05d.png", ret_img_dir.c_str(), g_det_cnt);
                cv::imwrite(depth_img, g_depthImage);            
                
                cv::Mat mat8;
                cv::normalize(g_depthImage, mat8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                depth_img = string_format("%s%05d-gray.jpg", ret_img_dir.c_str(), g_det_cnt);
                cv::imwrite(depth_img, mat8);
            }
            
            // json 里也保存一下文件
            std::string src_img_name = string_format("%05d.jpg", g_det_cnt);
            j["filename"] = src_img_name;
        }


    }

    if ("tray_zero"==g_det_type)
    {
        ret = ai_det_tray(j,ret_json,json_len);
    }
    else if ("top_face" == g_det_type)
    {
        j["ret-msg"] = "error: top_face detect failed";
        if (0 != top_face_init(g_appPath))
        {
            j["ret-msg"] = "top_face init failed: " + top_face_init_error();
            j["ret-code"] = -1;
            make_ret_json(j, ret_json, json_len);
            return -1;
        }

        json request = json::object();
        if (g_rect.size() == 8)
            request["rect"] = g_rect;
        if (g_reload_background)
            request["reload_background"] = true;

        json det_json;
        ret = top_face_detect(image, request, det_json);
        copy_top_face_diag(det_json, j);
        if (0 != ret)
        {
            if (det_json.contains("ret-msg"))
                j["ret-msg"] = det_json["ret-msg"];
            j["ret-code"] = -1;
            make_ret_json(j, ret_json, json_len);
            return -1;
        }

        ret = det_to_pts_top_face(image, det_json, j);
        if (0 != ret)
        {
            make_ret_json(j, ret_json, json_len);
            return ret;
        }
        j["ret-code"] = ret;
        make_ret_json(j, ret_json, json_len);
    }
    else
    {
        //=====================================
        // 开始检测
        j["ret-msg"] = "error: det catch error";
        std::vector< std::vector<Detection_pan>> detections;

        if (g_rect.size()==8 &&
            (startWith_str(g_det_type,"feilun")||
                        "feilun" == g_model_name )
            )
        {

            // 在检测feilun时，需要按区域检测时，请将该区域图像抠出来检测
            // 当"feilun"==g_model_name 或 g_det_type 是"feilun" 开头的情况下，如果8==g_rect.size()，
            // 检测结束后的绘制和坐标处理，还在原图（image）上进行。
            int offset_x = std::min({g_rect[0], g_rect[2], g_rect[4], g_rect[6]});
            int offset_y = std::min({g_rect[1], g_rect[3], g_rect[5], g_rect[7]});
            int x_max = std::max({g_rect[0], g_rect[2], g_rect[4], g_rect[6]});
            int y_max = std::max({g_rect[1], g_rect[3], g_rect[5], g_rect[7]});

            // 裁剪区域限制在图像范围内
            offset_x = std::max(offset_x, 0);
            offset_y = std::max(offset_y, 0);
            x_max = std::min(x_max, image.cols);
            y_max = std::min(y_max, image.rows);

            cv::Rect crop_rect(offset_x, offset_y, x_max - offset_x, y_max - offset_y);
            if (crop_rect.width <= 0 || crop_rect.height <= 0)
            {
                // 围栏区域完全在图像外，当作没有检测结果
                j["ret-msg"] = "error: feilun rect out of image";
                ret = -1;
            }
            else
            {
                // 用裁剪出来的局部图做检测，原图 image 保持不动
                cv::Mat det_img = image(crop_rect).clone();

                ret = det_image(det_img, detections, j);

                // 将检测结果的坐标加上偏移量，换算回原图坐标
                for (auto& det_list : detections)
                {
                    for (auto& det : det_list)
                    {
                        det.bbox.x += offset_x;
                        det.bbox.y += offset_y;
                        for (auto& pt : det.pts)
                        {
                            pt.x += offset_x;
                            pt.y += offset_y;
                        }
                    }
                }
            }

        }

        else
        {
            ret = det_image(image, detections, j);
        }
       
        
        //auto start = std::chrono::high_resolution_clock::now();
        //int loop = 100;

        //for (size_t i = 0; i < loop; i++)
        //{
        //    std::vector< std::vector<Detection_pan>> test;
        //    json j_test;
        //    det_image(image, test, j_test);
        //}

        //auto end = std::chrono::high_resolution_clock::now();

        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "det_image " << loop << "  times,  completed in " << elapsed.count() / loop << " seconds." << std::endl;
        
        
        if (0 != ret)
        {
            make_ret_json(j, ret_json, json_len);
            return ret;
        }

        if ("xialiao" == g_model_name || "feilun"==g_model_name || "screen"==g_model_name)
        {
            // 处理下料检测结果
            //=====================================
            // 分析检测结果
            j["ret-msg"] = "error: det no result";
            if (detections.size() > 0)
            {
                ret = handle_ret_xialiao(
                    image, detections, j, tray_edge_source);
                if (0 != ret)
                {
                    make_ret_json(j, ret_json, json_len);
                    return ret;
                }
            }
            else
            {
                ret = -1;
                make_ret_json(j, ret_json, json_len);
                return ret;
            }

            j["ret-code"] = ret;

            make_ret_json(j, ret_json, json_len);

        }

    }


    //=====================================
    // 保存检测结果
    if (!g_clean_mode && 0 == ret && g_save_ret > 0)
    {
        // save file
        {
            std::string ret_img_dir = g_appPath + "ret_img/";
            if (!fs::exists(ret_img_dir)) fs::create_directory(ret_img_dir);
            std::string ret_img = string_format("%s%05d.jpg", ret_img_dir.c_str(), g_det_cnt);
            cv::imwrite(ret_img, image);

            if (g_save_ret > 2)
            {
                ofstream ofile(string_format("%s%05d.json", ret_img_dir.c_str(), g_det_cnt));
                ofile /*<< std::setw(2) */ << *ret_json << std::endl;
                ofile.close();
            }

            j["ret-img"] = ret_img;
            make_ret_json(j, ret_json, json_len);


            if (g_save_csv > 0)
            {
                std::string csv_file = string_format("%sret.csv", ret_img_dir.c_str());
                save_csv(csv_file, j);
            }
        }
    }

    return ret;
}


std::vector<int> findCloseValuePairs(const std::vector<Detection_pan>& vec, float tolerance = 0.12f) {
    std::vector<std::pair<int, int>> result;
    int n = vec.size();
    std::set<int> selected_indices;
    std::vector<int> list_idx;
    std::vector<std::pair<int, int>> same_pair;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::fabs(vec[i].angle - vec[j].angle) <= tolerance) {
                same_pair.push_back({ i, j });

                if (selected_indices.find(i) == selected_indices.end()) {
                    selected_indices.insert(i);
                    list_idx.push_back(i);
                }

                if (selected_indices.find(j) == selected_indices.end()) {
                    selected_indices.insert(j);
                    list_idx.push_back(j);
                }
            }
        }
    }

    return list_idx;
}

void fillOutsideQuadrilateralFast(cv::Mat& image, 
    const std::vector<int>& rect, const float expandRatio = 1.1f,
    const cv::Scalar& grayColor = cv::Scalar(128, 128, 128))
{
    if (image.empty() || rect.size() != 8) {
        return;
    }
    
    // 将rect转换为cv::Point数组
    std::vector<cv::Point> vertices(4);
    for (int i = 0; i < 4; ++i) {
        vertices[i] = cv::Point(rect[i*2], rect[i*2+1]);
    }
    
    // 计算原始四边形的中心点
    cv::Point center(0, 0);
    for (const auto& pt : vertices) {
        center.x += pt.x;
        center.y += pt.y;
    }
    center.x /= 4;
    center.y /= 4;
    
    // 扩展后的顶点
    std::vector<cv::Point> expandedRect(4);
    for (int i = 0; i < 4; ++i) {
        // 计算相对于中心的偏移
        int dx = vertices[i].x - center.x;
        int dy = vertices[i].y - center.y;
        
        // 扩展偏移量
        int newDx = static_cast<int>(dx * expandRatio);
        int newDy = static_cast<int>(dy * expandRatio);
        
        // 计算新顶点坐标
        int newX = center.x + newDx;
        int newY = center.y + newDy;
        
        // 边界裁剪，不超出图像范围
        newX = std::max(0, std::min(newX, image.cols - 1));
        newY = std::max(0, std::min(newY, image.rows - 1));
        
        expandedRect[i] = cv::Point(newX, newY);
    }
    
    // 创建掩码
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(expandedRect);
    cv::drawContours(mask, contours, 0, cv::Scalar(255), -1);
    
    // 创建灰色图像
    cv::Mat grayImage(image.size(), image.type(), grayColor);
    
    // 使用掩码进行复制：扩展后的四边形区域保留原图，外部使用灰色
    cv::Mat result;
    grayImage.copyTo(result);
    image.copyTo(result, mask);
    
    // 将结果复制回原图
    result.copyTo(image);
}

int __cdecl
AI_det_xialiao(char** img, int* img_w, int* img_h, const char* input_json, char** ret_json, int* json_len)
{
    json j;
    j["ret-code"] = -1;
    int ret = -1;

    g_rect.clear();
    g_confirm = 0;
    g_ref_depth = -1;
    g_top_face_workpiece_height_mm = std::numeric_limits<double>::quiet_NaN();
    g_exposure_list.clear();
    g_exposure_time_det = -1;
    g_cache_slot = 0;
    g_top_capture = -1;
    g_lunzi_only = false;
    g_lunzi_radius = 0;
    g_cur_feilun_id.clear();
    g_clean_mode = false;
    g_reload_background = false;
    try {
        g_top_face_workpiece_height_mm=top_face_request_height(json::parse(input_json));
    } catch(const std::exception& e) {
        j["ret-msg"]=e.what();make_ret_json(j,ret_json,json_len);return -1;
    }

    //=====================================
    // 看看这回检测啥
    try
    {
        json j_input = json::parse(input_json);

        if (j_input.contains("xialiao_type")) {
            j_input["xialiao_type"].get_to(g_det_type);
        }
        if (j_input.contains("test_mode")) {
            j_input["test_mode"].get_to(g_test_mode);
        }

        if (j_input.contains("rect")) {
            j_input["rect"].get_to(g_rect);
        }        

        if (j_input.contains("confirm")) {
            j_input["confirm"].get_to(g_confirm);
        }    

        if (j_input.contains("ref_depth")) {
            j_input["ref_depth"].get_to(g_ref_depth);
        }

        if (j_input.contains("exposure_list")) {
            j_input["exposure_list"].get_to(g_exposure_list);
        }

        if (j_input.contains("exposure_time")) {
            j_input["exposure_time"].get_to(g_exposure_time_det);
        }

        if (j_input.contains("cache_slot")) {
            j_input["cache_slot"].get_to(g_cache_slot);
        }

        if (j_input.contains("top_capture")) {
            j_input["top_capture"].get_to(g_top_capture);
        }

        if (j_input.contains("lunzi_only")) {
            j_input["lunzi_only"].get_to(g_lunzi_only);
        }

        if (j_input.contains("lunzi_radius")) {
            j_input["lunzi_radius"].get_to(g_lunzi_radius);
        }

        if (j_input.contains("feilun_id")) {
            auto& val = j_input["feilun_id"];
            if (val.is_string()) {
                g_cur_feilun_id = val.get<std::string>();
            } else if (val.is_number_integer()) {
                g_cur_feilun_id = std::to_string(val.get<int>());
            }
        }

        if (j_input.contains("clean_mode")) {
            j_input["clean_mode"].get_to(g_clean_mode);
        }

        if (j_input.contains("reload_background")) {
            j_input["reload_background"].get_to(g_reload_background);
        }

    }
    catch (...) {}

    if (1==g_confirm)
    {
        float centor_dist = -1;
        if (g_rect.size()!=8)
        {
            g_rect = {0,0,g_cam_param.width-1,0,
                g_cam_param.width-1,g_cam_param.height-1,0,g_cam_param.height-1};
        }
        
        get_centor_depth_by_area_and_cap_depth_4times(g_rect,centor_dist);

        j["ret-msg"] = "error: get_centor_depth_by_area failed"; 

        int ret = -1;
        if (centor_dist>0)
        {
            j["ret-msg"] = "get_centor_depth_by_area successful"; 
            j["ref_depth"] = centor_dist; 
        }
        
        j["ret-code"] = ret;
        make_ret_json(j, ret_json, json_len);
        return ret;
    }
    



    //=====================================
    // 读取图像
    j["ret-msg"] = "error: read image from camera"; 

    // ret = g_camera->captureOneFrameWithDepth(g_bgrImage, g_depthFrame);
    if (1==g_test_mode)
    {
        // 测试图片必须是一次请求一帧。若本次路径无效或缺少同名深度图，
        // 不能沿用上一张测试图片的缓存，否则会以错误的深度返回假成功。
        g_bgrImage.release();
        g_depthImage.release();

        std::string color_image_name,depth_image_name;
        try
        {
            json j_input = json::parse(input_json);

            if (j_input.contains("color_image")) {
                j_input["color_image"].get_to(color_image_name);
            }
            if (j_input.contains("depth_image")) {
                j_input["depth_image"].get_to(depth_image_name);
            }
        }
        catch (...) {}

        if(""!=color_image_name && fs::exists(color_image_name))
        {
            g_bgrImage = cv::imread(color_image_name);
            g_bgrImage_undistorted = false;
        }

#ifdef useMVS

#else
        if(""!=depth_image_name && fs::exists(depth_image_name))
        {
            cv::Mat depth_raw = cv::imread(depth_image_name, cv::IMREAD_ANYDEPTH);
            if (depth_raw.empty()) {
                std::cerr << "[detectCenterRect] 无法读取深度图: " << depth_image_name << std::endl;
                return -1;
            }
            cv::Mat depth(depth_raw.rows, depth_raw.cols, CV_16UC1, depth_raw.data);

            if (depth.rows != g_bgrImage.rows || depth.cols != g_bgrImage.cols) {
                cv::resize(depth, depth, cv::Size(g_bgrImage.cols, g_bgrImage.rows), 0, 0, cv::INTER_NEAREST);
            } 

            g_depthImage = depth.clone();
        }
#endif // useMVS       


    }
    else
    {
        // top_face 多曝光会覆盖彩色；深度仍用这一帧。关掉自适应补拍，避免和融合背景错位。
        int fuse_first = g_fuse_image;
        if ("top_face" == g_det_type && !g_exposure_list.empty())
            fuse_first = 0;
        ret = get_good_camera_image(g_bgrImage, g_depthImage, fuse_first);

        if (g_bgrImage.empty() || 
            ((g_cfg_camera.enableDepth)&&g_depthImage.empty()) )
        {
            make_ret_json(j, ret_json, json_len);
            return ret;
        }

    }

    // xike 模型(screen) screen_centor 检测：传入了 exposure_list 则拍摄多曝光融合图
    // top_face 也走这条通路：背景与检测必须同一 exposure_list + Mertens，禁用按格拼贴。
    // test_mode 读的是文件，不能融合，保持单帧。
    if (("screen" == g_model_name ||
         ("top_face" == g_det_type && 1 != g_test_mode))
        && !g_exposure_list.empty()
        && g_camera && g_camera->isOpened())
    {
        std::vector<cv::Mat> list_img = g_camera->captureMultiExposureImages(g_exposure_list);
        if (list_img.empty())
        {
            j["ret-msg"] = "multi-exposure capture failed";
            make_ret_json(j, ret_json, json_len);
            return -1;
        }

            MultiExposureFusion fusion;

            // mosaic mertens；top_face 强制 Mertens（与背景同一融合方式）
            if("mertens"==g_fuse_mode || "top_face"==g_det_type)
            {
                g_bgrImage = fusion.method_mertens(list_img);
            }
            else
            {
                fusion.grid_rows = 8;
                fusion.grid_cols = 8;
                fusion.cache_slot = g_cache_slot;
                g_bgrImage = fusion.fuseImagesWithBlending(list_img, 20);
            }

        if (g_bgrImage.empty())
        {
            j["ret-msg"] = "multi-exposure fusion failed";
            make_ret_json(j, ret_json, json_len);
            return -1;
        }
    }

    // 检测/显示通路统一去畸变：test_mode 读图、实时取图、screen/top_face 多曝光融合
    // 的结果此时都已写入 g_bgrImage。必须放在下面移交裸指针之前，
    // 也必须放在 tray_edge_source、围栏遮罩之前，让它们拿到的是矫正图。
    g_bgrImage_undistorted = undistort_frame(g_bgrImage, &g_depthImage);

    // 行对边缘裁决必须使用围栏灰色遮罩前、且属于本次请求的原图。
    // 该局部图仅在料盘粗定位模式创建，不跨请求缓存。
    cv::Mat tray_edge_source;
    if (needsTrayRowEdgeSelection(g_det_type) &&
        g_top_capture != 1 && !g_bgrImage.empty())
    {
        tray_edge_source = g_bgrImage.clone();
    }

    // 让围栏生效
    if (g_rect.size()==8)
    {

        // 在检测feilun时，需要按区域检测时，暂时保留原图
        if (startWith_str(g_det_type,"feilun")||
            "feilun" == g_model_name ||
            "top_face" == g_det_type )
        {
        }
        else
        {
            float expandRatio = 1.1f;
            cv::Scalar grayColor = cv::Scalar(128, 128, 128);
            
            if ( startWith_str(g_det_type,"lunzi") )
            {
                expandRatio = 1.0f;
            } 
            
            fillOutsideQuadrilateralFast(g_bgrImage,g_rect,expandRatio,grayColor);
        }


    }
    

    *img = (char*)g_bgrImage.data;
    *img_w = g_bgrImage.cols;
    *img_h = g_bgrImage.rows;

    if (g_cfg_camera.enableDepth)
    {
        if (!g_depthImage.empty()) {
            g_depth_info.from_depthMat(g_depthImage);
        }
        else if ("tray_zero" != g_det_type) {
            // tray_zero(pallet) 可以用纯视觉边缘检测，深度图非必须
            ret = -1;
            j["ret-msg"] = "Failed to get depth frame (is depth stream enabled?)";
            make_ret_json(j, ret_json, json_len);
            return ret;
        }
#ifdef MY_LOG
        else {
            std::cout << "[AI_det] tray_zero mode: no depth image, using visual-only detection" << std::endl;
        }
#endif

    }
    
    // 处理单张
    std::vector<Detection_pan> detections_pan;
    ret = ai_det_one_image(
        detections_pan, j, ret_json, json_len,
        tray_edge_source.empty() ? nullptr : &tray_edge_source);

    return ret;
}


/*
* AI_det
* 检测工件
* @img:
如果img_w>0 && img_h==0 ，图像jpeg数据，数据长度为img_w
如果img_w 和 img_h 都>1时，图像bgr数据
如果img_w 和 img_h 都为0时，图像文件名称，


* @input_json: 描述检测要求的json
* @ret_json: 检测结果json，该内存不用释放，最后核心退出时，统一释放
* @json_len: 返回的json的长度

* @return: 0: 成功，其他：失败
*/
ALVA_DLL_API int __cdecl
AI_det(char** img, int* img_w, int* img_h, const char* input_json,char** ret_json,int* json_len, AI_det_callback callback)
{
    json j;
    j["ret-code"] = -1;
    g_test_mode = -1;
    g_ref_depth = -1;
    g_top_face_workpiece_height_mm = std::numeric_limits<double>::quiet_NaN();
    int exposure_time = -1;
    vector<int> exposure_list;
    int cache_slot = 0;
    int ret = -1;
    bool save_top_face_background = false;
    std::string preview_xialiao_type;
    if (nullptr==G_DET)
    {
        j["ret-msg"] = "error: det is not init";
        make_ret_json(j, ret_json, json_len);
        return ret;
    }

    try {
        g_top_face_workpiece_height_mm=top_face_request_height(json::parse(input_json));
    } catch(const std::exception& e) {
        j["ret-msg"]=e.what();make_ret_json(j,ret_json,json_len);return -1;
    }

    //=====================================
    // 看看这回检测啥
    std::string calibration_type = "";
    try
    {
        json j_input = json::parse(input_json);

        if (j_input.contains("calibration_type")) {
            j_input["calibration_type"].get_to(calibration_type);
        }

        if (j_input.contains("ref_depth")) {
            j_input["ref_depth"].get_to(g_ref_depth);
        }           
        if (j_input.contains("exposure_time")) {
            j_input["exposure_time"].get_to(exposure_time);
        }

        if (j_input.contains("exposure_list")) {
            j_input["exposure_list"].get_to(exposure_list);
            g_exposure_list = exposure_list;
        }

        if (j_input.contains("save_top_face_background")) {
            j_input["save_top_face_background"].get_to(save_top_face_background);
        }
        if (j_input.contains("reload_background")) {
            bool rb = false;
            j_input["reload_background"].get_to(rb);
            if (rb)
                save_top_face_background = true;
        }
        if (j_input.contains("xialiao_type")) {
            j_input["xialiao_type"].get_to(preview_xialiao_type);
        }

        if (j_input.contains("cache_slot")) {
            j_input["cache_slot"].get_to(cache_slot);
            g_cache_slot = cache_slot;
        }

        if (j_input.contains("exposure_time")) {
            j_input["exposure_time"].get_to(exposure_time);
            g_exposure_time_det = exposure_time;
        }

        if (j_input.contains("test_mode")) {
            j_input["test_mode"].get_to(g_test_mode);

            if (1==g_test_mode)
            {
                std::string color_image_name,depth_image_name;
                {
                    if (j_input.contains("color_image")) {
                        j_input["color_image"].get_to(color_image_name);
                    }
                    if (j_input.contains("depth_image")) {
                        j_input["depth_image"].get_to(depth_image_name);
                    }
                }


                if(""!=color_image_name && fs::exists(color_image_name))
                {
                    g_bgrImage = cv::imread(color_image_name);
                    // 这是原图，后面 board_distance 之类的分支会用到 g_bgrImage
                    g_bgrImage_undistorted = false;
                }
            }

        }        
    }
    catch (...) {}

    //=====================================
    // 像素坐标 → 物理坐标（毫米）
    if ("pixel_to_physical"==calibration_type)
    {
        if (g_cam_param.fx <= 0)
        {
            j["ret-msg"] = "camera params not loaded, check hand_eye_result.yaml";
            make_ret_json(j, ret_json, json_len);
            return ret;
        }

        vector<float> pts_in;
        float z = g_ref_depth;
        try
        {
            json j_input2 = json::parse(input_json);
            if (j_input2.contains("ref_depth"))
                j_input2["ref_depth"].get_to(z);
            if (j_input2.contains("pts"))
                j_input2["pts"].get_to(pts_in);
        }
        catch (...) {}

        if (z < 10)
        {
            j["ret-msg"] = "ref_depth not set or invalid, please set ref_depth in input";
            make_ret_json(j, ret_json, json_len);
            return ret;
        }

        if (pts_in.empty() || pts_in.size() % 2 != 0)
        {
            j["ret-msg"] = "pts is empty or not paired (need even number of values)";
            make_ret_json(j, ret_json, json_len);
            return ret;
        }

        vector<float> pts_out;
        for (size_t i = 0; i + 1 < pts_in.size(); i += 2)
        {
            float px = pts_in[i];
            float py = pts_in[i + 1];
            
            float wx = 0;
            float wy = 0;

            g_depth_info.get_world_xy_by_z(px,py,wx,wy,z);

            pts_out.push_back(wx);
            pts_out.push_back(wy);
        }

        j["pts"] = pts_out;
        j["ret-msg"] = "ok";
        j["ret-code"] = 0;
        make_ret_json(j, ret_json, json_len);
        return 0;
    }

    // 预览：打开相机拍一张图，存盘后通过 callback 返回给调用者
    else if ("preview"==calibration_type)
    {
        *img = nullptr;
        *img_w = 0;
        *img_h = 0;

        j["ret-msg"] = "capture failed";
        if (g_camera && g_camera->isOpened())
        {
            // 如果传入了 exposure_list，用多个曝光值拍照并融合
            if (!exposure_list.empty())
            {
                std::vector<cv::Mat> list_img = g_camera->captureMultiExposureImages(exposure_list);
                if (list_img.empty())
                {
                    j["ret-msg"] = "multi-exposure capture failed";
                    make_ret_json(j, ret_json, json_len);
                    return -1;
                }
                    MultiExposureFusion fusion;
                    // screen 等仍可按 cfg 网格拼贴；top_face 背景必须 Mertens
                    bool use_mertens = ("mertens"==g_fuse_mode) ||
                        "top_face"==preview_xialiao_type || save_top_face_background;
                    if (use_mertens)
                    {
                        g_bgrImage = fusion.method_mertens(list_img);
                    }
                    else
                    {
                        fusion.grid_rows = 8;
                        fusion.grid_cols = 8;
                        fusion.cache_slot = g_cache_slot;
                        g_bgrImage = fusion.fuseImagesWithBlending(list_img, 20);
                    }
                if (g_bgrImage.empty())
                {
                    j["ret-msg"] = "multi-exposure fusion failed";
                    make_ret_json(j, ret_json, json_len);
                    return -1;
                }
            }
            else
            {
                if (exposure_time>100)
                {
                    g_camera->setAutoExposure(false);
                    g_camera->setColorExposureTime(exposure_time);
                }

                g_bgrImage = g_camera->captureOneFrame(2000);
            }

            if (!g_bgrImage.empty())
            {
                // 预览也要给矫正后的图，且在移交裸指针、存盘之前
                g_bgrImage_undistorted = undistort_frame(g_bgrImage);

                *img = (char*)g_bgrImage.data;
                *img_w = g_bgrImage.cols;
                *img_h = g_bgrImage.rows;

                std::string img_path = g_appPath + "preview.jpg";
                if (!exposure_list.empty())
                {
                    img_path = g_appPath + "preview_fused.jpg";
                }
                else if (exposure_time>100)
                {
                    img_path = g_appPath + "preview_" + std::to_string(exposure_time) + ".jpg";
                }
                cv::imwrite(img_path, g_bgrImage);

                if (!exposure_list.empty() &&
                    (save_top_face_background || "top_face"==preview_xialiao_type))
                {
                    std::string bg_path = g_appPath + "config/top_face/background.bmp";
                    fs::create_directories(fs::path(bg_path).parent_path());
                    if (cv::imwrite(bg_path, g_bgrImage))
                    {
                        j["background_img"] = bg_path;
                    }
                    else
                    {
                        j["ret-msg"] = "failed to write top_face background";
                        j["ret-code"] = -1;
                        make_ret_json(j, ret_json, json_len);
                        return -1;
                    }
                }

                j["preview_img"] = img_path;
                j["image_size"] = std::to_string(g_bgrImage.cols) + "x" + std::to_string(g_bgrImage.rows);
                j["ret-msg"] = "ok";
                j["ret-code"] = 0;
                ret = 0;
            }
            else
            {
                j["ret-msg"] = "captureOneFrame returned empty";
            }
        }
        else
        {
            j["ret-msg"] = "camera not opened";
        }
        make_ret_json(j, ret_json, json_len);
        return ret;
    }
    // 测量距离
    else if ("board_distance"==calibration_type)
    {
        json j_diag;
        double h = -1;
        float dist = estimate_board_distance(h, &j_diag);
        j["distance_mm"] = h;
        if (dist > 0)
        {
            j["ret-code"] = 0;
            j["ret-msg"] = "ok";
        }
        else
        {
            j["ret-code"] = -1;
            j["ret-msg"] = "board distance estimation failed";
        }
        if (!j_diag.empty()) j["diag"] = j_diag;
        ret = j["ret-code"].get<int>();
        make_ret_json(j, ret_json, json_len);
        return ret;
    }

    //=====================================
    // 看看这回检测啥
    if ("xialiao"==g_model_name || "feilun"==g_model_name
        || "screen"==g_model_name || "xike"==g_model_name)
    {
        ret = AI_det_xialiao(img, img_w, img_h, input_json, ret_json, json_len);
    }
    //else
    //{
    //    ret = AI_det_shangliao(img, img_w, img_h, input_json, ret_json, json_len);
    //}


    return ret;
}



ALVA_DLL_API int __cdecl 
AI_det_close()
{
    if (nullptr!= g_camera)
    {
        g_camera->close();
        g_camera = nullptr;
    }

    G_DET = nullptr;
    top_face_close();
    util::Log::CloseLog();

    // 序号在每次递增时已经落盘，这里再写一次兜底。
    // write_det_cnt_file() 在 g_appPath 为空时不写，避免没 init 过的进程
    // 把 0 写到当前工作目录的 ret_img/cnt.txt 上。
    write_det_cnt_file();

    return 0;    
}
