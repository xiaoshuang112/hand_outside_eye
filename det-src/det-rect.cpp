#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <Windows.h>

#else
#include<unistd.h>
#include <dlfcn.h>
#define __cdecl
#endif

#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread> // for std::thread
#include <chrono>
#include <cstring>
#include "json.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include "my_base.h"


namespace fs = std::filesystem;
using json = nlohmann::json;


/**
 * @brief 在深度图上多区域采样，选取最近的前景物体深度
 *
 * 在图像中心附近多个位置采样，取最小有效中位数深度。
 * 托盘不完全在几何中心也能找到托盘而非背景墙。
 */
static double estimateTargetDepth(const cv::Mat& depth, double center_ratio)
{
    int h = depth.rows, w = depth.cols;
    int roi_w = std::max(30, static_cast<int>(w * center_ratio));
    int roi_h = std::max(30, static_cast<int>(h * center_ratio));
    int cx = w / 2, cy = h / 2;
    double best_median = -1;

    for (int dy : {0, -roi_h/2, roi_h/2, -roi_h/4, roi_h/4,
                   -roi_h*3/4, roi_h*3/4}) {
        for (int dx : {0, -roi_w/4, roi_w/4}) {
            int cy_try = std::clamp(cy + dy, roi_h / 2, h - 1 - roi_h / 2);
            int cx_try = std::clamp(cx + dx, roi_w / 2, w - 1 - roi_w / 2);
            cv::Rect roi(
                std::max(0, cx_try - roi_w / 2),
                std::max(0, cy_try - roi_h / 2),
                std::min(roi_w, w - std::max(0, cx_try - roi_w / 2)),
                std::min(roi_h, h - std::max(0, cy_try - roi_h / 2)));
            cv::Mat region = depth(roi);
            std::vector<std::uint16_t> vals;
            vals.reserve(region.total());
            for (int r = 0; r < region.rows; r++) {
                const std::uint16_t* row = region.ptr<std::uint16_t>(r);
                for (int c = 0; c < region.cols; c++)
                    if (row[c] > 0) vals.push_back(row[c]);
            }
            if (vals.size() < (int)(region.total() * 0.15)) continue;
            size_t n = vals.size();
            std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
            double median = vals[n / 2];
            if (best_median < 0 || median < best_median)
                best_median = median;
        }
    }
    return best_median;
}

/**
 * @brief 用百分比容差生成二值 mask
 *
 * tol = max(50mm, center_depth * ratio)，区间 [center - tol, center + tol]。
 * 百分比容差自动适应不同距离：近处物体容差小，远处物体容差大（透视深度变化大）。
 */
static cv::Mat createDepthMask(const cv::Mat& depth, double center_depth, double ratio)
{
    cv::Mat mask(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));
    double tol = std::max(50.0, center_depth * ratio);
    double lo = center_depth - tol;
    double hi = center_depth + tol;

    for (int r = 0; r < depth.rows; r++) {
        const std::uint16_t* src = depth.ptr<std::uint16_t>(r);
        std::uint8_t* dst = mask.ptr<std::uint8_t>(r);
        for (int c = 0; c < depth.cols; c++) {
            if (src[c] > 0 && src[c] >= lo && src[c] <= hi)
                dst[c] = 255;
        }
    }
    return mask;
}

/**
 * @brief 将点集拟合为四边形，返回 4 个角点
 *
 * 依次尝试：凸包逼近 → 轮廓逼近 → 多 epsilon 搜索 → minAreaRect 兜底
 */
static std::vector<cv::Point> fitQuad(const std::vector<cv::Point>& contour, double approx_eps)
{
    std::vector<cv::Point> approx;

    // 凸包 + 逼近
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double peri = cv::arcLength(hull, true);
    cv::approxPolyDP(hull, approx, approx_eps * peri, true);

    if (approx.size() != 4) {
        double peri2 = cv::arcLength(contour, true);
        approx.clear();
        cv::approxPolyDP(contour, approx, approx_eps * peri2, true);
    }

    if (approx.size() != 4) {
        for (double eps : {0.01, 0.03, 0.05, 0.08, 0.10, 0.15}) {
            double p = cv::arcLength(contour, true);
            approx.clear();
            cv::approxPolyDP(contour, approx, eps * p, true);
            if (approx.size() == 4) break;
        }
    }

    if (approx.size() != 4) {
        cv::RotatedRect r = cv::minAreaRect(contour);
        cv::Point2f pts[4];
        r.points(pts);
        approx.clear();
        for (int i = 0; i < 4; i++)
            approx.push_back(cv::Point(static_cast<int>(pts[i].x), static_cast<int>(pts[i].y)));
    }

    return approx;
}

/**
 * @brief 将四个无序角点排序为 TL, TR, BR, BL
 */
static std::vector<cv::Point> orderCorners(const std::vector<cv::Point>& pts)
{
    std::vector<cv::Point> sorted = pts;
    std::sort(sorted.begin(), sorted.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

    cv::Point tl = (sorted[0].x <= sorted[1].x) ? sorted[0] : sorted[1];
    cv::Point tr = (sorted[0].x >  sorted[1].x) ? sorted[0] : sorted[1];
    cv::Point bl = (sorted[2].x <= sorted[3].x) ? sorted[2] : sorted[3];
    cv::Point br = (sorted[2].x >  sorted[3].x) ? sorted[2] : sorted[3];

    return {tl, tr, br, bl};
}

/**
 * @brief 利用深度梯度在角点邻域精化位置
 */
static void refineCornersWithDepthGradient(const cv::Mat& depth,
                                           std::vector<cv::Point>& corners,
                                           int search_win = 15)
{
    cv::Mat depth_f;
    depth.convertTo(depth_f, CV_32FC1);

    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(depth_f, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(depth_f, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    int half = search_win / 2;

    for (auto& pt : corners) {
        int x0 = std::max(half, std::min(pt.x, depth.cols - 1 - half));
        int y0 = std::max(half, std::min(pt.y, depth.rows - 1 - half));

        cv::Rect roi(x0 - half, y0 - half, search_win, search_win);
        cv::Mat patch = grad_mag(roi);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(patch, &minVal, &maxVal, &minLoc, &maxLoc);

        if (maxVal > 5.0) {
            pt.x = x0 - half + maxLoc.x;
            pt.y = y0 - half + maxLoc.y;
        }
    }
}

/**
 * @brief 在矩形内部采样 3D 点，拟合平面，用已知物理尺寸精化角点
 *
 * 1. 在矩形内部区域采样有效深度点，转到 3D 坐标
 * 2. RANSAC 拟合平面
 * 3. 将四条边的像素坐标投影到平面上，计算 3D 边长度
 * 4. 沿法线方向微调边位置，使 3D 宽度→1195mm、高度→600mm
 * 5. 四条边相交得到精化后的角点
 *
 * @param depth         深度图 (CV_16UC1, mm)
 * @param cameraMatrix  3x3 相机内参矩阵
 * @param corners       输入/输出：4 个角点 TL, TR, BR, BL
 * @param physWidth     物理宽度 (mm)
 * @param physHeight    物理高度 (mm)
 */
static void refineWithKnownSize(const cv::Mat& depth,
                                const cv::Mat& cameraMatrix,
                                std::vector<cv::Point>& corners,
                                double physWidth,
                                double physHeight)
{
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    // ---- 转 3D 坐标 ----
    auto to3D = [&](cv::Point2d p, double Z) -> cv::Point3d {
        return cv::Point3d((p.x - cx) * Z / fx, (p.y - cy) * Z / fy, Z);
    };

    // ---- 在矩形内部采样深度点，拟合平面 ----
    // 将矩形收缩 20% 以避开边缘混合像素
    cv::Point2f c[4];
    for (int i = 0; i < 4; i++) c[i] = cv::Point2f(corners[i]);
    cv::Point2f center = (c[0] + c[1] + c[2] + c[3]) * 0.25f;
    cv::Point2f inner[4];
    float s = 0.20f; // 收缩比例
    for (int i = 0; i < 4; i++)
        inner[i] = center + (c[i] - center) * (1.0f - s);

    // 收集内矩形中的有效 3D 点
    std::vector<cv::Point3d> pts3D;
    int min_x = std::max(0, (int)std::min({inner[0].x, inner[1].x, inner[2].x, inner[3].x}));
    int max_x = std::min(depth.cols - 1, (int)std::max({inner[0].x, inner[1].x, inner[2].x, inner[3].x}));
    int min_y = std::max(0, (int)std::min({inner[0].y, inner[1].y, inner[2].y, inner[3].y}));
    int max_y = std::min(depth.rows - 1, (int)std::max({inner[0].y, inner[1].y, inner[2].y, inner[3].y}));

    // 使用多边形掩码精确采样
    cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    std::vector<cv::Point> poly;
    for (int i = 0; i < 4; i++)
        poly.push_back(cv::Point((int)inner[i].x, (int)inner[i].y));
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));

    for (int r = min_y; r <= max_y; r++) {
        const uint16_t* d_row = depth.ptr<uint16_t>(r);
        const uint8_t* m_row = mask.ptr<uint8_t>(r);
        for (int col = min_x; col <= max_x; col++) {
            if (m_row[col] && d_row[col] > 0)
                pts3D.push_back(to3D(cv::Point2d(col, r), d_row[col]));
        }
    }

    if (pts3D.size() < 50) return; // 点太少，放弃精化

    // ---- RANSAC 平面拟合 ----
    cv::Vec3d planeNormal;
    double planeD;
    {
        cv::Mat ptsMat((int)pts3D.size(), 3, CV_64F);
        for (size_t i = 0; i < pts3D.size(); i++) {
            ptsMat.at<double>((int)i, 0) = pts3D[i].x;
            ptsMat.at<double>((int)i, 1) = pts3D[i].y;
            ptsMat.at<double>((int)i, 2) = pts3D[i].z;
        }

        // 简单 PCA 平面拟合（比 RANSAC 更快，点已经是内点）
        cv::Mat mean, centered;
        cv::reduce(ptsMat, mean, 0, cv::REDUCE_AVG);
        cv::repeat(mean, ptsMat.rows, 1, centered);
        cv::Mat diff = ptsMat - centered;

        cv::Mat cov = diff.t() * diff;
        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(cov, eigenvalues, eigenvectors);

        // 最小特征值对应的特征向量 = 平面法线
        planeNormal = cv::Vec3d(
            eigenvectors.at<double>(2, 0),
            eigenvectors.at<double>(2, 1),
            eigenvectors.at<double>(2, 2)
        );
        // 确保法线指向相机（z 分量为正）
        if (planeNormal[2] < 0) planeNormal = -planeNormal;
        planeD = -(planeNormal[0] * mean.at<double>(0) +
                   planeNormal[1] * mean.at<double>(1) +
                   planeNormal[2] * mean.at<double>(2));
    }

    // ---- 根据平面方程，由像素坐标计算 Z 值 ----
    // 射线 (u,v) 与平面 n·P + d = 0 的交点
    // P = ( (u-cx)*Z/fx, (v-cy)*Z/fy, Z )
    // n0*(u-cx)*Z/fx + n1*(v-cy)*Z/fy + n2*Z + d = 0
    // Z = -d / ( n0*(u-cx)/fx + n1*(v-cy)/fy + n2 )
    auto planeDepth = [&](cv::Point2d p) -> double {
        double denom = planeNormal[0] * (p.x - cx) / fx +
                       planeNormal[1] * (p.y - cy) / fy +
                       planeNormal[2];
        if (std::abs(denom) < 1e-10) return 0;
        return -planeD / denom;
    };

    // ---- 计算边的 3D 长度 ----
    // 对每条边采样 N 个点，计算 3D 距离
    auto edge3DLength = [&](cv::Point2f a, cv::Point2f b, int N = 20) -> double {
        double total = 0;
        for (int i = 0; i < N; i++) {
            double t1 = (double)i / N;
            double t2 = (double)(i + 1) / N;
            cv::Point2d p1(a.x + (b.x - a.x) * t1, a.y + (b.y - a.y) * t1);
            cv::Point2d p2(a.x + (b.x - a.x) * t2, a.y + (b.y - a.y) * t2);
            double Z1 = planeDepth(p1);
            double Z2 = planeDepth(p2);
            if (Z1 <= 0 || Z2 <= 0) continue;
            cv::Point3d P1 = to3D(p1, Z1);
            cv::Point3d P2 = to3D(p2, Z2);
            total += cv::norm(P1 - P2);
        }
        return total;
    };

    // ---- 计算两条平行边之间在 3D 中的平均距离 ----
    // 沿一条边采样，向对边做法线方向的 3D 测量
    auto edgePair3DDistance = [&](cv::Point2f a1, cv::Point2f b1,  // 边1
                                   cv::Point2f a2, cv::Point2f b2,  // 边2
                                   int N = 20) -> double {
        double total = 0;
        int valid = 0;
        for (int i = 0; i <= N; i++) {
            double t = (double)i / N;
            cv::Point2d p1(a1.x + (b1.x - a1.x) * t, a1.y + (b1.y - a1.y) * t);
            cv::Point2d p2(a2.x + (b2.x - a2.x) * t, a2.y + (b2.y - a2.y) * t);
            double Z1 = planeDepth(p1);
            double Z2 = planeDepth(p2);
            if (Z1 <= 0 || Z2 <= 0) continue;
            cv::Point3d P1 = to3D(p1, Z1);
            cv::Point3d P2 = to3D(p2, Z2);
            total += cv::norm(P1 - P2);
            valid++;
        }
        return valid > 0 ? total / valid : 0;
    };

    // ---- 迭代精化 ----
    // corners: TL, TR, BR, BL
    cv::Point2f c_out[4];
    for (int i = 0; i < 4; i++) c_out[i] = c[i];

    for (int iter = 0; iter < 10; iter++) {
        // 当前 3D 尺寸
        // 宽度 = 左边(TL→BL) 与 右边(TR→BR) 之间的 3D 距离
        double w_3d = edgePair3DDistance(c_out[0], c_out[3], c_out[1], c_out[2]);
        // 高度 = 上边(TL→TR) 与 下边(BL→BR) 之间的 3D 距离
        double h_3d = edgePair3DDistance(c_out[0], c_out[1], c_out[3], c_out[2]);

        if (w_3d <= 0 || h_3d <= 0) break;

        double w_err = w_3d - physWidth;
        double h_err = h_3d - physHeight;

        if (std::abs(w_err) < 0.3 && std::abs(h_err) < 0.3) break;

        // 估算像素→毫米的缩放因子（在物体中心深度处）
        double avgZ = planeDepth(center);
        if (avgZ <= 0) break;
        double mmPerPx = avgZ / fx;  // fx≈fy

        // 每个边需要移动的像素量（正=向外扩张，负=向内收缩）
        double dw_px = -w_err / (2.0 * mmPerPx); // 每边移一半
        double dh_px = -h_err / (2.0 * mmPerPx);

        // 限制每次迭代的最大移动量，防止发散
        dw_px = std::clamp(dw_px, -20.0, 20.0);
        dh_px = std::clamp(dh_px, -20.0, 20.0);

        // 计算每条边的单位法线（指向矩形外部）
        auto outwardNormal = [&](cv::Point2f a, cv::Point2f b) -> cv::Vec2f {
            cv::Vec2f dir(b.x - a.x, b.y - a.y);
            float len = cv::norm(dir);
            if (len < 1) return {0, 0};
            dir /= len;
            // 候选法线（顺时针90度）
            cv::Vec2f n(-dir[1], dir[0]);
            // 取指向外部的方向
            cv::Point2f mid((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
            cv::Vec2f to_c(center.x - mid.x, center.y - mid.y);
            if (n.dot(to_c) > 0) n = -n;
            return n;
        };

        cv::Vec2f n_top   = outwardNormal(c_out[0], c_out[1]); // TL→TR
        cv::Vec2f n_right = outwardNormal(c_out[1], c_out[2]); // TR→BR
        cv::Vec2f n_bot   = outwardNormal(c_out[2], c_out[3]); // BR→BL
        cv::Vec2f n_left  = outwardNormal(c_out[3], c_out[0]); // BL→TL

        // 统一规则: 向外扩张 → +normal * dw_px; 向内收缩 → -normal * |dw_px|
        // dw_px > 0: 扩张 (w_err<0, 太窄)
        // dw_px < 0: 收缩 (w_err>0, 太宽)

        // 宽度调整：移动左右边
        c_out[0] = cv::Point2f(c_out[0].x + n_left[0] * dw_px, c_out[0].y + n_left[1] * dw_px);
        c_out[3] = cv::Point2f(c_out[3].x + n_left[0] * dw_px, c_out[3].y + n_left[1] * dw_px);
        c_out[1] = cv::Point2f(c_out[1].x + n_right[0] * dw_px, c_out[1].y + n_right[1] * dw_px);
        c_out[2] = cv::Point2f(c_out[2].x + n_right[0] * dw_px, c_out[2].y + n_right[1] * dw_px);

        // 高度调整：移动上下边
        c_out[0] = cv::Point2f(c_out[0].x + n_top[0] * dh_px, c_out[0].y + n_top[1] * dh_px);
        c_out[1] = cv::Point2f(c_out[1].x + n_top[0] * dh_px, c_out[1].y + n_top[1] * dh_px);
        c_out[3] = cv::Point2f(c_out[3].x + n_bot[0] * dh_px, c_out[3].y + n_bot[1] * dh_px);
        c_out[2] = cv::Point2f(c_out[2].x + n_bot[0] * dh_px, c_out[2].y + n_bot[1] * dh_px);
    }

    // 输出精化后的角点
    for (int i = 0; i < 4; i++) {
        corners[i] = cv::Point(
            std::clamp((int)std::round(c_out[i].x), 0, depth.cols - 1),
            std::clamp((int)std::round(c_out[i].y), 0, depth.rows - 1)
        );
    }
}

/**
 * @brief 用自适应颜色阈值过滤深度 mask，去除背景像素
 *
 * 在深度 mask 有效区域采样主色调，自适应建立 HSV 范围，
 * 然后将深度 mask 中不在颜色范围内的像素剔除。
 * 解决托盘与背景深度相近时深度分割泄漏的问题。
 */
static void refineDepthMaskWithColor(const cv::Mat& color,
                                     const cv::Mat& depth,
                                     cv::Mat& mask)
{
    cv::Mat hsv;
    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask_inner;
    cv::erode(mask, mask_inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    cv::Mat& sm = (cv::countNonZero(mask_inner) > 500) ? mask_inner : mask;

    std::vector<cv::Vec3b> samples;
    for (int r = 0; r < mask.rows; r += 3) {
        const uint8_t* mr = sm.ptr<uint8_t>(r);
        const cv::Vec3b* hr = hsv.ptr<cv::Vec3b>(r);
        for (int c = 0; c < mask.cols; c += 3)
            if (mr[c] > 0) samples.push_back(hr[c]);
    }
    if (samples.size() < 100) return;

    std::vector<int> hv, sv, vv;
    for (auto& s : samples) { hv.push_back(s[0]); sv.push_back(s[1]); vv.push_back(s[2]); }
    std::sort(hv.begin(), hv.end());
    std::sort(sv.begin(), sv.end());
    std::sort(vv.begin(), vv.end());

    int h_med = hv[hv.size()/2];
    std::vector<int> ad;
    for (int v : hv) ad.push_back(std::abs(v - h_med));
    std::sort(ad.begin(), ad.end());
    int h_mad = ad[ad.size()/2];

    int h_lo = std::max(0, h_med - std::max(20, 3 * h_mad + 15));
    int h_hi = std::min(180, h_med + std::max(20, 3 * h_mad + 15));
    int s_lo = std::max(15, sv[sv.size()/10]);
    int v_lo = std::max(20, vv[vv.size()/10]);

    for (int r = 0; r < mask.rows; r++) {
        uint8_t* mr = mask.ptr<uint8_t>(r);
        const cv::Vec3b* hr = hsv.ptr<cv::Vec3b>(r);
        for (int c = 0; c < mask.cols; c++) {
            if (mr[c] == 0) continue;
            if (hr[c][0] < h_lo || hr[c][0] > h_hi ||
                hr[c][1] < s_lo || hr[c][2] < v_lo)
                mr[c] = 0;
        }
    }
}

/**
 * @brief 边缘检测：用 Canny + 轮廓评分提取矩形
 *
 * 当深度和颜色方法都失败时使用，利用托盘边缘。
 */
static bool detectByEdges(const cv::Mat& color,
                          const cv::Mat& depth,
                          std::vector<cv::Point>& corners_out,
                          const CenterRectConfig& cfg)
{
    int w = color.cols, h = color.rows;
    int cx = w / 2, cy = h / 2;
    cv::Mat gray, blurred;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    std::vector<cv::Mat> edge_maps;
    for (double t1 : {15.0, 25.0, 40.0, 60.0}) {
        cv::Mat edges;
        cv::Canny(blurred, edges, t1, t1 * 2.5, 3);
        edge_maps.push_back(edges);
    }

    double exp_size_px = -1;
    if (cfg.phys_width_mm > 0) {
        double est = estimateTargetDepth(depth, cfg.center_ratio);
        if (est > 0) exp_size_px = cfg.phys_width_mm / (est / cfg.cam_fx);
    }

    struct EC { std::vector<cv::Point> c; cv::RotatedRect rr; double s; };
    std::vector<EC> cands;
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    double min_area = color.total() * cfg.min_area_ratio;

    for (auto& edges : edge_maps) {
        cv::Mat dilated;
        cv::dilate(edges, dilated, k3);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < min_area) continue;
            cv::RotatedRect rr = cv::minAreaRect(c);
            double rw = rr.size.width, rh = rr.size.height;
            if (rw < rh) std::swap(rw, rh);
            if (rh < 20) continue;
            double extent = area / std::max(rw * rh, 1.0);
            double aspect = rw / std::max(rh, 1.0);
            if (extent < 0.3 || aspect > 4.0) continue;
            cv::Point2f rc = rr.center;
            double dc = std::sqrt((rc.x-cx)*(rc.x-cx)+(rc.y-cy)*(rc.y-cy));
            double cs = 1.0 - std::min(1.0, dc/(std::min(w,h)*0.6));
            double ss = 0.5;
            if (exp_size_px > 0) {
                double se = std::abs(std::sqrt(area)-exp_size_px)/std::max(exp_size_px,1.0);
                ss = 1.0 - std::min(1.0, se);
            }
            double as = 1.0 - std::min(1.0, std::abs(aspect-1.0)/1.5);
            double score = cs*0.3 + ss*0.35 + as*0.2 + extent*0.15;
            cands.push_back({c, rr, score});
        }
    }
    if (cands.empty()) return false;

    std::sort(cands.begin(), cands.end(),
              [](const EC& a, const EC& b) { return a.s > b.s; });

    for (auto& cand : cands) {
        cv::Point2f pts[4]; cand.rr.points(pts);
        std::vector<cv::Point> rc;
        for (int i = 0; i < 4; i++)
            rc.push_back(cv::Point(std::clamp((int)std::round(pts[i].x),0,w-1),
                                   std::clamp((int)std::round(pts[i].y),0,h-1)));
        auto ordered = orderCorners(rc);
        double wp = cv::norm(cv::Point2f(ordered[1])-cv::Point2f(ordered[0]));
        double hp = cv::norm(cv::Point2f(ordered[3])-cv::Point2f(ordered[0]));
        double asp = wp/std::max(hp,1.0);
        if (asp < 0.25 || asp > 4.0) continue;
        double qcx=0, qcy=0;
        for (auto& p : ordered) { qcx+=p.x; qcy+=p.y; }
        qcx/=4; qcy/=4;
        if (std::sqrt((qcx-cx)*(qcx-cx)+(qcy-cy)*(qcy-cy)) > std::min(w,h)*0.5) continue;

        corners_out = ordered;
        cv::Mat gray2;
        cv::cvtColor(color, gray2, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> cf;
        for (auto& p : corners_out)
            cf.push_back(cv::Point2f((float)std::clamp(p.x,3,w-4),(float)std::clamp(p.y,3,h-4)));
        cv::cornerSubPix(gray2, cf, cv::Size(5,5), cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,30,0.01));
        for (size_t i=0; i<corners_out.size(); i++)
            corners_out[i]=cv::Point(std::clamp((int)std::round(cf[i].x),0,w-1),
                                     std::clamp((int)std::round(cf[i].y),0,h-1));
        if (cfg.phys_width_mm>0&&cfg.phys_height_mm>0) {
            cv::Mat cm = (cv::Mat_<double>(3,3)<<cfg.cam_fx,0,cfg.cam_cx,0,cfg.cam_fy,cfg.cam_cy,0,0,1);
            refineWithKnownSize(depth, cm, corners_out, cfg.phys_width_mm, cfg.phys_height_mm);
        }
        return true;
    }
    return false;
}

/**
 * @brief YOLO 引导的边缘检测：在 YOLO 给出的 ROI 区域内精化托盘边界
 *
 * YOLO 检测给出托盘的大体位置（bbox），此函数在扩展后的 ROI 内进行多阈值
 * Canny 边缘检测，寻找最优四边形作为托盘的精确边界。
 * 真实边界距 YOLO 检测区域不应太远 —— 超过 ROI 对角线一半的候选会被过滤。
 *
 * @param color       彩色图
 * @param depth       深度图（可选，用于已知尺寸精化）
 * @param yolo_roi    YOLO 检测给出的托盘区域
 * @param corners_out 输出的四个角点 (TL, TR, BR, BL)
 * @param cfg         检测配置参数
 * @return true 检测成功
 */
bool detectPalletByEdgesInROI(const cv::Mat& color,
                                      const cv::Mat& depth,
                                      const cv::Rect& yolo_roi,
                                      std::vector<cv::Point>& corners_out,
                                      const CenterRectConfig& cfg)
{
    int w = color.cols, h = color.rows;

    // ---- 1. 扩展 ROI，留出搜索余量 ----------------------------------------
    int margin_x = std::max(15, (int)(yolo_roi.width * 0.25));
    int margin_y = std::max(15, (int)(yolo_roi.height * 0.25));
    cv::Rect roi(
        std::max(0, yolo_roi.x - margin_x),
        std::max(0, yolo_roi.y - margin_y),
        std::min(w - std::max(0, yolo_roi.x - margin_x), yolo_roi.width + 2 * margin_x),
        std::min(h - std::max(0, yolo_roi.y - margin_y), yolo_roi.height + 2 * margin_y)
    );

    // YOLO 给出的中心点
    cv::Point2f yolo_center(yolo_roi.x + yolo_roi.width * 0.5f,
                             yolo_roi.y + yolo_roi.height * 0.5f);

    // ---- 2. 在 ROI 内做边缘检测 ------------------------------------------
    cv::Mat color_roi = color(roi);
    cv::Mat gray, blurred;
    cv::cvtColor(color_roi, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    // 多阈值 Canny 增加鲁棒性
    std::vector<cv::Mat> edge_maps;
    for (double t1 : {15.0, 25.0, 40.0, 60.0, 80.0}) {
        cv::Mat edges;
        cv::Canny(blurred, edges, t1, t1 * 2.5, 3);
        edge_maps.push_back(edges);
    }

    // ---- 3. 估计期望尺寸（如果有物理尺寸） --------------------------------
    double exp_size_px = -1;
    if (cfg.phys_width_mm > 0 && !depth.empty()) {
        // 在 YOLO 中心附近采样深度
        int sx = std::clamp((int)yolo_center.x, 10, w - 10);
        int sy = std::clamp((int)yolo_center.y, 10, h - 10);
        int sample_r = 20;
        std::vector<uint16_t> vals;
        for (int dy = -sample_r; dy <= sample_r; dy++) {
            for (int dx = -sample_r; dx <= sample_r; dx++) {
                uint16_t d = depth.at<uint16_t>(std::clamp(sy + dy, 0, h - 1),
                                                 std::clamp(sx + dx, 0, w - 1));
                if (d > 0) vals.push_back(d);
            }
        }
        if (!vals.empty()) {
            size_t n = vals.size();
            std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
            double est = vals[n / 2];
            if (est > 0) exp_size_px = cfg.phys_width_mm / (est / cfg.cam_fx);
        }
    }

    // ---- 4. 在边缘图中寻找候选四边形 --------------------------------------
    struct EC { std::vector<cv::Point> c; cv::RotatedRect rr; double s; };
    std::vector<EC> cands;
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // ROI 内最小面积阈值适当放低
    double min_area = color_roi.total() * std::max(cfg.min_area_ratio * 0.3, 0.003);

    for (auto& edges : edge_maps) {
        cv::Mat dilated;
        cv::dilate(edges, dilated, k3);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < min_area) continue;
            cv::RotatedRect rr = cv::minAreaRect(c);
            double rw = rr.size.width, rh = rr.size.height;
            if (rw < rh) std::swap(rw, rh);
            if (rh < 15) continue;
            double extent = area / std::max(rw * rh, 1.0);
            double aspect = rw / std::max(rh, 1.0);
            if (extent < 0.3 || aspect > 4.0) continue;

            // 坐标转回全图
            cv::Point2f rc_full(rr.center.x + roi.x, rr.center.y + roi.y);

            // 距 YOLO 中心的距离评分（越近越好）
            double dc = std::sqrt((rc_full.x - yolo_center.x) * (rc_full.x - yolo_center.x) +
                                   (rc_full.y - yolo_center.y) * (rc_full.y - yolo_center.y));
            double max_dist = std::sqrt((double)(roi.width * roi.width + roi.height * roi.height)) * 0.5;
            double cs = 1.0 - std::min(1.0, dc / std::max(max_dist, 1.0));

            // 尺寸匹配评分
            double ss = 0.5;
            if (exp_size_px > 0) {
                double se = std::abs(std::sqrt(area) - exp_size_px) / std::max(exp_size_px, 1.0);
                ss = 1.0 - std::min(1.0, se);
            }

            // 长宽比评分
            double as = 1.0 - std::min(1.0, std::abs(aspect - 1.0) / 1.5);

            // 综合评分：距离权重最高（YOLO 引导的核心）
            double score = cs * 0.45 + ss * 0.25 + as * 0.15 + extent * 0.15;
            cands.push_back({c, rr, score});
        }
    }

    if (cands.empty()) return false;

    // 按评分降序
    std::sort(cands.begin(), cands.end(),
              [](const EC& a, const EC& b) { return a.s > b.s; });

    // ---- 5. 逐个候选验证 -------------------------------------------------
    for (auto& cand : cands) {
        cv::Point2f pts[4]; cand.rr.points(pts);
        std::vector<cv::Point> rc;
        for (int i = 0; i < 4; i++)
            rc.push_back(cv::Point(
                std::clamp((int)std::round(pts[i].x) + roi.x, 0, w - 1),
                std::clamp((int)std::round(pts[i].y) + roi.y, 0, h - 1)));

        auto ordered = orderCorners(rc);

        double wp = cv::norm(cv::Point2f(ordered[1]) - cv::Point2f(ordered[0]));
        double hp = cv::norm(cv::Point2f(ordered[3]) - cv::Point2f(ordered[0]));
        double asp = wp / std::max(hp, 1.0);
        if (asp < 0.25 || asp > 4.0) continue;

        // 四边形中心必须在 YOLO ROI 扩展后的合理范围内
        double qcx = 0, qcy = 0;
        for (auto& p : ordered) { qcx += p.x; qcy += p.y; }
        qcx /= 4; qcy /= 4;
        double dist_from_yolo = std::sqrt((qcx - yolo_center.x) * (qcx - yolo_center.x) +
                                           (qcy - yolo_center.y) * (qcy - yolo_center.y));
        double allowed_dist = std::max(yolo_roi.width, yolo_roi.height) * 0.7;
        if (dist_from_yolo > allowed_dist) continue;

        corners_out = ordered;

        // ---- 6. 亚像素精化 ------------------------------------------------
        cv::Mat gray_full;
        cv::cvtColor(color, gray_full, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> cf;
        for (auto& p : corners_out)
            cf.push_back(cv::Point2f((float)std::clamp(p.x, 3, w - 4),
                                      (float)std::clamp(p.y, 3, h - 4)));
        cv::cornerSubPix(gray_full, cf, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
        for (size_t i = 0; i < corners_out.size(); i++)
            corners_out[i] = cv::Point(std::clamp((int)std::round(cf[i].x), 0, w - 1),
                                        std::clamp((int)std::round(cf[i].y), 0, h - 1));

        // ---- 7. 已知物理尺寸 3D 精化 --------------------------------------
        if (cfg.phys_width_mm > 0 && cfg.phys_height_mm > 0 && !depth.empty()) {
            cv::Mat cm = (cv::Mat_<double>(3, 3) << cfg.cam_fx, 0, cfg.cam_cx,
                           0, cfg.cam_fy, cfg.cam_cy, 0, 0, 1);
            refineWithKnownSize(depth, cm, corners_out, cfg.phys_width_mm, cfg.phys_height_mm);
        }

        return true;
    }

    return false;
}

/**
 * @brief 颜色备选检测：用自适应 HSV 颜色分割提取矩形物体
 *
 * 当深度分割因物体与背景深度重叠而失败时使用。
 * 流程：HSV 颜色筛选 → 形态学去噪 → 最大轮廓 → 四边形拟合 → 亚像素精化
 */
static bool detectByColor(const cv::Mat& color,
                          const cv::Mat& depth,
                          std::vector<cv::Point>& corners_out,
                          const CenterRectConfig& cfg)
{
    cv::Mat hsv;
    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);
    int w = color.cols, h = color.rows;
    int cx = w / 2, cy = h / 2;

    // ---- 1. 自适应采样主色调 ----
    int roi_sz = std::max(30, (int)(std::min(w, h) * cfg.center_ratio));
    int h_center = -1, s_lo = cfg.color_s_min, v_lo = cfg.color_v_min;

    for (int dy : {0, -roi_sz/2, -roi_sz, roi_sz/2, -roi_sz/4}) {
        int ry = std::clamp(cy + dy, roi_sz/2, h - 1 - roi_sz/2);
        cv::Rect roi(std::max(0, cx-roi_sz/2), std::max(0, ry-roi_sz/2),
                     std::min(roi_sz, w-std::max(0,cx-roi_sz/2)),
                     std::min(roi_sz, h-std::max(0,ry-roi_sz/2)));
        std::vector<int> hv, sv, vv;
        for (int r = roi.y; r < roi.y+roi.height; r++) {
            const cv::Vec3b* row = hsv.ptr<cv::Vec3b>(r);
            for (int c = roi.x; c < roi.x+roi.width; c++) {
                hv.push_back(row[c][0]); sv.push_back(row[c][1]); vv.push_back(row[c][2]);
            }
        }
        if (hv.size() < 100) continue;
        std::sort(hv.begin(), hv.end());
        std::sort(sv.begin(), sv.end());
        std::sort(vv.begin(), vv.end());
        int hm = hv[hv.size()/2], sm = sv[sv.size()/2], vm = vv[vv.size()/2];
        if (hm >= 85 && hm <= 145 && sm > 25) {
            h_center = hm;
            s_lo = std::max(15, sm/3);
            v_lo = std::max(20, vm/3);
            break;
        }
    }
    if (h_center < 0) h_center = (cfg.color_h_low + cfg.color_h_high) / 2;

    // ---- 2. 颜色 mask + flood fill 隔离中心区域 ----
    cv::Mat color_mask;
    {
        int hl = std::max(0, h_center - 18);
        int hh = std::min(180, h_center + 18);
        cv::inRange(hsv, cv::Scalar(hl, s_lo, v_lo), cv::Scalar(hh, 255, 255), color_mask);
        cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, k5);
        cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, k3);
    }

    // 连通分量分析，取包含中心的分量
    cv::Mat labels, stats, centroids;
    int nl = cv::connectedComponentsWithStats(color_mask, labels, stats, centroids, 8);
    if (nl <= 1) return false;

    int cl = labels.at<int>(cy, cx);
    if (cl <= 0) {
        double md = std::min(w, h) * 0.3;
        for (int i = 1; i < nl; i++) {
            double d = std::sqrt((centroids.at<double>(i,0)-cx)*(centroids.at<double>(i,0)-cx)
                               + (centroids.at<double>(i,1)-cy)*(centroids.at<double>(i,1)-cy));
            if (d < md && stats.at<int>(i,cv::CC_STAT_AREA) > color.total()*cfg.min_area_ratio) {
                cl = i; md = d;
            }
        }
    }
    if (cl <= 0) return false;

    cv::Mat cmask = cv::Mat::zeros(color_mask.size(), CV_8UC1);
    for (int r = 0; r < h; r++) {
        const int* lr = labels.ptr<int>(r);
        uint8_t* mr = cmask.ptr<uint8_t>(r);
        for (int c = 0; c < w; c++)
            if (lr[c] == cl) mr[c] = 255;
    }
    cv::Mat k7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(cmask, cmask, cv::MORPH_CLOSE, k7);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    auto& bc = *std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
    if (cv::contourArea(bc) < color.total() * cfg.min_area_ratio) return false;

    // ---- 3. minAreaRect 获取矩形角点 ----
    cv::RotatedRect rr = cv::minAreaRect(bc);
    cv::Point2f pts[4]; rr.points(pts);
    std::vector<cv::Point> rc;
    for (int i = 0; i < 4; i++)
        rc.push_back(cv::Point(std::clamp((int)std::round(pts[i].x),0,w-1),
                               std::clamp((int)std::round(pts[i].y),0,h-1)));
    corners_out = orderCorners(rc);

    // 验证尺寸
    {
        double wp = cv::norm(cv::Point2f(corners_out[1])-cv::Point2f(corners_out[0]));
        double hp = cv::norm(cv::Point2f(corners_out[3])-cv::Point2f(corners_out[0]));
        double asp = wp/std::max(hp,1.0);
        if (asp < 0.3 || asp > 3.0) return false;
        if (cfg.phys_width_mm > 0) {
            double est = estimateTargetDepth(depth, cfg.center_ratio);
            if (est > 0) {
                double exp_px = cfg.phys_width_mm / (est/cfg.cam_fx);
                double err = std::abs(std::max(wp,hp) - exp_px) / exp_px;
                if (err > 0.30) return false; // 尺寸偏差太大，让边缘检测接力
            }
        }
    }

    // ---- 4. 亚像素精化 ----
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> cf;
    for (auto& p : corners_out)
        cf.push_back(cv::Point2f((float)std::clamp(p.x,3,w-4),(float)std::clamp(p.y,3,h-4)));
    cv::cornerSubPix(gray, cf, cv::Size(5,5), cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,30,0.01));
    for (size_t i=0; i<corners_out.size(); i++)
        corners_out[i]=cv::Point(std::clamp((int)std::round(cf[i].x),0,w-1),
                                 std::clamp((int)std::round(cf[i].y),0,h-1));

    // ---- 5. 3D 精化 ----
    if (cfg.phys_width_mm>0 && cfg.phys_height_mm>0) {
        cv::Mat cm = (cv::Mat_<double>(3,3)<<cfg.cam_fx,0,cfg.cam_cx,0,cfg.cam_fy,cfg.cam_cy,0,0,1);
        refineWithKnownSize(depth, cm, corners_out, cfg.phys_width_mm, cfg.phys_height_mm);
    }
    return true;
}

/**
 * @brief 结合彩色图和深度图，检测图像中心处的矩形物体
 *
 * 算法流程：
 *   1. 从深度图中心区域采样，取中位数作为目标物体深度
 *   2. 百分比容差（±15%）生成二值 mask，触碰边界过多则自动收紧重试
 *   3. 形态学去噪后在 mask 中寻找包含图像中心的最大轮廓
 *   4. 四边形拟合得初始角点
 *   5. 深度梯度在角点邻域精化位置
 *   6. 彩色图 cornerSubPix 亚像素精化
 *   7. 按 TL, TR, BR, BL 排序输出
 *
 * 百分比容差自适应不同距离场景：中心深度 × 15%，最少 ±50mm。
 *
 * @param color_path  彩色图路径 (.jpg)
 * @param depth_path  深度图路径 (.png, 16-bit 单通道, 单位毫米)
 * @param corners_out 输出 4 个角点，顺序 TL, TR, BR, BL
 * @param cfg         可调配的参数
 * @param out_image   可选，绘制检测结果
 * @return true 检测成功, false 失败
 */

 bool detectCenterRect(const cv::Mat& color,
                      const cv::Mat& depth,
                      std::vector<cv::Point>& corners_out,
                      const CenterRectConfig& cfg = CenterRectConfig(),
                      cv::Mat* out_image = nullptr,double* c_depth_avg = nullptr)
{

    int h = depth.rows, w = depth.cols;
    int cx = w / 2, cy = h / 2;

    // ---- 2. 估算中心物体的深度 ------------------------------------------------
    double center_depth = estimateTargetDepth(depth, cfg.center_ratio);
    if (center_depth < 0) {
        std::cerr << "[detectCenterRect] 中心区域无有效深度值" << std::endl;
        return false;
    }

    if (nullptr != c_depth_avg)
    {
        *c_depth_avg=center_depth;
    }
    

    // ---- 3. 深度 mask（百分比容差分割） ---------------------------------------
    // 如果轮廓触碰图像边界太多，逐步收紧容差重试
    std::vector<cv::Point> best_contour;
    cv::Point2f center_pt(static_cast<float>(cx), static_cast<float>(cy));
    double used_ratio = cfg.depth_tolerance_ratio;

    cv::Mat final_mask;
    for (int retry = 0; retry < 3; retry++) {
        cv::Mat mask = createDepthMask(depth, center_depth, used_ratio);
        refineDepthMaskWithColor(color, depth, mask);

        cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k5);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k5);
        final_mask = mask;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) { used_ratio *= 0.7; continue; }

        best_contour.clear();
        double best_area = 0;
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < color.total() * cfg.min_area_ratio) continue;
            if (cv::pointPolygonTest(c, center_pt, false) >= 0 && area > best_area) {
                best_area = area; best_contour = c;
            }
        }
        if (best_contour.empty()) {
            auto it = std::max_element(contours.begin(), contours.end(),
                [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
            if (it != contours.end() && cv::contourArea(*it) >= color.total() * cfg.min_area_ratio)
                best_contour = *it;
        }
        if (best_contour.empty()) { used_ratio *= 0.7; continue; }

        cv::Rect br = cv::boundingRect(best_contour);
        int border_touches = 0;
        if (br.x <= 2) border_touches++;
        if (br.y <= 2) border_touches++;
        if (br.x + br.width >= w - 2) border_touches++;
        if (br.y + br.height >= h - 2) border_touches++;
        if (border_touches >= 3) { used_ratio *= 0.6; continue; }
        break;
    }

    // ---- 5. 判断深度分割并提取角点 ----
    bool depth_ok = !best_contour.empty();
    if (depth_ok) {
        cv::Rect br = cv::boundingRect(best_contour);
        double bbox_coverage = (double)(br.width * br.height) / color.total();
        int border_touches = 0;
        if (br.x <= 2) border_touches++; if (br.y <= 2) border_touches++;
        if (br.x + br.width >= w - 2) border_touches++;
        if (br.y + br.height >= h - 2) border_touches++;
        if (bbox_coverage > 0.75 || border_touches >= 3) depth_ok = false;
    }

    if (depth_ok) {
        // CC 分析 + 腐蚀断开粘连
        cv::Mat lbl, st, ct;
        int nl = cv::connectedComponentsWithStats(final_mask, lbl, st, ct, 8);
        bool cc_ok = false;

        auto tryCC = [&](const cv::Mat& m) -> bool {
            cv::Mat l, s, c;
            int n = cv::connectedComponentsWithStats(m, l, s, c, 8);
            if (n <= 1) return false;
            double mmp = center_depth / cfg.cam_fx;
            double exp = cfg.phys_width_mm / mmp;
            double bs = 0; int bl = -1; std::vector<cv::Point> ba;
            for (int i = 1; i < n; i++) {
                int ca = s.at<int>(i, cv::CC_STAT_AREA);
                if (ca < color.total()*cfg.min_area_ratio) continue;
                cv::Mat cm = cv::Mat::zeros(m.size(), CV_8UC1);
                for (int r = 0; r < h; r++) {
                    const int* lr = l.ptr<int>(r); uint8_t* mr = cm.ptr<uint8_t>(r);
                    for (int cc = 0; cc < w; cc++)
                        if (lr[cc] == i) mr[cc] = 255;
                }
                std::vector<std::vector<cv::Point>> ccs;
                cv::findContours(cm, ccs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (ccs.empty()) continue;
                auto& ccc = *std::max_element(ccs.begin(), ccs.end(),
                    [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
                cv::RotatedRect rr = cv::minAreaRect(ccc);
                double rw = rr.size.width, rh = rr.size.height;
                if (rw < rh) std::swap(rw, rh);
                double asp = rw/std::max(rh,1.0);
                if (asp > 2.5) continue;
                double se = std::abs(std::max(rw,rh)-exp)/exp;
                if (se > 0.4) continue;
                double dx = c.at<double>(i,0)-cx, dy = c.at<double>(i,1)-cy;
                double di = std::sqrt(dx*dx+dy*dy);
                double sc = (1.0-std::min(1.0,se))*0.5
                          + (1.0-std::min(1.0,di/(std::min(w,h)*0.4)))*0.3
                          + (1.0-std::min(1.0,std::abs(asp-1.0)))*0.2;
                if (sc > bs) {
                    bs=sc; bl=i;
                    cv::Point2f pts[4]; rr.points(pts);
                    ba.clear();
                    for (int j=0;j<4;j++)
                        ba.push_back(cv::Point(std::clamp((int)std::round(pts[j].x),0,w-1),
                                               std::clamp((int)std::round(pts[j].y),0,h-1)));
                }
            }
            if (bl>0 && ba.size()==4) { corners_out=orderCorners(ba); return true; }
            return false;
        };

        if (cfg.phys_width_mm > 0) {
            cc_ok = tryCC(final_mask);
            if (!cc_ok && nl <= 3) {
                cv::Mat er;
                cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11));
                cv::erode(final_mask, er, ker);
                cc_ok = tryCC(er);
            }
        }

        if (!cc_ok) {
            std::vector<cv::Point> approx = fitQuad(best_contour, cfg.approx_eps);
            if (approx.size() == 4) corners_out = orderCorners(approx);
        }

        if (corners_out.size() == 4 && cfg.phys_width_mm > 0 && cfg.phys_height_mm > 0) {
            double mmPerPx = center_depth / cfg.cam_fx;
            double ew = cfg.phys_width_mm / mmPerPx;
            double eh = cfg.phys_height_mm / mmPerPx;
            double dw = cv::norm(cv::Point2f(corners_out[1])-cv::Point2f(corners_out[0]));
            double dh = cv::norm(cv::Point2f(corners_out[3])-cv::Point2f(corners_out[0]));
            double we = std::abs(dw-ew)/ew, he = std::abs(dh-eh)/eh;
            if (we > 0.25 || he > 0.25) depth_ok = false;
            if (depth_ok) {
                double asp = dw/std::max(dh,1.0);
                if (asp < 0.5 || asp > 1.8) depth_ok = false;
            }
        }
    }

    // --- 颜色备选方案 ---
    if (!depth_ok && cfg.use_color_fallback) {
        if (detectByColor(color, depth, corners_out, cfg)) {
            if (out_image) {
                *out_image = color.clone();
                cv::polylines(*out_image, corners_out, true, cv::Scalar(0,255,0), 3);
                const char* ls[] = {"TL","TR","BR","BL"};
                for (int i=0;i<4;i++) {
                    cv::circle(*out_image, corners_out[i], 10, cv::Scalar(0,0,255), -1);
                    cv::putText(*out_image, ls[i], corners_out[i]+cv::Point(15,5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
                }
            }
            return true;
        }
        if (detectByEdges(color, depth, corners_out, cfg)) {
            if (out_image) {
                *out_image = color.clone();
                cv::polylines(*out_image, corners_out, true, cv::Scalar(0,255,0), 3);
                const char* ls[] = {"TL","TR","BR","BL"};
                for (int i=0;i<4;i++) {
                    cv::circle(*out_image, corners_out[i], 10, cv::Scalar(0,0,255), -1);
                    cv::putText(*out_image, ls[i], corners_out[i]+cv::Point(15,5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
                }
            }
            return true;
        }
    }

    if (!depth_ok) {
        std::cerr << "[detectCenterRect] 各方法均未找到有效目标" << std::endl;
        return false;
    }

    // ---- 6. 深度梯度精化边缘 --------------------------------------------------
    refineCornersWithDepthGradient(depth, corners_out);

    // ---- 7. 彩色图亚像素精化 --------------------------------------------------
    {
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners_f;
        for (auto& p : corners_out) corners_f.push_back(cv::Point2f(p));
        cv::cornerSubPix(gray, corners_f, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
        for (size_t i = 0; i < corners_out.size(); i++)
            corners_out[i] = cv::Point(static_cast<int>(corners_f[i].x), static_cast<int>(corners_f[i].y));
    }

    // ---- 8. 已知物理尺寸 3D 精化 -----------------------------------------------
    if (cfg.phys_width_mm > 0 && cfg.phys_height_mm > 0) {
        cv::Mat camMat = (cv::Mat_<double>(3, 3)
            << cfg.cam_fx, 0, cfg.cam_cx,
               0, cfg.cam_fy, cfg.cam_cy,
               0, 0, 1);
        refineWithKnownSize(depth, camMat, corners_out, cfg.phys_width_mm, cfg.phys_height_mm);
    }

    // ---- 9. 可选：绘制结果 ----------------------------------------------------
    if (out_image) {
        *out_image = color.clone();
        cv::polylines(*out_image, corners_out, true, cv::Scalar(0, 255, 0), 3);
        const char* labels[] = {"TL", "TR", "BR", "BL"};
        for (int i = 0; i < 4; i++) {
            cv::circle(*out_image, corners_out[i], 10, cv::Scalar(0, 0, 255), -1);
            cv::putText(*out_image, labels[i], corners_out[i] + cv::Point(15, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }
    }

    return true;
}


bool detectCenterRect(const std::string& color_path,
                      const std::string& depth_path,
                      std::vector<cv::Point>& corners_out,
                      const CenterRectConfig& cfg = CenterRectConfig(),
                      cv::Mat* out_image = nullptr)
{
    // ---- 1. 读取图像 --------------------------------------------------------
    cv::Mat color = cv::imread(color_path);
    if (color.empty()) {
        std::cerr << "[detectCenterRect] 无法读取彩色图: " << color_path << std::endl;
        return false;
    }

    cv::Mat depth_raw = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
    if (depth_raw.empty()) {
        std::cerr << "[detectCenterRect] 无法读取深度图: " << depth_path << std::endl;
        return false;
    }
    cv::Mat depth(depth_raw.rows, depth_raw.cols, CV_16UC1, depth_raw.data);

    if (depth.rows != color.rows || depth.cols != color.cols) {
        cv::resize(depth, depth, cv::Size(color.cols, color.rows), 0, 0, cv::INTER_NEAREST);
    }


    return detectCenterRect(color,depth,corners_out,cfg,out_image);
}

/**
 * @brief 检测矩形物体并输出四条边的坐标
 *
 * @param color_path     彩色图路径
 * @param depth_path     深度图路径
 * @param output_path    可选，结果可视化图的保存路径
 * @return true 检测成功
 */
bool detectCenterRectAndPrint(const std::string& color_path,
                              const std::string& depth_path,
                              const std::string& output_path = "",
                              double phys_w = 0,
                              double phys_h = 0)
{
    CenterRectConfig cfg;
    if (phys_w > 0 && phys_h > 0) {
        cfg.phys_width_mm = phys_w;
        cfg.phys_height_mm = phys_h;
    }
    std::vector<cv::Point> corners;
    if (!detectCenterRect(color_path, depth_path, corners, cfg)) {
        std::cout << "{\"error\": \"detection failed\"}" << std::endl;
        return false;
    }

    // 四条边
    struct Edge { cv::Point p1, p2; const char* name; };
    Edge edges[4] = {
        {corners[0], corners[1], "Top"},
        {corners[1], corners[2], "Right"},
        {corners[2], corners[3], "Bottom"},
        {corners[3], corners[0], "Left"}
    };

    std::cout << "=== Rectangle Edges ===" << std::endl;
    for (const auto& e : edges) {
        std::cout << "  " << e.name << ": (" << e.p1.x << "," << e.p1.y
                  << ") -> (" << e.p2.x << "," << e.p2.y << ")" << std::endl;
    }

    // JSON 输出
    nlohmann::json j;
    j["edges"] = nlohmann::json::array();
    for (const auto& e : edges) {
        j["edges"].push_back({
            {"name",  e.name},
            {"x1",    e.p1.x}, {"y1", e.p1.y},
            {"x2",    e.p2.x}, {"y2", e.p2.y}
        });
    }
    j["corners"] = {
        {{"label", "TL"}, {"x", corners[0].x}, {"y", corners[0].y}},
        {{"label", "TR"}, {"x", corners[1].x}, {"y", corners[1].y}},
        {{"label", "BR"}, {"x", corners[2].x}, {"y", corners[2].y}},
        {{"label", "BL"}, {"x", corners[3].x}, {"y", corners[3].y}}
    };
    std::cout << j.dump(2) << std::endl;

    if (!output_path.empty()) {
        cv::Mat result;
        detectCenterRect(color_path, depth_path, corners, cfg, &result);
        if (!result.empty())
            cv::imwrite(output_path, result);
    }

    return true;
}

/**
 * @brief 按序号批量检测（00001, 00002, 00003 等）
 *
 * @param dir_path   图像所在目录
 * @param seq        序号，如 "00001"
 * @param output_dir 结果输出目录（空则不保存图片）
 */
bool detectCenterRectBySeq(const std::string& dir_path,
                           const std::string& seq,
                           const std::string& output_dir = "",
                           double phys_w = 0,
                           double phys_h = 0)
{
    std::string color_path = dir_path + "/" + seq + ".jpg";
    std::string depth_path = dir_path + "/" + seq + ".png";
    std::string out_path;
    if (!output_dir.empty())
        out_path = output_dir + "/" + seq + "_result.jpg";

    return detectCenterRectAndPrint(color_path, depth_path, out_path, phys_w, phys_h);
}
