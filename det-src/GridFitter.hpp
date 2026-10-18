#ifndef GRID_FITTER_H
#define GRID_FITTER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <array>
#include <functional>
#include <string>

#include <opencv2/opencv.hpp>


#ifndef M_PI
const double M_PI = 3.14159265358979323846;
#endif



// 检测点结构体
struct DetectionPoint {
    int id;           // 检测框的ID（4或5）
    float confidence; // 置信度
    float x;          // 中心点x坐标（从边界框计算）
    float y;          // 中心点y坐标（从边界框计算）

    DetectionPoint()
        : id(0), confidence(0.0f), x(0.0f), y(0.0f) {
    }
    DetectionPoint(int id_, float conf_, float x_, float y_)
        : id(id_), confidence(conf_), x(x_), y(y_) {
    }
};

enum class RowPairSelectionSource {
    None,
    OnlyPair,
    Rect,
    Edge,
    EdgeTieBreak
};

struct RowPairScore {
    int pair_index = -1;
    int rect_score = -1;
    float edge_top = -1.0f;
    float edge_bottom = -1.0f;
    float edge_combined = -1.0f;
};

struct RowPairSelectionResult {
    bool success = false;
    int selected_pair = -1;
    RowPairSelectionSource source = RowPairSelectionSource::None;
    std::string failure_reason;
    std::vector<float> row_centers;
    std::vector<RowPairScore> pair_scores;
};

struct LocalRow {
    std::vector<DetectionPoint> original_points;
    float center = 0.0f;
};


// RANSAC直线参数
struct Line {
    float a, b, c;  // 直线方程: ax + by + c = 0
    float inlierCount;

    Line(float a_ = 0, float b_ = 0, float c_ = 0, float count = 0)
        : a(a_), b(b_), c(c_), inlierCount(count) {
    }

    // 计算点到直线的垂足（交点）
    cv::Point2f getFootOfPerpendicular(const cv::Point2f& p) const {
        // 防止除以零
        float denominator = a * a + b * b;
        if (std::abs(denominator) < 1e-10f) {
            std::cerr << "警告：直线参数a和b都接近零，可能不是有效直线" << std::endl;
            return cv::Point2f(0,0); // 返回原點作为默认值
        }

        // 计算直线上的垂足坐标
        float numerator = a * p.x + b * p.y + c;
        float footX = p.x - a * numerator / denominator;
        float footY = p.y - b * numerator / denominator;

        return cv::Point2f(footX, footY);
    }

    // 计算点到直线的距离
    float distanceToPoint(const cv::Point2f& p) const {
        float denominator = std::sqrt(a * a + b * b);
        if (denominator < 1e-10f) return std::numeric_limits<float>::max();

        return std::abs(a * p.x + b * p.y + c) / denominator;
    }

    /**
         * @brief 获取直线上距离给定点p为d的两个点
         * @param p 直线上的已知点
         * @param d 距离（正数）
         * @return 包含两个点的向量，[沿方向向量的点，反方向向量的点]
         */
    std::vector<cv::Point2f> getPointsAtDistance(const cv::Point2f& p, float d) const {
        std::vector<cv::Point2f> result;

        // 计算直线法向量(a,b)的模长
        float norm = std::sqrt(a * a + b * b);

        if (norm < 1e-10f) {
            std::cerr << "警告：直线参数无效，a和b都接近零" << std::endl;
            result.push_back(p);
            result.push_back(p);
            return result;
        }

        if (d < 0) {
            d = std::abs(d); // 确保距离为正
        }

        // 直线的方向向量：垂直于法向量(a,b)，有两个方向：(-b, a) 或 (b, -a)
        // 我们选择其中一个方向
        cv::Point2f dir(-b, a); // 垂直于法向量(a,b)的方向向量

        // 单位化方向向量
        float dirNorm = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (dirNorm < 1e-10f) {
            // 如果(-b, a)接近零向量，尝试另一个方向
            dir = cv::Point2f(b, -a);
            dirNorm = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        }

        if (dirNorm < 1e-10f) {
            std::cerr << "错误：无法计算有效的方向向量" << std::endl;
            result.push_back(p);
            result.push_back(p);
            return result;
        }

        // 单位方向向量
        cv::Point2f unitDir = dir / dirNorm;

        // 计算两个点：沿着正方向和反方向
        cv::Point2f q1 = p + d * unitDir;
        cv::Point2f q2 = p - d * unitDir;

        // 验证点是否在直线上（可选，用于调试）
        if (!isPointOnLine(q1) || !isPointOnLine(q2)) {
            std::cerr << "警告：计算点可能不在精确直线上（浮点误差）" << std::endl;
        }

        result.push_back(q1);
        result.push_back(q2);

        return result;
    }

    /**
     * @brief 检查点是否在直线上（考虑浮点误差）
     * @param p 要检查的点
     * @param tolerance 容差
     * @return 如果点在直线上返回true
     */
    bool isPointOnLine(const cv::Point2f& p, float tolerance = 1e-5f) const {
        float value = a * p.x + b * p.y + c;
        return std::abs(value) < tolerance;
    }

    /**
     * @brief 计算直线上两个点之间的距离
     * @param p1 直线上的点1
     * @param p2 直线上的点2
     * @return 两点之间的直线距离
     */
    float distanceBetweenPoints(const cv::Point2f& p1, const cv::Point2f& p2) const {
        // 验证两点是否在直线上（可选）
        if (!isPointOnLine(p1) || !isPointOnLine(p2)) {
            std::cerr << "警告：点可能不在精确直线上" << std::endl;
        }

        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    // 转换为两点式便于绘制
    void getTwoPoints(cv::Point2f& p1, cv::Point2f& p2, const cv::Size& imgSize) {
        if (std::abs(b) > 1e-6) {
            // 直线不垂直，从x=0和x=imgSize.width-1计算y
            p1.x = 0;
            p1.y = -c / b;

            p2.x = imgSize.width - 1;
            p2.y = -(a * p2.x + c) / b;
        }
        else if (std::abs(a) > 1e-6) {
            // 垂直线，从y=0和y=imgSize.height-1计算x
            p1.y = 0;
            p1.x = -c / a;

            p2.y = imgSize.height - 1;
            p2.x = p1.x; // 垂直线x坐标不变
        }
        else {
            // 无效直线
            p1 = cv::Point2f(0, 0);
            p2 = cv::Point2f(0, 0);
        }
    }


    // 归一化直线参数
    void normalize() {
        float norm = std::sqrt(a * a + b * b);
        if (norm > 1e-6f) {
            a /= norm;
            b /= norm;
            c /= norm;
        }
    }

    // 计算点到直线的投影点（垂足）
    cv::Point2f projectPoint(const DetectionPoint& p) const {
        float denominator = a * a + b * b;
        if (denominator < 1e-10f) {
            return cv::Point2f(p.x, p.y);
        }

        float numerator = a * p.x + b * p.y + c;
        float projX = p.x - a * numerator / denominator;
        float projY = p.y - b * numerator / denominator;

        return cv::Point2f(projX, projY);
    }

    // 获取直线的方向向量
    cv::Point2f getDirectionVector() const {
        cv::Point2f dir(-b, a);
        float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (norm < 1e-6f) {
            return cv::Point2f(1.0f, 0.0f);
        }
        return cv::Point2f(dir.x / norm, dir.y / norm);
    }
};


// 点集聚类结果
struct Cluster {
    std::vector<DetectionPoint> points;
    std::vector<DetectionPoint> pts_ret;
    Line fittedLine;
    float centerX, centerY;
};


class FourPointEstimator {
private:
    // 投影点信息
    struct ProjectedPoint {
        DetectionPoint original;    // 原始检测点
        cv::Point2f projected;      // 投影到直线上的点
        float t;                    // 直线参数（沿方向向量的距离）
        int clusterId;              // 所属聚类ID

        ProjectedPoint(const DetectionPoint& orig, const cv::Point2f& proj, float t_val)
            : original(orig), projected(proj), t(t_val), clusterId(-1) {
        }
    };

public:
    /**
     * @brief 估计直线上的4个等间距点
     * @param line 直线
     * @param points 检测点
     * @param expectedSpacing 期望的点间距（如果为0则自动估计）
     * @return 4个理想位置的点（可能包含估计点）
     */
    std::vector<DetectionPoint> estimateFourPoints(
        const Line& line,
        const std::vector<DetectionPoint>& points,
        float expectedSpacing = 0.0f) {

        std::vector<DetectionPoint> result(4);

        if (points.empty()) {
            std::cerr << "警告：没有检测点，返回估计位置" << std::endl;
            return estimateFromLineOnly(line);
        }

        // 步骤1：将所有点投影到直线上
        std::vector<ProjectedPoint> projectedPoints = projectPointsToLine(line, points);

        if (projectedPoints.empty()) {
            std::cerr << "错误：投影点计算失败" << std::endl;
            return estimateFromLineOnly(line);
        }

        // 步骤2：沿着直线方向参数化（计算t值）
        parameterizePoints(projectedPoints, line);

        // 步骤3：聚类到4个位置
        std::vector<int> clusterAssignments = clusterToFourPositions(projectedPoints);

        // 步骤4：分析每个聚类
        std::vector<ClusterInfo> clusters = analyzeClusters(projectedPoints, clusterAssignments);

        // 步骤5：估计或计算间距
        float spacing = (expectedSpacing > 0) ? expectedSpacing : estimateSpacing(clusters);

        // 步骤6：生成4个理想位置的点
        result = generateFourPoints(line, clusters, spacing, projectedPoints);

        // 步骤7：验证和调整
        validateAndAdjust(result, line, spacing);

        return result;
    }

public:
    struct ClusterInfo {
        int clusterId = 0;
        float centerT = 0;           // 聚类中心的t值
        float minT = FLT_MAX;
        float maxT = -FLT_MAX;        // 聚类范围
        std::vector<DetectionPoint> points;
        float avgConfidence = 0;
        bool hasDetection = false;
    };

    // 步骤1：投影点到直线
    std::vector<ProjectedPoint> projectPointsToLine(
        const Line& line,
        const std::vector<DetectionPoint>& points) {

        std::vector<ProjectedPoint> projected;

        for (const auto& point : points) {
            cv::Point2f proj = line.projectPoint(point);
            projected.emplace_back(point, proj, 0.0f);
        }

        return projected;
    }

    // 步骤2：参数化点（计算沿直线方向的t值）
    void parameterizePoints(std::vector<ProjectedPoint>& projectedPoints, const Line& line) {
        if (projectedPoints.empty()) return;

        // 获取直线方向
        cv::Point2f dir = line.getDirectionVector();

        // 选择一个参考点（使用第一个投影点）
        cv::Point2f refPoint = projectedPoints[0].projected;

        for (auto& projPoint : projectedPoints) {
            // 计算从参考点到投影点的向量
            cv::Point2f vec(projPoint.projected.x - refPoint.x,
                projPoint.projected.y - refPoint.y);

            // 计算在方向上的投影长度（t值）
            projPoint.t = vec.x * dir.x + vec.y * dir.y;
        }

        // 可选的：重新中心化，使t值以0为中心
        float avgT = 0.0f;
        for (const auto& projPoint : projectedPoints) {
            avgT += projPoint.t;
        }
        avgT /= projectedPoints.size();

        for (auto& projPoint : projectedPoints) {
            projPoint.t -= avgT;
        }
    }

    // 步骤3：聚类到4个位置
    std::vector<int> clusterToFourPositions(std::vector<ProjectedPoint>& projectedPoints) {
        if (projectedPoints.size() <= 4) {
            // 点少，直接分配
            std::vector<int> assignments(projectedPoints.size());
            for (size_t i = 0; i < projectedPoints.size(); ++i) {
                assignments[i] = i;
                projectedPoints[i].clusterId = i;
            }
            return assignments;
        }

        // 按t值排序
        std::sort(projectedPoints.begin(), projectedPoints.end(),
            [](const ProjectedPoint& a, const ProjectedPoint& b) {
            return a.t < b.t;
        });

        // 使用K-means思想找到4个中心
        std::vector<float> centers(4, 0.0f);

        // 初始化中心：均匀分布在t值范围内
        float minT = projectedPoints.front().t;
        float maxT = projectedPoints.back().t;
        float range = maxT - minT;

        for (int i = 0; i < 4; ++i) {
            centers[i] = minT + (i + 0.5f) * range / 4.0f;
        }

        // 简单迭代分配
        std::vector<int> assignments(projectedPoints.size());
        bool changed;
        int maxIterations = 10;

        for (int iter = 0; iter < maxIterations; ++iter) {
            changed = false;

            // 分配点到最近的中心
            for (size_t i = 0; i < projectedPoints.size(); ++i) {
                float minDist = FLT_MAX;
                int bestCluster = -1;

                for (int j = 0; j < 4; ++j) {
                    float dist = std::abs(projectedPoints[i].t - centers[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }

                if (assignments[i] != bestCluster) {
                    assignments[i] = bestCluster;
                    projectedPoints[i].clusterId = bestCluster;
                    changed = true;
                }
            }

            if (!changed) break;

            // 更新中心
            std::vector<float> sumT(4, 0.0f);
            std::vector<int> count(4, 0);

            for (size_t i = 0; i < projectedPoints.size(); ++i) {
                int cluster = assignments[i];
                sumT[cluster] += projectedPoints[i].t;
                count[cluster]++;
            }

            for (int j = 0; j < 4; ++j) {
                if (count[j] > 0) {
                    centers[j] = sumT[j] / count[j];
                }
            }
        }

        return assignments;
    }

    // 步骤4：分析聚类
    std::vector<ClusterInfo> analyzeClusters(
        const std::vector<ProjectedPoint>& projectedPoints,
        const std::vector<int>& assignments) {

        std::vector<ClusterInfo> clusters(4);
        for (int i = 0; i < 4; ++i) {
            ClusterInfo clusterInfo;
            clusterInfo.clusterId = i;
            clusters[i] = clusterInfo;
        }

        // 收集每个聚类的信息
        for (size_t i = 0; i < projectedPoints.size(); ++i) {
            int clusterId = assignments[i];
            ClusterInfo& cluster = clusters[clusterId];

            cluster.points.push_back(projectedPoints[i].original);
            cluster.centerT += projectedPoints[i].t;
            cluster.minT = std::min(cluster.minT, projectedPoints[i].t);
            cluster.maxT = std::max(cluster.maxT, projectedPoints[i].t);
            cluster.avgConfidence += projectedPoints[i].original.confidence;
            cluster.hasDetection = true;
        }

        // 计算平均值
        for (auto& cluster : clusters) {
            if (!cluster.points.empty()) {
                cluster.centerT /= cluster.points.size();
                cluster.avgConfidence /= cluster.points.size();
            }
            else {
                // 没有检测点的聚类，标记为缺失
                cluster.hasDetection = false;
            }
        }

        // 按t值排序聚类
        std::sort(clusters.begin(), clusters.end(),
            [](const ClusterInfo& a, const ClusterInfo& b) {
            return a.centerT < b.centerT;
        });

        // 重新编号
        for (int i = 0; i < 4; ++i) {
            clusters[i].clusterId = i;
        }

        return clusters;
    }

    // 步骤5：估计间距
    float estimateSpacing(const std::vector<ClusterInfo>& clusters) {
        // 收集所有有检测点的聚类的t值
        std::vector<float> validCenters;
        for (const auto& cluster : clusters) {
            if (cluster.hasDetection) {
                validCenters.push_back(cluster.centerT);
            }
        }

        if (validCenters.size() < 2) {
            // 不足2个点，返回默认间距
            return 50.0f;
        }

        // 计算相邻聚类中心的平均间距
        std::sort(validCenters.begin(), validCenters.end());

        float totalGap = 0.0f;
        int gapCount = 0;

        for (size_t i = 1; i < validCenters.size(); ++i) {
            float gap = validCenters[i] - validCenters[i - 1];
            totalGap += gap;
            gapCount++;
        }

        if (gapCount > 0) {
            return totalGap / gapCount;
        }

        return 50.0f; // 默认值
    }

    // 步骤6：生成4个理想位置的点
    std::vector<DetectionPoint> generateFourPoints(
        const Line& line,
        const std::vector<ClusterInfo>& clusters,
        float spacing,
        const std::vector<ProjectedPoint>& allProjectedPoints) {

        std::vector<DetectionPoint> fourPoints(4);

        // 确定第一个点的t值（使4个点以0为中心）
        float firstT = -1.5f * spacing;

        // 获取直线方向向量和参考点
        cv::Point2f dir = line.getDirectionVector();
        cv::Point2f refPoint = allProjectedPoints.empty() ?
            cv::Point2f(0, 0) :
            allProjectedPoints[0].projected;

        for (int i = 0; i < 4; ++i) {
            float targetT = firstT + i * spacing;

            // 查找最接近这个位置的聚类
            int bestCluster = -1;
            float minDist = FLT_MAX;

            for (const auto& cluster : clusters) {
                if (cluster.hasDetection) {
                    float dist = std::abs(cluster.centerT - targetT);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = cluster.clusterId;
                    }
                }
            }

            if (bestCluster != -1 && minDist < spacing * 0.5f) {
                // 使用该聚类的最佳点
                const auto& cluster = clusters[bestCluster];

                // 选择置信度最高的点
                auto bestPointIt = std::max_element(cluster.points.begin(), cluster.points.end(),
                    [](const DetectionPoint& a, const DetectionPoint& b) {
                    return a.confidence < b.confidence;
                });

                fourPoints[i] = *bestPointIt;
            }
            else {
                // 需要估计这个位置的点
                fourPoints[i] = estimatePointAtT(line, refPoint, dir, targetT, i);
            }
        }

        return fourPoints;
    }

    // 估计在特定t值处的点
    DetectionPoint estimatePointAtT(const Line& line,
        const cv::Point2f& refPoint,
        const cv::Point2f& dir,
        float t,
        int positionIndex) {
        // 计算坐标
        float x = refPoint.x + dir.x * t;
        float y = refPoint.y + dir.y * t;

        // 创建估计点
        // ID可以根据位置设定，比如位置1、2用ID4，位置3、4用ID5
        int estimatedId = (positionIndex < 2) ? 4 : 5;
        float estimatedConfidence = 0.5f; // 估计点的默认置信度
        DetectionPoint dp = DetectionPoint(estimatedId, estimatedConfidence, x, y);
        return dp;
    }

    // 仅从直线信息估计4个点
    std::vector<DetectionPoint> estimateFromLineOnly(const Line& line) {
        std::vector<DetectionPoint> points(4);

        cv::Point2f dir = line.getDirectionVector();
        cv::Point2f startPoint;

        // 找一个直线上的点作为起点
        if (std::abs(line.b) > 1e-6f) {
            startPoint = cv::Point2f(0, -line.c / line.b);
        }
        else if (std::abs(line.a) > 1e-6f) {
            startPoint = cv::Point2f(-line.c / line.a, 0);
        }
        else {
            startPoint = cv::Point2f(100, 100);
        }

        float spacing = 60.0f; // 默认间距

        for (int i = 0; i < 4; ++i) {
            float x = startPoint.x + dir.x * (i - 1.5f) * spacing;
            float y = startPoint.y + dir.y * (i - 1.5f) * spacing;
            int id = (i < 2) ? 4 : 5;

            points[i] = DetectionPoint(id, 0.3f, x, y);
        }

        return points;
    }

    // 步骤7：验证和调整
    void validateAndAdjust(std::vector<DetectionPoint>& points,
        const Line& line,
        float spacing) {
        // 检查间距是否大致相等
        float minSpacing = FLT_MAX;
        float maxSpacing = 0.0f;

        for (int i = 1; i < 4; ++i) {
            float dx = points[i].x - points[i - 1].x;
            float dy = points[i].y - points[i - 1].y;
            float dist = std::sqrt(dx * dx + dy * dy);

            minSpacing = std::min(minSpacing, dist);
            maxSpacing = std::max(maxSpacing, dist);
        }

        float spacingRatio = (maxSpacing - minSpacing) / spacing;

        if (spacingRatio > 0.3f) {
            std::cerr << "警告：点间距不均匀，进行平滑调整" << std::endl;

            // 简单平滑：重新计算使间距更均匀
            float avgX = 0.0f, avgY = 0.0f;
            for (const auto& point : points) {
                avgX += point.x;
                avgY += point.y;
            }
            avgX /= 4.0f;
            avgY /= 4.0f;

            // 以平均点为中心重新分布
            cv::Point2f dir = line.getDirectionVector();

            for (int i = 0; i < 4; ++i) {
                float offset = (i - 1.5f) * spacing;
                points[i].x = avgX + dir.x * offset;
                points[i].y = avgY + dir.y * offset;
            }
        }

        // 确保点在直线上（精确投影）
        for (auto& point : points) {
            cv::Point2f proj = line.projectPoint(point);
            point.x = proj.x;
            point.y = proj.y;
        }
    }
};



class TrayFitting {
public:
    std::vector<DetectionPoint> all_pts;
    std::vector<Cluster> clusters;
    cv::Rect bbox;
    std::size_t cell_size;
    float radians;
    cv::Point2f center_pt;
    cv::Mat* p_img_dbg = NULL;

    // 建议抓取点
    std::vector<cv::Point2f> list_pts;

    // 排序点
    std::vector<cv::Point2f> list_pts_sort;
    

    // RANSAC参数
    const int ransacIterations = 1000;
    const float inlierThreshold = 10.0f;  // 内点距离阈值（像素）
    const int minInliers = 3;             // 最少内点数量

public:
    // 添加检测点
    void addDetPT(const DetectionPoint& point) {
        all_pts.push_back(point);
    }

    // 计算两点距离
    float distancePointToLine(float x, float y, const Line& line) {
        return std::abs(line.a * x + line.b * y + line.c) /
            std::sqrt(line.a * line.a + line.b * line.b);
    }

    // RANSAC拟合直线
    Line ransacFitLine(const std::vector<DetectionPoint>& points) {
        if (points.size() < 2) {
            return Line();
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points.size() - 1);

        Line bestLine;
        int maxInliers = 0;

        for (int iter = 0; iter < ransacIterations; ++iter) {
            // 随机选择两个点
            int idx1 = dis(gen);
            int idx2 = dis(gen);

            while (idx2 == idx1) {
                idx2 = dis(gen);
            }

            const auto& p1 = points[idx1];
            const auto& p2 = points[idx2];

            // 计算直线参数
            float a = p2.y - p1.y;
            float b = p1.x - p2.x;
            float c = p2.x * p1.y - p1.x * p2.y;

            // 归一化
            float norm = std::sqrt(a * a + b * b);
            if (norm < 1e-6) continue;

            a /= norm;
            b /= norm;
            c /= norm;

            Line currentLine(a, b, c);

            // 统计内点
            int inliers = 0;
            for (const auto& p : points) {
                float dist = distancePointToLine(p.x, p.y, currentLine);
                if (dist < inlierThreshold) {
                    inliers++;
                }
            }

            // 更新最佳直线
            if (inliers > maxInliers) {
                maxInliers = inliers;
                bestLine = Line(a, b, c, inliers);
            }
        }

        // 如果找到足够内点，用所有内点重新拟合直线（最小二乘法）
        if (maxInliers >= minInliers) {
            std::vector<DetectionPoint> inliers;
            for (const auto& p : points) {
                if (distancePointToLine(p.x, p.y, bestLine) < inlierThreshold) {
                    inliers.push_back(p);
                }
            }

            // 最小二乘法拟合直线
            if (inliers.size() >= 2) {
                float sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
                for (const auto& p : inliers) {
                    sumX += p.x;
                    sumY += p.y;
                    sumXX += p.x * p.x;
                    sumXY += p.x * p.y;
                }

                float n = static_cast<float>(inliers.size());
                float denominator = n * sumXX - sumX * sumX;

                if (std::abs(denominator) > 1e-6) {
                    float slope = (n * sumXY - sumX * sumY) / denominator;
                    float intercept = (sumY - slope * sumX) / n;

                    // 转换为标准形式: ax + by + c = 0
                    bestLine.a = slope;
                    bestLine.b = -1.0f;
                    bestLine.c = intercept;
                    bestLine.inlierCount = inliers.size();
                }
            }
        }

        return bestLine;
    }


    // 单方向聚类核心函数
    std::vector<std::vector<DetectionPoint>> clusterByDirection(
        const std::vector<DetectionPoint>& points,
        char direction, float min_dist) {

        if (points.empty()) return {};

        // 按指定方向排序
        std::vector<DetectionPoint> sortedPoints = points;
        if (direction == 'x' || direction == 'X') {
            std::sort(sortedPoints.begin(), sortedPoints.end(),
                [](const DetectionPoint& a, const DetectionPoint& b) {
                return a.x < b.x;
            });
        }
        else {
            std::sort(sortedPoints.begin(), sortedPoints.end(),
                [](const DetectionPoint& a, const DetectionPoint& b) {
                return a.y < b.y;
            });
        }

        std::vector<std::vector<DetectionPoint>> clusters;
        std::vector<DetectionPoint> currentCluster;

        float lastValue = (direction == 'x') ? sortedPoints[0].x : sortedPoints[0].y;
        currentCluster.push_back(sortedPoints[0]);

        for (size_t i = 1; i < sortedPoints.size(); ++i) {
            float currentValue = (direction == 'x') ? sortedPoints[i].x : sortedPoints[i].y;

            if (currentValue - lastValue <= min_dist) {
                currentCluster.push_back(sortedPoints[i]);
            }
            else {
                clusters.push_back(currentCluster);
                currentCluster.clear();
                currentCluster.push_back(sortedPoints[i]);
            }

            lastValue = currentValue;
        }

        if (!currentCluster.empty()) {
            clusters.push_back(currentCluster);
        }

        return clusters;
    }

    // 聚类点集（按y坐标分成上下两行）
    void cluster_pts_2_cls(std::vector<Cluster>& list_c, std::vector<DetectionPoint>& pts, char dim = 'y') {
        if (pts.empty()) return;

        // 如果点太少，直接返回
        if (pts.size() < 4) {
            std::cout << "警告：检测点太少，无法准确拟合" << std::endl;
            return;
        }


        if ('y' == dim)
        {
            // 按y坐标排序
            std::sort(pts.begin(), pts.end(),
                [](const DetectionPoint& a, const DetectionPoint& b) {
                return a.y < b.y;
            });
        }
        else
        {
            // 按x坐标排序
            std::sort(pts.begin(), pts.end(),
                [](const DetectionPoint& a, const DetectionPoint& b) {
                return a.x < b.x;
            });
        }

        // 找到y坐标的自然分界点（简单方法：中位数分割）
        float median_v = pts[pts.size() / 2].y;
        if ('y' != dim)
        {
            median_v = pts[pts.size() / 2].x;
        }

        Cluster topRow, bottomRow;


        if ('y' == dim)
        {
            // 按y坐标排序
            for (const auto& p : pts) {
                if (p.y < median_v) {
                    topRow.points.push_back(p);
                }
                else {
                    bottomRow.points.push_back(p);
                }
            }
        }
        else
        {
            // 按x坐标排序
            for (const auto& p : pts) {
                if (p.x < median_v) {
                    topRow.points.push_back(p);
                }
                else {
                    bottomRow.points.push_back(p);
                }
            }
        }

        list_c.clear();
        if (!topRow.points.empty()) list_c.push_back(topRow);
        if (!bottomRow.points.empty()) list_c.push_back(bottomRow);
    }



    // 绕中心点旋转
    DetectionPoint rotatePointAroundCenter(const DetectionPoint& pt, const cv::Point2f& center, float radians) {

        // 计算相对中心点的坐标
        float dx = pt.x - center.x;
        float dy = pt.y - center.y;

        // 应用旋转矩阵
        // x' = x * cosθ - y * sinθ
        // y' = x * sinθ + y * cosθ
        float cosTheta = std::cos(radians);
        float sinTheta = std::sin(radians);

        float dx_rotated = dx * cosTheta - dy * sinTheta;
        float dy_rotated = dx * sinTheta + dy * cosTheta;

        // 转换回绝对坐标
        float x_rotated = center.x + dx_rotated;
        float y_rotated = center.y + dy_rotated;

        // 返回旋转后的点（保持id和confidence不变）
        return DetectionPoint(pt.id, pt.confidence, x_rotated, y_rotated);
    }


    // 聚类点集（按y坐标分成上下两行）
    void clusterPoints() {
        if (all_pts.empty()) return;
        // 如果点太少，直接返回
        if (all_pts.size() < 4) {
            std::cout << "警告：检测点太少，无法准确拟合" << std::endl;
            return;
        }


        std::vector<DetectionPoint> list_r;
        for (auto& d : all_pts)
        {
            DetectionPoint pt_r = rotatePointAroundCenter(d, center_pt, radians);
            list_r.push_back(pt_r);
        }

#ifdef REL_DBG
        //if (NULL != p_img_dbg)
        //{
        //    cv::Scalar cl_p = Scalar(255, 0, 255);

        //    for (const auto& dp : list_r) {
        //        cv::circle(*p_img_dbg, cv::Point(dp.x, dp.y), 20,
        //            cl_p, -1);
        //    }
        //}
#endif  

        std::vector<std::vector<DetectionPoint>> list_cls_y = clusterByDirection(list_r, 'y', cell_size * 0.5);
        std::vector<std::vector<DetectionPoint>> list_cls_x = clusterByDirection(list_r, 'x', cell_size * 0.5);
        
        std::vector<std::vector<DetectionPoint>> list_cls;
        if (list_cls_y.size() <= list_cls_x.size()) list_cls = list_cls_y;
        else list_cls = list_cls_x;

        clusters.clear();
        for (auto&cls : list_cls)
        {
            Cluster c;
            for (auto& d : cls)
            {
                // 把坐标转回来
                DetectionPoint pt_r = rotatePointAroundCenter(d, center_pt, -radians);
                c.points.push_back(pt_r);
            }
            clusters.push_back(c);
        }


#ifdef REL_DBG
        if (NULL != p_img_dbg)
        {
            const std::vector<std::vector<unsigned int>> COLORS_HOLE = {
                {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
                {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0} };
            
            int idx = 0;
            for (auto& c: clusters)
            {
                cv::Scalar color = cv::Scalar(COLORS_HOLE[idx][0], COLORS_HOLE[idx][1], COLORS_HOLE[idx][2]);
                idx++;
                for (const auto& dp : c.points) {
                    cv::circle(*p_img_dbg, cv::Point(dp.x, dp.y), 20,
                        color, -1);
                }
            }

        }
#endif  
        
        //if (list_cls_y.size() <= list_cls_x.size()) cluster_pts_2_cls(clusters, all_pts, 'y');
        //else cluster_pts_2_cls(clusters, all_pts, 'x');

    }

    // 获取指定方向的值
    float get_dp_value_by_direction(const DetectionPoint& dp, char direction) {
        if (direction == 'x' || direction == 'X') {
            return dp.x;
        }
        else if (direction == 'y' || direction == 'Y') {
            return dp.y;
        }
        else {
            std::cerr << "错误：方向必须是 'x' 或 'y'" << std::endl;
            return 0.0f;
        }
    }

    /**
     * @brief 使用简单阈值法聚类（更快速）
     * @param points 检测点向量
     * @param direction 聚类方向
     * @param minGap 聚类之间的最小间隔
     * @return 聚类结果
     */
    std::vector<std::vector<DetectionPoint>> clusterSimple(
        const std::vector<DetectionPoint>& points,
        char direction,
        float minGap) {

        if (points.empty()) {
            return {};
        }

        // 复制并按指定方向排序
        std::vector<DetectionPoint> sortedPoints = points;
        if (direction == 'x' || direction == 'X') {
            std::sort(sortedPoints.begin(), sortedPoints.end(),
                [](const DetectionPoint& a, const DetectionPoint& b) {
                return a.x < b.x;
            });
        }
        else {
            std::sort(sortedPoints.begin(), sortedPoints.end(),
                [](const DetectionPoint& a, const DetectionPoint& b) {
                return a.y < b.y;
            });
        }

        std::vector<std::vector<DetectionPoint>> clusters;
        std::vector<DetectionPoint> currentCluster;

        float lastValue = get_dp_value_by_direction(sortedPoints[0], direction);
        currentCluster.push_back(sortedPoints[0]);

        for (size_t i = 1; i < sortedPoints.size(); ++i) {
            float currentValue = get_dp_value_by_direction(sortedPoints[i], direction);

            // 如果当前点与上一个点的距离小于最小间隔，属于同一聚类
            if (currentValue - lastValue <= minGap) {
                currentCluster.push_back(sortedPoints[i]);
            }
            else {
                // 开始新的聚类
                clusters.push_back(currentCluster);
                currentCluster.clear();
                currentCluster.push_back(sortedPoints[i]);
            }

            lastValue = currentValue;
        }

        // 添加最后一个聚类
        if (!currentCluster.empty()) {
            clusters.push_back(currentCluster);
        }

        return clusters;
    }

    // 2 行 × 4 列料盘的两个抓取点：左右各一个，位于上下两行之间。
    // grid 顺序固定为上排 0..3、下排 4..7。
    static std::array<cv::Point2f, 2> computeGrabPoints2x4(
        const std::array<cv::Point2f, 8>& grid)
    {
        return {
            (grid[0] + grid[1] + grid[4] + grid[5]) * 0.25f,
            (grid[2] + grid[3] + grid[6] + grid[7]) * 0.25f
        };
    }

    static const char* rowPairSelectionSourceName(RowPairSelectionSource source)
    {
        switch (source)
        {
        case RowPairSelectionSource::OnlyPair: return "only_pair";
        case RowPairSelectionSource::Rect: return "rect";
        case RowPairSelectionSource::Edge: return "edge";
        case RowPairSelectionSource::EdgeTieBreak: return "edge_tiebreak";
        default: return "none";
        }
    }

    static std::vector<RowPairScore> scoreRowPairsByPolygon(
        const std::vector<LocalRow>& rows,
        const std::vector<cv::Point2f>& polygon)
    {
        std::vector<RowPairScore> scores;
        scores.reserve(rows.size() > 1 ? rows.size() - 1 : 0);
        for (size_t i = 0; i + 1 < rows.size(); ++i)
        {
            RowPairScore score;
            score.pair_index = static_cast<int>(i);
            score.rect_score = 0;
            for (int row_offset = 0; row_offset < 2; ++row_offset)
            {
                for (const auto& point : rows[i + row_offset].original_points)
                {
                    if (cv::pointPolygonTest(
                            polygon, cv::Point2f(point.x, point.y), false) >= 0.0)
                        ++score.rect_score;
                }
            }
            scores.push_back(score);
        }
        return scores;
    }

    bool scoreRowPairsByTrayEdges(
        const std::vector<LocalRow>& rows,
        bool rows_are_y,
        const std::vector<int>& allowed_pairs,
        const cv::Mat& edge_image,
        std::vector<RowPairScore>* pair_scores,
        int* selected_pair,
        std::string* failure_reason,
        bool resolving_rect_tie)
    {
        if (selected_pair) *selected_pair = -1;
        if (edge_image.empty())
        {
            if (failure_reason) *failure_reason =
                "rect row-pair tie and edge image is unavailable";
            return false;
        }

        cv::Mat edge_bgr;
        if (edge_image.channels() == 3)
            edge_bgr = edge_image.clone();
        else if (edge_image.channels() == 4)
            cv::cvtColor(edge_image, edge_bgr, cv::COLOR_BGRA2BGR);
        else if (edge_image.channels() == 1)
            cv::cvtColor(edge_image, edge_bgr, cv::COLOR_GRAY2BGR);
        else
        {
            if (failure_reason) *failure_reason = "unsupported edge image format";
            return false;
        }
        cv::GaussianBlur(edge_bgr, edge_bgr, cv::Size(5, 5), 0.0);

        auto local_to_image = [&](float column, float row_coord) {
            DetectionPoint local;
            if (rows_are_y)
            {
                local.x = column;
                local.y = row_coord;
            }
            else
            {
                local.x = row_coord;
                local.y = column;
            }
            DetectionPoint original =
                rotatePointAroundCenter(local, center_pt, -radians);
            return cv::Point2f(original.x, original.y);
        };

        auto color_difference = [&](const cv::Point2f& a,
                                    const cv::Point2f& b) {
            int ax = cvRound(a.x);
            int ay = cvRound(a.y);
            int bx = cvRound(b.x);
            int by = cvRound(b.y);
            if (ax < 0 || ax >= edge_bgr.cols ||
                ay < 0 || ay >= edge_bgr.rows ||
                bx < 0 || bx >= edge_bgr.cols ||
                by < 0 || by >= edge_bgr.rows)
                return -1.0f;
            const cv::Vec3b ca = edge_bgr.at<cv::Vec3b>(ay, ax);
            const cv::Vec3b cb = edge_bgr.at<cv::Vec3b>(by, bx);
            return (std::abs(static_cast<int>(ca[0]) - cb[0]) +
                    std::abs(static_cast<int>(ca[1]) - cb[1]) +
                    std::abs(static_cast<int>(ca[2]) - cb[2])) / 3.0f;
        };

        auto boundary_score = [&](float expected_row,
                                  float search_radius,
                                  float column_begin,
                                  float column_end) {
            constexpr int sample_count = 64;
            constexpr float gradient_radius = 4.0f;
            float best_score = -1.0f;
            constexpr float search_step = 2.0f;
            for (float offset = -search_radius;
                 offset <= search_radius; offset += search_step)
            {
                std::vector<float> differences;
                differences.reserve(sample_count);
                for (int sample = 0; sample < sample_count; ++sample)
                {
                    float ratio = (sample + 0.5f) / sample_count;
                    float column = column_begin +
                        (column_end - column_begin) * ratio;
                    cv::Point2f outside = local_to_image(
                        column, expected_row + offset - gradient_radius);
                    cv::Point2f inside = local_to_image(
                        column, expected_row + offset + gradient_radius);
                    float diff = color_difference(outside, inside);
                    if (diff >= 0.0f)
                        differences.push_back(diff);
                }
                if (differences.size() < sample_count * 3 / 4)
                    continue;

                std::sort(differences.begin(), differences.end());
                float median = differences[differences.size() / 2];
                float upper_quartile =
                    differences[differences.size() * 3 / 4];
                int strong_count = 0;
                for (float diff : differences)
                {
                    if (diff >= 14.0f)
                        ++strong_count;
                }
                float continuity = static_cast<float>(strong_count) /
                    differences.size();
                float score = median * 0.45f +
                    upper_quartile * 0.35f + continuity * 20.0f;
                best_score = std::max(best_score, score);
            }
            return best_score;
        };

        std::vector<RowPairScore> edge_scores;
        edge_scores.reserve(allowed_pairs.size());
        for (int pair_index : allowed_pairs)
        {
            if (pair_index < 0 ||
                pair_index + 1 >= static_cast<int>(rows.size()))
                continue;

            const float row_spacing =
                rows[pair_index + 1].center - rows[pair_index].center;
            if (row_spacing <= 1.0f)
                continue;

            std::vector<float> columns;
            for (int row_offset = 0; row_offset < 2; ++row_offset)
            {
                for (const auto& point :
                     rows[pair_index + row_offset].original_points)
                {
                    DetectionPoint local = rotatePointAroundCenter(
                        point, center_pt, radians);
                    columns.push_back(rows_are_y ? local.x : local.y);
                }
            }
            if (columns.size() < 4)
                continue;
            std::sort(columns.begin(), columns.end());

            std::vector<float> unique_columns;
            for (float column : columns)
            {
                if (unique_columns.empty() ||
                    std::abs(column - unique_columns.back()) > cell_size * 0.35f)
                    unique_columns.push_back(column);
                else
                    unique_columns.back() =
                        (unique_columns.back() + column) * 0.5f;
            }
            if (unique_columns.size() < 2)
                continue;

            std::vector<float> column_gaps;
            for (size_t c = 1; c < unique_columns.size(); ++c)
                column_gaps.push_back(unique_columns[c] - unique_columns[c - 1]);
            std::sort(column_gaps.begin(), column_gaps.end());
            const float column_spacing =
                column_gaps[column_gaps.size() / 2];
            const float column_padding = column_spacing * 0.42f;
            const float column_begin = unique_columns.front() - column_padding;
            const float column_end = unique_columns.back() + column_padding;

            constexpr float edge_offset_ratio = 0.55f;
            const float search_radius = std::max(8.0f, row_spacing * 0.16f);
            RowPairScore score;
            score.pair_index = pair_index;
            score.edge_top = boundary_score(
                rows[pair_index].center - row_spacing * edge_offset_ratio,
                search_radius, column_begin, column_end);
            score.edge_bottom = boundary_score(
                rows[pair_index + 1].center + row_spacing * edge_offset_ratio,
                search_radius, column_begin, column_end);

            float strongest = std::max(score.edge_top, score.edge_bottom);
            float secondary = std::min(score.edge_top, score.edge_bottom);
            score.edge_combined = strongest;
            if (secondary >= 0.0f)
                score.edge_combined += secondary * 0.35f;

            if (pair_scores)
            {
                auto existing = std::find_if(
                    pair_scores->begin(), pair_scores->end(),
                    [&](const RowPairScore& item) {
                        return item.pair_index == pair_index;
                    });
                if (existing == pair_scores->end())
                    pair_scores->push_back(score);
                else
                {
                    existing->edge_top = score.edge_top;
                    existing->edge_bottom = score.edge_bottom;
                    existing->edge_combined = score.edge_combined;
                }
            }
            edge_scores.push_back(score);
        }

        std::sort(edge_scores.begin(), edge_scores.end(),
            [](const RowPairScore& a, const RowPairScore& b) {
                if (a.edge_combined != b.edge_combined)
                    return a.edge_combined > b.edge_combined;
                return a.pair_index < b.pair_index;
            });

#ifdef REL_DBG
        for (const auto& score : edge_scores)
        {
            std::cout << "grid edge pair=" << score.pair_index
                      << " top=" << score.edge_top
                      << " bottom=" << score.edge_bottom
                      << " combined=" << score.edge_combined << std::endl;
        }
#endif

        constexpr float min_edge_score = 24.0f;
        constexpr float min_score_margin = 6.0f;
        if (edge_scores.empty() ||
            edge_scores.front().edge_combined < min_edge_score)
        {
            if (failure_reason) *failure_reason =
                "tray outer edge score is too low";
            return false;
        }
        if (edge_scores.size() > 1 &&
            edge_scores.front().edge_combined -
                edge_scores[1].edge_combined < min_score_margin)
        {
            if (failure_reason) *failure_reason = resolving_rect_tie
                ? "tray outer edge does not resolve rect row-pair tie"
                : "tray outer edge does not uniquely select one row pair";
            return false;
        }

        if (selected_pair)
            *selected_pair = edge_scores.front().pair_index;
        return true;
    }

    // 旁路：固定输出 8 个有序网格点，不修改旧 fitCenterPoints()。
    //
    // all_pts 可以来自多行料盘。此函数只会选择一对相邻的 4 孔行：
    // - 恰好两行时直接使用；
    // - 多于两行时优先由 selection_polygon 唯一指定目标行对；
    // - 没有 selection_polygon 时，可由 edge_image 中的料盘长边选择行对；
    // - 不允许把多行合并后再截断成两行，否则会把 3x4 误报为完整 2x4。
    bool fitOrderedGrid2x4(
        std::array<cv::Point2f, 8>* ordered_grid = nullptr,
        const std::vector<cv::Point2f>* selection_polygon = nullptr,
        std::string* failure_reason = nullptr,
        const cv::Mat* edge_image = nullptr,
        RowPairSelectionResult* selection_result = nullptr)
    {
        clusters.clear();
        list_pts.clear();
        list_pts_sort.clear();

        if (failure_reason) failure_reason->clear();
        if (selection_result) *selection_result = RowPairSelectionResult{};
        auto fail = [&](const std::string& reason) {
            if (failure_reason) *failure_reason = reason;
            if (selection_result)
            {
                selection_result->success = false;
                selection_result->failure_reason = reason;
            }
            return false;
        };

        if (cell_size <= 0 || all_pts.size() < 4)
            return fail("too few tray candidates");

        // 在料盘局部坐标系中聚类。正常情况下长边（4 列）为局部 x，
        // 行为局部 y；保留 x 方向作为旋转角退化时的兜底。
        std::vector<DetectionPoint> local_pts;
        local_pts.reserve(all_pts.size());
        for (const auto& p : all_pts)
            local_pts.push_back(rotatePointAroundCenter(p, center_pt, radians));

        std::vector<std::vector<DetectionPoint>> rows_y =
            clusterByDirection(local_pts, 'y', cell_size * 0.5f);
        std::vector<std::vector<DetectionPoint>> rows_x =
            clusterByDirection(local_pts, 'x', cell_size * 0.5f);

        // 2x4 / 3x4 中行簇数应小于列簇数。若两者相等，优先局部 y，
        // 保持与标准料盘坐标系一致。
        const bool rows_are_y = rows_y.size() <= rows_x.size();
        const auto& local_rows = rows_are_y ? rows_y : rows_x;
        if (local_rows.size() < 2)
            return fail("fewer than two tray rows");

        std::vector<LocalRow> all_rows;
        all_rows.reserve(local_rows.size());
        for (const auto& local_row : local_rows)
        {
            LocalRow row;
            for (const auto& p : local_row)
            {
                row.center += rows_are_y ? p.y : p.x;
                row.original_points.push_back(
                    rotatePointAroundCenter(p, center_pt, -radians));
            }
            row.center /= static_cast<float>(local_row.size());
            all_rows.push_back(row);
        }
        std::sort(all_rows.begin(), all_rows.end(),
            [](const LocalRow& a, const LocalRow& b) {
                return a.center < b.center;
            });

        if (selection_result)
        {
            selection_result->row_centers.reserve(all_rows.size());
            for (const auto& row : all_rows)
                selection_result->row_centers.push_back(row.center);
        }

#ifdef REL_DBG
        std::cout << "grid rows=" << all_rows.size() << " centers=[";
        for (size_t i = 0; i < all_rows.size(); ++i)
        {
            if (i) std::cout << ',';
            std::cout << all_rows[i].center;
        }
        std::cout << "]" << std::endl;
#endif

        int selected_row = 0;
        RowPairSelectionSource selection_source =
            RowPairSelectionSource::OnlyPair;
        std::vector<int> rect_tied_pairs;
        if (all_rows.size() > 2)
        {
            if (selection_polygon != nullptr && selection_polygon->size() >= 3)
            {
                int best_score = -1;
                std::vector<RowPairScore> pair_scores =
                    scoreRowPairsByPolygon(all_rows, *selection_polygon);
                std::vector<int> tied_pairs;
                for (const auto& score : pair_scores)
                {
                    if (score.rect_score > best_score)
                    {
                        best_score = score.rect_score;
                        tied_pairs.clear();
                        tied_pairs.push_back(score.pair_index);
                    }
                    else if (score.rect_score == best_score)
                        tied_pairs.push_back(score.pair_index);
                }
                rect_tied_pairs = tied_pairs;

#ifdef REL_DBG
                for (const auto& score : pair_scores)
                    std::cout << "grid rect pair=" << score.pair_index
                              << " score=" << score.rect_score << std::endl;
                std::cout << "grid rect tied_pairs=[";
                for (size_t i = 0; i < tied_pairs.size(); ++i)
                {
                    if (i) std::cout << ',';
                    std::cout << tied_pairs[i];
                }
                std::cout << "]" << std::endl;
#endif

                if (selection_result)
                    selection_result->pair_scores = pair_scores;

                if (best_score < 4)
                    return fail("rect contains too few tray candidates");

                if (tied_pairs.size() == 1)
                {
                    selected_row = tied_pairs.front();
                    selection_source = RowPairSelectionSource::Rect;
                }
                else
                {
                    if (selection_result)
                        selection_result->source =
                            RowPairSelectionSource::EdgeTieBreak;
                    if (edge_image == nullptr || edge_image->empty())
                        return fail(
                            "rect row-pair tie and edge image is unavailable");

                    std::string edge_failure_reason;
                    if (!scoreRowPairsByTrayEdges(
                            all_rows, rows_are_y, tied_pairs, *edge_image,
                            &pair_scores, &selected_row,
                            &edge_failure_reason, true))
                    {
                        if (selection_result)
                            selection_result->pair_scores = pair_scores;
                        return fail(edge_failure_reason);
                    }
                    if (selection_result)
                        selection_result->pair_scores = pair_scores;
                    selection_source = RowPairSelectionSource::EdgeTieBreak;
                }
            }
            else if (edge_image != nullptr && !edge_image->empty())
            {
                if (selection_result)
                    selection_result->source = RowPairSelectionSource::Edge;
                std::vector<int> allowed_pairs;
                std::vector<RowPairScore> pair_scores;
                for (size_t i = 0; i + 1 < all_rows.size(); ++i)
                {
                    allowed_pairs.push_back(static_cast<int>(i));
                    RowPairScore score;
                    score.pair_index = static_cast<int>(i);
                    pair_scores.push_back(score);
                }
                std::string edge_failure_reason;
                if (!scoreRowPairsByTrayEdges(
                        all_rows, rows_are_y, allowed_pairs, *edge_image,
                        &pair_scores, &selected_row,
                        &edge_failure_reason, false))
                {
                    if (selection_result)
                        selection_result->pair_scores = pair_scores;
                    return fail(edge_failure_reason);
                }
                if (selection_result)
                    selection_result->pair_scores = pair_scores;
                selection_source = RowPairSelectionSource::Edge;
            }
            else
                return fail(
                    "multiple tray row pairs; rect or outer edge is required");
        }

        if (selection_result)
        {
            selection_result->selected_pair = selected_row;
            selection_result->source = selection_source;
        }

#ifdef REL_DBG
        std::cout << "grid selected_pair=" << selected_row
                  << " source=" << rowPairSelectionSourceName(selection_source)
                  << std::endl;

        if (p_img_dbg != nullptr && !p_img_dbg->empty())
        {
            float column_begin = std::numeric_limits<float>::max();
            float column_end = std::numeric_limits<float>::lowest();
            for (const auto& row : all_rows)
            {
                for (const auto& point : row.original_points)
                {
                    DetectionPoint local = rotatePointAroundCenter(
                        point, center_pt, radians);
                    float column = rows_are_y ? local.x : local.y;
                    column_begin = std::min(column_begin, column);
                    column_end = std::max(column_end, column);
                }
            }
            column_begin -= cell_size * 0.4f;
            column_end += cell_size * 0.4f;

            auto draw_row_center = [&](int row_index,
                                       const cv::Scalar& color,
                                       int thickness) {
                if (row_index < 0 ||
                    row_index >= static_cast<int>(all_rows.size()))
                    return;
                DetectionPoint local_begin;
                DetectionPoint local_end;
                if (rows_are_y)
                {
                    local_begin.x = column_begin;
                    local_begin.y = all_rows[row_index].center;
                    local_end.x = column_end;
                    local_end.y = all_rows[row_index].center;
                }
                else
                {
                    local_begin.x = all_rows[row_index].center;
                    local_begin.y = column_begin;
                    local_end.x = all_rows[row_index].center;
                    local_end.y = column_end;
                }
                DetectionPoint image_begin = rotatePointAroundCenter(
                    local_begin, center_pt, -radians);
                DetectionPoint image_end = rotatePointAroundCenter(
                    local_end, center_pt, -radians);
                cv::line(*p_img_dbg,
                    cv::Point2f(image_begin.x, image_begin.y),
                    cv::Point2f(image_end.x, image_end.y),
                    color, thickness, cv::LINE_AA);
            };

            for (size_t i = 0; i < all_rows.size(); ++i)
                draw_row_center(static_cast<int>(i),
                    cv::Scalar(128, 128, 128), 1);
            for (int pair : rect_tied_pairs)
            {
                draw_row_center(pair, cv::Scalar(0, 255, 255), 2);
                draw_row_center(pair + 1, cv::Scalar(0, 255, 255), 2);
            }
            draw_row_center(selected_row, cv::Scalar(0, 255, 0), 3);
            draw_row_center(selected_row + 1, cv::Scalar(0, 255, 0), 3);
        }
#endif

        clusters.clear();
        for (int i = 0; i < 2; ++i)
        {
            const auto& row = all_rows[selected_row + i];
            // 多检行不能靠截断伪装成 4 格；漏检行可在已确定的行对内
            // 补齐网格，以便仍返回已确认的格位和 unknown 槽位。
            if (row.original_points.size() < 2 || row.original_points.size() > 4)
            {
                clusters.clear();
                return fail(
                    "selected tray row must contain two to four candidates");
            }
            Cluster cluster;
            cluster.points = row.original_points;
            clusters.push_back(cluster);
        }

        // 两行共用同一组列坐标。旧实现逐行独立补点；当两行恰好同时
        // 漏掉同一列时，会把一个双倍列距误当成普通列距，生成的 8 点
        // 与真实检测交错。这里先合并两行的列证据，再枚举其对应的 4 个
        // 标准列索引，用尺寸先验和料盘中心约束选择唯一解。
        std::vector<float> observed_columns;
        for (const auto& cluster : clusters)
        {
            for (const auto& p : cluster.points)
            {
                DetectionPoint local =
                    rotatePointAroundCenter(p, center_pt, radians);
                observed_columns.push_back(rows_are_y ? local.x : local.y);
            }
        }
        std::sort(observed_columns.begin(), observed_columns.end());
        std::vector<float> column_centers;
        std::vector<int> column_counts;
        const float column_merge_threshold = cell_size * 0.42f;
        for (float column : observed_columns)
        {
            if (column_centers.empty() ||
                column - column_centers.back() > column_merge_threshold)
            {
                column_centers.push_back(column);
                column_counts.push_back(1);
            }
            else
            {
                const int count = ++column_counts.back();
                column_centers.back() +=
                    (column - column_centers.back()) / count;
            }
        }
        if (column_centers.size() < 2 || column_centers.size() > 4)
            return fail("selected tray rows do not form two to four columns");

        DetectionPoint tray_center_local;
        tray_center_local.x = bbox.x + bbox.width * 0.5f;
        tray_center_local.y = bbox.y + bbox.height * 0.5f;
        tray_center_local = rotatePointAroundCenter(
            tray_center_local, center_pt, radians);
        const float expected_grid_center = rows_are_y ?
            tray_center_local.x : tray_center_local.y;

        std::vector<int> best_assignment;
        float best_origin = 0.0f;
        float best_spacing = 0.0f;
        float best_cost = std::numeric_limits<float>::max();
        std::vector<int> assignment(column_centers.size(), 0);
        std::function<void(size_t, int)> enumerate_assignments;
        enumerate_assignments = [&](size_t pos, int next_index) {
            if (pos == assignment.size())
            {
                const float n = static_cast<float>(assignment.size());
                float sum_i = 0.0f;
                float sum_x = 0.0f;
                float sum_ii = 0.0f;
                float sum_ix = 0.0f;
                for (size_t i = 0; i < assignment.size(); ++i)
                {
                    const float index = static_cast<float>(assignment[i]);
                    sum_i += index;
                    sum_x += column_centers[i];
                    sum_ii += index * index;
                    sum_ix += index * column_centers[i];
                }
                const float denom = n * sum_ii - sum_i * sum_i;
                if (std::abs(denom) < 1e-5f)
                    return;
                const float spacing =
                    (n * sum_ix - sum_i * sum_x) / denom;
                const float origin = (sum_x - spacing * sum_i) / n;
                if (spacing < cell_size * 0.65f ||
                    spacing > cell_size * 1.45f)
                    return;

                float residual = 0.0f;
                for (size_t i = 0; i < assignment.size(); ++i)
                {
                    const float predicted = origin +
                        spacing * static_cast<float>(assignment[i]);
                    const float error = column_centers[i] - predicted;
                    residual += error * error;
                }
                residual = std::sqrt(residual / n) /
                    std::max(1.0f, static_cast<float>(cell_size));
                const float spacing_penalty =
                    std::abs(spacing - static_cast<float>(cell_size)) /
                    std::max(1.0f, static_cast<float>(cell_size));
                const float center_penalty = std::abs(
                    origin + 1.5f * spacing - expected_grid_center) /
                    std::max(1.0f, static_cast<float>(cell_size));
                const float cost = residual +
                    spacing_penalty * 0.35f + center_penalty * 0.12f;
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_origin = origin;
                    best_spacing = spacing;
                    best_assignment = assignment;
                }
                return;
            }

            const int remaining = static_cast<int>(
                assignment.size() - pos - 1);
            for (int index = next_index; index <= 3 - remaining; ++index)
            {
                assignment[pos] = index;
                enumerate_assignments(pos + 1, index + 1);
            }
        };
        enumerate_assignments(0, 0);
        if (best_assignment.empty() || best_spacing <= 0.0f)
            return fail("failed to resolve four tray columns");

#ifdef REL_DBG
        std::cout << "grid columns observed=[";
        for (size_t i = 0; i < column_centers.size(); ++i)
        {
            if (i) std::cout << ',';
            std::cout << column_centers[i] << "->" << best_assignment[i];
        }
        std::cout << "] origin=" << best_origin
                  << " spacing=" << best_spacing
                  << " cost=" << best_cost << std::endl;
#endif

        for (int row_index = 0; row_index < 2; ++row_index)
        {
            auto& cluster = clusters[row_index];
            cluster.pts_ret.clear();
            for (int column_index = 0; column_index < 4; ++column_index)
            {
                const float target_column =
                    best_origin + best_spacing * column_index;
                const DetectionPoint* nearest = nullptr;
                float nearest_distance = std::numeric_limits<float>::max();
                for (const auto& p : cluster.points)
                {
                    DetectionPoint local = rotatePointAroundCenter(
                        p, center_pt, radians);
                    const float column = rows_are_y ? local.x : local.y;
                    const float distance = std::abs(column - target_column);
                    if (distance < nearest_distance)
                    {
                        nearest_distance = distance;
                        nearest = &p;
                    }
                }
                if (nearest != nullptr &&
                    nearest_distance <= best_spacing * 0.42f)
                {
                    cluster.pts_ret.push_back(*nearest);
                    continue;
                }

                DetectionPoint local_estimate;
                if (rows_are_y)
                {
                    local_estimate.x = target_column;
                    local_estimate.y = all_rows[selected_row + row_index].center;
                }
                else
                {
                    local_estimate.x = all_rows[selected_row + row_index].center;
                    local_estimate.y = target_column;
                }
                DetectionPoint estimate = rotatePointAroundCenter(
                    local_estimate, center_pt, -radians);
                estimate.id = -1;
                estimate.confidence = 0.0f;
                cluster.pts_ret.push_back(estimate);
            }
        }

        auto sort_row_by_local_column = [&](std::vector<DetectionPoint>& row) {
            std::sort(row.begin(), row.end(),
                [&](const DetectionPoint& a, const DetectionPoint& b) {
                    DetectionPoint local_a =
                        rotatePointAroundCenter(a, center_pt, radians);
                    DetectionPoint local_b =
                        rotatePointAroundCenter(b, center_pt, radians);
                    float col_a = rows_are_y ? local_a.x : local_a.y;
                    float col_b = rows_are_y ? local_b.x : local_b.y;
                    return col_a < col_b;
                });
        };

        sort_row_by_local_column(clusters[0].pts_ret);
        sort_row_by_local_column(clusters[1].pts_ret);

        list_pts_sort.clear();
        for (int i = 0; i < 4; ++i)
            list_pts_sort.emplace_back(clusters[0].pts_ret[i].x, clusters[0].pts_ret[i].y);
        for (int i = 0; i < 4; ++i)
            list_pts_sort.emplace_back(clusters[1].pts_ret[i].x, clusters[1].pts_ret[i].y);

        if (list_pts_sort.size() != 8)
            return fail("selected tray grid does not contain eight points");

        if (ordered_grid) {
            for (int i = 0; i < 8; ++i)
                (*ordered_grid)[i] = list_pts_sort[i];
        }

        std::array<cv::Point2f, 8> grid;
        for (int i = 0; i < 8; ++i)
            grid[i] = list_pts_sort[i];
        auto grab_pts = computeGrabPoints2x4(grid);
        list_pts.assign(grab_pts.begin(), grab_pts.end());
        if (selection_result)
        {
            selection_result->success = true;
            selection_result->failure_reason.clear();
        }
        return true;
    }

    // 拟合每条边的中心点
    void fitCenterPoints() {

        // 聚类
        clusterPoints();

        for (auto& cluster : clusters) {
            if (cluster.points.size() < 2) {
                std::cout << "警告：该行点数不足，跳过拟合" << std::endl;
                continue;
            }

            // 使用RANSAC拟合直线
            cluster.fittedLine = ransacFitLine(cluster.points);

            if (cluster.fittedLine.inlierCount < minInliers) {
                std::cout << "警告：内点不足，拟合可能不准确" << std::endl;
            }
        }

        if (cell_size > 0 && clusters.size() >= 2) {
            // 使用估计器
            FourPointEstimator estimator;
            for (auto& c : clusters)
            {
                std::vector<DetectionPoint> det_pts;
                for (const auto& p : c.points) {
                    if (distancePointToLine(p.x, p.y, c.fittedLine) < inlierThreshold) {
                        det_pts.push_back(p);
                    }

                    c.pts_ret = estimator.estimateFourPoints(c.fittedLine, det_pts, cell_size);
                }

            }


            if (clusters[0].pts_ret.size() == 4 && clusters[1].pts_ret.size() == 4)
            {
                DetectionPoint* dp0 = clusters[0].pts_ret.data();
                DetectionPoint* dp1 = clusters[1].pts_ret.data();
                // 选取要抓取的2个点
                list_pts.clear();

                if (bbox.width > bbox.height)
                {
                    cv::Point2f pt1((dp0[1].x + dp1[0].x) / 2, (dp0[1].y + dp1[0].y) / 2);
                    cv::Point2f pt2((dp0[2].x + dp1[3].x) / 2, (dp0[2].y + dp1[3].y) / 2);

                    list_pts.push_back(pt1);
                    list_pts.push_back(pt2);


                    list_pts_sort.push_back(cv::Point2f(dp0[1].x,dp0[1].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[2].x,dp1[2].y));
                    list_pts_sort.push_back(cv::Point2f(dp0[2].x,dp0[2].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[1].x,dp1[1].y));

                    list_pts_sort.push_back(cv::Point2f(dp0[0].x,dp0[0].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[3].x,dp1[3].y));
                    list_pts_sort.push_back(cv::Point2f(dp0[3].x,dp0[3].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[0].x,dp1[0].y));                 

                }
                else
                {
                    cv::Point2f pt1((dp0[1].x + dp1[0].x) / 2, (dp0[1].y + dp1[0].y) / 2);
                    cv::Point2f pt2((dp0[2].x + dp1[3].x) / 2, (dp0[2].y + dp1[3].y) / 2);

                    //cv::Point2f pt1 = (grid[0][0] + grid[1][1]) / 2;
                    //cv::Point2f pt2 = (grid[2][0] + grid[3][1]) / 2;

                    list_pts.push_back(pt1);
                    list_pts.push_back(pt2);


                    list_pts_sort.push_back(cv::Point2f(dp0[1].x,dp0[1].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[2].x,dp1[2].y));
                    list_pts_sort.push_back(cv::Point2f(dp0[2].x,dp0[2].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[1].x,dp1[1].y));

                    list_pts_sort.push_back(cv::Point2f(dp0[0].x,dp0[0].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[3].x,dp1[3].y));
                    list_pts_sort.push_back(cv::Point2f(dp0[3].x,dp0[3].y));
                    list_pts_sort.push_back(cv::Point2f(dp1[0].x,dp1[0].y));                        
                }
            }
        }

    }

    // 获取拟合结果
    const std::vector<Cluster>& getResults() const {
        return clusters;
    }

    // 打印结果
    void printResults() const {
        std::cout << "=== 料盘长边中心点拟合结果 ===" << std::endl;
        std::cout << "检测到总点数: " << all_pts.size() << std::endl;
        std::cout << "聚类数量: " << clusters.size() << std::endl;

        for (size_t i = 0; i < clusters.size(); ++i) {
            std::cout << "\n第 " << i + 1 << " 行:" << std::endl;
            std::cout << "  点数: " << clusters[i].points.size() << std::endl;
            std::cout << "  内点数: " << clusters[i].fittedLine.inlierCount << std::endl;
            std::cout << "  直线方程: " << clusters[i].fittedLine.a << "x + "
                << clusters[i].fittedLine.b << "y + "
                << clusters[i].fittedLine.c << " = 0" << std::endl;
            std::cout << "  中心点坐标: (" << clusters[i].centerX << ", "
                << clusters[i].centerY << ")" << std::endl;
        }
    }
};


#endif // GRID_FITTER_H
