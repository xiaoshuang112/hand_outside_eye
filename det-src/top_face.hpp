#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "json.hpp"

int top_face_init(const std::string& appDir);
int top_face_detect(const cv::Mat& bgrOrGray, const nlohmann::json& request, nlohmann::json& result);
void top_face_close();

// Internal conversion: false preserves legacy XY; enabled configuration errors throw.
bool top_face_tcp_xy(double u, double v, cv::Size imageSize, double& x, double& y);
// Height-aware mapping uses the actual undistorted image intrinsics, not the calibration-image K.
bool top_face_tcp_xy(double u, double v, cv::Size imageSize, double& x, double& y,
                     double workpieceHeightMm, const cv::Matx33d& imageCamera,
                     bool unflippedUndistorted = true);
double top_face_request_height(const nlohmann::json& request);
std::string top_face_init_error();
