#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>

/**
 * 将 BGR 图像转换为 NV12 格式
 * NV12: Y 平面 (W×H) + UV 交错平面 (W/2 × H/2 × 2)
 * 使用 BT.601 色彩空间转换
 *
 * @param bgr    输入 BGR 图像 (CV_8UC3)
 * @param nv12   输出 buffer (需预分配 W×H×3/2 字节)
 * @param w      目标宽度 (通常为模型 input_w)
 * @param h      目标高度 (通常为模型 input_h)
 * @param keep_aspect  保持宽高比 (letterbox, 灰色填充)
 * @param scale_w      出参: 水平缩放因子
 * @param scale_h      出参: 垂直缩放因子
 * @param margin_left  出参: 左边距
 * @param margin_top   出参: 上边距
 */
void bgr_to_nv12(const cv::Mat& bgr, uint8_t* nv12,
                 int w, int h, bool keep_aspect,
                 float& scale_w, float& scale_h,
                 float& margin_left, float& margin_top);
