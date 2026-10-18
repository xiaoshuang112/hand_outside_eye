#include "nv12_utils.h"
#include <cmath>
#include <algorithm>

// BT.601 full range: Y = 0.299R + 0.587G + 0.114B
// We use integer arithmetic for speed
static inline uint8_t rgb_to_y(uint8_t r, uint8_t g, uint8_t b) {
    // Y = ( 66R + 129G + 25B + 128) >> 8 + 16  (studio swing)
    // Y = ( 77R + 150G + 29B + 128) >> 8       (full swing, standard)
    // Let's use BT.601 full-range coefficients
    return (uint8_t)((77 * r + 150 * g + 29 * b + 128) >> 8);
}

static inline uint8_t rgb_to_u(uint8_t r, uint8_t g, uint8_t b) {
    // U = (-43R - 85G + 128B + 128) >> 8 + 128
    return (uint8_t)(((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128);
}

static inline uint8_t rgb_to_v(uint8_t r, uint8_t g, uint8_t b) {
    // V = (128R - 107G - 21B + 128) >> 8 + 128
    return (uint8_t)(((128 * r - 107 * g - 21 * b + 128) >> 8) + 128);
}

void bgr_to_nv12(const cv::Mat& bgr, uint8_t* nv12,
                 int w, int h, bool keep_aspect,
                 float& scale_w, float& scale_h,
                 float& margin_left, float& margin_top)
{
    int src_w = bgr.cols;
    int src_h = bgr.rows;

    // 计算缩放和 letterbox
    float scale = 1.0f;
    if (keep_aspect) {
        scale = std::min((float)w / src_w, (float)h / src_h);
        margin_left = (w - src_w * scale) * 0.5f;
        margin_top  = (h - src_h * scale) * 0.5f;
    } else {
        scale_w = (float)w / src_w;
        scale_h = (float)h / src_h;
        scale = 1.0f;  // not used
        margin_left = 0;
        margin_top = 0;
    }

    scale_w = scale;
    scale_h = scale;

    // Resize to target size first
    cv::Mat resized;
    if (keep_aspect && (src_w != w || src_h != h)) {
        int new_w = (int)(src_w * scale);
        int new_h = (int)(src_h * scale);
        cv::resize(bgr, resized, cv::Size(new_w, new_h));
    } else if (!keep_aspect && (src_w != w || src_h != h)) {
        cv::resize(bgr, resized, cv::Size(w, h));
    } else {
        resized = bgr;
    }

    uint8_t* y_plane  = nv12;
    uint8_t* uv_plane = nv12 + w * h;

    int roi_x = (int)margin_left;
    int roi_y = (int)margin_top;
    int roi_w = resized.cols;
    int roi_h = resized.rows;

    // Fill Y plane: letterbox area = 0 (black, full-range), ROI from resized image
    for (int dy = 0; dy < h; dy++) {
        uint8_t* y_row = y_plane + dy * w;
        if (dy < roi_y || dy >= roi_y + roi_h) {
            // Padding row — fill with black (Y=16)
            memset(y_row, 0, w);
        } else {
            int sy = dy - roi_y;
            const uint8_t* src_row = resized.ptr<uint8_t>(sy);
            for (int dx = 0; dx < w; dx++) {
                if (dx < roi_x || dx >= roi_x + roi_w) {
                    y_row[dx] = 0;  // padding
                } else {
                    int sx = dx - roi_x;
                    const uint8_t* px = src_row + sx * 3;
                    y_row[dx] = rgb_to_y(px[2], px[1], px[0]);  // BGR -> RGB order
                }
            }
        }
    }

    // Fill UV plane (subsampled 2x2)
    int uv_w = w / 2;
    int uv_h = h / 2;
    int uv_roi_x = roi_x / 2;
    int uv_roi_y = roi_y / 2;
    int uv_roi_w = roi_w / 2;
    int uv_roi_h = roi_h / 2;

    for (int dy = 0; dy < uv_h; dy++) {
        uint8_t* uv_row = uv_plane + dy * w;  // UV interleaved: U,V,U,V... (w bytes = w/2 U + w/2 V)
        if (dy < uv_roi_y || dy >= uv_roi_y + uv_roi_h) {
            // Padding row — fill with gray (U=128, V=128)
            memset(uv_row, 128, w);
        } else {
            int sy = dy * 2 - roi_y;
            if (sy < 0) sy = 0;
            const uint8_t* src_row0 = resized.ptr<uint8_t>(std::min(sy, resized.rows - 1));
            const uint8_t* src_row1 = resized.ptr<uint8_t>(std::min(sy + 1, resized.rows - 1));

            for (int dx = 0; dx < uv_w; dx++) {
                if (dx < uv_roi_x || dx >= uv_roi_x + uv_roi_w) {
                    uv_row[dx * 2]     = 128;
                    uv_row[dx * 2 + 1] = 128;
                } else {
                    int sx = dx * 2 - roi_x;
                    if (sx < 0) sx = 0;
                    // Average 2x2 block for UV
                    const uint8_t* p00 = src_row0 + std::min(sx, resized.cols - 1) * 3;
                    const uint8_t* p01 = src_row0 + std::min(sx + 1, resized.cols - 1) * 3;
                    const uint8_t* p10 = src_row1 + std::min(sx, resized.cols - 1) * 3;
                    const uint8_t* p11 = src_row1 + std::min(sx + 1, resized.cols - 1) * 3;

                    // Average R,G,B over 2x2
                    int r = ((int)p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                    int g = ((int)p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                    int b = ((int)p00[0] + p01[0] + p10[0] + p11[0]) / 4;

                    uv_row[dx * 2]     = rgb_to_u(r, g, b);
                    uv_row[dx * 2 + 1] = rgb_to_v(r, g, b);
                }
            }
        }
    }
}
