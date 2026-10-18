#include "USBCamera.hpp"
#include "exposure_fusion.hpp"
#include <iostream>
#include <thread>
#include <chrono>

USBCamera::USBCamera(const CONFIG& config) : AICamera(config) {
    // cam_sn 用作设备索引 (e.g. "0" for /dev/video0)
    if (!config.cam_sn.empty()) {
        try {
            m_deviceIndex = std::stoi(config.cam_sn);
        } catch (...) {
            m_deviceIndex = 0;
        }
    }
}

USBCamera::~USBCamera() {
    if (m_isOpened) close();
}

bool USBCamera::open() {
    if (m_isOpened) return true;

    m_cap.open(m_deviceIndex, cv::CAP_V4L2);
    if (!m_cap.isOpened()) {
        // 尝试用默认后端
        m_cap.open(m_deviceIndex);
        if (!m_cap.isOpened()) {
            std::cerr << "[USBCamera] Failed to open device " << m_deviceIndex << std::endl;
            return false;
        }
    }

    // 设置分辨率
    if (m_config.cam_width > 0 && m_config.cam_height > 0) {
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, m_config.cam_width);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_config.cam_height);
    }

    // 设置 FPS
    if (m_config.fps > 0) {
        m_cap.set(cv::CAP_PROP_FPS, m_config.fps);
    }

    // 关闭自动曝光，改为手动控制
    m_cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);  // 0 = manual

    m_isOpened = true;
    std::cout << "[USBCamera] Opened device " << m_deviceIndex
              << " (" << (int)m_cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << (int)m_cap.get(cv::CAP_PROP_FRAME_HEIGHT) << ")" << std::endl;
    return true;
}

void USBCamera::close() {
    if (m_isOpened) {
        m_cap.release();
        m_isOpened = false;
    }
}

cv::Mat USBCamera::captureOneFrame(int timeoutMs) {
    if (!m_isOpened) return cv::Mat();

    cv::Mat frame;

    // 用循环实现超时重试
    auto start = std::chrono::steady_clock::now();
    while (true) {
        m_cap >> frame;
        if (!frame.empty()) break;

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeoutMs) {
            std::cerr << "[USBCamera] captureOneFrame timeout" << std::endl;
            return cv::Mat();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return frame;
}

int USBCamera::captureOneFrameWithDepth(cv::Mat& bgrImage, cv::Mat& depthImage, int timeoutMs) {
    bgrImage = captureOneFrame(timeoutMs);
    depthImage = cv::Mat();  // USB 摄像头无深度
    return bgrImage.empty() ? -1 : 0;
}

// ========== 曝光控制 ==========

bool USBCamera::setColorExposureTime(int exposureUs) {
    if (!m_isOpened) return false;
    // V4L2 的曝光单位可能不是微秒；OpenCV 中的 CAP_PROP_EXPOSURE 通常在 V4L2 下以 100us 或 1ms 为单位
    // 这里假设单位为 100us，可酌情调整
    double value = exposureUs / 100.0;
    bool ok = m_cap.set(cv::CAP_PROP_EXPOSURE, value);
    if (ok) m_exposureCached = exposureUs;
    return ok;
}

int USBCamera::getColorExposureTime() {
    if (!m_isOpened) return -1;
    double value = m_cap.get(cv::CAP_PROP_EXPOSURE);
    if (value < 0) return -1;
    return (int)(value * 100);  // 转换回 us
}

bool USBCamera::setAutoExposure(bool enable) {
    if (!m_isOpened) return false;
    // 0 = manual, 1 = auto (V4L2 约定)
    bool ok = m_cap.set(cv::CAP_PROP_AUTO_EXPOSURE, enable ? 1.0 : 0.0);
    if (ok) m_autoExposureCached = enable;
    return ok;
}

bool USBCamera::getAutoExposure() {
    if (!m_isOpened) return false;
    double value = m_cap.get(cv::CAP_PROP_AUTO_EXPOSURE);
    return (value > 0.5);
}

bool USBCamera::getExposureRange(int& minUs, int& maxUs) {
    // OpenCV 不提供曝光范围查询，返回典型值
    minUs = 1;      // ~1us
    maxUs = 500000; // ~500ms
    return true;
}

// ========== 增益控制 ==========

bool USBCamera::setColorGain(int gain) {
    if (!m_isOpened) return false;
    return m_cap.set(cv::CAP_PROP_GAIN, (double)gain);
}

int USBCamera::getColorGain() {
    if (!m_isOpened) return -1;
    double value = m_cap.get(cv::CAP_PROP_GAIN);
    return (int)value;
}

bool USBCamera::getGainRange(int& min, int& max) {
    min = 0;
    max = 100;
    return true;
}

// ========== 多曝光融合 ==========

std::vector<cv::Mat> USBCamera::captureMultiExposureImages(const std::vector<int>& exposureValuesUs) {
    return captureMultiExposureImagesImpl(exposureValuesUs);
}

std::vector<cv::Mat> USBCamera::captureMultiExposureImagesImpl(const std::vector<int>& exposureValuesUs) {
    std::vector<cv::Mat> images;
    if (!m_isOpened) return images;

    bool wasAuto = getAutoExposure();
    setAutoExposure(false);
    int prevExposure = getColorExposureTime();

    for (int exposure : exposureValuesUs) {
        setColorExposureTime(exposure);
        // 等待曝光生效
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cv::Mat frame = captureOneFrame(500);
        if (!frame.empty()) {
            images.push_back(frame);
        }
    }

    // 恢复
    setColorExposureTime(prevExposure);
    if (wasAuto) setAutoExposure(true);

    return images;
}

// ========== 内参 ==========

std::vector<std::vector<float>> USBCamera::getCameraIntrinsics(const int imgWidth, const int imgHight) {
    // USB 摄像头无内置内参，返回空
    (void)imgWidth;
    (void)imgHight;
    return {};
}

// ========== 翻转 ==========

bool USBCamera::setFlip(int flip_v, int flip_h) {
    // OpenCV VideoCapture 不支持翻转设置
    (void)flip_v;
    (void)flip_h;
    std::cerr << "[USBCamera] flip not supported" << std::endl;
    return false;
}
