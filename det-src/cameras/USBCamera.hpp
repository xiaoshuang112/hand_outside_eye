#ifndef USB_CAMERA_HPP
#define USB_CAMERA_HPP

#include "AICamera.hpp"
#include <opencv2/videoio.hpp>

class USBCamera : public AICamera {
public:
    explicit USBCamera(const CONFIG& config);
    ~USBCamera() override;

    bool open() override;
    void close() override;
    cv::Mat captureOneFrame(int timeoutMs = 1000) override;
    int captureOneFrameWithDepth(cv::Mat& bgrImage, cv::Mat& depthImage, int timeoutMs = 2000) override;
    bool isOpened() const override { return m_isOpened; }

    bool setColorExposureTime(int exposureUs) override;
    int getColorExposureTime() override;
    bool setAutoExposure(bool enable) override;
    bool getAutoExposure() override;
    bool getExposureRange(int& minUs, int& maxUs) override;

    bool setColorGain(int gain) override;
    int getColorGain() override;
    bool getGainRange(int& min, int& max) override;

    std::vector<cv::Mat> captureMultiExposureImages(const std::vector<int>& exposureValuesUs) override;
    std::vector<std::vector<float>> getCameraIntrinsics(const int imgWidth, const int imgHight) override;

    bool setFlip(int flip_v, int flip_h) override;

protected:
    std::vector<cv::Mat> captureMultiExposureImagesImpl(const std::vector<int>& exposureValuesUs) override;

private:
    bool m_isOpened = false;
    int m_deviceIndex = 0;
    cv::VideoCapture m_cap;
};

#endif // USB_CAMERA_HPP
