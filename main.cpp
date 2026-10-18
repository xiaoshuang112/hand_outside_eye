#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <Windows.h>

#else
#include<unistd.h>
#endif

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread> // for std::thread
#include <chrono>

namespace fs = std::filesystem;


#define VERSION "1.0.0"
#define NOTE ""

std::string CurDir;
std::string LogDir;
std::string ExeName;
std::string g_directory = "";
std::string g_det_flag = "";
std::string g_thd_num_flag = "";
std::string g_det_huatu = "";


int main(int argc, char* argv[]) {

	std::cout << "=========================================" << std::endl;
	std::cout << "@@@@@@@@  TEST OPENCV480 TOOLS  @@@@@@@@@" << std::endl;
	std::cout << "=========================================" << std::endl;


	cv::Mat image = cv::imread("1.jpg");

	cv::putText(image, "12345678", cv::Point(100,100), cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 5, 0);

	// Display the image (optional)
	cv::imshow("Inference", image);
	cv::waitKey(0); // Wait indefinitely until a key is pressed

	// Save the output image
	std::string outputImagePath = "output_image.jpg";
	cv::imwrite(outputImagePath, image);

	return 0;
}
