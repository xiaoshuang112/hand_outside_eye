#include "hand_eye_calibration.hpp"
#include "auto_eye_in_hand_charuco_calibration.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #include <io.h>
    #include <fcntl.h>

#else
    #include<unistd.h>
#endif

#include "my_base.h"
#include "cameras/AICamera.hpp"
#include "robotapi.h"
#include "basestruct.h"


extern std::string g_posPath;
extern std::string g_calibrationPath;

extern CONFIG g_cfg_camera;
extern std::string g_paramPath;
extern std::string g_appPath;
extern std::shared_ptr<AICamera> g_camera;
extern RobotHandle robotHandle;
extern std::string robotIp;
extern int cameraToTcpEnd;
extern AICameraIntrinsic g_cam_param;

namespace fs = std::filesystem;


std::string string_format(const char* format, ...) {
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



// 机器人控制相关的函数
namespace RobotControl {

    // 模拟从机器人控制器读取位姿
    static cv::Mat getRobotPose(HandEyeCalibration& calibrator) {
        // 这里应该从机器人控制器读取实时位姿
        // 示例：返回随时间变化的位姿
        static double angle = 0;
        angle += 5.0;
        if (angle > 360)
         angle = 0;

        return calibrator.poseEulerToTransformMatrix(
            0.5 + 0.1 * sin(angle * CV_PI / 180.0),  // x坐标有小幅变化
            0.0 + 0.1 * cos(angle * CV_PI / 180.0),  // y坐标有小幅变化
            0.3,                                      // z坐标固定
            0, 0, angle, true);                      // 绕Z轴旋转
    }

    // 显示机器人位姿信息
    static void displayRobotPoseInfo(const cv::Mat& pose_matrix) {
        if (pose_matrix.empty()) {
            std::cout << "无效的位姿矩阵" << std::endl;
            return;
        }

        double x = pose_matrix.at<double>(0, 3);
        double y = pose_matrix.at<double>(1, 3);
        double z = pose_matrix.at<double>(2, 3);

        std::cout << "机器人位姿:" << std::endl;
        std::cout << "  位置: (" << std::fixed << std::setprecision(3)
            << x << ", " << y << ", " << z << ") m" << std::endl;
        std::cout << "  位置: (" << std::fixed << std::setprecision(1)
            << x * 1000 << ", " << y * 1000 << ", " << z * 1000 << ") mm" << std::endl;
    }
}

// 相机相关的函数
namespace CameraControl {

    // 模拟从3D相机获取点云
    static std::vector<cv::Point3f> getCameraPoints() {
        // 这里应该是实际的点云获取代码
        // 示例：返回固定模式的点

        std::vector<cv::Point3f> points;

        // 模拟一个在 XY 平面上的网格点
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                points.push_back(cv::Point3f(
                    i * 0.05,
                    j * 0.05,
                    0.5  // 固定的深度
                ));
            }
        }

        return points;
    }
}

std::vector<double> parse_robot_joint_pose(const std::string& pose_str) {
    std::vector<double> values;
    std::stringstream ss(pose_str);
    std::string token;

    // 去除空格
    std::string trimmed = pose_str;
    trimmed.erase(std::remove(trimmed.begin(), trimmed.end(), ' '), trimmed.end());
    ss = std::stringstream(trimmed);

    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stod(token));
        }
        catch (const std::exception& e) {
            std::cerr << "解析位姿失败: " << e.what() << std::endl;

            values.clear();
        }
    }

    return values;
}

std::string getCurrentTime_to_path_str(std::time_t t=0) {
    std::time_t now = t;
    if (0==t)
    {
        now = std::time(nullptr);
    }
        
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
    return buf;
}


void goto_jointpos(double* jointpos)
{
    // double jointpos[6] = { 184.89, 12.75, -73, -28, 87.34, 173 }; //目标点示例数据
    double pose[6] = { 0,0,0,0,0,0 }; //在MoveJ中无影响
    PointControlParaSimple  pointControlParaSimple;
    for (int index = 0; index < ROB_AXIS_NUM; index++)
    {
        pointControlParaSimple.pose[index] = pose[index];
        pointControlParaSimple.jointpos[index] = jointpos[index];
        pointControlParaSimple.tcpOffset[index] = 0;
        pointControlParaSimple.coordinatePose[index] = 0;
        pointControlParaSimple.speed[index] = 30;
        pointControlParaSimple.acc[index] = 30;
    }
    pointControlParaSimple.coordinateType = jointCoordinate;
    pointControlParaSimple.pointTransType = pointTransStop;
    pointControlParaSimple.motiontriggerMode = MovetriggerbyOnlyRpc;
    PointControlPara  pointControlPara;
    CRresult result = cr_move_pointControlPara_transfer(robotHandle, pointControlParaSimple, &pointControlPara);
    assert(result == sucess);
    result = cr_move_joint(robotHandle, pointControlPara, TRUE);
    assert(result == sucess);
}

std::string get_pos_dir(int pos){
    return g_posPath+std::to_string(pos) + "/";
}

void convert_joint_2_pos(int pos){
   std::string pos_dir = get_pos_dir(pos);
   std::string rob_joint_file = pos_dir + "rob_joint.csv";
   std::string rob_pos_file = pos_dir + "rob_pos.csv";
   std::ifstream file(rob_joint_file);
   std::string line;
   while (std::getline(file, line)) {
        // 假设CSV格式: x,y,z,rx,ry,rz (单位mm,度)
        std::vector<double> actual_pose(ROB_AXIS_NUM);                
        std::vector<double> joint_pose = parse_robot_joint_pose(line);
        if (joint_pose.empty()) 
           continue;
        cr_kineForward(robotHandle, joint_pose.data(), actual_pose.data());
        VectorWriter writer_tcp(rob_pos_file, 6);
        writer_tcp.appendDoubleVector(actual_pose);
   }
}

void convert_pos_2_joint(int pos)
{
    std::string pos_dir = get_pos_dir(pos);
   std::string rob_joint_file = pos_dir + "rob_joint.csv";
   std::string rob_joint_file_n = pos_dir + "rob_joint_n.csv";
   std::string rob_pos_file = pos_dir + "rob_pos.csv";
   std::ifstream i_joint(rob_joint_file), i_pos(rob_pos_file);
   std::string joint_line, pos_line;
   if(fs::exists(rob_joint_file_n))
      std::remove(rob_joint_file_n.data());
   int i = 0;
   while (std::getline(i_joint, joint_line)) {
        i++;
        std::getline(i_pos, pos_line);
        // 假设CSV格式: x,y,z,rx,ry,rz (单位mm,度)
        std::vector<double> new_joint(ROB_AXIS_NUM);                
        std::vector<double> joint_pose = parse_robot_joint_pose(joint_line);
        std::vector<double> line_pose = parse_robot_joint_pose(pos_line);
        if (joint_pose.empty()) 
           continue;
        cr_kineInverse(robotHandle, line_pose.data(), joint_pose.data(), new_joint.data());
        VectorWriter writer_tcp(rob_joint_file_n, 6);
        writer_tcp.appendDoubleVector(new_joint);
   }
   i_joint.close();
   i_pos.close();
   std::string rob_joint_file_old = pos_dir + "rob_joint_old.csv";
   if(fs::exists(rob_joint_file_old))
      std::remove(rob_joint_file_old.data());
   std::rename(rob_joint_file.data(), rob_joint_file_old.data());
   std::rename(rob_joint_file_n.data(), rob_joint_file.data());
}
void convert_pos_euler_2_joint(int pos)
{
    std::string pos_dir = get_pos_dir(pos);
   std::string rob_joint_file = pos_dir + "rob_joint.csv";
   std::string rob_joint_file_n = pos_dir + "rob_joint_n.csv";
   std::string rob_eule_file = pos_dir + "rob_pos_eule.csv";
   std::string axis_angle_file = pos_dir + "rob_pos.csv";
   std::ifstream i_joint(rob_joint_file), i_eule(rob_eule_file);
   std::string joint_line, eule_line, axis_line;
   if(fs::exists(rob_joint_file_n))
      std::remove(rob_joint_file_n.data());
   std::getline(i_joint, joint_line);
   int i = 0;
   while (std::getline(i_eule, eule_line)) {
        i++;       
        // 假设CSV格式: x,y,z,rx,ry,rz (单位mm,度)
        std::vector<double> new_joint(ROB_AXIS_NUM), axisAngle_pose(ROB_AXIS_NUM);                
        std::vector<double> joint_pose = parse_robot_joint_pose(joint_line);
        std::vector<double> eule_pose = parse_robot_joint_pose(eule_line);
        cr_Eule2AxisAngle(eule_pose.data(), axisAngle_pose.data());
        if (joint_pose.empty()) 
           continue;
        CRresult ret = cr_kineInverse(robotHandle, axisAngle_pose.data(), joint_pose.data(), new_joint.data());
        if( 0 != ret)
           continue;

        VectorWriter writer_tcp(rob_joint_file_n, 6);
        writer_tcp.appendDoubleVector(new_joint);
        VectorWriter writer_eule(axis_angle_file, 6);
        writer_eule.appendDoubleVector(axisAngle_pose);// axis-angle pose
   }
   i_joint.close();
   i_eule.close();
   std::string rob_joint_file_old = pos_dir + "rob_joint_old.csv";
   if(fs::exists(rob_joint_file_old))
      std::remove(rob_joint_file_old.data());
   std::rename(rob_joint_file.data(), rob_joint_file_old.data());
   std::rename(rob_joint_file_n.data(), rob_joint_file.data());
}
void checkAndRestoreRobotHand(){
    // 检查机械臂是否上电
    enum RobotModes robotMode;
    CRresult res = cr_get_robotMode(robotHandle, &robotMode);
    if (res == sucess)
    {
        if (robotMode == JointPowerOff)
        {
            std::cout << "机械臂未上电，上电使能" << std::endl;
            res = cr_FaultReset(robotHandle);
            if (res == sucess)
            {
                std::cout << "机械臂故障复位" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            else
                std::cout << "故障复位失败" << std::endl;
            
            if (res == sucess)
                res = cr_poweron(robotHandle); // 上电
            if (res == sucess)
            {
                std::cout << "机械臂开始上电" << std::endl;
                while (res == sucess)
                {
                    res = cr_get_robotMode(robotHandle, &robotMode);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    if (robotMode == JointIdle)
                    {
                        break;
                    }
                }

                res = cr_enable(robotHandle); // 使能
                if (res == sucess)
                {
                    std::cout << "机械臂开始使能" << std::endl;
                    while (res == sucess)
                    {
                        res = cr_get_robotMode(robotHandle, &robotMode);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        if (robotMode == ProgramStop)
                        {
                            break;
                        }
                    }
                }
                else
                    std::cout << "机械臂使能失败 res: " << res << std::endl;
            }
            else
                std::cout << "机械臂上电失败 res: " << res << std::endl;
        }
    }
    else
        std::cout << "获取机械臂模式失败" << std::endl;
}

// ============================================================
// 自动手眼标定入口（菜单项 5）
// ============================================================
//void moveToJointPos(int robot_handle, const double joint[6]);

void auto_handeye_calibration(CalibrationResult& calibrat_ret, AI_det_callback callback) {

    // 尝试加载已有相机内参
    cv::Mat camera_matrix, dist_coeffs;
    std::string cam_file = g_paramPath + "camera_params.yaml";
    cv::FileStorage fs(cam_file, cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs.release();
        if (!camera_matrix.empty())
            std::cout << "[autoCalib] 加载已有相机内参: " << cam_file << std::endl;
    }


    {
        namespace fs = std::filesystem;
        if (fs::exists(g_calibrationPath))
            fs::remove_all(g_calibrationPath);
        fs::create_directories(g_calibrationPath);
    }    

    // 实时预览 — 等待用户拖到最佳位姿后按 's'
    // cv::namedWindow("Auto Calibration", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Auto Calibration", 960, 600);

    CharucoBoardConfig board_cfg;
    CharucoDetector detector(board_cfg, cam_file);

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "  请拖拽机器人到任意标定板可见的位姿" << std::endl;
    std::cout << "  系统将自动调整到最佳位姿" << std::endl;
    std::cout << "  按 's' 开始自动标定" << std::endl;
    std::cout << "  按 'q' 取消" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;
    
    checkAndRestoreRobotHand();   

    double init_tcp_pose[6] = {209.396,	-38.517, -29.509,-43.52,95.22,239.616};
    TCPMsg tcp_msg;
    bool confirmed = false;
    bool detected = false;
    // moveToJointPos(robotHandle, init_tcp_pose);   
    CRresult ret = cr_get_tcpActualPose(robotHandle, init_tcp_pose);

    while (true) {
        cv::Mat frame = g_camera->captureOneFrame(1000);
        if (frame.empty()) continue;

        // 检测 ChArUco 并叠加显示
        cv::Mat display;
        cv::Mat rvec, tvec;
        cv::Mat empty_depth;
        if (detector.detectAndDraw(frame, display, rvec, tvec)) {
            cv::putText(display, "ChArUco OK  Press 's' to start",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 0), 2);
            detected = true;
        } else {
            frame.copyTo(display);
            cv::putText(display, "No board detected",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                3, cv::Scalar(0, 0, 255), 3);
        }

        if (nullptr!=callback) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = now.time_since_epoch();
            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

            std::string file_id = std::to_string(nanoseconds);

            std::string img_name = g_calibrationPath + "/" +  file_id  + ".jpg";
            cv::imwrite(img_name, display);

            int call_ret = callback((char*)img_name.c_str(), (int)img_name.length(), 1);

            std::this_thread::sleep_for(std::chrono::milliseconds(200));  
        }
        
        if (detected) {
            // 记录初始位姿
            CRresult ret = cr_get_tcpActualPose(robotHandle, init_tcp_pose);
            if (ret != sucess) {
                std::cerr << "[autoCalib] 获取 TCP 位姿失败" << std::endl;
                continue;
            }
            ret = cr_get_currentTCPmsg(robotHandle, &tcp_msg);
            if (ret != sucess) {
                std::cerr << "[autoCalib] 获取 TCP 偏移失败" << std::endl;
                continue;
            }
            confirmed = true;

            // std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            break;
        }
    }

    // cv::destroyWindow("Auto Calibration");

    if (!confirmed) return;

    // 输出初始位姿
    std::cout << "[autoCalib] 初始 TCP 位姿 (轴角, mm/deg): ";
    for (int i = 0; i < 6; i++)
        std::cout << init_tcp_pose[i] << (i < 5 ? ", " : "\n");

    // 使用与 HandEyeCalibration 一致的默认值（7x5 方格, 40mm 边长, 28mm 标记）
    // 调用标定模块
    CalibrationResult result;
    bool ok = autoEyeInHandCharucoCalibration(
        robotHandle,
        (char*)robotIp.c_str(),
        g_camera,
        board_cfg,
        camera_matrix,
        dist_coeffs,
        cameraToTcpEnd,
        result,
        g_calibrationPath,
        cam_file,
        callback
    );



    // 输出/保存结果
    if (ok) {
        calibrat_ret = result;
        calibrat_ret.T_flange_camera = result.T_flange_camera.clone();
        calibrat_ret.camera_matrix = result.camera_matrix.clone();
        calibrat_ret.dist_coeffs = result.dist_coeffs.clone();
    

        std::cout << "\n========== 标定结果 ==========" << std::endl;
        std::cout << "手眼矩阵 T_flange_camera:\n" << result.T_flange_camera << std::endl;
        std::cout << "相机内参:\n" << result.camera_matrix << std::endl;
        std::cout << "畸变系数: " << result.dist_coeffs.t() << std::endl;
        std::cout << "有效帧数: " << result.valid_frames << std::endl;
        std::cout << "平均误差: " << (result.reproj_error * 1000.0) << " mm" << std::endl;

        // 保存到文件
        cv::FileStorage out(g_paramPath + "auto_handeye_result.yaml", cv::FileStorage::WRITE);
        out << "T_flange_camera" << result.T_flange_camera;
        out << "camera_matrix" << result.camera_matrix;
        out << "distortion_coefficients" << result.dist_coeffs;
        out << "reprojection_error_mm" << result.reproj_error * 1000.0;
        out << "valid_frames" << result.valid_frames;
        out.release();
        std::cout << "结果已保存到: auto_handeye_result.yaml" << std::endl;

    } else {
        std::cerr << "\n[autoCalib] 标定失败: " << result.message << std::endl;
    }

    //cv::destroyWindow("Auto Calibration");
}
