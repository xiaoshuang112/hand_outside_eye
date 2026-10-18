// Loads the real ai-det library/BPU. Never calls AI_det_init, AI_det, or camera/robot functions.
#include "top_face.hpp"
#include "my_base.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
using json=nlohmann::json;
extern int det_to_pts_top_face(cv::Mat&,const json&,json&);
extern AICameraIntrinsic g_cam_param;
extern cv::Mat g_hand_eye_matrix;
extern std::vector<cv::Point2f> g_undistortedPoints;
extern bool g_undistort_ready;
extern int g_undistort_enable;
extern float g_ref_depth;
extern int g_top_capture;
extern bool g_clean_mode;
extern std::string g_det_type;
int main(int argc,char** argv){try{
    if(argc<5)throw std::runtime_error("usage: top-face-x3-offline-runner APP_DIR CAMERA_HAND_EYE_YAML_OR_- REF_DEPTH_MM IMAGE...");
    cv::Mat k;
    if(std::string(argv[2])!="-"){
        cv::FileStorage fs(argv[2],cv::FileStorage::READ);
        if(!fs.isOpened())throw std::runtime_error("cannot read calibration YAML");
        fs["camera_matrix"]>>k;
        if(k.rows!=3||k.cols!=3||k.type()!=CV_64F||k.at<double>(0,0)<=0||k.at<double>(1,1)<=0)
            throw std::runtime_error("invalid camera_matrix");
        if(!fs["T_base_camera"].empty())fs["T_base_camera"]>>g_hand_eye_matrix;
        else if(!fs["T_flange_camera"].empty())fs["T_flange_camera"]>>g_hand_eye_matrix;
        if(!g_hand_eye_matrix.empty()&&(g_hand_eye_matrix.rows!=4||g_hand_eye_matrix.cols!=4||g_hand_eye_matrix.type()!=CV_64F))
            throw std::runtime_error("invalid hand-eye matrix");
        g_ref_depth=std::stof(argv[3]);
        if(!std::isfinite(g_ref_depth)||g_ref_depth<100)throw std::runtime_error("invalid REF_DEPTH_MM");
        g_cam_param={float(k.at<double>(0,0)),float(k.at<double>(1,1)),float(k.at<double>(0,2)),float(k.at<double>(1,2)),1280,1024};
        // Match AI_det_init -> init_undistort -> ensure_undistort_map exactly:
        // calibration K enters float camera fields before alpha=1 newK, then
        // newK enters float fields again. Saved images are already undistorted.
        cv::Mat d;fs["distortion_coefficients"]>>d;
        g_undistort_enable=1;
        g_undistort_ready=!d.empty();
        if(!d.empty()){
            cv::Mat originalK=(cv::Mat_<double>(3,3)<<g_cam_param.fx,0,g_cam_param.cx,0,g_cam_param.fy,g_cam_param.cy,0,0,1);
            cv::Mat newK=cv::getOptimalNewCameraMatrix(originalK,d,{1280,1024},1.0,{1280,1024});
            g_cam_param.fx=float(newK.at<double>(0,0));g_cam_param.fy=float(newK.at<double>(1,1));
            g_cam_param.cx=float(newK.at<double>(0,2));g_cam_param.cy=float(newK.at<double>(1,2));
        }
        // Production clears the table for undistorted inputs so DEPTH_INFO
        // uses the same (pixel-c)*z/f pinhole branch, without double correction.
        g_undistortedPoints.clear();
    }
    if(top_face_init(argv[1]))throw std::runtime_error(top_face_init_error());
    g_clean_mode=false;g_det_type="top_face";g_top_capture=0;
    for(int i=4;i<argc;i++){
        cv::Mat image=cv::imread(argv[i]);
        if(image.empty()||image.size()!=cv::Size(1280,1024))throw std::runtime_error("input must be existing undistorted1280x1024 image");
        json detected;
        int rc=top_face_detect(image,json::object(),detected);
        json output={{"image",argv[i]},{"detection",detected}};
        if(rc){std::cout<<output.dump()<<'\n';return 1;}
        if(detected.value("feature_backend",std::string())!="horizon_x3_bpu")throw std::runtime_error("library is not the X3 BPU build");
        if(!k.empty()){
            json converted;
            if(det_to_pts_top_face(image,detected,converted))throw std::runtime_error(converted.value("ret-msg",std::string("conversion failed")));
            output["conversion"]=converted;
            g_top_capture=1;json first;
            if(det_to_pts_top_face(image,detected,first))throw std::runtime_error("top_capture conversion failed");
            g_top_capture=0;
            if(converted["list_pts_base"].size()>0)
                for(const char* key:{"list_pts_base","pts_lunzi","pts_top_face","angle_deg"})
                    if(first[key].size()!=1||first[key][0]!=converted[key][0])throw std::runtime_error("top_capture correspondence failed");
            output["top_capture_1"]=first;
        }else{
            output["mapped_tcp_xy_mm"]=json::array();
            for(const auto& o:detected["objects"]){
                if(o["status"]!="ok")continue;
                double x=0,y=0;
                bool enabled=top_face_tcp_xy(o["center_px"][0],o["center_px"][1],image.size(),x,y);
                output["mapped_tcp_xy_mm"].push_back({{"id",o["id"]},{"enabled",enabled},{"xy",{x,y}}});
            }
        }
        std::cout<<output.dump()<<'\n';
    }
    top_face_close();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';top_face_close();return 1;}}
