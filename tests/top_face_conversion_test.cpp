// CPU integration check; compile det.cpp with -DUSE_RDK -ffunction-sections -fdata-sections and link with --gc-sections.
// No AI_det_init, camera, model or robot functions run.
#include "top_face.hpp"
#include "my_base.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using json=nlohmann::json;
namespace fs=std::filesystem;
extern int det_to_pts_top_face(cv::Mat&,const json&,json&);
extern AICameraIntrinsic g_cam_param;
extern cv::Mat g_hand_eye_matrix;
extern float g_ref_depth;
extern double g_top_face_workpiece_height_mm;
extern bool g_bgrImage_undistorted;
extern int g_top_capture;
extern bool g_clean_mode;
extern std::string g_det_type;
void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(int argc,char** argv){try{
    check(argc==2,"usage: top-face-conversion-test APP_DIR");
    fs::path source=fs::path(argv[1])/"config"/"top_face";
    fs::path app=fs::temp_directory_path()/("top-face-conversion-"+std::to_string(getpid()));
    auto dir=app/"config"/"top_face";fs::create_directories(dir);
    json algorithm,mapping;std::ifstream(source/"algorithm.json")>>algorithm;std::ifstream(source/"tcp_xy_mapping.json")>>mapping;
    algorithm["use_center_model"]=false;
    algorithm["background_image"]=fs::absolute(source/"background.bmp").string();std::ofstream(dir/"algorithm.json")<<algorithm;
    auto load=[&](bool enabled){top_face_close();mapping["enabled"]=enabled;std::ofstream(dir/"tcp_xy_mapping.json")<<mapping;check(top_face_init(app.string())==0,"init failed");};
    g_cam_param={800,810,640,512,1280,1024};g_ref_depth=1200;g_top_capture=0;g_clean_mode=false;g_det_type="top_face";
    g_top_face_workpiece_height_mm=60;g_bgrImage_undistorted=true;
    g_hand_eye_matrix=(cv::Mat_<double>(4,4)<<1,0,0,.1,0,.98,-.2,-.2,0,.2,.98,.05,0,0,0,1);
    json originalMapping=mapping;
    json projective={{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},
        {"matrix",{{.8,.02,-300},{-.03,-.7,1400},{.0002,-.0001,1}}}};
    json affine={{"enabled",true},{"model_type","affine"},{"image_size_wh",{1280,1024}},
        {"matrix",{{.8,.02,-300},{-.03,-.7,1400}}}};
    for(const auto& variant:std::vector<json>{originalMapping,affine,projective}){
    mapping=variant;g_top_capture=0;
    json detections={{"ret-code",0},{"top_face_count",2},{"objects",json::array({
        {{"status","ok"},{"center_px",{960.25,700.75}},{"angle_deg",130.},{"apparent_length_px",114.}},
        {{"status","ok"},{"center_px",{640.125,512.875}},{"angle_deg",-90.},{"apparent_length_px",115.}},
        {{"status","not_rectangular"},{"center_px",nullptr}}
    })}};
    cv::Mat image(1024,1280,CV_8UC3,cv::Scalar(0));json legacy,mapped,captured;
    load(false);check(det_to_pts_top_face(image,detections,legacy)==0,"legacy conversion failed");
    load(true);check(det_to_pts_top_face(image,detections,mapped)==0,"mapped conversion failed");
    check(mapped["pts_lunzi"]==legacy["pts_lunzi"],"camera coordinates changed");
    check(mapped["angle_deg"]==legacy["angle_deg"]&&mapped["angle_deg"]==json::array({90.,-50.}),"image angle convention/order changed");
    check(mapped["pts_top_face"]==legacy["pts_top_face"]&&mapped["pts_top_face"][0][0]==640.125,"center sorting/correspondence changed");
    for(int i=0;i<2;i++){
        const auto& base=mapped["list_pts_base"][i];const auto& old=legacy["list_pts_base"][i];
        check(base.size()==4&&base[2]==old[2]&&base[3]==old[3],"Z/length/schema changed");
        double u=mapped["pts_top_face"][i][0],v=mapped["pts_top_face"][i][1];
        double expectedX=0,expectedY=0;
        // Numeric mapping has separate tests; here check caller correspondence.
        check(top_face_tcp_xy(u,v,image.size(),expectedX,expectedY,g_top_face_workpiece_height_mm,cv::Matx33d(g_cam_param.fx,0,g_cam_param.cx,0,g_cam_param.fy,g_cam_param.cy,0,0,1)),"mapping disabled during integration check");
        check(std::abs(base[0].get<double>()-expectedX)<1e-10&&
              std::abs(base[1].get<double>()-expectedY)<1e-10,"XY precision/correspondence failed");
    }
    check(std::abs(mapped["list_pts_base"][0][2].get<double>()-1200)>1,"hand-eye Z path not exercised");
    if(mapping.value("model_type",std::string())=="calibrated_camera_ray_plane"){
        json raised;g_top_face_workpiece_height_mm=120;
        check(det_to_pts_top_face(image,detections,raised)==0,"second requested height failed");
        check(raised["workpiece_height_mm"]==120&&raised["list_pts_base"][0][0]!=mapped["list_pts_base"][0][0],"height did not update XY/diagnostics");
        check(raised["pts_lunzi"]==mapped["pts_lunzi"]&&raised["angle_deg"]==mapped["angle_deg"],"height changed camera output/angle");
        g_top_face_workpiece_height_mm=std::numeric_limits<double>::quiet_NaN();json missing;
        check(det_to_pts_top_face(image,detections,missing)!=0&&missing["ret-code"]==-1,"missing next height reused previous result");
        g_top_face_workpiece_height_mm=60;
    }
    g_top_capture=1;check(det_to_pts_top_face(image,detections,captured)==0,"top capture failed");
    for(const char* key:{"pts_lunzi","list_pts_base","angle_deg","pts_top_face"})check(captured[key].size()==1&&captured[key][0]==mapped[key][0],"top capture correspondence changed");
    cv::Mat badSize(512,640,CV_8UC3,cv::Scalar(0));json failed;
    check(det_to_pts_top_face(badSize,detections,failed)!=0&&failed["ret-code"]==-1,"mapping input dimension error did not propagate");
    }
    top_face_close();fs::remove_all(app);std::cout<<"top-face conversion integration checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
