// CPU check with a controlled NN test double; production NN implementation is unchanged.
// g++ -std=c++17 -O2 -Iinclude -Idet-src tests/top_face_center_mapping_test.cpp det-src/top_face.cpp $(pkg-config --cflags --libs opencv4) -o top-face-center-mapping-test
// Run: ./top-face-center-mapping-test MATCHING_APP_DIR [00197.jpg]
#include "top_face.hpp"
#include "center_heatmap_infer.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using json=nlohmann::json;
namespace fs=std::filesystem;
int nnMode=0; // 0=success,1=false predict,2=throw predict,3=false load,4=throw load,5=nonfinite predict.
bool CenterHeatmapInfer::load(const std::string& path,int,int){if(nnMode==4)throw std::runtime_error("test load exception");if(nnMode==3)return false;path_=path;backend_="test_double";impl_=this;return true;}
bool CenterHeatmapInfer::ready()const{return impl_!=nullptr;}
bool CenterHeatmapInfer::predict(const cv::Mat&,cv::Point2d& center,double* mass)const{
    if(nnMode==2)throw std::runtime_error("test predict exception");if(nnMode==1)return false;
    center={.25,.75};if(nnMode==5)center.x=std::numeric_limits<double>::infinity();if(mass)*mass=2;return true;
}
void CenterHeatmapInfer::close(){impl_=nullptr;}
void check(bool value,const char* msg){if(!value)throw std::runtime_error(msg);}
void save(const fs::path& p,const json& value){std::ofstream(p)<<value.dump(2);}
int main(int argc,char** argv){try{
    check(argc==2||argc==3,"usage: top-face-center-mapping-test APP_DIR [IMAGE]");
    fs::path source=fs::absolute(argv[1])/"config"/"top_face";
    fs::path app=fs::temp_directory_path()/("top-face-center-mapping-"+std::to_string(getpid())),dir=app/"config"/"top_face";
    fs::create_directories(dir);
    json algorithm;std::ifstream(source/"algorithm.json")>>algorithm;
    algorithm["background_image"]=fs::absolute(source/"background.bmp").string();algorithm["use_center_model"]=false;
    save(dir/"algorithm.json",algorithm);
    auto load=[&](const json& mapping){top_face_close();save(dir/"tcp_xy_mapping.json",mapping);return top_face_init(app.string());};
    json affine={{"enabled",true},{"image_size_wh",{1280,1024}},{"matrix",{{.8,.02,-300},{-.03,-.7,1400}}}};
    double x=17,y=19,u=471.123456789,v=556.987654321;
    check(load(affine)==0,"legacy affine load failed");check(top_face_tcp_xy(u,v,{1280,1024},x,y),"affine disabled");
    check(std::abs(x-(.8*u+.02*v-300))<1e-10&&std::abs(y-(-.03*u-.7*v+1400))<1e-10,"legacy affine numeric changed");
    affine["model_type"]="affine";check(load(affine)==0,"explicit affine failed");
    json hom={{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},{"matrix",{{.8,.02,-300},{-.03,-.7,1400},{.0002,-.0001,1}}}};
    double expectedX=(.8*u+.02*v-300)/(.0002*u-.0001*v+1),expectedY=(-.03*u-.7*v+1400)/(.0002*u-.0001*v+1);
    for(double scale:{1.,1e-250,-1e250,1e-15}){
        json scaled=hom;for(auto& row:scaled["matrix"])for(auto& value:row)value=value.get<double>()*scale;
        check(load(scaled)==0,"scale-equivalent homography rejected");
        check(top_face_tcp_xy(u,v,{1280,1024},x,y),"homography disabled");
        check(std::abs(x-expectedX)<1e-8&&std::abs(y-expectedY)<1e-8,"homogeneous division/scale failed");
    }
    check(load(hom)==0,"homography reload failed");
    check(top_face_tcp_xy(1700.25,1300.75,{1280,1024},x,y),"extrapolation blocked");
    for(cv::Point2d p:{cv::Point2d(-5000,0),cv::Point2d(-5000+1e-9,0),cv::Point2d(std::numeric_limits<double>::infinity(),0)}){
        x=17;y=19;bool failed=false;try{top_face_tcp_xy(p.x,p.y,{1280,1024},x,y);}catch(const std::exception&){failed=true;}
        check(failed&&x==17&&y==19,"unsafe denominator/input accepted or mutated outputs");
    }
    for(json bad:std::vector<json>{
        {{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},{"matrix",{{1,0,0},{0,1,0}}}},
        {{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},{"matrix",{{1,0,0},{2,0,0},{0,0,1}}}},
        {{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},{"matrix",{{1,0,0},{0,1,0},{.002,0,-1}}}},
        {{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},{"matrix",{{1,0,1},{0,1,0},{.001,0,0}}}},
        {{"enabled",true},{"model_type","homography"},{"image_size_wh",{1280,1024}},{"matrix",{{1,0,nullptr},{0,1,0},{0,0,1}}}},
        {{"enabled",true},{"model_type","unknown"},{"image_size_wh",{1280,1024}},{"matrix",{{1,0,0},{0,1,0}}}}
    }){
        check(load(bad)!=0,"invalid homography accepted");check(top_face_init_error().find("tcp_xy_mapping.json")!=std::string::npos,"mapping error not clear");
    }
    json overflow=hom;overflow["matrix"]={{1,1,0},{0,1,0},{0,0,1}};
    check(load(overflow)==0,"finite nonsingular homography rejected");bool failed=false;x=17;y=19;
    try{top_face_tcp_xy(1.7e308,1.7e308,{1280,1024},x,y);}catch(const std::exception&){failed=true;}
    check(failed&&x==17&&y==19,"nonfinite projected XY accepted or mutated outputs");
    json ray={{"enabled",true},{"model_type","ray_plane"},{"image_size_wh",{1280,1024}},
        {"undistorted_camera_matrix",{{1000,0,640},{0,1000,512},{0,0,1}}},
        {"camera_to_base_rotation",{{1,0,0},{0,-1,0},{0,0,-1}}},
        {"reference_top_base_z_mm",244.6},{"current_top_base_z_mm",244.6},
        {"effective_camera_base_z_mm",812.8328821212844},{"xy_offset_mm",{40.17518861452055,1279.5135905558327}},
        {"radial_residual_mm",20.58666314522111}};
    // Synthetic axis-aligned rays have an independent closed-form expected XY.
    check(load(ray)==0,"ray plane load failed");
    check(top_face_tcp_xy(740,712,{1280,1024},x,y),"ray plane disabled");
    double sameX=40.17518861452055+(812.8328821212844-244.6)*.1-20.58666314522111*.1*.05;
    double sameY=1279.5135905558327-(812.8328821212844-244.6)*.2+20.58666314522111*.2*.05;
    check(std::abs(x-sameX)<1e-8&&std::abs(y-sameY)<1e-8,"same height ray formula changed");
    ray["current_top_base_z_mm"]=304.6;check(load(ray)==0,"raised plane rejected");
    check(top_face_tcp_xy(740,712,{1280,1024},x,y),"raised ray plane disabled");
    check(std::abs(x-sameX+6)<1e-8&&std::abs(y-sameY-12)<1e-8,"60mm top change not equal to 60*q");
    for(cv::Point2d point:{cv::Point2d(-.1,0),cv::Point2d(1280,0),cv::Point2d(0,1024),cv::Point2d(0,std::numeric_limits<double>::infinity()),cv::Point2d(std::numeric_limits<double>::quiet_NaN(),0)}){
        x=17;y=19;bool rejected=false;try{top_face_tcp_xy(point.x,point.y,{1280,1024},x,y);}catch(const std::exception&){rejected=true;}
        check(rejected&&x==17&&y==19,"ray invalid pixel changed outputs");
    }
    x=17;y=19;bool sizeRejected=false;try{top_face_tcp_xy(740,712,{640,512},x,y);}catch(const std::exception&){sizeRejected=true;}
    check(sizeRejected&&x==17&&y==19,"ray wrong image size changed outputs");
    std::vector<json> invalid;
    auto bad=ray;bad["undistorted_camera_matrix"][0][0]=0;invalid.push_back(bad);
    bad=ray;bad["undistorted_camera_matrix"][0][1]=1;invalid.push_back(bad);
    bad=ray;bad["undistorted_camera_matrix"][2][2]=2;invalid.push_back(bad);
    bad=ray;bad["camera_to_base_rotation"][0][0]=2;invalid.push_back(bad);
    bad=ray;bad["camera_to_base_rotation"]={{1,0,0},{0,1,0},{0,0,-1}};invalid.push_back(bad);
    bad=ray;bad["camera_to_base_rotation"]={{1,0,0},{0,1,0},{0,0,1}};invalid.push_back(bad);
    bad=ray;bad["camera_to_base_rotation"]={{0,0,1},{0,1,0},{-1,0,0}};invalid.push_back(bad);
    bad=ray;bad["current_top_base_z_mm"]=900;invalid.push_back(bad);
    bad=ray;bad["reference_top_base_z_mm"]=900;invalid.push_back(bad);
    bad=ray;bad["radial_residual_mm"]=10000;invalid.push_back(bad);
    bad=ray;bad["xy_offset_mm"][0]=nullptr;invalid.push_back(bad);
    bad=ray;bad["image_size_wh"]={640,512};invalid.push_back(bad);
    for(const auto& badMapping:invalid){
        check(load(badMapping)!=0,"invalid ray config accepted");
        x=17;y=19;bool rejected=false;try{top_face_tcp_xy(740,712,{1280,1024},x,y);}catch(const std::exception&){rejected=true;}
        check(rejected&&x==17&&y==19,"failed ray init retained runtime or changed outputs");
    }
    bad=ray;bad["radial_residual_mm"]=0;check(load(bad)==0,"zero radial residual rejected");
    // Fitted fixture is checked against independently generated Python values.
    const fs::path fixture=fs::path(__FILE__).parent_path()/"ray_plane_reference.json";
    check(fs::exists(fixture),"missing fitted ray reference fixture");
    {
        json reference;std::ifstream(fixture)>>reference;
        check(reference.at("points").size()==8,"fitted reference must contain eight points");
        check(load(reference.at("mapping"))==0,"fitted ray config rejected");
        for(const auto& point:reference.at("points")){
            check(top_face_tcp_xy(point["uv"][0],point["uv"][1],{1280,1024},x,y),"fitted ray disabled");
            check(std::abs(x-point["xy"][0].get<double>())<1e-8&&std::abs(y-point["xy"][1].get<double>())<1e-8,"fitted ray differs from Python reference");
        }
        std::cout<<"8 fitted Python ray reference points agree within 1e-8 mm\n";
    }
    // Same rays in original-K and production alpha=1 pixels must map identically at each height.
    {
        json reference;std::ifstream(fs::path(__FILE__).parent_path()/"coin_height_reference.json")>>reference;
        const auto coin=reference.at("mapping");check(load(coin)==0,"coin height model rejected");
        for(const auto& point:reference.at("points")){
            cv::Matx33d camera;
            for(int row=0;row<3;row++)for(int col=0;col<3;col++)camera(row,col)=point["image_camera"][row][col].get<double>();
            check(top_face_tcp_xy(point["uv"][0],point["uv"][1],{1280,1024},x,y,point["height_mm"],camera),"coin model disabled");
            check(std::abs(x-point["xy"][0].get<double>())<1e-8&&std::abs(y-point["xy"][1].get<double>())<1e-8,"coin height/K conversion differs from Python");
        }
        cv::Matx33d camera(740.42150408757766,0,615.92682315059858,0,740.86801562742789,530.66704271527044,0,0,1);
        for(double height:{-1.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),585.5378234020575,1000.}){
            bool rejected=false;x=17;y=19;try{top_face_tcp_xy(640,512,{1280,1024},x,y,height,camera);}catch(const std::exception&){rejected=true;}
            check(rejected&&x==17&&y==19,"invalid/missing height mutated XY");
        }
        bool rejected=false;try{top_face_tcp_xy(640,512,{1280,1024},x,y);}catch(const std::exception&){rejected=true;}
        check(rejected,"legacy call silently supplied zero height to coin model");
        rejected=false;try{top_face_tcp_xy(640,512,{1280,1024},x,y,60,camera,false);}catch(const std::exception&){rejected=true;}
        check(rejected,"raw/flipped pixels accepted");
        auto badCamera=camera;badCamera(0,0)=0;rejected=false;
        try{top_face_tcp_xy(640,512,{1280,1024},x,y,60,badCamera);}catch(const std::exception&){rejected=true;}
        check(rejected,"invalid live intrinsics accepted");
        for(json value:std::vector<json>{nullptr,true,"60",-1}){
            rejected=false;try{top_face_request_height(json{{"workpiece_height_mm",value}});}catch(const std::exception&){rejected=true;}
            check(rejected,"invalid request height accepted");
        }
        check(top_face_request_height(json{{"workpiece_height_mm",60}})==60,"request height not parsed");
        check(std::isnan(top_face_request_height(json::object())),"missing height reused previous value");
        check(top_face_request_height(json{{"workpiece_height_mm",0}})==0,"coin/table height zero rejected");
        auto invalidCoin=coin;invalidCoin["camera_center_base_mm"][2]=coin.at("table_base_z_mm");
        check(load(invalidCoin)!=0,"camera at table height accepted");
        std::cout<<"coin model Python references, live intrinsics, request heights and invalid inputs passed\n";
    }
    json disabled={{"enabled",false},{"model_type","homography"},{"matrix","ignored"}};
    check(load(disabled)==0,"disabled mapping failed");x=17;y=19;check(!top_face_tcp_xy(u,v,{1280,1024},x,y)&&x==17&&y==19,"disabled mapping changed XY");
    if(argc==3){
        cv::Mat image=cv::imread(argv[2]);check(!image.empty(),"cannot read image");json baseline;
        check(top_face_detect(image,json::object(),baseline)==0&&baseline["top_face_count"].get<int>()>0,"baseline image failed");
        check(load(ray)==0,"ray detection config rejected");json rayDetected;
        check(top_face_detect(image,json::object(),rayDetected)==0&&rayDetected["pts_top_face"]==baseline["pts_top_face"]&&rayDetected["objects"]==baseline["objects"],"ray mapping changed detector geometry or output");
        algorithm["use_center_model"]=true;algorithm["center_model_required"]=true;algorithm["center_model"]="test-center.onnx";
        std::ofstream(dir/"test-center.onnx")<<"test double";
        for(nnMode=0;nnMode<=5;nnMode++){
            save(dir/"algorithm.json",algorithm);check(load(disabled)==0,"optional NN load made init fail");json detected;
            check(top_face_detect(image,json::object(),detected)==0,"NN failure made geometric detection fail");
            if(nnMode!=0)check(detected["top_face_count"]==baseline["top_face_count"]&&detected["pts_top_face"]==baseline["pts_top_face"],"failed NN changed geometric fallback");
            for(size_t i=0;i<detected["objects"].size();i++){
                const auto& o=detected["objects"][i];if(o["status"]!="ok")continue;
                check(o["center_source"]=="diagonal_intersection"&&o["center_px"]==o["geometric_center_px"]&&o["center_px"]==o["top_center_px"],"center is not geometric");
                if(nnMode!=0)check(o["center_px"]==baseline["objects"][i]["center_px"]&&o["corners_px"]==baseline["objects"][i]["corners_px"],"failed NN changed geometry");
                const auto& q=o["corners_px"];
                auto point=[&](int k){return cv::Vec3d(q[k][0].get<double>(),q[k][1].get<double>(),1);};
                auto p=point(0).cross(point(2)).cross(point(1).cross(point(3)));
                check(std::abs(p[0]/p[2]-o["center_px"][0].get<double>())<1e-8&&std::abs(p[1]/p[2]-o["center_px"][1].get<double>())<1e-8,"center differs from diagonal intersection");
                if(nnMode==0){check(o["center_model_status"]=="ok"&&o.contains("center_model_center_px")&&o.contains("center_model_offset_px"),"NN diagnostic missing");check(o["center_model_center_px"]!=o["center_px"],"test NN failed to propose different center");}
                else check(o["center_model_status"]!="ok","NN failure reported as success");
            }
        }
        nnMode=0;save(dir/"algorithm.json",algorithm);check(load(disabled)==0,"NN reload failed");
        const auto& center=baseline["pts_top_face"][0];double cx=center[0],cy=center[1];
        json roi={{"rect",{cx-8,cy-8,cx+8,cy-8,cx+8,cy+8,cx-8,cy+8}}};json inside;
        check(top_face_detect(image,roi,inside)==0&&inside["top_face_count"]==1,"NN changed geometric ROI membership");
        algorithm["center_model"]="missing-center.onnx";save(dir/"algorithm.json",algorithm);check(load(disabled)==0,"missing required NN made init fail");
        json detected;check(top_face_detect(image,json::object(),detected)==0&&detected["pts_top_face"]==baseline["pts_top_face"],"missing NN changed valid detection");
    }
    top_face_close();fs::remove_all(app);std::cout<<"diagonal center, optional fusion prior, legacy mapping and ray plane checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
