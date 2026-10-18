// Native CPU check: g++ -std=c++17 -O2 -Iinclude -Idet-src tests/top_face_offline_test.cpp det-src/top_face.cpp $(pkg-config --cflags --libs opencv4) -o top-face-offline-test
// Run: ./top-face-offline-test APP_DIR [00197.jpg 00199.jpg 00218.jpg]
#include "top_face.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using json=nlohmann::json;
namespace fs=std::filesystem;
void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
void write(const fs::path& p,const json& j){std::ofstream(p)<<j.dump(2);}
int main(int argc,char** argv){
 try {
    check(argc==2||argc==5,"usage: top-face-offline-test APP_DIR [00197.jpg 00199.jpg 00218.jpg]");
    const fs::path source=fs::path(argv[1])/"config"/"top_face";
    const fs::path app=fs::temp_directory_path()/("top-face-check-"+std::to_string(getpid()));
    const auto dir=app/"config"/"top_face";
    fs::create_directories(dir);
    json algorithm,mapping;
    std::ifstream(source/"algorithm.json")>>algorithm;
    // Keep the legacy affine compatibility check independent of the deployed calibration.
    mapping={{"enabled",true},{"image_size_wh",{1280,1024}},{"matrix",{
        {0.7831809905663851,-0.035313363576290384,-591.5982689569211},
        {-0.0419130996690261,-0.7761799076977126,1853.9665478613795}}}};
    algorithm["use_center_model"]=false;
    algorithm["background_image"]=fs::absolute(source/"background.bmp").string();
    write(dir/"algorithm.json",algorithm);
    auto load=[&](){top_face_close();return top_face_init(app.string());};
    double x=17,y=19;
    check(load()==0,"missing mapping init failed");
    check(!top_face_tcp_xy(471.1,556.8,{1280,1024},x,y)&&x==17&&y==19,"missing mapping did not preserve XY");
    write(dir/"tcp_xy_mapping.json",{{"enabled",false},{"matrix","ignored while disabled"}});
    check(load()==0,"disabled mapping init failed");
    check(!top_face_tcp_xy(471.1,556.8,{1280,1024},x,y)&&x==17&&y==19,"disabled mapping did not preserve XY");
    write(dir/"tcp_xy_mapping.json",mapping);
    check(load()==0,"valid mapping init failed");
    check(top_face_tcp_xy(471.0922638573122,556.7930921259205,{1280,1024},x,y),"mapping not applied");
    check(std::abs(x+242.31)<1e-8&&std::abs(y-1402.05)<1e-8,"calibration pair failed");
    check(top_face_tcp_xy(472.123456789,555.987654321,{1280,1024},x,y),"fractional mapping not applied");
    check(std::abs(x+241.47394658025246)<1e-8&&std::abs(y-1402.6319441489554)<1e-8,"full precision mapping failed");
    check(top_face_tcp_xy(1014.09365449,558.10888621,{1280,1024},x,y),"extrapolation unexpectedly blocked");
    bool threw=false;
    try{top_face_tcp_xy(1,2,{640,512},x,y);}catch(const std::exception&){threw=true;}
    check(threw,"wrong input dimensions accepted");
    for(const json& bad:std::vector<json>{
        {{"enabled",true},{"image_size_wh",{1280,1024}},{"matrix",{{1,2},{3,4}}}},
        {{"enabled",true},{"image_size_wh",{640,512}},{"matrix",{{1,2,3},{4,5,6}}}},
        {{"enabled",true},{"image_size_wh",{1280,1024}},{"matrix",{{1,2,nullptr},{4,5,6}}}},
        {{"enabled","true"},{"image_size_wh",{1280,1024}},{"matrix",{{1,2,3},{4,5,6}}}}
    }){
        write(dir/"tcp_xy_mapping.json",bad);
        check(load()!=0,"invalid mapping accepted");
        check(top_face_init_error().find("tcp_xy_mapping.json")!=std::string::npos,"mapping failure lacks clear error");
    }
    std::ofstream(dir/"tcp_xy_mapping.json")<<R"({"enabled":true,"image_size_wh":[1280,1024],"matrix":[[1e999,2,3],[4,5,6]]})";
    check(load()!=0,"nonfinite mapping accepted");
    write(dir/"tcp_xy_mapping.json",mapping);
    check(load()==0,"mapping reload failed");
    if(argc==5){
        const std::vector<std::vector<cv::Point2d>> refs={
            {{471.09226386,556.79309213},{901.19304728,606.54076353},{628.84836331,804.33615653}},
            {{1014.09365449,558.10888621},{277.14548533,632.96403337}},
            {{683.46135,759.33308},{310.91709,813.83943}}
        };
        for(int frame=0;frame<3;frame++){
            json result;
            check(top_face_detect(cv::imread(argv[frame+2]),json::object(),result)==0,"image detection failed");
            check(result["top_face_count"]==refs[frame].size(),"wrong top count");
            size_t hit=0;
            for(const auto& o:result["objects"]){
                if(o["status"]!="ok")continue;
                cv::Point2d center(o["center_px"][0].get<double>(),o["center_px"][1].get<double>());
                double error=cv::norm(center-refs[frame][hit]);
                check(error<(frame==2&&hit==1?8.:3.),"automatic center differs excessively from offline reference");
                check(o["apparent_length_px"].get<double>()>97&&o["apparent_length_px"].get<double>()<130,"invalid top length");
                check(o["apparent_width_px"].get<double>()<76,"ledge included in wide envelope");
                check(o["angle_deg"].get<double>()>=0&&o["angle_deg"].get<double>()<180,"image angle convention changed");
                check(o["seed_method"]!="manual","manual seed used");
                check(o["center_source"]=="diagonal_intersection"&&o["center_px"]==o["geometric_center_px"],"non-geometric center used");
                for(const auto& side:o["sides"])check(side["support"].get<double>()>=.55,"unsupported edge accepted");
                std::cout<<"frame="<<frame<<" hit="<<hit<<" center="<<o["center_px"]<<" error_px="<<error<<" angle="<<o["angle_signed_deg"]<<'\n';
                hit++;
            }
        }
    }
    top_face_close();fs::remove_all(app);
    std::cout<<"top_face offline checks passed\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
