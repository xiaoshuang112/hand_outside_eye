#include "top_face.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace fs=std::filesystem;
using json=nlohmann::json;
void require(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
struct TemporaryApp {
    fs::path path=fs::temp_directory_path()/("cylinder-top-check-"+std::to_string(getpid()));
    TemporaryApp(){fs::create_directories(path/"config/top_face");}
    ~TemporaryApp(){top_face_close();std::error_code error;fs::remove_all(path,error);}
};
void write(const fs::path& path,const json& value){std::ofstream(path)<<value.dump(2);}
json run(const cv::Mat& image,const json& request=json::object()){
    json result;
    const int code=top_face_detect(image,request,result);
    require(code==0,result.value("ret-msg",std::string("detection failed")));
    return result;
}
std::vector<json> hits(const json& result){
    std::vector<json> out;
    for(const auto& o:result.at("objects"))if(o.at("status")=="ok")out.push_back(o);
    return out;
}
void validateContract(const json& result){
    const auto objects=hits(result);
    require(result.at("top_face_count")==objects.size(),"count does not match valid objects");
    require(result.at("pts_top_face").size()==objects.size(),"pixel tuple count changed");
    for(size_t i=0;i<objects.size();++i){
        const auto& o=objects[i];
        require(o.at("center_px")==o.at("top_center_px")&&o.at("center_px")==o.at("geometric_center_px"),"center contract changed");
        require(o.at("center_source")=="circle_fit"&&o.at("shape")=="circle","not using circle geometry");
        require(o.at("angle_deg")==0&&o.at("angle_signed_deg")==0&&!o.at("angle_defined").get<bool>(),"circle heading is not fixed zero");
        require(o.at("center_model_status")=="disabled","obsolete rectangle model is active");
        require(o.at("corners_px").size()==4&&o.at("corners_definition")=="circle_bounding_box","legacy bounds field missing");
        require(o.at("apparent_length_px")==o.at("diameter_px")&&o.at("apparent_width_px")==o.at("diameter_px"),"size field is not cylinder diameter");
        const auto& tuple=result.at("pts_top_face")[i];
        require(tuple.size()==3&&tuple[0]==o.at("center_px")[0]&&tuple[1]==o.at("center_px")[1]&&tuple[2]==0,"pixel tuple correspondence changed");
    }
}

int main(int argc,char** argv){try{
    require(argc==2||argc==4,"usage: cylinder-top-face-test APP_DIR [UNDISTORTED_DATA_DIR OUTPUT_DIR]");
    TemporaryApp temp;
    const auto dir=temp.path/"config/top_face";
    json algorithm={{"workpiece_shape","circle"},{"background_image","background.png"},
                    {"alignment_mode","fixed"},{"use_center_model",true},
                    {"center_model_required",true},{"center_model","missing-rectangle-model.bin"}};
    cv::Mat background(360,480,CV_8UC3,cv::Scalar(30,30,30));
    cv::imwrite((dir/"background.png").string(),background);write(dir/"algorithm.json",algorithm);
    require(top_face_init(temp.path.string())==0,top_face_init_error());
    cv::Mat scene=background.clone();
    const std::vector<cv::Point2d> centers={{95.35,90.65},{230.6,150.15},{360.2,260.8}};
    cv::rectangle(scene,{95,73},{134,109},cv::Scalar(170,170,170),cv::FILLED);
    for(auto c:centers)cv::circle(scene,{cvRound(c.x*256),cvRound(c.y*256)},cvRound(26.2*256),cv::Scalar(235,235,235),cv::FILLED,cv::LINE_AA,8);
    auto result=run(scene);validateContract(result);auto objects=hits(result);
    cv::Mat gray;cv::cvtColor(scene,gray,cv::COLOR_BGR2GRAY);
    require(run(gray)["pts_top_face"]==result["pts_top_face"],"grayscale and BGR inputs disagree");
    require(objects.size()==3,"synthetic circles including a bright sidewall were not all detected");
    for(size_t i=0;i<centers.size();++i){
        const auto& o=objects[i];cv::Point2d c(o["center_px"][0],o["center_px"][1]);
        require(cv::norm(c-centers[i])<.8,"subpixel circle center pulled towards the sidewall");
        require(std::abs(o["radius_px"].get<double>()-26.2)<1,"incorrect fitted top radius");
    }
    const auto c=centers[1];
    result=run(scene,{{"rect",{c.x-5,c.y-5,c.x+5,c.y-5,c.x+5,c.y+5,c.x-5,c.y+5}}});
    require(result["top_face_count"]==1,"ROI did not select only the requested circle");
    require(run(background)["top_face_count"]==0,"empty frame reused old circles");
    cv::Mat distractors=background.clone();
    cv::rectangle(distractors,{60,60},{105,105},cv::Scalar(235,235,235),cv::FILLED);
    cv::circle(distractors,{200,80},26,cv::Scalar(235,235,235),2);
    cv::ellipse(distractors,{320,230},{70,16},0,0,360,cv::Scalar(220,220,220),cv::FILLED);
    cv::circle(distractors,{95,250},28,cv::Scalar(5,5,5),cv::FILLED);
    require(run(distractors)["top_face_count"]==0,"rectangle, ring, elongated reflection or dark stain was detected as a solid circular top");
    cv::Mat partial=background.clone();cv::circle(partial,{2,200},26,cv::Scalar(235,235,235),cv::FILLED);
    require(run(partial)["top_face_count"]==0,"clipped circle was accepted");
    json error;require(top_face_detect(cv::Mat(),json::object(),error)!=0&&error["top_face_count"]==0&&error["objects"].empty(),"empty input retained old detections");
    require(top_face_detect(scene,{{"rect",{0,0,1,1}}},error)!=0,"malformed ROI accepted");
    require(top_face_detect(cv::Mat(200,200,CV_8UC1),json::object(),error)!=0,"background dimension mismatch accepted");
    top_face_close();
    auto invalid=algorithm;invalid["circle_min_radius_px"]=40;invalid["circle_max_radius_px"]=20;
    write(dir/"algorithm.json",invalid);require(top_face_init(temp.path.string())!=0,"invalid circle size bounds accepted");
    invalid=algorithm;invalid["circle_max_radius_px"]=1000;
    write(dir/"algorithm.json",invalid);require(top_face_init(temp.path.string())!=0,"oversized circle radius accepted");
    std::cout<<"Synthetic cylinders, sidewalls, distractors, fixed angle, ROI and invalid inputs passed\n";

    if(argc==4){
        const fs::path data=fs::absolute(argv[2]),output=fs::absolute(argv[3]);fs::create_directories(output);
        const std::string bg="capture_20261019_163657_840_94154fa5_fused_undistorted.png";
        json fieldAlgorithm;std::ifstream(fs::path(argv[1])/"config/top_face/algorithm.json")>>fieldAlgorithm;
        fieldAlgorithm["background_image"]=(data/bg).string();
        // No NN asset is copied into this isolated runtime.
        fieldAlgorithm["use_center_model"]=true;fieldAlgorithm["center_model"]="missing-old-model.bin";
        write(dir/"algorithm.json",fieldAlgorithm);require(top_face_init(temp.path.string())==0,top_face_init_error());
        const std::vector<std::pair<std::string,std::vector<cv::Point2d>>> fixtures={
            {"capture_20261019_163254_574_3a777eab",{{462,564},{610,578},{792,560}}},
            {"capture_20261019_163356_853_2ed07fbb",{{403,728},{558,399},{845,754}}},
            {"capture_20261019_163422_677_fa7a4b1c",{{349,508},{658,728},{890,583}}},
            {"capture_20261019_163454_154_c9e8e838",{{357,798},{664,791},{1002,798}}},
            {"capture_20261019_163552_575_2771fe7f",{{172,600},{683,676},{1148,756}}},
            {"capture_20261019_163627_352_b29fd665",{}},
            {"capture_20261019_163657_840_94154fa5",{}}
        };
        json summary=json::array();
        for(const auto& fixture:fixtures){
            const std::string name=fixture.first+"_fused_undistorted.png";
            cv::Mat image=cv::imread((data/name).string());require(!image.empty(),"cannot read "+name);
            auto detected=run(image);validateContract(detected);auto circles=hits(detected);
            require(circles.size()==fixture.second.size(),"wrong cylinder count in "+name);
            json centersOut=json::array();
            for(size_t i=0;i<circles.size();++i){
                const auto& o=circles[i];cv::Point2d center(o["center_px"][0],o["center_px"][1]);
                // Rough, independently inspected image locations; not metrology ground truth.
                require(cv::norm(center-fixture.second[i])<4,"wrong top rim or sidewall chosen in "+name);
                require(o["radius_px"].get<double>()>23&&o["radius_px"].get<double>()<29,"unexpected field top diameter");
                cv::circle(image,center,cvRound(o["radius_px"].get<double>()),{0,255,0},2,cv::LINE_AA);
                cv::drawMarker(image,center,{0,0,255},cv::MARKER_CROSS,13,2);
                cv::putText(image,std::to_string(i+1)+" ("+std::to_string(cvRound(center.x))+","+std::to_string(cvRound(center.y))+")",center+cv::Point2d(-35,-36),cv::FONT_HERSHEY_SIMPLEX,.5,{0,255,255},1,cv::LINE_AA);
                centersOut.push_back(o["center_px"]);
            }
            cv::imwrite((output/name).string(),image);write(output/(fixture.first+".json"),detected);
            summary.push_back({{"image",name},{"count",circles.size()},{"centers_px",centersOut},{"elapsed_ms",detected["elapsed_ms"]}});
            std::cout<<name<<" count="<<circles.size()<<" elapsed_ms="<<detected["elapsed_ms"]<<'\n';
        }
        write(output/"summary.json",summary);
        std::cout<<"All 15 cylinders detected; both empty frames returned zero\n";
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
