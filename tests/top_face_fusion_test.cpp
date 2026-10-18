#include "../det-src/top_face.cpp"
#include <cstdlib>

static void require(bool ok,const char* msg){if(!ok){std::cerr<<msg<<'\n';std::exit(1);}}
static FaceCandidate candidate(P center,double quality){Coarse o{};o.center=center;o.angle=0;Fit f;f.corners=rectangle({0,0,0,114,64},o);f.quality=quality;return {f,"test","test",0};}
int main(){
    cv::Mat image(200,200,CV_8U,cv::Scalar(40));
    CenterPrior missing;
    std::vector<FaceCandidate> choices={candidate({90,90},.75),candidate({100,90},.755)};
    require(selectCandidate(image,choices,missing).index==1,"NN failure must retain geometry maximum");
    CenterPrior stable;stable.available=true;stable.center={90,90};
    require(selectCandidate(image,choices,stable).index==0,"stable NN must rank eligible geometry");
    stable.spread=100;
    require(selectCandidate(image,choices,stable).index==1,"unstable crop prediction must lose influence");
    stable.spread=0;choices[0].fit.quality=.70;
    require(selectCandidate(image,choices,stable).index==1,"NN must not promote geometry outside .04 shortlist");
    std::vector<FaceCandidate> distinct;Model model{"test",114,64};
    addCandidate(distinct,candidate({90,90},.75).fit,model,"test");
    addCandidate(distinct,candidate({90.8,90},.75).fit,model,"test");
    require(distinct.size()==2,"different nearby geometric centers must survive deduplication");
    int calls=0;
    auto predictor=[&](const cv::Mat& crop,P& center,double& mass,double& ms){++calls;require(crop.cols<=200&&crop.rows<=200,"crop must be clamped");center={5,6};mass=.8;ms=1;return true;};
    auto prior=coarseCenterPrior(image,{1,2,20,30},predictor);
    require(prior.available&&calls==3&&prior.samples.size()==3,"three independent padded predictions required");
    require(prior.center==P(5,6)&&prior.spread==0,"clamped crop offset or component median incorrect");
    require(prior.samples[0]["crop_bbox_xyxy"]==json({0,0,29,40})&&prior.samples[2]["crop_bbox_xyxy"]==json({0,0,45,56}),"padding clamped asymmetrically");
    calls=0;require(!coarseCenterPrior(image,{-1,2,20,30},predictor).available&&calls==0,"invalid bbox must not infer");
    auto failed=coarseCenterPrior(image,{40,40,80,80},[](const cv::Mat&,P&,double&,double&){return false;});
    require(!failed.available&&failed.samples.size()==3,"NN failure must not prevent geometric fallback");
    calls=0;auto partial=coarseCenterPrior(image,{40,40,80,80},[&](const cv::Mat&,P& p,double& mass,double&){p={5,6};mass=.8;return ++calls<3;});
    require(!partial.available&&partial.samples.size()==3,"partial crop group cannot establish NN reliability");
    auto invalid=coarseCenterPrior(image,{40,40,80,80},[](const cv::Mat&,P& p,double& m,double&){p={NAN,2};m=.8;return true;});
    require(!invalid.available,"nonfinite NN output must be ignored");
    auto outside=coarseCenterPrior(image,{40,40,80,80},[](const cv::Mat& m,P& p,double& mass,double&){p={double(m.cols),2};mass=.8;return true;});
    require(!outside.available,"NN padding outside original crop must be rejected");
    for(double angle:{13.,83.,139.}){
        cv::Mat gray(256,256,CV_8U,cv::Scalar(30));Coarse o{};o.center={128,128};o.angle=angle;o.bbox={20,20,216,216};
        o.box=rectangle({0,0,0,114,64},o);std::vector<cv::Point> points;for(P p:o.box)points.emplace_back(cvRound(p.x),cvRound(p.y));cv::fillConvexPoly(gray,points,cv::Scalar(220));
        cv::Mat im=smooth(gray);std::vector<FaceCandidate> fits;
        for(double direction:candidateAngles(o)){auto search=searchOrigin(o,o.center,direction);for(const auto& seed:integralSearch(search,model,im,cv::Mat()))addCandidate(fits,refine(seed,im,model),model,"test");}
        require(!fits.empty(),"rotated synthetic rectangle unresolved");const auto chosen=selectCandidate(gray,fits,missing);const auto& fit=fits[chosen.index];
        require(cv::norm(diagonalCenter(fit.fit.corners)-P(128,128))<2,"synthetic diagonal center incorrect");
        require(std::abs(angleDiff(fit.angle,angle))<2,"orthogonal candidate selected on synthetic rectangle");
    }
    std::cout<<"PASS: fusion shortlist/reliability, geometric center retention, NN failure/nonfinite/outside, bbox/pad clamping, 3 rotated rectangles\n";
}
