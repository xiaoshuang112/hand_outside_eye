#include "top_face.hpp"
#include "center_heatmap_infer.hpp"
#include "json.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <queue>
#include <stdexcept>
#ifdef USE_X3_NPU
#include "x3_features.h"
#endif

namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
using P = cv::Point2d;
using Quad = std::array<P, 4>;
constexpr double pi = 3.14159265358979323846;
#ifdef USE_X3_NPU
constexpr const char* featureBackend="horizon_x3_bpu";
#else
constexpr const char* featureBackend="cpu";
#endif
double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
double angleDiff(double a, double b) { return std::remainder(a-b, 180.); }
double axisAngle(P p) { double a=std::atan2(p.y,p.x)*180/pi; return a<0?a+180:a; }
double median(std::vector<double> a) {
    if(a.empty()) return 0;
    std::sort(a.begin(),a.end());
    return (a[(a.size()-1)/2]+a[a.size()/2])/2;
}
// Match OpenCV INTER_LINEAR's 1/32-pixel interpolation used by Python calipers.
double sample(const cv::Mat& im, P p) {
    double x=std::nearbyint(clamp(p.x,0,im.cols-1)*32)/32;
    double y=std::nearbyint(clamp(p.y,0,im.rows-1)*32)/32;
    int ix=int(x), iy=int(y), jx=std::min(ix+1,im.cols-1), jy=std::min(iy+1,im.rows-1);
    double dx=x-ix,dy=y-iy;
    return (1-dy)*((1-dx)*im.at<float>(iy,ix)+dx*im.at<float>(iy,jx))
         +dy*((1-dx)*im.at<float>(jy,ix)+dx*im.at<float>(jy,jx));
}
cv::Mat smooth(const cv::Mat& a, double sigma=.85) {
    cv::Mat f,r; a.convertTo(f,CV_32F); cv::GaussianBlur(f,r,cv::Size(),sigma); return r;
}
cv::Vec3d lineFit(const std::vector<P>& points, double tolerance=.001) {
    std::vector<cv::Point2f> ps; for(P p:points) ps.emplace_back(p);
    cv::Vec4f l; cv::fitLine(ps,l,cv::DIST_HUBER,0,tolerance,tolerance);
    return cv::Vec3d(-l[1],l[0],l[1]*l[2]-l[0]*l[3]);
}
P intersect(cv::Vec3d a,cv::Vec3d b) {
    auto p=a.cross(b);
    if(std::abs(p[2])<1e-9) throw std::runtime_error("degenerate quadrilateral");
    return P(p[0]/p[2],p[1]/p[2]);
}
cv::Vec3d through(P a,P b) { return cv::Vec3d(a.y-b.y,b.x-a.x,a.x*b.y-a.y*b.x); }
json pointsJson(const Quad& c) {
    json a=json::array(); for(P p:c) a.push_back({p.x,p.y}); return a;
}
struct Model { std::string name; double length,width; };
// Specular bloom inflates the difference blob. Reject only undersized debris;
// a larger envelope still has to contain the known rectangle. refine() keeps
// the 0.86–1.14 model-size lock so returned millimetres stay on the part, not the halo.
bool envelopeFits(const Model& m,double length,double width) {
    return length>=.82*m.length&&length<=2.5*m.length&&width>=.75*m.width&&width<=2.5*m.width&&m.length<=1.05*length&&m.width<=1.10*width;
}
struct Side { double support,rms; int scans,inliers; };
struct Fit { Quad corners; std::array<Side,4> sides; double quality=-1e9; };
struct Coarse { std::vector<cv::Point> contour; cv::Rect bbox; Quad box; P center; double length,width,angle; bool partial; std::vector<cv::Vec4f> lines; };
double alongSpan(const Model& m,const Coarse& o) {
    return std::min(std::max(.2*m.length,.5*std::max(0.,o.length-m.length)+8),.8*m.length);
}
double acrossSpan(const Model& m,const Coarse& o) {
    return std::min(std::max(.6*m.width,.5*std::max(0.,o.width-m.width)+8),1.2*m.width);
}
struct Runtime {
    fs::path app;
    fs::path backgroundPath;
    fs::file_time_type backgroundTime;
    uintmax_t backgroundBytes=0;
    unsigned backgroundGeneration=0;
    bool autoReload=true,alignBackground=false;
    bool circles=true;
    double circleMinRadius=18, circleMaxRadius=38, circleThreshold=30;
    double circleMinIntensity=105, circleTolerance=1.3, circleCoverage=.75, circleFill=.75;
    double circleFixedAngle=0;
    cv::Mat background,descriptors;
    cv::Mat backgroundFloat,backgroundSmall,backgroundMean,backgroundVariance,backgroundSmooth;
    std::vector<cv::KeyPoint> keys;
    cv::Ptr<cv::ORB> orb=cv::ORB::create(2500);
    std::vector<Model> models={{"small_rectangle",207,96},{"large_rectangle",294,157}};
    int iterations=90;
    bool fast=true;
#ifdef USE_X3_NPU
    X3Features npu;
#endif
    bool tcpXYEnabled=false;
    cv::Size tcpXYImageSize;
    cv::Matx23d tcpXY;
    bool tcpXYHomography=false;
    cv::Matx33d tcpXYH;
    bool tcpXYRayPlane=false, tcpXYHeightInput=false;
    cv::Matx33d tcpXYCamera, tcpXYRotation;
    cv::Vec2d tcpXYOffset;
    double tcpXYPlaneDifference=0, tcpXYRadialResidual=0;
    json profile;
    json geometry;
    double difference=15,residual=10,correlation=.6;
    bool useCenterModel=false;
    bool centerModelRequired=false;
    std::string centerModelPath;
    std::string centerModelError;
    int centerModelImageSize=256;
    int centerModelDecodeRadius=4;
    CenterHeatmapInfer centerModel;
};
// ponytail: requests serialize around the shared background/ORB state;
// use per-instance contexts if multiple cameras need concurrent inference.
std::mutex runtimeMutex;
std::unique_ptr<Runtime> runtime;
thread_local std::string reply;

bool loadBackground(Runtime& r,bool force=false) {
    auto time=fs::last_write_time(r.backgroundPath);auto bytes=fs::file_size(r.backgroundPath);
    if(!force&&r.backgroundGeneration&&time==r.backgroundTime&&bytes==r.backgroundBytes)return false;
    cv::Mat background=cv::imread(r.backgroundPath.string(),cv::IMREAD_GRAYSCALE);
    if(background.empty())throw std::runtime_error("cannot read background: "+r.backgroundPath.string());
    if(time!=fs::last_write_time(r.backgroundPath)||bytes!=fs::file_size(r.backgroundPath))throw std::runtime_error("background changed while loading; replace it atomically and retry");
#ifdef USE_X3_NPU
    if(!r.circles&&background.size()!=cv::Size(1280,1024))throw std::runtime_error("X3 feature model requires 1280x1024 background");
#endif
    cv::Mat bgFloat,bgSmall,mean,variance;
    background.convertTo(bgFloat,CV_32F);
    auto bgSmooth=smooth(background);
    if(r.fast)cv::resize(bgFloat,bgSmall,{},.5,.5,cv::INTER_AREA);else bgSmall=bgFloat;
    int window=r.fast?7:15;
    cv::boxFilter(bgSmall,mean,-1,{window,window});
    cv::boxFilter(bgSmall.mul(bgSmall),variance,-1,{window,window});
    variance-=mean.mul(mean);cv::max(variance,1,variance);
    // Commit only a completely read image. Missing/partial files return an error.
    r.background=background;r.backgroundTime=time;r.backgroundBytes=bytes;r.backgroundGeneration++;
    r.backgroundFloat=bgFloat;r.backgroundSmall=bgSmall;r.backgroundMean=mean;
    r.backgroundVariance=variance;r.backgroundSmooth=bgSmooth;
    r.keys.clear();r.descriptors.release();
    if(r.alignBackground){
        cv::Mat feature=background;if(r.fast)cv::resize(background,feature,{},.5,.5,cv::INTER_AREA);
        r.orb=cv::ORB::create(r.fast?1200:2500);r.orb->detectAndCompute(feature,cv::noArray(),r.keys,r.descriptors);
    }
    return true;
}

std::vector<Coarse> foreground(Runtime& r,const cv::Mat& image,cv::Mat& normalized,cv::Mat& aligned) {
    r.profile=json::object();
    auto mark=std::chrono::steady_clock::now();
    auto tick=[&](const char* name){auto now=std::chrono::steady_clock::now();r.profile[name]=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;};
    if(image.size()!=r.background.size()) throw std::runtime_error("background/image size mismatch");
    cv::Mat affine=cv::Mat::eye(2,3,CV_64F);
    if(r.alignBackground&&!r.descriptors.empty()){
    std::vector<cv::KeyPoint> keys; cv::Mat descriptors;
    cv::Mat featureImage;
    if(r.fast)cv::resize(image,featureImage,{},.5,.5,cv::INTER_AREA);else featureImage=image;
    r.orb->detectAndCompute(featureImage,cv::noArray(),keys,descriptors);
    if(descriptors.empty()) throw std::runtime_error("insufficient features for optional alignment; use fixed alignment for a stationary camera");
    std::vector<std::vector<cv::DMatch>> matches;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(r.descriptors,descriptors,matches,2);
    std::vector<cv::Point2f> a,b;
    for(const auto& ms:matches) if(ms.size()==2&&ms[0].distance<.72*ms[1].distance) {
        a.push_back(r.keys[ms[0].queryIdx].pt); b.push_back(keys[ms[0].trainIdx].pt);
    }
    if(a.size()<15) throw std::runtime_error("insufficient background matches");
    cv::Mat inliers;
    affine=cv::estimateAffinePartial2D(a,b,inliers,cv::RANSAC,r.fast?.75:1.5);
    if(affine.empty()||cv::countNonZero(inliers)<15) throw std::runtime_error("background alignment failed");
    if(r.fast){affine.at<double>(0,2)*=2;affine.at<double>(1,2)*=2;}
    }
    tick("background_alignment");
    double scale=std::hypot(affine.at<double>(0,0),affine.at<double>(1,0));
    if(scale<.95||scale>1.05||std::abs(affine.at<double>(0,2))>40||std::abs(affine.at<double>(1,2))>40)
        throw std::runtime_error("background moved outside alignment limits; recapture background");
    if(r.alignBackground&&!r.descriptors.empty())cv::warpAffine(r.background,aligned,affine,image.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    else aligned=r.background;
    bool fixed=!(r.alignBackground&&!r.descriptors.empty());
    cv::Mat bg,im;if(fixed)bg=r.backgroundFloat;else aligned.convertTo(bg,CV_32F);
    image.convertTo(im,CV_32F);
    double gain=1,bias=0;std::vector<double> offsets;
    for(int y=0;y<im.rows;y+=16)for(int x=0;x<im.cols;x+=16)offsets.push_back(im.at<float>(y,x)-bg.at<float>(y,x));
    bias=median(offsets);
    for(int iter=0;iter<5;++iter) {
        double sx=0,sy=0,sxx=0,sxy=0,n=0;
        for(int y=0;y<im.rows;y+=8) for(int x=0;x<im.cols;x+=8) {
            double bv=bg.at<float>(y,x),iv=im.at<float>(y,x);
            if(std::abs(iv-(gain*bv+bias))<15) { sx+=bv;sy+=iv;sxx+=bv*bv;sxy+=bv*iv;n++; }
        }
        if(n<100)throw std::runtime_error("insufficient unchanged background pixels for exposure compensation");
        if((n*sxx-sx*sx)/(n*n)<1){gain=1;bias=(sy-sx)/n;}
        else{gain=(n*sxy-sx*sy)/(n*sxx-sx*sx);bias=(sy-gain*sx)/n;}
    }
    if(gain<.5||gain>2) throw std::runtime_error("exposure outside supported range");
    im=(im-bias)/gain;
    tick("warp_photometry");
    cv::Mat segIm=im,segBg=bg;
    if(r.fast){cv::resize(im,segIm,{},.5,.5,cv::INTER_AREA);if(fixed)segBg=r.backgroundSmall;else cv::resize(bg,segBg,{},.5,.5,cv::INTER_AREA);}
    cv::Mat featureIm,featureBg;
    if(r.fast){
        featureIm=segIm.clone();featureBg=segBg.clone();
#ifdef USE_X3_NPU
        r.npu.run(featureIm,featureBg);
#else
        // Same fixed convolution as the BPU model, for the desktop preview.
        featureIm-=128;featureBg-=128;
        cv::GaussianBlur(featureIm,featureIm,{3,3},0,0,cv::BORDER_CONSTANT);
        cv::GaussianBlur(featureBg,featureBg,{3,3},0,0,cv::BORDER_CONSTANT);
        featureIm+=128;featureBg+=128;
#endif
    }
    tick("image_features");
    auto box=[](const cv::Mat& f) { cv::Mat z;cv::boxFilter(f,z,-1,{15,15});return z; };
    auto localBox=[&](const cv::Mat& f){if(!r.fast)return box(f);cv::Mat z;cv::boxFilter(f,z,-1,{7,7});return z;};
    cv::Mat mb=fixed?r.backgroundMean:localBox(segBg),mi=localBox(segIm);
    cv::Mat vb=fixed?r.backgroundVariance:localBox(segBg.mul(segBg))-mb.mul(mb),vi=localBox(segIm.mul(segIm))-mi.mul(mi);
    cv::max(vb,1,vb);cv::max(vi,1,vi);
    cv::Mat cov=localBox(segBg.mul(segIm))-mb.mul(mi),diff;
    if(r.fast)cv::absdiff(featureIm,featureBg,diff);
    else{cv::absdiff(segIm,segBg,diff);cv::GaussianBlur(diff,diff,{5,5},0);}
    cv::Mat mask=cv::Mat::zeros(segIm.size(),CV_8U);
    for(int y=0;y<mask.rows;++y) for(int x=0;x<mask.cols;++x) {
        float bv=vb.at<float>(y,x),iv=vi.at<float>(y,x),c=cov.at<float>(y,x);
        float residual=float(r.fast?7:r.residual),corr=float(r.correlation);
        bool decorrelated=c<0||c*c<corr*corr*bv*iv;
        bool changedTexture=iv*bv-c*c>residual*residual*bv&&decorrelated;
        // A smooth/clipped part can replace background texture without adding
        // local variance. Strong decorrelated differences still identify it.
        bool strongContrast=bv<=16&&iv<=4&&diff.at<float>(y,x)>3*r.difference&&decorrelated;
        // Low-texture floors and smooth parts need direct difference evidence.
        if(diff.at<float>(y,x)>r.difference&&(bv<=1.05||changedTexture||strongContrast))
            mask.at<uchar>(y,x)=255;
    }
    cv::morphologyEx(mask,mask,cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_ELLIPSE,r.fast?cv::Size(3,3):cv::Size(7,7)));
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,r.fast?cv::Size(11,11):cv::Size(23,23)));
    tick("texture_segmentation");
    im.convertTo(normalized,CV_8U);
    std::vector<std::vector<cv::Point>> contours;cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    std::vector<Coarse> objects;
    auto lsd=cv::createLineSegmentDetector();
    for(auto& c:contours) {
        if(r.fast)for(auto& p:c)p*=2;
        auto rect=cv::minAreaRect(c);
        if(cv::contourArea(c)<=800||std::min(rect.size.width,rect.size.height)<=25) continue;
        cv::Point2f corners[4];rect.points(corners);Quad quad;
        for(int i=0;i<4;i++) quad[i]=corners[i];
        P longest;double length=0,width=1e9;
        for(int i=0;i<4;i++) {P d=quad[(i+1)%4]-quad[i];double len=cv::norm(d);if(len>length){length=len;longest=d;}width=std::min(width,len);}
        double angle=axisAngle(longest);
        auto bbox=cv::boundingRect(c);
        cv::Rect roi=(bbox+cv::Size(20,20)-cv::Point(10,10))&cv::Rect(0,0,image.cols,image.rows);
        bool eligible=false;
        for(const auto& m:r.models)if(envelopeFits(m,length,width))eligible=true;
        std::vector<cv::Vec4f> segments;if(!r.fast||eligible){cv::Mat tile=normalized(roi);if(r.fast)cv::resize(tile,tile,{},.5,.5,cv::INTER_AREA);lsd->detect(tile,segments);if(r.fast)for(auto& l:segments)l*=2;}
        std::vector<std::pair<double,double>> votes;
        for(auto l:segments) {P d(l[2]-l[0],l[3]-l[1]);double len=cv::norm(d),ang=axisAngle(d);if(len>30&&std::abs(angleDiff(ang,angle))<25)votes.emplace_back(ang,len);}
        double best=-1,peak=angle;
        for(auto v:votes) {double score=0;for(auto z:votes)score+=z.second*std::exp(-.5*std::pow(angleDiff(z.first,v.first)/2,2));if(score>best){best=score;peak=v.first;}}
        double sum=0,weight=0;
        for(auto v:votes) if(std::abs(angleDiff(v.first,peak))<3){sum+=v.second*angleDiff(v.first,peak);weight+=v.second;}
        if(weight) {angle=std::fmod(peak+sum/weight+180,180);}
        bool partial=bbox.x<=2||bbox.y<=2||bbox.br().x>=image.cols-2||bbox.br().y>=image.rows-2;
        for(auto& l:segments){l[0]+=roi.x;l[2]+=roi.x;l[1]+=roi.y;l[3]+=roi.y;}
        objects.push_back({c,bbox,quad,rect.center,length,width,angle,partial,std::move(segments)});
    }
    std::sort(objects.begin(),objects.end(),[](const auto& a,const auto& b){return std::tie(a.bbox.y,a.bbox.x)<std::tie(b.bbox.y,b.bbox.x);});
    tick("coarse_geometry");
    return objects;
}

Quad rectangle(const std::array<double,5>& p,const Coarse& o) {
    double theta=o.angle*pi/180;P u(std::cos(theta),std::sin(theta)),v(-u.y,u.x);
    double a=p[2]*pi/180;P axis=std::cos(a)*u+std::sin(a)*v,normal(-axis.y,axis.x);
    P center=o.center+p[0]*u+p[1]*v;
    return {center-.5*p[3]*axis-.5*p[4]*normal,center+.5*p[3]*axis-.5*p[4]*normal,
            center+.5*p[3]*axis+.5*p[4]*normal,center-.5*p[3]*axis+.5*p[4]*normal};
}
double searchScore(const Quad& corners,const cv::Mat& im,const cv::Mat& bg,const Model& m,double l,double w,int samples=80) {
    double total=0,minimum=1;
    for(int i=0;i<4;i++) {
        P e=corners[(i+1)%4]-corners[i];P n(e.y,-e.x);n*=1/cv::norm(n);double score=0;
        for(int j=0;j<samples;j++) {
            P p=corners[i]+(.08+.84*j/(samples-1))*e;
            double g=sample(im,p+2*n)-sample(im,p-2*n),gb=sample(bg,p+2*n)-sample(bg,p-2*n);
            score+=clamp((std::abs(g)-1)/8,0,1)*(.4+.6*clamp((std::abs(g-gb)-1)/6,0,1));
        }
        score/=samples;total+=score;minimum=std::min(minimum,score);
    }
    double prior=std::pow((l-m.length)/(.06*m.length),2)+std::pow((w-m.width)/(.10*m.width),2);
    return .72*total/4+.28*minimum-.025*prior;
}
Quad search(const Coarse& o,const Model& m,const cv::Mat& im,const cv::Mat& bg,int iterations) {
    // Same bounded differential-evolution family as Python; deterministic seed.
    const double xSpan=alongSpan(m,o),ySpan=acrossSpan(m,o);
    const std::array<double,5> lo={-xSpan,-ySpan,-4,.91*m.length,.89*m.width};
    const std::array<double,5> hi={ xSpan, ySpan, 4,1.09*m.length,1.11*m.width};
    std::mt19937 rng(31);std::uniform_real_distribution<double> uniform(0,1);
    std::array<std::array<double,5>,45> population;std::array<double,45> scores;
    int best=0;
    for(int d=0;d<5;d++) {
        std::array<int,45> perm;std::iota(perm.begin(),perm.end(),0);std::shuffle(perm.begin(),perm.end(),rng);
        for(int i=0;i<45;i++)population[i][d]=lo[d]+(perm[i]+uniform(rng))/45*(hi[d]-lo[d]);
    }
    for(int i=0;i<45;i++){scores[i]=searchScore(rectangle(population[i],o),im,bg,m,population[i][3],population[i][4]);if(scores[i]>scores[best])best=i;}
    for(int it=0;it<iterations;it++) {
        double mutation=.5+.5*uniform(rng);
        for(int i=0;i<45;i++) {
            int a,b;do{a=int(rng()%45);}while(a==i||a==best);do{b=int(rng()%45);}while(b==i||b==best||b==a);
            auto trial=population[i];int forced=rng()%5;
            for(int d=0;d<5;d++)if(d==forced||uniform(rng)<.7){trial[d]=population[best][d]+mutation*(population[a][d]-population[b][d]);if(trial[d]<lo[d]||trial[d]>hi[d])trial[d]=lo[d]+uniform(rng)*(hi[d]-lo[d]);}
            double score=searchScore(rectangle(trial,o),im,bg,m,trial[3],trial[4]);
            if(score>=scores[i]){population[i]=trial;scores[i]=score;if(score>scores[best])best=i;}
        }
        double mean=std::accumulate(scores.begin(),scores.end(),0.)/45,variance=0;
        for(double s:scores)variance+=(s-mean)*(s-mean)/45;
        if(std::sqrt(variance)<.0001*std::abs(mean))break;
    }
    return rectangle(population[best],o);
}

std::vector<Quad> integralSearch(const Coarse& o,const Model& m,const cv::Mat& im,const cv::Mat& bg) {
    // The same four-edge objective is accumulated along rows/columns once.
    // Each candidate then costs four prefix-sum queries, rather than 1280 samples.
    struct Candidate {double score;Quad quad;};
    std::vector<Candidate> winners;
    const double xSpan=alongSpan(m,o),ySpan=acrossSpan(m,o);
    const int cx=int(std::ceil(.5*1.03*m.length+xSpan))+4,cy=int(std::ceil(.5*1.10*m.width+ySpan))+4;
    const int cols=2*cx+1,rows=2*cy+1;
    for(double degrees:{0.}) {
        double theta=(o.angle+degrees)*pi/180;P u(std::cos(theta),std::sin(theta)),v(-u.y,u.x);
        cv::Mat transform=(cv::Mat_<double>(2,3)<<u.x,v.x,o.center.x-cx*u.x-cy*v.x,u.y,v.y,o.center.y-cx*u.y-cy*v.y);
        cv::Mat ri,rb;cv::warpAffine(im,ri,transform,{cols,rows},cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);if(!bg.empty())cv::warpAffine(bg,rb,transform,{cols,rows},cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
        cv::Mat bi,bb;cv::copyMakeBorder(ri,bi,2,2,2,2,cv::BORDER_REPLICATE);if(!rb.empty())cv::copyMakeBorder(rb,bb,2,2,2,2,cv::BORDER_REPLICATE);
        auto evidence=[&](bool horizontal){
            cv::Rect plus(horizontal?2:4,horizontal?4:2,cols,rows),minus(horizontal?2:0,horizontal?0:2,cols,rows);
            cv::Mat g=bi(plus)-bi(minus),a;
            cv::absdiff(g,cv::Scalar(0),a);
            a=(a-1)/8;cv::max(a,0,a);cv::min(a,1,a);
            if(bb.empty())return a;
            cv::Mat gb=bb(plus)-bb(minus),d;cv::absdiff(g,gb,d);
            d=(d-1)/6;cv::max(d,0,d);cv::min(d,1,d);
            cv::Mat result=a.mul(.4+.6*d);return result;
        };
        cv::Mat he=evidence(true),ve=evidence(false);
        cv::Mat hp=cv::Mat::zeros(rows,cols+1,CV_32F),vp=cv::Mat::zeros(rows+1,cols,CV_32F);
        for(int y=0;y<rows;y++){
            auto* h=hp.ptr<float>(y);const auto* e=he.ptr<float>(y);
            for(int x=0;x<cols;x++)h[x+1]=h[x]+e[x];
            auto* next=vp.ptr<float>(y+1);const auto* prev=vp.ptr<float>(y);const auto* vertical=ve.ptr<float>(y);
            for(int x=0;x<cols;x++)next[x]=prev[x]+vertical[x];
        }
        for(double ls:{.97,1.,1.03})for(double ws:{.94,1.02,1.10}) {
            double length=ls*m.length,width=ws*m.width;
            double prior=std::pow((length-m.length)/(.06*m.length),2)+std::pow((width-m.width)/(.10*m.width),2);
            Candidate best{-1e9,{}};
            struct Range {double shift;int low,high,start,end;};
            std::vector<Range> xs,ys;
            for(double dx=-xSpan;dx<=xSpan;dx+=2){int lo=int(std::lround(cx+dx-length/2)),hi=int(std::lround(cx+dx+length/2));xs.push_back({dx,lo,hi,lo+int(.08*length),hi-int(.08*length)});}
            for(double dy=-ySpan;dy<=ySpan;dy+=2){int lo=int(std::lround(cy+dy-width/2)),hi=int(std::lround(cy+dy+width/2));ys.push_back({dy,lo,hi,lo+int(.08*width),hi-int(.08*width)});}
            for(const auto& y:ys)for(const auto& x:xs) {
                int x0=x.low,x1=x.high,y0=y.low,y1=y.high,a=x.start,b=x.end,c=y.start,d=y.end;
                if(x0<0||y0<0||x1>=cols||y1>=rows||b<=a||d<=c)continue;
                std::array<double,4> scores={(hp.at<float>(y0,b+1)-hp.at<float>(y0,a))/(b-a+1),
                    (vp.at<float>(d+1,x1)-vp.at<float>(c,x1))/(d-c+1),
                    (hp.at<float>(y1,b+1)-hp.at<float>(y1,a))/(b-a+1),
                    (vp.at<float>(d+1,x0)-vp.at<float>(c,x0))/(d-c+1)};
                double score=.72*std::accumulate(scores.begin(),scores.end(),0.)/4+.28* *std::min_element(scores.begin(),scores.end())-.025*prior;
                if(score>best.score){P center=o.center+x.shift*u+y.shift*v;best={score,{center-.5*length*u-.5*width*v,center+.5*length*u-.5*width*v,center+.5*length*u+.5*width*v,center-.5*length*u+.5*width*v}};}
            }
            winners.push_back(best);
        }
    }
    std::sort(winners.begin(),winners.end(),[](const auto& a,const auto& b){return a.score>b.score;});
    std::vector<Quad> result;
    for(const auto& candidate:winners) {
        P center=(candidate.quad[0]+candidate.quad[2])*.5;bool duplicate=false;
        for(const auto& q:result)if(cv::norm(center-(q[0]+q[2])*.5)<2&&std::abs(angleDiff(axisAngle(candidate.quad[1]-candidate.quad[0]),axisAngle(q[1]-q[0])))<1)duplicate=true;
        if(!duplicate)result.push_back(candidate.quad);
        if(result.size()==6)break;
    }
    return result;
}

struct Edge {cv::Vec3d line;double offset;};
std::vector<Quad> proposals(const cv::Mat& image,const cv::Mat& im,const cv::Mat& bg,const Coarse& o,const Model& m) {
    cv::Rect roi=(o.bbox+cv::Size(30,30)-cv::Point(15,15))&cv::Rect(0,0,image.cols,image.rows);
    cv::Mat mask=cv::Mat::zeros(roi.size(),CV_8U);cv::fillPoly(mask,std::vector<std::vector<cv::Point>>{o.contour},cv::Scalar(1),cv::LINE_8,0,-roi.tl());cv::dilate(mask,mask,cv::Mat::ones(7,7,CV_8U));
    double theta=o.angle*pi/180;P u(std::cos(theta),std::sin(theta)),v(-u.y,u.x);
    auto local=[&](P p){p-=o.center;return P(p.dot(u),p.dot(v));};
    auto world=[&](P p){return o.center+p.x*u+p.y*v;};
    struct Segment {std::vector<P> points;double offset;};
    std::array<std::vector<Segment>,2> groups;
    for(auto l:o.lines) {
        P a(l[0],l[1]),b(l[2],l[3]),d=b-a;
        double len=cv::norm(d);if(len<12)continue;P n(-d.y/len,d.x/len);
        std::vector<double> gs,novel;int covered=0,count=std::max(10,std::min(32,int(len)));
        for(int i=0;i<count;i++) {P p=a+(.05+.9*i/(count-1))*d;int x=int(std::lround(p.x))-roi.x,y=int(std::lround(p.y))-roi.y;if(x>=0&&y>=0&&x<mask.cols&&y<mask.rows&&mask.at<uchar>(y,x))covered++;double g=sample(im,p+2*n)-sample(im,p-2*n),gb=sample(bg,p+2*n)-sample(bg,p-2*n);gs.push_back(std::abs(g));novel.push_back(std::abs(g-gb));}
        if(double(covered)/count<.9||median(gs)<2.5||median(novel)<std::max(2.5,.4*median(gs)))continue;
        P al=local(a),bl=local(b),dl=bl-al;int axis=std::abs(dl.y)>std::abs(dl.x);
        double along=axis?dl.y:dl.x,across=axis?dl.x:dl.y;
        if(std::abs(across/along)>.15)continue;
        std::vector<P> ps;for(int i=0;i<=int(len);i++)ps.push_back(al+double(i)/int(len)*(bl-al));
        auto line=through(al,bl);
        groups[axis].push_back({std::move(ps),-line[2]/line[1-axis]});
    }
    std::array<std::vector<Edge>,2> edges;
    for(int axis=0;axis<2;axis++) {
        auto offset=[&](cv::Vec3d line){return -line[2]/line[1-axis];};
        std::sort(groups[axis].begin(),groups[axis].end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
        std::vector<P> merged;
        double offsetSum=0,weight=0;
        auto flush=[&](){if(!merged.empty()){auto l=lineFit(merged,.01);edges[axis].push_back({l,offset(l)});merged.clear();offsetSum=0;weight=0;}};
        for(const auto& g:groups[axis]) {if(!merged.empty()&&std::abs(offsetSum/weight-g.offset)>2.5)flush();merged.insert(merged.end(),g.points.begin(),g.points.end());offsetSum+=g.offset*g.points.size();weight+=g.points.size();}flush();
    }
    std::vector<std::pair<double,Quad>> candidates;
    for(size_t a=0;a<edges[0].size();a++)for(size_t b=a+1;b<edges[0].size();b++) {
        double width=edges[0][b].offset-edges[0][a].offset;if(width<.82*m.width||width>1.18*m.width)continue;
        for(size_t c=0;c<edges[1].size();c++)for(size_t d=c+1;d<edges[1].size();d++) {
            double length=edges[1][d].offset-edges[1][c].offset;if(length<.89*m.length||length>1.11*m.length)continue;
            try {
                Quad q={world(intersect(edges[0][a].line,edges[1][c].line)),world(intersect(edges[0][a].line,edges[1][d].line)),world(intersect(edges[0][b].line,edges[1][d].line)),world(intersect(edges[0][b].line,edges[1][c].line))};
                double score=searchScore(q,im,bg,m,length,width,16);
                candidates.emplace_back(score,q);
            } catch(const std::runtime_error&) {}
        }
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.first>b.first;});
    std::vector<Quad> result;for(size_t i=0;i<std::min(size_t(2),candidates.size());i++)result.push_back(candidates[i].second);return result;
}

Fit refine(Quad corners,const cv::Mat& im,const Model& model) {
    Fit result;Quad initial=corners;
    for(int iteration=0;iteration<3;iteration++) {
        std::array<cv::Vec3d,4> lines;
        for(int i=0;i<4;i++) {
            P p=corners[i],q=corners[(i+1)%4],e=q-p;double length=cv::norm(e);e*=1/length;P n(e.y,-e.x);
            int count=std::max(32,int(length*.8));std::vector<P> scans;std::vector<double> base;
            for(int j=0;j<count;j++){P pos=p+(.1+.8*j/(count-1))*(q-p);scans.push_back(pos);base.push_back(sample(im,pos+2*n)-sample(im,pos-2*n));}
            double polarity=median(base)>=0?1:-1;std::vector<P> good;
            // Batch the caliper samples through OpenCV's SIMD interpolation.
            cv::Mat mx(count,82,CV_32F),my(count,82,CV_32F),pixels;
            for(int j=0;j<count;j++)for(int k=0;k<41;k++)for(int d=0;d<2;d++){
                P x=scans[j]+(-5+.25*k+(d?1:-1))*n;
                mx.at<float>(j,2*k+d)=float(x.x);my.at<float>(j,2*k+d)=float(x.y);
            }
            cv::remap(im,pixels,mx,my,cv::INTER_LINEAR,cv::BORDER_REPLICATE);
            static const auto weights=[](){std::array<double,41> a{};for(int k=0;k<41;k++)a[k]=std::exp(-.5*std::pow((-5+.25*k)/2.8,2));return a;}();
            for(int j=0;j<count;j++) {P pos=scans[j];
                std::array<double,41> scores,grad;int best=0;
                for(int k=0;k<41;k++){grad[k]=(pixels.at<float>(j,2*k+1)-pixels.at<float>(j,2*k))*polarity;scores[k]=std::max(0.,grad[k])*weights[k];if(scores[k]>scores[best])best=k;}
                if(best==0||best==40||grad[best]<=1.5)continue;
                double den=scores[best-1]-2*scores[best]+scores[best+1];
                double delta=std::abs(den)>1e-6?clamp(.5*(scores[best-1]-scores[best+1])/den,-.5,.5):0;
                good.push_back(pos+(-5+.25*(best+delta))*n);
            }
            if(good.size()<16)return result;
            auto line=lineFit(good);std::vector<double> distances;
            for(P p:good)distances.push_back(std::abs(line[0]*p.x+line[1]*p.y+line[2]));
            double threshold=std::max(1.2,std::min(2.5,2.5*median(distances)));std::vector<P> kept;
            for(size_t j=0;j<good.size();j++)if(distances[j]<threshold)kept.push_back(good[j]);
            if(kept.size()<16)return result;
            line=lineFit(kept);P vec(line[1],-line[0]);double angle=std::acos(clamp(std::abs(vec.dot(e)),0,1))*180/pi;
            if(angle>6)return result;
            double rms=0;for(P p:kept)rms+=std::pow(line[0]*p.x+line[1]*p.y+line[2],2);
            result.sides[i]={double(kept.size())/count,std::sqrt(rms/kept.size()),count,int(kept.size())};lines[i]=line;
        }
        try {for(int i=0;i<4;i++)corners[i]=intersect(lines[(i+3)%4],lines[i]);}catch(const std::runtime_error&){return result;}
        for(int i=0;i<4;i++)if(!std::isfinite(corners[i].x)||!std::isfinite(corners[i].y)||cv::norm(corners[i]-initial[i])>12)return result;

    }
    std::vector<cv::Point2f> c;for(P p:corners)c.emplace_back(p);if(!cv::isContourConvex(c))return result;
    std::array<double,4> sizes;std::array<P,4> units;
    for(int i=0;i<4;i++){P e=corners[(i+1)%4]-corners[i];sizes[i]=cv::norm(e);units[i]=e*(1/sizes[i]);}
    double l=(sizes[0]+sizes[2])/2,w=(sizes[1]+sizes[3])/2;
    if(l<.86*model.length||l>1.14*model.length||w<.82*model.width||w>1.18*model.width)return result;
    double disagreement=0,maximum=0;
    for(int i=0;i<2;i++){double a=std::acos(clamp(std::abs(units[i].dot(units[i+2])),0,1))*180/pi;disagreement+=a/2;maximum=std::max(maximum,a);}
    double support=0,minSupport=1,rms=0,maxRms=0;
    for(auto s:result.sides){support+=s.support/4;minSupport=std::min(minSupport,s.support);rms+=s.rms/4;maxRms=std::max(maxRms,s.rms);}
    if(maximum>8||minSupport<.55)return result;
    double prior=std::pow((l-model.length)/(.06*model.length),2)+std::pow((w-model.width)/(.10*model.width),2);
    result.corners=corners;result.quality=.55*support+.25*minSupport-.12*rms-.05*maxRms-.025*prior-.003*disagreement;
    return result;
}

struct CenterPrior {
    bool available=false;
    P center;
    double spread=0,mass=0,inferMs=0;
    json samples=json::array();
};

// Predictions are independent of fitted top faces: always crop the coarse component.
template<typename Predict>
CenterPrior coarseCenterPrior(const cv::Mat& image,cv::Rect bbox,Predict predict) {
    CenterPrior result;std::vector<double> xs,ys,masses;
    if(bbox.empty()||(bbox&cv::Rect(0,0,image.cols,image.rows))!=bbox)return result;
    for(int pad:{8,16,24}) {
        int x0=std::max(0,bbox.x-pad),y0=std::max(0,bbox.y-pad);
        int x1=std::min(image.cols,bbox.br().x+pad),y1=std::min(image.rows,bbox.br().y+pad);
        cv::Rect crop(x0,y0,x1-x0,y1-y0);
        json sample={{"pad",pad},{"crop_bbox_xyxy",{x0,y0,x1,y1}},{"status","failed"}};
        try {
            P center;double mass=0,ms=0;
            bool ok=crop.width>=8&&crop.height>=8&&predict(image(crop),center,mass,ms);
            result.inferMs+=ms;
            if(ok&&std::isfinite(center.x)&&std::isfinite(center.y)&&std::isfinite(mass)&&mass>0&&
               center.x>=0&&center.y>=0&&center.x<crop.width&&center.y<crop.height) {
                center+=P(crop.tl());xs.push_back(center.x);ys.push_back(center.y);masses.push_back(mass);
                sample["status"]="ok";sample["center_px"]={center.x,center.y};sample["mass"]=mass;
            }
        } catch(const std::exception& e){sample["error"]=e.what();}
        result.samples.push_back(sample);
    }
    // A partial group cannot establish three-crop stability; use geometry alone.
    if(xs.size()!=3)return result;
    result.available=true;result.center=P(median(xs),median(ys));result.mass=median(masses);
    for(size_t i=0;i<xs.size();i++)result.spread=std::max(result.spread,cv::norm(P(xs[i],ys[i])-result.center));
    return result;
}

std::vector<double> candidateAngles(const Coarse& o) {
    auto modulo90=[](double a){return std::fmod(a+180,90.);};
    auto distance=[](double a,double b){return std::abs(std::remainder(a-b,90.));};
    P longest;double size=0;
    for(int i=0;i<4;i++){P e=o.box[(i+1)%4]-o.box[i];if(cv::norm(e)>size){size=cv::norm(e);longest=e;}}
    std::vector<double> bases{modulo90(axisAngle(longest))};
    std::vector<std::pair<double,double>> votes;
    for(auto l:o.lines){P e(l[2]-l[0],l[3]-l[1]);double length=cv::norm(e);if(length>=20)votes.emplace_back(modulo90(axisAngle(e)),length);}
    std::vector<std::pair<double,double>> peaks;
    for(auto v:votes){double score=0;for(auto z:votes)score+=z.second*std::exp(-.5*std::pow(distance(v.first,z.first)/2,2));peaks.emplace_back(score,v.first);}
    std::stable_sort(peaks.begin(),peaks.end(),[](const auto& a,const auto& b){return a.first>b.first;});
    for(auto peak:peaks){bool different=true;for(double a:bases)if(distance(peak.second,a)<=8)different=false;
        if(different)bases.push_back(peak.second);if(bases.size()==2)break;}
    std::vector<double> angles;
    for(double a:bases)for(double turn:{0.,90.})for(double delta:{0.,-2.,2.})angles.push_back(std::fmod(a+turn+delta+180,180.));
    return angles;
}

Coarse searchOrigin(const Coarse& o,P origin,double angle) {
    Coarse result=o;result.center=origin;result.angle=angle;
    double theta=angle*pi/180,c=std::abs(std::cos(theta)),s=std::abs(std::sin(theta));
    result.length=c*o.bbox.width+s*o.bbox.height;result.width=s*o.bbox.width+c*o.bbox.height;
    return result;
}
P diagonalCenter(const Quad& c){return intersect(through(c[0],c[2]),through(c[1],c[3]));}
double faceAngle(const Quad& c) {
    std::vector<cv::Point2f> unit={{0,0},{1,0},{1,1},{0,1}},quad;for(P p:c)quad.emplace_back(p);
    auto h=cv::getPerspectiveTransform(unit,quad);std::vector<cv::Point2f> axis={{0,.5},{1,.5}},mapped;
    cv::perspectiveTransform(axis,mapped,h);return axisAngle(P(mapped[1]-mapped[0]));
}
struct FaceCandidate {Fit fit;std::string model,source;double angle;};
void addCandidate(std::vector<FaceCandidate>& candidates,Fit fit,const Model& model,const char* source) {
    if(fit.quality<=-1e8)return;
    P center=diagonalCenter(fit.corners);double angle=faceAngle(fit.corners);
    for(auto& old:candidates)if(cv::norm(center-diagonalCenter(old.fit.corners))<.75&&std::abs(angleDiff(angle,old.angle))<1) {
        if(fit.quality>old.fit.quality)old={fit,model.name,source,angle};return;
    }
    candidates.push_back({fit,model.name,source,angle});
}

double transverseSeam(const cv::Mat& gray,const Quad& c) {
    std::vector<cv::Point2f> src,dst={{0,0},{113,0},{113,63},{0,63}};for(P p:c)src.emplace_back(p);
    cv::Mat f,tile;gray.convertTo(f,CV_32F);
    cv::warpPerspective(f,tile,cv::getPerspectiveTransform(src,dst),{114,64});
    cv::GaussianBlur(tile,tile,{},.85);
    double maximum=0;
    // gradient column j compares source columns j+4 and j, exactly Python [20:90].
    for(int x=20;x<90;x++) {double coherent=0;
        for(int y=8;y<56;y++)coherent+=clamp((std::abs(tile.at<float>(y,x+4)-tile.at<float>(y,x))-6)/24,0,1);
        maximum=std::max(maximum,coherent/48);
    }
    return maximum;
}

struct FusionSelection {size_t index=0;double quality=-1e9,score=-1e9,seam=0,weight=0,margin=0;int shortlisted=0;};
FusionSelection selectCandidate(const cv::Mat& gray,const std::vector<FaceCandidate>& candidates,const CenterPrior& prior) {
    FusionSelection selection;
    for(const auto& c:candidates)selection.quality=std::max(selection.quality,c.fit.quality);
    selection.weight=prior.available?.10/(1+std::pow(prior.spread/4,2)):0;
    double runnerUp=-1e9;
    for(size_t i=0;i<candidates.size();i++) {
        const auto& c=candidates[i];if(c.fit.quality<=.35||c.fit.quality<selection.quality-.04)continue;
        ++selection.shortlisted;
        double seam=transverseSeam(gray,c.fit.corners);
        double distance=prior.available?cv::norm(diagonalCenter(c.fit.corners)-prior.center):0;
        // Missing/failed NN is a geometry-only fallback, without changing its score semantics.
        double score=prior.available?c.fit.quality-selection.weight*std::pow(std::min(distance/16,3.),2)-.03*seam:c.fit.quality;
        if(score>selection.score){runnerUp=selection.score;selection.index=i;selection.score=score;selection.seam=seam;}
        else runnerUp=std::max(runnerUp,score);
    }
    selection.margin=selection.shortlisted>1?selection.score-runnerUp:0;
    return selection;
}

// Cylinders have no observable in-plane heading. Fit the top rim, not the
// centroid/bounding ellipse of a blob that can also contain a bright sidewall.
struct CircleFit {
    P center;
    double radius=0, support=0, coverage=0, rms=0;
    bool edgeRefined=false;
};

bool circleThrough(P a,P b,P c,CircleFit& fit) {
    P u=b-a,v=c-a;
    const double cross=u.x*v.y-u.y*v.x;
    if(std::abs(cross)<1e-6)return false;
    const double uu=u.dot(u)/2,vv=v.dot(v)/2;
    fit.center=a+P((uu*v.y-vv*u.y)/cross,(u.x*vv-v.x*uu)/cross);
    fit.radius=cv::norm(fit.center-a);
    return std::isfinite(fit.radius)&&std::isfinite(fit.center.x)&&std::isfinite(fit.center.y);
}

bool circleLeastSquares(const std::vector<P>& points,CircleFit& fit) {
    if(points.size()<6)return false;
    P origin;
    for(P p:points)origin+=p;
    origin*=1./points.size();
    cv::Mat a(int(points.size()),3,CV_64F),b(int(points.size()),1,CV_64F),solution;
    for(size_t i=0;i<points.size();++i){
        P p=points[i]-origin;
        a.at<double>(int(i),0)=2*p.x;a.at<double>(int(i),1)=2*p.y;
        a.at<double>(int(i),2)=1;b.at<double>(int(i))=p.dot(p);
    }
    if(!cv::solve(a,b,solution,cv::DECOMP_SVD))return false;
    P relative(solution.at<double>(0),solution.at<double>(1));
    const double squared=solution.at<double>(2)+relative.dot(relative);
    if(!(squared>0))return false;
    fit.center=origin+relative;fit.radius=std::sqrt(squared);
    return std::isfinite(fit.radius)&&std::isfinite(fit.center.x)&&std::isfinite(fit.center.y);
}

double circleSupport(const std::vector<P>& points,CircleFit& fit,double tolerance,
                     std::vector<P>* inliers=nullptr) {
    std::array<int,24> bins{};int count=0;double squared=0;
    if(inliers)inliers->clear();
    for(P p:points){
        P d=p-fit.center;const double error=std::abs(cv::norm(d)-fit.radius);
        if(error>tolerance)continue;
        ++count;squared+=error*error;
        int bin=int((std::atan2(d.y,d.x)+pi)*24/(2*pi));
        ++bins[std::max(0,std::min(23,bin))];
        if(inliers)inliers->push_back(p);
    }
    int covered=0,balanced=0;
    for(int n:bins){covered+=n>0;balanced+=std::min(n,5);}
    fit.support=points.empty()?0:double(count)/points.size();
    fit.coverage=covered/24.;fit.rms=count?std::sqrt(squared/count):1e9;
    return balanced-.001*fit.rms;
}

bool fitCircleRim(const Runtime& r,const std::vector<cv::Point>& contour,
                  const cv::Mat& smoothed,CircleFit& best) {
    if(contour.size()<12)return false;
    std::vector<P> points;points.reserve(contour.size());
    for(auto p:contour)points.emplace_back(p);
    std::mt19937 random(248);
    std::uniform_int_distribution<size_t> choose(0,points.size()-1);
    double bestScore=-1;
    for(int iteration=0;iteration<240;++iteration){
        size_t a=choose(random),b=choose(random),c=choose(random);
        if(a==b||a==c||b==c)continue;
        CircleFit candidate;
        if(!circleThrough(points[a],points[b],points[c],candidate)||
           candidate.radius<r.circleMinRadius||candidate.radius>r.circleMaxRadius)continue;
        double score=circleSupport(points,candidate,r.circleTolerance);
        if(candidate.coverage<r.circleCoverage||candidate.support<.55)continue;
        if(score>bestScore){bestScore=score;best=candidate;}
    }
    if(bestScore<0)return false;
    for(int iteration=0;iteration<5;++iteration){
        std::vector<P> inliers;
        circleSupport(points,best,r.circleTolerance,&inliers);
        CircleFit next=best;
        if(!circleLeastSquares(inliers,next)||next.radius<r.circleMinRadius||
           next.radius>r.circleMaxRadius)break;
        circleSupport(points,next,r.circleTolerance);
        if(next.coverage<r.circleCoverage||next.support<.55)break;
        best=next;
    }
    // Subpixel calipers are restricted to the already supported top rim. A
    // remote sidewall-to-floor edge cannot drag the grasp center towards it.
    std::vector<P> edgePoints;
    for(int i=0;i<120;++i){
        const double angle=i*(2*pi/120);P direction(std::cos(angle),std::sin(angle));
        double bestGradient=6,bestOffset=0;bool found=false;
        for(double offset=-2.5;offset<=2.5;offset+=.25){
            P p=best.center+direction*(best.radius+offset);
            double gradient=sample(smoothed,p-direction*.9)-sample(smoothed,p+direction*.9);
            double score=gradient-.6*std::abs(offset);
            if(score>bestGradient){bestGradient=score;bestOffset=offset;found=true;}
        }
        if(found)edgePoints.push_back(best.center+direction*(best.radius+bestOffset));
    }
    if(edgePoints.size()>=80){
        CircleFit refined=best;
        for(int iteration=0;iteration<4;++iteration){
            std::vector<P> inliers;circleSupport(edgePoints,refined,r.circleTolerance,&inliers);
            if(!circleLeastSquares(inliers,refined))break;
        }
        circleSupport(edgePoints,refined,r.circleTolerance);
        if(refined.coverage>=r.circleCoverage&&refined.support>=.7&&
           cv::norm(refined.center-best.center)<=1.5&&std::abs(refined.radius-best.radius)<=2&&
           refined.radius>=r.circleMinRadius&&refined.radius<=r.circleMaxRadius){
            refined.edgeRefined=true;best=refined;
        }
    }
    return best.coverage>=r.circleCoverage&&best.support>=.55;
}

json detectCircles(Runtime& r,const cv::Mat& image,const json& request) {
    const auto started=std::chrono::steady_clock::now();
    bool updated=false;
    if(r.autoReload||request.value("reload_background",false))
        updated=loadBackground(r,request.value("reload_background",false));
    if(image.type()!=CV_8UC1)throw std::runtime_error("circle input must be 8-bit grayscale or BGR");
    if(image.size()!=r.background.size())throw std::runtime_error("background/image size mismatch");
    std::vector<cv::Point2f> roi;
    if(request.contains("rect")){
        const auto& a=request.at("rect");
        if(!a.is_array()||a.size()!=8)throw std::runtime_error("rect must contain eight pixel coordinates");
        for(int i=0;i<8;i+=2){
            double x=a[i].get<double>(),y=a[i+1].get<double>();
            if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("rect coordinates must be finite");
            roi.emplace_back(float(x),float(y));
        }
        if(!cv::isContourConvex(roi)||std::abs(cv::contourArea(roi))<1)
            throw std::runtime_error("rect must be a convex quadrilateral");
    }
    // A robust exposure offset removes global brightness drift without making
    // dark drawn circles/stains into positive cylinder candidates.
    std::vector<double> offsets;
    for(int y=0;y<image.rows;y+=16)for(int x=0;x<image.cols;x+=16)
        offsets.push_back(double(image.at<uchar>(y,x))-r.background.at<uchar>(y,x));
    const double bias=median(offsets);
    cv::Mat normalized;image.convertTo(normalized,CV_32F,1,-bias);
    cv::Mat difference=normalized-r.backgroundFloat,mask,bright;
    cv::compare(difference,r.circleThreshold,mask,cv::CMP_GT);
    cv::compare(normalized,r.circleMinIntensity,bright,cv::CMP_GT);
    cv::bitwise_and(mask,bright,mask);
    cv::morphologyEx(mask,mask,cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_ELLIPSE,{3,3}));
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,{5,5}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    std::sort(contours.begin(),contours.end(),[](const auto& a,const auto& b){
        return cv::boundingRect(a).x<cv::boundingRect(b).x;
    });
    const cv::Mat smoothed=smooth(normalized);
    json output=json::array(),pts=json::array();int id=0,count=0;
    for(const auto& contour:contours){
        const double area=cv::contourArea(contour);
        if(area<.5*pi*r.circleMinRadius*r.circleMinRadius)continue;
        const auto box=cv::boundingRect(contour);
        json rec={{"id",++id},{"shape","circle"},{"status","not_circular"},
                  {"model","cylinder_top"},{"center_px",nullptr},{"top_center_px",nullptr},
                  {"angle_deg",r.circleFixedAngle},{"long_edge_angle_deg",r.circleFixedAngle},
                  {"angle_defined",false},{"angle_source","fixed_rotational_symmetry"},
                  {"corners_px",nullptr},{"coarse_bbox_xyxy",{box.x,box.y,box.br().x,box.br().y}},
                  {"center_model_status","disabled"}};
        if(request.value("include_foreground_contours",false)){
            rec["foreground_contour_px"]=json::array();
            for(auto p:contour)rec["foreground_contour_px"].push_back({p.x,p.y});
        }
        if(box.x<=1||box.y<=1||box.br().x>=image.cols-1||box.br().y>=image.rows-1){
            rec["status"]="partial";output.push_back(rec);continue;
        }
        if(area>1.6*pi*r.circleMaxRadius*r.circleMaxRadius){
            rec["status"]="unmodeled_size";output.push_back(rec);continue;
        }
        const double perimeter=cv::arcLength(contour,true);
        const double circularity=perimeter>0?4*pi*area/(perimeter*perimeter):0;
        rec["circularity"]=circularity;
        CircleFit fit;
        if(circularity<.60||!fitCircleRim(r,contour,smoothed,fit)){
            output.push_back(rec);continue;
        }
        int interior=0,foreground=0;
        const double innerRadius=.65*fit.radius;
        for(int y=int(std::floor(fit.center.y-innerRadius));y<=int(std::ceil(fit.center.y+innerRadius));++y)
            for(int x=int(std::floor(fit.center.x-innerRadius));x<=int(std::ceil(fit.center.x+innerRadius));++x){
                if(x<0||y<0||x>=mask.cols||y>=mask.rows||cv::norm(P(x,y)-fit.center)>innerRadius)continue;
                ++interior;foreground+=mask.at<uchar>(y,x)>0;
            }
        const double fill=interior?double(foreground)/interior:0;
        rec["interior_fill"]=fill;
        if(fill<r.circleFill){rec["status"]="not_solid_top";output.push_back(rec);continue;}
        if(fit.center.x-fit.radius<1||fit.center.y-fit.radius<1||
           fit.center.x+fit.radius>=image.cols-1||fit.center.y+fit.radius>=image.rows-1){
            rec["status"]="partial";output.push_back(rec);continue;
        }
        if(!roi.empty()&&cv::pointPolygonTest(roi,fit.center,false)<0){
            rec["status"]="outside_roi";output.push_back(rec);continue;
        }
        const double x=fit.center.x,y=fit.center.y,radius=fit.radius;
        rec["status"]="ok";rec["center_px"]=rec["top_center_px"]=rec["geometric_center_px"]={x,y};
        rec["center_source"]="circle_fit";rec["radius_px"]=radius;rec["diameter_px"]=2*radius;
        rec["apparent_length_px"]=rec["apparent_width_px"]=2*radius;
        // Keep the legacy four-point field as bounds, not physical top corners.
        rec["corners_px"]={{x-radius,y-radius},{x+radius,y-radius},{x+radius,y+radius},{x-radius,y+radius}};
        rec["corners_definition"]="circle_bounding_box";
        rec["fit_support"]=fit.support;rec["fit_coverage"]=fit.coverage;rec["fit_rms_px"]=fit.rms;
        rec["edge_refined"]=fit.edgeRefined;rec["angle_signed_deg"]=r.circleFixedAngle;
        output.push_back(rec);pts.push_back({x,y,r.circleFixedAngle});++count;
    }
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return {{"ret-code",0},{"ret-msg","ok"},{"best_img_idx",count?0:-1},
            {"image_size_wh",{image.cols,image.rows}},{"stage_ms",{{"circle_detection",elapsed}}},
            {"geometry_ms",json::object()},{"feature_backend","cpu_circle_geometry"},
            {"background_image",r.backgroundPath.string()},{"background_generation",r.backgroundGeneration},
            {"background_updated",updated},{"alignment_mode","fixed"},
            {"coordinate_frame","current_input_image_pixels"},{"center_definition","fitted_circular_top_rim"},
            {"angle_definition","fixed_for_rotationally_symmetric_cylinder"},
            {"foreground_object_count",output.size()},{"top_face_count",count},
            {"objects",output},{"pts_top_face",pts},{"elapsed_ms",elapsed}};
}

json detectRectangles(Runtime& r,const cv::Mat& image,const json& request) {
    auto started=std::chrono::steady_clock::now();cv::Mat normalized,aligned;
    bool backgroundUpdated=false;
    if(r.autoReload||request.value("reload_background",false))backgroundUpdated=loadBackground(r,request.value("reload_background",false));
    double reloadMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    auto coarse=foreground(r,image,normalized,aligned);
    r.profile["background_loading"]=reloadMs;
    r.geometry=json::object();
    auto mark=std::chrono::steady_clock::now();
    auto tick=[&](const char* name){auto now=std::chrono::steady_clock::now();r.geometry[name]=r.geometry.value(name,0.)+std::chrono::duration<double,std::milli>(now-mark).count();mark=now;};
    json output=json::array(),pts=json::array();int id=0,count=0;
    std::vector<cv::Point2f> roi;
    if(request.contains("rect")) {
        const auto& a=request.at("rect");if(!a.is_array()||a.size()!=8)throw std::runtime_error("rect must contain eight pixel coordinates");
        for(int i=0;i<8;i+=2)roi.emplace_back(a[i].get<float>(),a[i+1].get<float>());
        if(!cv::isContourConvex(roi)||std::abs(cv::contourArea(roi))<1)throw std::runtime_error("rect must be a convex quadrilateral");
    }
    for(const auto& o:coarse) {
        json rec={{"id",++id},{"status","not_rectangular"},{"model",nullptr},{"center_px",nullptr},{"top_center_px",nullptr},{"angle_deg",nullptr},{"long_edge_angle_deg",nullptr},{"corners_px",nullptr},
                  {"coarse_bbox_xyxy",{o.bbox.x,o.bbox.y,o.bbox.br().x,o.bbox.br().y}},{"old_envelope_center_px",{o.center.x,o.center.y}},{"old_envelope_corners_px",pointsJson(o.box)}};
        if(request.value("include_foreground_contours",false)){rec["foreground_contour_px"]=json::array();for(const auto& p:o.contour)rec["foreground_contour_px"].push_back({p.x,p.y});}
        if(o.partial){rec["status"]="partial";output.push_back(rec);continue;}
        // A top plus its ledge can form a square envelope; require a resolved
        // four-edge model before accepting it instead of rejecting the envelope.
        const bool squareEnvelope=o.length/o.width<1.1;
        auto eligible=[&](const Model& model){return envelopeFits(model,o.length,o.width);};
        bool plausible=false;for(const auto& model:r.models)if(eligible(model))plausible=true;
        if(!plausible){rec["status"]="unmodeled_size";output.push_back(rec);continue;}
        mark=std::chrono::steady_clock::now();
        CenterPrior prior;
        rec["center_model_status"]=r.centerModelError.empty()?"disabled":"unavailable";
        if(!r.centerModelError.empty())rec["center_model_error"]=r.centerModelError;
        if(r.useCenterModel&&r.centerModel.ready()) {
            prior=coarseCenterPrior(image,o.bbox,[&](const cv::Mat& crop,P& center,double& mass,double& ms){
                bool ok=r.centerModel.predict(crop,center,&mass);ms=r.centerModel.last_infer_ms();return ok;
            });
            rec["center_model_status"]=prior.available?"ok":"failed";
            rec["center_model_crop_source"]="independent_coarse_bbox";
            rec["center_model_samples"]=prior.samples;
            rec["center_model_backend"]=r.centerModel.backend();rec["center_model_infer_ms"]=prior.inferMs;
            if(prior.available){rec["center_model_center_px"]={prior.center.x,prior.center.y};rec["center_model_mass"]=prior.mass;
                rec["center_model_crop_spread_px"]=prior.spread;}
        }
        tick("independent_center_inference");
        auto angles=candidateAngles(o);std::vector<P> origins{P(o.bbox.x+.5*o.bbox.width,o.bbox.y+.5*o.bbox.height)};
        if(prior.available&&cv::norm(prior.center-origins[0])>=2)origins.push_back(prior.center);
        std::vector<cv::Point2f> bounds;
        for(const auto& model:r.models)if(eligible(model))for(double angle:angles)for(P center:origins) {
            auto search=searchOrigin(o,center,angle);double theta=angle*pi/180;P u(std::cos(theta),std::sin(theta)),v(-u.y,u.x);
            double hx=.5*1.03*model.length+alongSpan(model,search)+16,hy=.5*1.10*model.width+acrossSpan(model,search)+16;
            for(int x:{-1,1})for(int y:{-1,1})bounds.emplace_back(center+x*hx*u+y*hy*v);
        }
        auto tile=cv::boundingRect(bounds)&cv::Rect(0,0,image.cols,image.rows);P origin(tile.tl());
        Coarse local=o;local.center-=origin;local.bbox-=tile.tl();
        for(auto& p:local.contour)p-=tile.tl();for(auto& p:local.box)p-=origin;
        for(auto& l:local.lines){l[0]-=tile.x;l[2]-=tile.x;l[1]-=tile.y;l[3]-=tile.y;}
        mark=std::chrono::steady_clock::now();
        cv::Mat pixels=image(tile),im=smooth(pixels);
        cv::Mat bg=(r.alignBackground&&!r.descriptors.empty())?smooth(aligned(tile)):r.backgroundSmooth(tile);
        tick("smoothing");
        std::vector<FaceCandidate> candidates;
        for(const auto& model:r.models)if(eligible(model)) {
            mark=std::chrono::steady_clock::now();
            for(double angle:angles)for(P center:origins) {
                auto direction=searchOrigin(local,center-origin,angle);
                auto seeds=integralSearch(direction,model,im,bg);
                for(const auto& seed:seeds)addCandidate(candidates,refine(seed,im,model),model,"integral_edge_search");
            }
            tick("multi_direction_search_refinement");
            double ceiling=-1e9;for(const auto& c:candidates)ceiling=std::max(ceiling,c.fit.quality);
            if(!r.fast||ceiling<.60)for(double angle:angles) {
                auto direction=searchOrigin(local,origins[0]-origin,angle);
                for(const auto& seed:proposals(pixels,im,bg,direction,model))addCandidate(candidates,refine(seed,im,model),model,"four_line_proposal");
                if(!r.fast)addCandidate(candidates,refine(search(direction,model,im,bg,r.iterations),im,model),model,"bounded_rectangle_search");
            }
            tick("line_proposals_refinement");
        }
        CenterPrior localPrior=prior;localPrior.center-=origin;
        const auto selection=selectCandidate(pixels,candidates,localPrior);
        if(selection.quality<=.35){rec["status"]=squareEnvelope?"not_rectangular":"top_face_not_resolved";output.push_back(rec);continue;}
        Fit best=candidates[selection.index].fit;std::string modelName=candidates[selection.index].model,source=candidates[selection.index].source;
        for(auto& p:best.corners)p+=origin;
        const auto& c=best.corners;P geometricCenter=diagonalCenter(c);const P center=geometricCenter;
        if(prior.available)rec["center_model_offset_px"]={prior.center.x-center.x,prior.center.y-center.y};
        rec["fusion_method"]=prior.available?"nn_plus_seam":"geometry_only";
        rec["fusion_score"]=selection.score;rec["fusion_nn_weight"]=selection.weight;
        rec["fusion_transverse_seam"]=selection.seam;rec["fusion_geometry_ceiling"]=selection.quality;
        rec["fusion_candidate_count"]=candidates.size();rec["fusion_shortlisted_count"]=selection.shortlisted;
        rec["fusion_selection_margin"]=selection.shortlisted>1?json(selection.margin):json(nullptr);
        rec["fusion_selection_ambiguous"]=selection.shortlisted>1&&selection.margin<.01;
        if(request.value("include_fusion_candidates",false)) {
            rec["fusion_candidates"]=json::array();
            for(const auto& candidate:candidates){auto q=candidate.fit.corners;for(auto& p:q)p+=origin;
                P cp=diagonalCenter(q);rec["fusion_candidates"].push_back({{"quality",candidate.fit.quality},{"center_px",{cp.x,cp.y}},
                    {"angle_deg",std::fmod(180.-candidate.angle,180.)},{"corners_px",pointsJson(q)}});}
        }
        if(!roi.empty()&&cv::pointPolygonTest(roi,center,false)<0){rec["status"]="outside_roi";output.push_back(rec);continue;}
        std::vector<cv::Point2f> unit={{0,0},{1,0},{1,1},{0,1}},quad;for(P p:c)quad.emplace_back(p);
        auto H=cv::getPerspectiveTransform(unit,quad);std::vector<cv::Point2f> axis={{0,.5},{1,.5}},transformed;cv::perspectiveTransform(axis,transformed,H);
        double angle=axisAngle(P(transformed[1]-transformed[0]));
        // Internal geometry is clockwise-positive (image y points down); report the opposite.
        angle=std::fmod(180.-angle,180.);
        rec["status"]="ok";rec["model"]=modelName;rec["center_px"]=rec["top_center_px"]={center.x,center.y};
        rec["geometric_center_px"]={geometricCenter.x,geometricCenter.y};
        rec["center_source"]="diagonal_intersection";
        rec["angle_deg"]=rec["long_edge_angle_deg"]=angle;rec["angle_signed_deg"]=std::fmod(angle+90,180)-90;rec["corners_px"]=pointsJson(c);
        rec["apparent_length_px"]=(cv::norm(c[1]-c[0])+cv::norm(c[2]-c[3]))/2;rec["apparent_width_px"]=(cv::norm(c[3]-c[0])+cv::norm(c[2]-c[1]))/2;
        rec["selection_score"]=best.quality;rec["seed_method"]=source;rec["sides"]=json::array();
        for(int i=0;i<4;i++){auto s=best.sides[i];rec["sides"].push_back({{"side",i},{"support",s.support},{"rms_px",s.rms},{"scans",s.scans},{"inliers",s.inliers}});}
        output.push_back(rec);pts.push_back({center.x,center.y,angle});count++;
    }
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    double foregroundMs=0;for(const auto& item:r.profile.items())foregroundMs+=item.value().get<double>();r.profile["top_geometry"]=ms-foregroundMs;
    return {{"ret-code",0},{"ret-msg","ok"},{"best_img_idx",count?0:-1},{"image_size_wh",{image.cols,image.rows}},{"stage_ms",r.profile},{"geometry_ms",r.geometry},{"feature_backend",featureBackend},
            {"background_image",r.backgroundPath.string()},{"background_generation",r.backgroundGeneration},{"background_updated",backgroundUpdated},{"alignment_mode",r.alignBackground?"orb":"fixed"},
            {"coordinate_frame","current_input_image_pixels"},{"center_definition","intersection_of_top_face_diagonals"},
            {"angle_definition","projected_long_axis_counterclockwise_from_image_positive_x_mod_180"},
            {"foreground_object_count",output.size()},{"top_face_count",count},{"objects",output},{"pts_top_face",pts},{"elapsed_ms",ms}};
}

// Homogeneous scale is arbitrary; reject cancellation relative to this denominator's terms.
double homographyDenominator(const cv::Matx33d& h,double u,double v){
    double a=h(2,0)*u,b=h(2,1)*v,c=h(2,2),denominator=a+b+c;
    double scale=std::abs(a)+std::abs(b)+std::abs(c);
    if(!std::isfinite(denominator)||!std::isfinite(scale)||std::abs(denominator)<=1e-12*scale)
        throw std::runtime_error("homography denominator is zero or near zero");
    return denominator;
}

} // namespace

int top_face_init(const std::string& appDir) {
    std::lock_guard<std::mutex> lock(runtimeMutex);
    if (runtime)
        return 0;
    try {
        auto r = std::make_unique<Runtime>();
        r->app = appDir.empty() ? fs::current_path() : fs::absolute(appDir);
        auto cfg = r->app / "config" / "top_face" / "algorithm.json";
        if (!fs::exists(cfg)) cfg = r->app / "config" / "algorithm.json";
        json config = json::object();
        if (fs::exists(cfg)) {
            std::ifstream f(cfg);
            f >> config;
        }
        auto background = fs::path(config.value("background_image", std::string("background.bmp")));
        if (background.is_relative())
            background = (fs::exists(cfg) ? cfg.parent_path() : r->app / "config" / "top_face") / background;
        r->backgroundPath = fs::absolute(background);
        r->autoReload = config.value("background_auto_reload", true);
        const auto shape=config.value("workpiece_shape",std::string("circle"));
        if(shape!="circle"&&shape!="rectangle")
            throw std::runtime_error("workpiece_shape must be circle or rectangle");
        r->circles=shape=="circle";
        auto alignment = config.value("alignment_mode", std::string("fixed"));
        if (alignment != "fixed" && alignment != "orb")
            throw std::runtime_error("alignment_mode must be fixed or orb");
        r->alignBackground = alignment == "orb";
        r->fast = config.value("fast", true);
        r->iterations = config.value("search_iterations", 90);
        r->difference = config.value("difference_threshold", 15.);
        r->residual = config.value("residual_threshold", 10.);
        r->correlation = config.value("correlation_threshold", .6);
        int threads = config.value("opencv_threads", 2);
        if (r->iterations < 1 || r->iterations > 500 || threads < 1 || threads > 64 ||
            r->difference <= 0 || r->residual <= 0 || r->correlation <= 0 || r->correlation >= 1)
            throw std::runtime_error("invalid algorithm configuration");
        cv::setNumThreads(threads);
        if(r->circles){
            if(r->alignBackground)throw std::runtime_error("circle detection requires fixed alignment");
            r->circleMinRadius=config.value("circle_min_radius_px",18.);
            r->circleMaxRadius=config.value("circle_max_radius_px",38.);
            r->circleThreshold=config.value("circle_difference_threshold",30.);
            r->circleMinIntensity=config.value("circle_min_intensity",105.);
            r->circleTolerance=config.value("circle_fit_tolerance_px",1.3);
            r->circleCoverage=config.value("circle_min_coverage",.75);
            r->circleFill=config.value("circle_min_fill_ratio",.75);
            r->circleFixedAngle=config.value("circle_angle_deg",0.);
            for(double value:{r->circleMinRadius,r->circleMaxRadius,r->circleThreshold,
                              r->circleMinIntensity,r->circleTolerance,r->circleCoverage,r->circleFill,r->circleFixedAngle})
                if(!std::isfinite(value))throw std::runtime_error("circle parameters must be finite");
            if(r->circleMinRadius<3||r->circleMaxRadius<=r->circleMinRadius||
               r->circleThreshold<=0||r->circleThreshold>=255||
               r->circleMinIntensity<0||r->circleMinIntensity>=255||
               r->circleTolerance<=0||r->circleTolerance>5||
               r->circleCoverage<.6||r->circleCoverage>1||
               r->circleFill<=0||r->circleFill>1||
               r->circleFixedAngle<=-90||r->circleFixedAngle>90)
                throw std::runtime_error("invalid circle detector configuration");
        }
        // The former rectangle path is available only by explicit selection.
        // Circle mode never loads the rectangle center NN or BPU feature model.
        if(!r->circles){
        if (config.contains("models")) {
            r->models.clear();
            for (const auto& m : config["models"]) {
                Model model{m.at("name"), m.at("length_px"), m.at("width_px")};
                if (!std::isfinite(model.length) || !std::isfinite(model.width) ||
                    model.length <= model.width || model.width < 15)
                    throw std::runtime_error("model requires length > width >= 15 px");
                r->models.push_back(model);
            }
        }
        if (r->models.empty()) throw std::runtime_error("no face models configured");
        // Optional heatmap prior ranks geometric candidates; failure keeps geometric detection.
        try {
            fs::path cfgDir = fs::exists(cfg) ? cfg.parent_path() : r->app / "config" / "top_face";
            std::string modelRel = config.value("center_model", std::string("center_only.onnx"));
            bool explicitUse = config.contains("use_center_model");
            r->useCenterModel = config.value("use_center_model", true);
            r->centerModelRequired = config.value("center_model_required", false); // Legacy flag, diagnostics remain optional.
            r->centerModelImageSize = config.value("center_model_image_size", 256);
            r->centerModelDecodeRadius = config.value("center_model_decode_radius", 4);
            fs::path modelPath = fs::path(modelRel);
            if (modelPath.is_relative()) modelPath = cfgDir / modelPath;
            r->centerModelPath = modelPath.string();
            if (r->useCenterModel) {
                if (fs::exists(modelPath)) {
                    if (!r->centerModel.load(modelPath.string(), r->centerModelImageSize, r->centerModelDecodeRadius))
                        throw std::runtime_error("failed to load center_model: " + modelPath.string());
                } else {
                    r->useCenterModel = false;
                    if (explicitUse || r->centerModelRequired)
                        r->centerModelError = "center_model not found: " + modelPath.string();
                }
            }
        } catch (const std::exception& e) {
            r->useCenterModel = false;
            r->centerModelError = e.what();
        }
        }
        auto mappingPath = r->app / "config" / "top_face" / "tcp_xy_mapping.json";
        if (fs::exists(mappingPath)) {
            try {
                std::ifstream f(mappingPath); json mapping; f >> mapping;
                r->tcpXYEnabled = mapping.at("enabled").get<bool>();
                if (r->tcpXYEnabled) {
                    const auto& size = mapping.at("image_size_wh");
                    if (!size.is_array() || size.size()!=2 || !size[0].is_number_integer() ||
                        !size[1].is_number_integer() || size[0]!=1280 || size[1]!=1024)
                        throw std::runtime_error("image_size_wh must be [1280,1024]");
                    r->tcpXYImageSize = cv::Size(size[0].get<int>(),size[1].get<int>());
                    const auto type=mapping.value("model_type",std::string("affine"));
                    if(type!="affine"&&type!="homography"&&type!="ray_plane"&&type!="calibrated_camera_ray_plane")
                        throw std::runtime_error("unsupported TCP XY model_type");
                    r->tcpXYHeightInput=type=="calibrated_camera_ray_plane";
                    r->tcpXYRayPlane=type=="ray_plane" || r->tcpXYHeightInput;
                    if(r->tcpXYRayPlane){
                        auto number=[](const json& value){
                            if(!value.is_number() || !std::isfinite(value.get<double>()))
                                throw std::runtime_error("ray_plane coefficients must be finite numbers");
                            return value.get<double>();
                        };
                        auto matrix=[&](const char* key, cv::Matx33d& out){
                            const auto& values=mapping.at(key);
                            if(!values.is_array() || values.size()!=3)throw std::runtime_error(std::string(key)+" must be 3x3");
                            for(int row=0;row<3;row++){
                                if(!values[row].is_array() || values[row].size()!=3)throw std::runtime_error(std::string(key)+" must be 3x3");
                                for(int col=0;col<3;col++)out(row,col)=number(values[row][col]);
                            }
                        };
                        matrix(r->tcpXYHeightInput?"camera_matrix":"undistorted_camera_matrix",r->tcpXYCamera);
                        matrix("camera_to_base_rotation",r->tcpXYRotation);
                        const auto& k=r->tcpXYCamera;
                        if(k(0,0)<=0 || k(1,1)<=0 || k(0,1)!=0 || k(1,0)!=0 || k(2,0)!=0 || k(2,1)!=0 || k(2,2)!=1 ||
                           k(0,2)<0 || k(0,2)>=r->tcpXYImageSize.width || k(1,2)<0 || k(1,2)>=r->tcpXYImageSize.height)
                            throw std::runtime_error("invalid pinhole camera matrix");
                        const auto& rotation=r->tcpXYRotation;
                        auto gram=rotation.t()*rotation;
                        for(int row=0;row<3;row++)for(int col=0;col<3;col++)
                            if(std::abs(gram(row,col)-(row==col?1.:0.))>1e-6)throw std::runtime_error("rotation must be orthonormal");
                        if(std::abs(cv::determinant(rotation)-1.)>1e-6)throw std::runtime_error("rotation determinant must be +1");
                        if(r->tcpXYHeightInput){
                            double table=number(mapping.at("table_base_z_mm"));
                            const auto& center=mapping.at("camera_center_base_mm");
                            if(!center.is_array() || center.size()!=3)throw std::runtime_error("camera_center_base_mm must have three values");
                            r->tcpXYOffset={number(center[0]),number(center[1])};
                            double camera=number(center[2]);
                            if(camera<=table)throw std::runtime_error("camera must be above table");
                            r->tcpXYPlaneDifference=table-camera;
                        }else{
                            double reference=number(mapping.at("reference_top_base_z_mm"));
                            double current=number(mapping.at("current_top_base_z_mm"));
                            double camera=number(mapping.at("effective_camera_base_z_mm"));
                            if(camera<=current || camera<=reference)throw std::runtime_error("effective camera must be above both top planes");
                            r->tcpXYPlaneDifference=current-camera;
                            r->tcpXYRadialResidual=number(mapping.at("radial_residual_mm"));
                            const auto& offset=mapping.at("xy_offset_mm");
                            if(!offset.is_array() || offset.size()!=2)throw std::runtime_error("xy_offset_mm must have two values");
                            r->tcpXYOffset={number(offset[0]),number(offset[1])};
                        }
                        if(!std::isfinite(r->tcpXYPlaneDifference))throw std::runtime_error("nonfinite plane difference");
                        double maxRho2=0;
                        for(double u:{0.,double(r->tcpXYImageSize.width-1)})for(double v:{0.,double(r->tcpXYImageSize.height-1)}){
                            cv::Vec3d ray((u-k(0,2))/k(0,0),(v-k(1,2))/k(1,1),1);
                            auto base=rotation*ray;
                            double rho2=ray[0]*ray[0]+ray[1]*ray[1];
                            if(!std::isfinite(rho2) || !std::isfinite(base[2]) || base[2]>=-1e-6)throw std::runtime_error("ray must point down without image-domain poles");
                            maxRho2=std::max(maxRho2,rho2);
                        }
                        if(!std::isfinite(rotation(2,2)) || rotation(2,2)>=-1e-6)throw std::runtime_error("optical axis must point down");
                        if(std::abs(r->tcpXYRadialResidual/r->tcpXYPlaneDifference)*maxRho2>.1)
                            throw std::runtime_error("radial residual exceeds 10 percent of plane scale");
                    }else{
                    r->tcpXYHomography=type=="homography";
                    const int rows=r->tcpXYHomography?3:2;
                    const std::string shape=r->tcpXYHomography?"3x3":"2x3";
                    const auto& matrix = mapping.at("matrix");
                    if (!matrix.is_array() || matrix.size()!=size_t(rows))
                        throw std::runtime_error("matrix must be "+shape);
                    for (int row=0;row<rows;row++) {
                        if (!matrix[row].is_array() || matrix[row].size()!=3)
                            throw std::runtime_error("matrix must be "+shape);
                        for (int col=0;col<3;col++) {
                            if (!matrix[row][col].is_number())
                                throw std::runtime_error("matrix coefficients must be finite numbers");
                            double value=matrix[row][col].get<double>();
                            if (!std::isfinite(value))
                                throw std::runtime_error("matrix coefficients must be finite numbers");
                            if(r->tcpXYHomography)r->tcpXYH(row,col)=value;
                            else r->tcpXY(row,col)=value;
                        }
                    }
                    if(r->tcpXYHomography){
                        auto& h=r->tcpXYH;
                        double scale=0;for(double value:h.val)scale=std::max(scale,std::abs(value));
                        if(scale==0)throw std::runtime_error("singular homography");
                        for(double& value:h.val)value/=scale;
                        // Compare determinant to its terms, not an absolute matrix scale.
                        double a=h(0,0)*h(1,1)*h(2,2),b=h(0,1)*h(1,2)*h(2,0),c=h(0,2)*h(1,0)*h(2,1);
                        double d=h(0,2)*h(1,1)*h(2,0),e=h(0,1)*h(1,0)*h(2,2),f=h(0,0)*h(1,2)*h(2,1);
                        double det=a+b+c-d-e-f,terms=std::abs(a)+std::abs(b)+std::abs(c)+std::abs(d)+std::abs(e)+std::abs(f);
                        if(terms==0||std::abs(det)<=1e-12*terms)throw std::runtime_error("singular homography");
                        int sign=0;
                        for(double u:{0.,double(r->tcpXYImageSize.width)})for(double v:{0.,double(r->tcpXYImageSize.height)}){
                            double denominator=homographyDenominator(h,u,v);
                            int current=denominator>0?1:-1;
                            if(sign&&current!=sign)throw std::runtime_error("homography denominator crosses zero in image domain");
                            sign=current;
                        }
                    }
                    }
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("tcp_xy_mapping.json: " + std::string(e.what()));
            }
        }
        loadBackground(*r, true);
        if(r->circles&&r->circleMaxRadius>=.5*std::min(r->background.cols,r->background.rows))
            throw std::runtime_error("circle radius bounds exceed the background image");
        if (r->tcpXYEnabled && r->background.size()!=r->tcpXYImageSize)
            throw std::runtime_error("tcp_xy_mapping.json: background dimensions mismatch");
#ifdef USE_X3_NPU
        if(!r->circles){
        if (!r->fast) throw std::runtime_error("X3 BPU requires fast mode");
        if (r->background.size() != cv::Size(1280, 1024))
            throw std::runtime_error("X3 feature model requires 1280x1024 background");
        r->npu.load((cfg.parent_path() / config.value("npu_model", std::string("background_features.bin"))).string());
        }
#endif
        reply.clear();
        runtime = std::move(r);
        return 0;
    } catch (const std::exception& e) {
        reply=e.what();
        std::cerr << "top_face_init: " << e.what() << '\n';
        runtime.reset();
        return -1;
    }
}

int top_face_detect(const cv::Mat& bgrOrGray, const nlohmann::json& request, nlohmann::json& result) {
    std::lock_guard<std::mutex> lock(runtimeMutex);
    try {
        if (!runtime) throw std::runtime_error("top_face is not initialized");
        cv::Mat image;
        if (bgrOrGray.empty()) throw std::runtime_error("empty image");
        if (bgrOrGray.channels() == 3)
            cv::cvtColor(bgrOrGray, image, cv::COLOR_BGR2GRAY);
        else if (bgrOrGray.channels() == 1)
            image = bgrOrGray;
        else
            throw std::runtime_error("image must be grayscale or BGR");
        result = runtime->circles ? detectCircles(*runtime,image,request)
                                  : detectRectangles(*runtime,image,request);
        return result.value("ret-code", 0);
    } catch (const std::exception& e) {
        result = {{"ret-code", -1},
                  {"ret-msg", e.what()},
                  {"best_img_idx", -1},
                  {"top_face_count", 0},
                  {"objects", nlohmann::json::array()},
                  {"pts_top_face", nlohmann::json::array()}};
        return -1;
    }
}

void top_face_close() {
    std::lock_guard<std::mutex> lock(runtimeMutex);
    runtime.reset();
}

std::string top_face_init_error() { return reply; }

double top_face_request_height(const json& request) {
    if(!request.contains("workpiece_height_mm"))return std::numeric_limits<double>::quiet_NaN();
    const auto& value=request.at("workpiece_height_mm");
    if(!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>()<0)
        throw std::runtime_error("workpiece_height_mm must be a finite nonnegative number in mm");
    return value.get<double>();
}

bool top_face_tcp_xy(double u, double v, cv::Size imageSize, double& x, double& y) {
    return top_face_tcp_xy(u,v,imageSize,x,y,std::numeric_limits<double>::quiet_NaN(),cv::Matx33d::eye());
}

bool top_face_tcp_xy(double u, double v, cv::Size imageSize, double& x, double& y,
                     double height, const cv::Matx33d& imageCamera, bool unflippedUndistorted) {
    std::lock_guard<std::mutex> lock(runtimeMutex);
    if (!runtime) throw std::runtime_error("top_face is not initialized");
    if (!runtime->tcpXYEnabled) return false;
    if (imageSize!=runtime->tcpXYImageSize)
        throw std::runtime_error("tcp_xy_mapping.json: input dimensions mismatch");
    if (!std::isfinite(u) || !std::isfinite(v))
        throw std::runtime_error("tcp_xy_mapping.json: nonfinite pixel center");
    double mappedX,mappedY;
    if(runtime->tcpXYRayPlane){
        if(u<0 || v<0 || u>imageSize.width-1 || v>imageSize.height-1)
            throw std::runtime_error("tcp_xy_mapping.json: pixel center outside image domain");
        const auto& k=runtime->tcpXYHeightInput?imageCamera:runtime->tcpXYCamera;
        double planeDifference=runtime->tcpXYPlaneDifference;
        if(runtime->tcpXYHeightInput){
            if(!std::isfinite(height) || height<0)throw std::runtime_error("workpiece_height_mm is required and must be nonnegative");
            if(!unflippedUndistorted)throw std::runtime_error("height mapping requires unflipped undistorted image pixels");
            for(double value:k.val)if(!std::isfinite(value))throw std::runtime_error("nonfinite image camera matrix");
            if(k(0,0)<=0 || k(1,1)<=0 || k(0,1)!=0 || k(1,0)!=0 || k(2,0)!=0 || k(2,1)!=0 || k(2,2)!=1)
                throw std::runtime_error("invalid image camera matrix");
            planeDifference+=height;
            if(!std::isfinite(planeDifference) || planeDifference>=0)throw std::runtime_error("workpiece top must be below camera");
        }
        const auto& rotation=runtime->tcpXYRotation;
        cv::Vec3d ray((u-k(0,2))/k(0,0),(v-k(1,2))/k(1,1),1);
        auto base=rotation*ray;
        if(!std::isfinite(base[2]) || base[2]>=-1e-6)
            throw std::runtime_error("tcp_xy_mapping.json: invalid downward ray");
        double qx=base[0]/base[2],qy=base[1]/base[2];
        double residual=runtime->tcpXYRadialResidual*(ray[0]*ray[0]+ray[1]*ray[1]);
        mappedX=runtime->tcpXYOffset[0]+planeDifference*qx+residual*(qx-rotation(0,2)/rotation(2,2));
        mappedY=runtime->tcpXYOffset[1]+planeDifference*qy+residual*(qy-rotation(1,2)/rotation(2,2));
    }else if(runtime->tcpXYHomography){
        const auto& h=runtime->tcpXYH;
        double denominator;
        try {denominator=homographyDenominator(h,u,v);}
        catch(const std::exception& e){throw std::runtime_error("tcp_xy_mapping.json: "+std::string(e.what()));}
        mappedX=(h(0,0)*u+h(0,1)*v+h(0,2))/denominator;
        mappedY=(h(1,0)*u+h(1,1)*v+h(1,2))/denominator;
    }else{
        const auto& a=runtime->tcpXY;
        mappedX=a(0,0)*u+a(0,1)*v+a(0,2);
        mappedY=a(1,0)*u+a(1,1)*v+a(1,2);
    }
    if (!std::isfinite(mappedX) || !std::isfinite(mappedY))
        throw std::runtime_error("tcp_xy_mapping.json: nonfinite mapped coordinates");
    x=mappedX;y=mappedY;
    return true;
}
