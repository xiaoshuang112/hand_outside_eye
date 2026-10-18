#pragma once
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <cstring>

// Persistent model and cached tensors; calls are serialized by the ABI mutex.
struct X3Features {
    hbPackedDNNHandle_t packed=nullptr;
    hbDNNHandle_t model=nullptr;
    hbDNNTensor input{},output{};
    static void check(int code,const char* operation) {
        if(code)throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(code));
    }
    ~X3Features(){
        if(input.sysMem[0].virAddr)hbSysFreeMem(&input.sysMem[0]);
        if(output.sysMem[0].virAddr)hbSysFreeMem(&output.sysMem[0]);
        if(packed)hbDNNRelease(packed);
    }
    void load(const std::string& path){
        const char* file=path.c_str();
        check(hbDNNInitializeFromFiles(&packed,&file,1),"load BPU model");
        const char** names=nullptr;int count=0;
        check(hbDNNGetModelNameList(&names,&count,packed),"model names");
        if(count!=1)throw std::runtime_error("BPU file must contain one feature model");
        check(hbDNNGetModelHandle(&model,packed,names[0]),"model handle");
        check(hbDNNGetInputTensorProperties(&input.properties,model,0),"input properties");
        check(hbDNNGetOutputTensorProperties(&output.properties,model,0),"output properties");
        {
            const auto* p=&output.properties;
            if(p->tensorLayout!=HB_DNN_LAYOUT_NCHW||p->tensorType!=HB_DNN_TENSOR_TYPE_S32||p->validShape.numDimensions!=4||p->quantiType!=SCALE||p->scale.scaleLen!=2||p->scale.zeroPointLen!=0)
                throw std::runtime_error("feature output requires NCHW int32 with per-channel scales");
            const int expected[]={1,2,512,640};
            for(int i=0;i<4;i++)if(p->validShape.dimensionSize[i]!=expected[i]||p->alignedShape.dimensionSize[i]!=expected[i])
                throw std::runtime_error("feature model shape must be 1x2x512x640 without padding");
            if(p->alignedByteSize!=2*512*640*int(sizeof(float)))throw std::runtime_error("unexpected feature tensor storage");
        }
        const auto& p=input.properties;
        const int valid[]={1,512,640,2},aligned[]={1,512,640,4};
        if(p.tensorLayout!=HB_DNN_LAYOUT_NHWC||p.tensorType!=HB_DNN_TENSOR_TYPE_S8||p.validShape.numDimensions!=4||p.quantiType!=SCALE||p.scale.scaleLen!=1||p.scale.scaleData[0]!=1||p.scale.zeroPointLen!=0)
            throw std::runtime_error("feature input requires NHWC signed int8 at unit scale");
        for(int i=0;i<4;i++)if(p.validShape.dimensionSize[i]!=valid[i]||p.alignedShape.dimensionSize[i]!=aligned[i])throw std::runtime_error("unexpected BPU input shape");
        if(p.alignedByteSize!=512*640*4)throw std::runtime_error("unexpected BPU input storage");
        check(hbSysAllocCachedMem(&input.sysMem[0],input.properties.alignedByteSize),"input allocation");
        std::memset(input.sysMem[0].virAddr,0,input.properties.alignedByteSize);
        check(hbSysAllocCachedMem(&output.sysMem[0],output.properties.alignedByteSize),"output allocation");
    }
    void run(cv::Mat& current,cv::Mat& background){
        if(current.size()!=cv::Size(640,512)||background.size()!=current.size())throw std::runtime_error("BPU requires 1280x1024 input images");
        cv::Mat a,b;current.convertTo(a,CV_8S,1,-128);background.convertTo(b,CV_8S,1,-128);
        cv::Mat packedInput(512,640,CV_8SC4,input.sysMem[0].virAddr);
        const cv::Mat sources[]={a,b};const int channels[]={0,0,1,1};
        cv::mixChannels(sources,2,&packedInput,1,channels,2);
        check(hbSysFlushMem(&input.sysMem[0],HB_SYS_MEM_CACHE_CLEAN),"input cache flush");
        hbDNNTaskHandle_t task=nullptr;hbDNNTensor* outputs=&output;
        hbDNNInferCtrlParam control;HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
        int code=hbDNNInfer(&task,&outputs,&input,model,&control);
        if(code){if(task)hbDNNReleaseTask(task);check(code,"BPU infer");}
        code=hbDNNWaitTaskDone(task,1000);hbDNNReleaseTask(task);check(code,"BPU wait");
        check(hbSysFlushMem(&output.sysMem[0],HB_SYS_MEM_CACHE_INVALIDATE),"output cache flush");
        auto* features=static_cast<int32_t*>(output.sysMem[0].virAddr);
        cv::Mat(512,640,CV_32S,features).convertTo(current,CV_32F,output.properties.scale.scaleData[0],128);
        cv::Mat(512,640,CV_32S,features+512*640).convertTo(background,CV_32F,output.properties.scale.scaleData[1],128);
    }
};
