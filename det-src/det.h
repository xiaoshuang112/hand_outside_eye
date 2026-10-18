#pragma once

// #define DEBUG_STITCH_IMAGE  // 全部的调试细节，输出大量中间信息，按需使用
// #define SAVE_STITCH_IMAGE_RESULT // 输出拼接结果，占用一定运行时间，调试时建议打开

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#else
  
#endif


#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <iostream>
#include <optional>


#ifdef _WIN32

#define ALVA_DLL_API extern "C" __declspec(dllexport) 

#else
#define ALVA_DLL_API extern "C"
#define __cdecl    
#endif

ALVA_DLL_API int __cdecl AI_det_init(const char* appDir);

typedef int (__cdecl *AI_det_callback)(char* msg, int msg_len, int flag);

/*
- AI_det
- 检测工件
- @img:
如果img_w>0 && img_h==0 ，图像jpeg数据，数据长度为img_w
如果img_w 和 img_h 都>1时，图像bgr数据
如果img_w 和 img_h 都为0时，图像文件名称，

- @callback: 回调函数，默认为nullptr。用于在采集图像保存后通知调用方
- input_json 可选字段 lunzi_only: 1/true 时同 lunzi 返回 pts_lunzi 为 [x,y,z,r]，但中心取 lunzi 矩形框中心像素坐标，深度不用 hole
- 如果算法返回 "best_img_idx":-1   意思是图像都不合格
- @json_len: 返回的json的长度
- @return: 0: 成功，其他：失败
*/
ALVA_DLL_API int __cdecl AI_det(char** img, int* img_w, int* img_h,
	const char* input_json,char** ret_json,int* json_len,
	AI_det_callback callback);

ALVA_DLL_API int __cdecl AI_det_close();
