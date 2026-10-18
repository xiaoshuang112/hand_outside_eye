#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <Windows.h>

#else
#include<unistd.h>
#include <dlfcn.h>
#define __cdecl
#endif

#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread> // for std::thread
#include <chrono>
#include <cstring>
#include "json.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include <sstream>



namespace fs = std::filesystem;
using json = nlohmann::json;

#define VERSION "1.0.0"
#define NOTE ""

std::string CurDir;
std::string LogDir;
std::string ExeName;
std::string g_directory = "";
std::string g_depth_str = "";
std::string g_exposure_list_str = "";


#ifdef _WIN32
HMODULE hMod_sp = nullptr;
#else
void* hMod_sp = nullptr;
#endif


typedef int(__cdecl* LP_AI_det_init)(const char* appDir);
LP_AI_det_init AI_det_init = NULL;


typedef int(__cdecl* AI_det_callback)(char* msg, int msg_len, int flag);

typedef int(__cdecl* LP_AI_det)(char** img,int* img_w,int* img_h,
	const char* input_json,char** ret_json,int* json_len,
	AI_det_callback callback);
LP_AI_det AI_det = NULL;

typedef int(__cdecl* LP_AI_det_close)();
LP_AI_det_close AI_det_close = NULL;


std::vector<std::string> str_split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {  // 可选：过滤空字符串
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<int> parse_exposure_list(const std::string& s) {
    std::vector<int> out;
    for (const auto& t : str_split(s, ',')) {
        try {
            size_t n = 0;
            int us = std::stoi(t, &n);
            if (n > 0 && us > 0)
                out.push_back(us);
        } catch (...) {}
    }
    return out;
}

bool GetBasePathInfo() {

	char szPath[2048] = { 0 };
	char* p = szPath;
#if WIN32
	GetModuleFileNameA(NULL, szPath, sizeof(szPath)); 
	char split = '\\';
#else
	readlink("/proc/self/exe", szPath, sizeof(szPath));
	char split = '/';
#endif

	while (strchr(p, split)) //找路径分隔符号
	{
		p = strchr(p, split);
		p++;
	}
	char t = *p;
	*p = 0; // 路径 放在 szPath里了
	CurDir = szPath;
	LogDir = CurDir + "logs/";
	*p = t;
	ExeName = p;

	return true;
}


int det_init(const std::string& str_path,const std::string& init_param)
{
#ifdef _WIN32
    std::string libPath = str_path + "ai-det.dll";
    hMod_sp = LoadLibraryA(libPath.c_str());
    if (hMod_sp) {

        AI_det_init = (LP_AI_det_init)GetProcAddress(hMod_sp, "AI_det_init");
        AI_det = (LP_AI_det)GetProcAddress(hMod_sp, "AI_det");
        AI_det_close = (LP_AI_det_close)GetProcAddress(hMod_sp, "AI_det_close");

        if (NULL == AI_det_init||NULL == AI_det||NULL == AI_det_close)
        {
            printf("Load ai-det functions failed\n");
            FreeLibrary(hMod_sp);
            hMod_sp = nullptr;
            return -2;
        }
    }
    else
    {
        printf("Load libsplit failed\n");
        return -2;
    }


#else
    std::string libPath = str_path + "libai-det.so";
    hMod_sp = dlopen(libPath.c_str(), RTLD_LAZY);
    if (hMod_sp) {
        AI_det_init = (LP_AI_det_init)dlsym(hMod_sp, "AI_det_init");
        AI_det = (LP_AI_det)dlsym(hMod_sp, "AI_det");
        AI_det_close = (LP_AI_det_close)dlsym(hMod_sp, "AI_det_close");

        if (NULL == AI_det_init||NULL == AI_det||NULL == AI_det_close)
        {
            printf("Load ai-det functions failed\n");
            dlclose(hMod_sp);
            hMod_sp = nullptr;
            return -2;
        }
    }
    else
    {
        printf("Load libai-det failed\n");
        return -2;
    }
#endif

    int ret = -1;
    if ("test"==init_param || "clean"==init_param)
        ret = AI_det_init(init_param.c_str());
    else
        ret = AI_det_init(str_path.c_str());

    return ret;
}

int det_close()
{
    AI_det_close();

#ifdef _WIN32

    if (hMod_sp) {
        FreeLibrary(hMod_sp);
        hMod_sp = nullptr;
    }
#else

    if (hMod_sp) {
        dlclose(hMod_sp);
        hMod_sp = nullptr;
    }
#endif

    return 0;
}

json det_file(std::string file_name, std::string xialiao_type="plate_empty",
    std::string test_mode="")
{
    bool clean_mode = ("clean" == test_mode);

    char* ret_json = NULL;
    int json_len = 0;

    json j;

    bool lunzi_only = (xialiao_type == "lunzi_only");
    if (lunzi_only)
        xialiao_type = "lunzi";

    if ("auto"==xialiao_type||"preview"==xialiao_type||
        "board_distance"==xialiao_type||
        "collect_data"==xialiao_type
    )
    {
        j["calibration_type"] = xialiao_type;
    }
    else
    {
        j["xialiao_type"] = xialiao_type;
    }

    if ("preview"==xialiao_type && ""!=test_mode)
    {
        // 预览模式下，传入的 test_mode 实际上是曝光时间
        std::vector<int> exposure_times = parse_exposure_list(test_mode);
        if (exposure_times.size() > 1) {
            j["exposure_list"] = exposure_times;
            // CLI 多曝光预览视为拍 top_face 背景，写入 background.bmp
            j["save_top_face_background"] = true;
            j["xialiao_type"] = "top_face";
        } else
        {
            j["exposure_time"] = std::atoi(test_mode.c_str());
        }
        
    }


    if ("preview"!=xialiao_type && ""!=g_exposure_list_str)
    {
        // 非预览检测：argv[5] 逗号分隔曝光档透传为 exposure_list（板端验证融合检测用）
        std::vector<int> exposure_times = parse_exposure_list(g_exposure_list_str);
        if (!exposure_times.empty())
            j["exposure_list"] = exposure_times;
    }

    if (lunzi_only)
        j["lunzi_only"] = 1;


    int img_w = 0;
    int img_h = 0;
    char* img = nullptr;
    cv::Mat image;
    if (fs::exists(file_name) && ("test"==test_mode || "clean"==test_mode))
    {
        // 调试/清理模式：读取图片文件
        j["test_mode"] = 1;
        j["color_image"] = file_name;
        if (file_name.length() >= 4) {
            file_name.erase(file_name.length() - 4);
            file_name += ".png";
        }
        j["depth_image"] = file_name;
        if (""!=g_depth_str && 1==std::atoi(g_depth_str.c_str()))
        {
            j["top_capture"] = 1;
        }        

        // j["top_capture"] = 1;
        // j["rect"] = {
        // 404,
        // 200,
        // 1186,
        // 200,
        // 1186,
        // 620,
        // 404,
        // 620};
    }

    if (clean_mode)
    {
        j["clean_mode"] = true;
    }


    if (""!=g_depth_str&& std::atoi(g_depth_str.c_str())>10)
    {
        j["ref_depth"] = std::atoi(g_depth_str.c_str());
    }


    AI_det(&img, &img_w, &img_h, j.dump().c_str(), &ret_json, &json_len, nullptr);

    std::cout<< ret_json<<std::endl;

    // 解析返回的JSON
    json ret_j;
    if (ret_json && json_len > 0)
    {
        try {
            ret_j = json::parse(ret_json);
        } catch (...) {}
    }
    return ret_j;
}


void det_dir(std::string dir_name,std::string flag="",
    std::string test_mode="")
{
    bool clean_mode = ("clean" == test_mode);
    int total = 0, deleted = 0;

    for (auto& p : fs::directory_iterator(dir_name))
    {
        if (p.path().extension().compare(".jpg") == 0
            || p.path().extension().compare(".jpeg") == 0
            || p.path().extension().compare(".png") == 0
            || p.path().extension().compare(".bmp") == 0)
        {
            total++;
            json ret_j = det_file(p.path().string(),flag,test_mode);

            if (clean_mode && !ret_j.empty())
            {
                int ret_code = ret_j.value("ret-code", -1);

                if (0 != ret_code)
                {
                    std::string img_path = p.path().string();
                    std::cout << "  [CLEAN] 删除无检测结果图片: " << img_path << std::endl;

                    // 删除图片
                    std::error_code ec;
                    fs::remove(img_path, ec);

                    // 同时删除对应的深度图(.png)
                    std::string depth_path = p.path().stem().string() + ".png";
                    auto depth_full = p.path().parent_path() / depth_path;
                    if (fs::exists(depth_full))
                    {
                        fs::remove(depth_full, ec);
                    }

                    if (!ec)
                        deleted++;
                }
            }
        }
    }

    if (clean_mode)
    {
        std::cout << "\n[CLEAN] 扫描: " << total << " 张, 删除: " << deleted << " 张" << std::endl;
    }
}


/**
 * @brief 检测矩形物体并输出四条边的坐标
 *
 * @param color_path     彩色图路径
 * @param depth_path     深度图路径
 * @param output_path    可选，结果可视化图的保存路径
 * @return true 检测成功
 */
extern bool detectCenterRectAndPrint(const std::string& color_path,
                              const std::string& depth_path,
                              const std::string& output_path = "",
                              double phys_w = 0,
                              double phys_h = 0);


/**
 * @brief 按序号批量检测（00001, 00002, 00003 等）
 *
 * @param dir_path   图像所在目录
 * @param seq        序号，如 "00001"
 * @param output_dir 结果输出目录（空则不保存图片）
 */
extern bool detectCenterRectBySeq(const std::string& dir_path,
                           const std::string& seq,
                           const std::string& output_dir = "",
                           double phys_w = 0,
                           double phys_h = 0);



int main(int argc, char* argv[]) {

    std::cout << "=========================================" << std::endl;
    std::cout << "@@@@@@@@  DET TOOLS  @@@@@@@@@" << std::endl;
    std::cout << "=========================================" << std::endl;

    if (argc > 1)
    {
        g_directory = argv[1];
    }

    if (!GetBasePathInfo()) {
        return -1;
    }

    std::string det_flag = "";
    if (argc > 2)
    {
        det_flag = argv[2];
    }

    // ---- 深度+彩色联合矩形检测模式（不依赖 AI 检测库） ----
    if (det_flag == "rect") {
        std::string out_path = "";
        if (argc > 3) out_path = argv[3];

        // g_directory 为图像所在目录，argc>3 时 argv[3] 为序号如 00001
        std::string seq = "";
        if (argc > 4) seq = argv[4];

        // 可选：物理尺寸 (mm), det-tool-lib <dir> rect <out> <seq> <phys_w> <phys_h>
        double phys_w = 0, phys_h = 0;
        if (argc > 6) { phys_w = std::atof(argv[5]); phys_h = std::atof(argv[6]); }

        if (!seq.empty()) {
            detectCenterRectBySeq(g_directory, seq, out_path, phys_w, phys_h);
        } else {
            // 按目录批量检测所有 .jpg 文件
            for (auto& p : fs::directory_iterator(g_directory)) {
                if (p.path().extension() == ".jpg") {
                    std::string name = p.path().stem().string();
                    // 跳过 gray 文件和已有结果文件
                    if (name.find("-gray") != std::string::npos) continue;
                    if (name.find("_result") != std::string::npos) continue;
                    std::string depth_path = g_directory + "/" + name + ".png";
                    if (fs::exists(depth_path)) {
                        std::cout << "\n--- " << name << " ---" << std::endl;
                        detectCenterRectAndPrint(p.path().string(), depth_path,
                            out_path.empty() ? "" : out_path + "/" + name + "_result.jpg",
                            phys_w, phys_h);
                    }
                }
            }
        }
        return 0;
    }

    // ---- 原 AI 检测模式 ----
    std::string init_param = "";
    if (argc > 3)
    {
        init_param = argv[3];
    }

    if (argc > 4)
    {
        g_depth_str = argv[4];
    }

    // 可选曝光档（逗号分隔，如 2000,5000,8000）→ j["exposure_list"]
    // det-tool-lib <dir> <type> [test|clean|""] [ref_depth] [exp1,exp2,exp3]
    if (argc > 5)
    {
        g_exposure_list_str = argv[5];
    }


    int ret = det_init(CurDir, init_param);
    if (0 != ret)
    {
        det_close();
        return -1;
    }

    if ("" == g_directory)
    {
        g_directory = CurDir + "1.jpg";
    }

    if (fs::is_directory(g_directory))
    {
        det_dir(g_directory, det_flag,init_param);
    }
    else
    {
        json ret_j = det_file(g_directory, det_flag,init_param);

        if ("clean" == init_param && !ret_j.empty())
        {
            int ret_code = ret_j.value("ret-code", -1);

            if (0 != ret_code)
            {
                std::cout << "  [CLEAN] 删除无检测结果图片: " << g_directory << std::endl;
                std::error_code ec;
                fs::remove(g_directory, ec);
            }
        }
    }

    det_close();

    return 0;
}
