#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

using InitFn = int (*)(const char*);
using DetFn = int (*)(
    char**, int*, int*, const char*, char**, int*, void*);
using CloseFn = int (*)();

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::cerr << "usage: ai-det-offline-runner COLOR DEPTH LIBRARY "
                     "DET_TYPE RECT_JSON REPEAT [TOP_CAPTURE]\n";
        return 2;
    }

    void* handle = dlopen(argv[3], RTLD_NOW);
    if (handle == nullptr)
    {
        std::cerr << dlerror() << '\n';
        return 3;
    }

    auto init = reinterpret_cast<InitFn>(dlsym(handle, "AI_det_init"));
    auto detect = reinterpret_cast<DetFn>(dlsym(handle, "AI_det"));
    auto close = reinterpret_cast<CloseFn>(dlsym(handle, "AI_det_close"));
    if (init == nullptr || detect == nullptr || close == nullptr)
    {
        std::cerr << "AI_det exports are incomplete\n";
        dlclose(handle);
        return 4;
    }
    if (init("test") != 0)
    {
        std::cerr << "AI_det_init failed\n";
        dlclose(handle);
        return 5;
    }

    const int top_capture = argc >= 8 ? std::atoi(argv[7]) : 0;
    const std::string request =
        "{\"xialiao_type\":\"" + std::string(argv[4]) +
        "\",\"top_capture\":" + std::to_string(top_capture) +
        ",\"test_mode\":1,\"color_image\":\"" +
        std::string(argv[1]) + "\",\"depth_image\":\"" +
        std::string(argv[2]) + "\",\"rect\":" + argv[5] + "}";
    const int repeat = std::max(1, std::atoi(argv[6]));
    int result = 0;
    for (int iteration = 0; iteration < repeat; ++iteration)
    {
        char* image = nullptr;
        int width = 0;
        int height = 0;
        char* output = nullptr;
        int output_size = 0;
        result = detect(
            &image, &width, &height, request.c_str(),
            &output, &output_size, nullptr);
        if (result != 0)
        {
            std::cerr << "AI_det returned " << result
                      << " at iteration " << iteration << '\n';
            break;
        }
        std::cout << "RESULT_JSON=";
        if (output != nullptr && output_size > 0)
            std::cout.write(output, output_size);
        std::cout << '\n';
    }

    close();
    dlclose(handle);
    return result == 0 ? 0 : 6;
}
