cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME det-tool)

option(USE_RDK "Build with RDK DNN API instead of TensorRT" ON)
option(BUILD_OFFLINE_TESTS "Build offline AI_det regression runners" OFF)

if(USE_RDK)
    project(${PROJECT_NAME} LANGUAGES C CXX)
    add_definitions(-DuseRDK)

    message("USE_RDK  Build with RDK DNN API instead of TensorRT")
else()
    project(${PROJECT_NAME} LANGUAGES C CXX CUDA)
endif()

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(useORB FALSE CACHE BOOL "Use Orbbec camera")
set(useOAK FALSE CACHE BOOL "Use OAK-D camera")
set(useMVS TRUE CACHE BOOL "Use MVS camera")
set(useUSB FALSE CACHE BOOL "Use USB camera")

if(WIN32)

else()
# 未显式指定时保留原先的 Debug 默认值；调用方传入
# -DCMAKE_BUILD_TYPE=Release 时不得被这里覆盖。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DREL_DBG)
endif()
endif()

# 下料标志
add_definitions(-DXIALIAO)


# 设置策略CMP0091为NEW，新策略
# 为了解决 [MSVC][Error] 检测到“RuntimeLibrary”的不匹配项_ 值“MT_StaticRelease”不匹配值“MD_DynamicRelease”
if (POLICY CMP0091)
    cmake_policy (SET CMP0091 NEW)
endif (POLICY CMP0091)

# 设置CMP0104策略为NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# 设置CMP0146策略为NEW
# if(POLICY CMP0146)
#   cmake_policy(SET CMP0146 NEW)
# endif()


# 获取系统架构信息
message("System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# 判断架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH_TYPE "x86_64")
    set(ARCH_FLAG "-DX86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_TYPE "ARM64")
    set(ARCH_FLAG "-DARM64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armv7l")
    set(ARCH_TYPE "ARMv7")
    set(ARCH_FLAG "-DARM")
else()
    set(ARCH_TYPE "Unknown")
endif()

message("Architecture Type: ${ARCH_TYPE} ${ARCH_FLAG}  ")
add_definitions(${ARCH_FLAG})

# Define the path to TensorRT installation based on architecture
if(WIN32)
	set(ORBBEC_SDK_V2_DIR "F:/Program Files/OrbbecSDK 2.4.11")
	set(CMAKE_CUDA_ARCHITECTURES 80 86)
    set(TENSORRT_PATH "F:/soft/TensorRT-8.6.1.6")

    set(CGX_SDK_LIBS )      
else()
    # # public
    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${SRC_DIR})


    # 根据架构设置不同的编译选项
    if(ARCH_TYPE STREQUAL "x86_64")
		set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
        set(TENSORRT_PATH "/work/soft/TensorRT-10.10.0.31")
        set(CUDA_DIR "/usr/local/cuda")
    elseif(ARCH_TYPE STREQUAL "ARM64")
		set(CMAKE_CUDA_ARCHITECTURES 87)
        set(TENSORRT_PATH "/usr/local/cuda")
        set(CUDA_DIR "/usr/local/cuda")
    endif()


endif()


# Define the path to OpenCV installation

# Allow overriding TensorRT and OpenCV paths via command line
# e.g., cmake -DTENSORRT_PATH="path/to/TensorRT" -DOpenCV_DIR="path/to/OpenCV" ..
# option(TENSORRT_PATH_OPTION "Path to TensorRT installation" ${TENSORRT_PATH})
# set(TENSORRT_PATH ${TENSORRT_PATH_OPTION} CACHE PATH "Path to TensorRT installation")

# Find OpenCV
find_package(OpenCV 4.10 REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR.")
endif()

if(NOT USE_RDK)
# Find CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()
endif()

if(useOAK)
    # DepthAI SDK v3.5.0
    set(depthai_DIR "/usr/local/")
    find_package(depthai REQUIRED CONFIG)
endif()



if(NOT USE_RDK)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
endif()

# Include directories for TensorRT (non-RDK only)
if(USE_RDK)
    include_directories(SYSTEM /usr/include/dnn)
    link_directories(/usr/lib)
else()
    include_directories(${TENSORRT_PATH}/include)
    include_directories(${CUDA_DIR}/include)
endif()

# Include directory for your project
include_directories(${CMAKE_SOURCE_DIR}/include)

# SpdLog
include_directories(${CMAKE_SOURCE_DIR}/det-src/3rdParty/spdlog/include/)

set(INI_DIR ${CMAKE_SOURCE_DIR}/det-src/inih)
include_directories(${INI_DIR})
include_directories(${CMAKE_SOURCE_DIR}/det-src/utils)

message("ORBBEC_SDK_V2_DIR: ${ORBBEC_SDK_V2_DIR}")
include_directories(${ORBBEC_SDK_V2_DIR}/include)
link_directories(${ORBBEC_SDK_V2_DIR}/lib)

# MVS SDK
if(useMVS)

    set(MVS_SDK_DIR "/opt/MVS" CACHE PATH "MVS SDK path")
    include_directories(${MVS_SDK_DIR}/include)

    message("MVS_SDK_DIR: ${MVS_SDK_DIR}")

    if(ARCH_TYPE STREQUAL "x86_64")
        if(ARCH_TYPE STREQUAL "x86_64")
            link_directories(${MVS_SDK_DIR}/lib/64)
        else()
            link_directories(${MVS_SDK_DIR}/lib/32)
        endif()
    elseif(ARCH_TYPE STREQUAL "ARM64")
        link_directories(${MVS_SDK_DIR}/lib/aarch64)
    endif()


endif()


# # public
set(SPLIT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/det-src)
include_directories(${SPLIT_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${SPLIT_DIR}/3rdParty/eigen-3.4.0/)
# include_directories(${SPLIT_DIR}/3rdParty/libjpeg-turbo64/include/)
# link_directories(${SPLIT_DIR}/3rdParty/libjpeg-turbo64/lib/)

# header-only spdlog 编进 so 时，禁止 GNU unique，避免和宿主
# libspdlog.so.1 合并同一份 registry（Qt app 测距/拍背景会直接崩）。
set_source_files_properties(det-src/Log.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-gnu-unique;-fvisibility=hidden")

SET(SRC_LISTS
    det-src/Log.cpp
    det-src/inih/ini.c
    det-src/inih/INIReader.cpp
    det-src/charuco_detector.cpp
  	det-src/det.cpp
    det-src/top_face.cpp
    det-src/center_heatmap_infer.cpp
    det-src/decode.c
    det-src/utils/putText.cpp
    det-src/utils/Texts.cpp
    det-src/utils/Util.cpp
)

if(NOT USE_RDK)
    list(APPEND SRC_LISTS
        det-src/det-rect.cpp
        det-src/exposure_fusion.cpp
    )
endif()

if(USE_RDK)
    list(APPEND SRC_LISTS
        det-src/yolov11_rdk.cpp
        det-src/nv12_utils.cpp
    )
else()
    list(APPEND SRC_LISTS
        det-src/yolov11.cpp
        det-src/preprocess.cu
    )
endif()

SET(SRC_LIBS 
)

if(useORB)
    add_definitions(-DuseORB)
    list(APPEND SRC_LISTS
        det-src/cameras/OrbbecCamera.cpp
    )
    list(APPEND SRC_LIBS
        OrbbecSDK
    )
endif()

if(useOAK)
    add_definitions(-DuseOAK)
    list(APPEND SRC_LISTS
        det-src/cameras/OAKCamera.cpp
    )
    list(APPEND SRC_LIBS
        depthai::core
    )
endif()

if(USE_RDK)
    add_definitions(-DUSE_RDK)
    if(CMAKE_SYSROOT MATCHES "rdk_x3" OR CMAKE_TOOLCHAIN_FILE MATCHES "rdk_x3")
        add_definitions(-DUSE_X3_NPU)
        message(STATUS "top_face: Horizon X3 BPU features enabled")
    endif()
else()
    
endif()

if(useMVS)
    add_definitions(-DuseMVS)
    list(APPEND SRC_LISTS
        det-src/cameras/MVSCamera.cpp
    )
    list(APPEND SRC_LIBS
        MvCameraControl
    )
endif()

if(useUSB)
    add_definitions(-DuseUSB)
    list(APPEND SRC_LISTS
        det-src/cameras/USBCamera.cpp
    )
endif()

add_library(
  ai-det SHARED
  ${SRC_LISTS}

)




# Runtime-loaded X3 library must find the deployed MVS SDK without shell environment changes.
if(USE_RDK)
    set_target_properties(ai-det PROPERTIES
        BUILD_RPATH "/opt/MVS/lib/aarch64;/usr/local/lib"
        INSTALL_RPATH "$ORIGIN;/opt/MVS/lib/aarch64;/usr/local/lib")
endif()

if(NOT USE_RDK)


if(WIN32)
    set(TENSORRT_LIBS
        "${TENSORRT_PATH}/lib/nvinfer.lib"
        "${TENSORRT_PATH}/lib/nvonnxparser.lib"
        "${TENSORRT_PATH}/lib/nvparsers.lib"
        "${TENSORRT_PATH}/lib/nvinfer_plugin.lib"
    )
else()

    if(ARCH_TYPE STREQUAL "x86_64")

        set(TENSORRT_LIBS
            "${TENSORRT_PATH}/lib/libnvinfer.so"
            "${TENSORRT_PATH}/lib/libnvonnxparser.so"
            "${TENSORRT_PATH}/lib/libnvinfer_plugin.so"
        )
    elseif(ARCH_TYPE STREQUAL "ARM64")


        set(TENSORRT_LIBS
            "/usr/lib/aarch64-linux-gnu/libnvinfer.so"
            "/usr/lib/aarch64-linux-gnu/libnvonnxparser.so"
            # "/usr/lib/aarch64-linux-gnu/libnvparsers.so.8"
            "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so"
        )    
    endif()


endif()


endif()



if(NOT USE_RDK)
# Enable separable compilation for CUDA (optional but recommended)
set_target_properties(ai-det PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
endif()


if(USE_RDK)
    target_link_libraries(ai-det
        ${OpenCV_LIBS}
        ${SRC_LIBS}        
        dnn
    )
else()
    target_link_libraries(ai-det
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBS}
        ${CGX_SDK_LIBS}
        ${SRC_LIBS}
    )
endif()


add_executable(
    ${PROJECT_NAME}-lib main-lib.cpp  det-src/det-rect.cpp
)

target_link_libraries(
    ${PROJECT_NAME}-lib ${OpenCV_LIBS}
)



# 默认将产物输出到当前构建目录。这样 Debug / Release 使用独立的
# build-debug / build-release 目录时，不会相互覆盖。
set(DET_OUTPUT_DIR "${CMAKE_BINARY_DIR}" CACHE PATH
    "Directory for det-tool runtime and library outputs")

set_target_properties(${PROJECT_NAME}-lib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}"
)

set_target_properties(ai-det PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}"
)

if(BUILD_OFFLINE_TESTS AND UNIX)
    add_executable(ai-det-offline-runner
        tests/ai_det_offline_runner.cpp)
    target_link_libraries(ai-det-offline-runner PRIVATE dl)
    set_target_properties(ai-det-offline-runner PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}")

    add_executable(plate-top-selector-test
        tests/plate_top_selector_test.cpp)
    target_link_libraries(plate-top-selector-test PRIVATE ${OpenCV_LIBS})
    set_target_properties(plate-top-selector-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${DET_OUTPUT_DIR}")
endif()


if(WIN32)

    # bigobj
    set(CMAKE_CXX_FLAGS "-DNOMINMAX ${CMAKE_CXX_FLAGS} /bigobj")

    # vs MT/MTd
    set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET ai-det PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
