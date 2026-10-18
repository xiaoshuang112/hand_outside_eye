# CMake Toolchain for RDK X3 (地平线旭日3) Cross-Compilation
# Usage: cmake -B build_rdk_x3 -DCMAKE_TOOLCHAIN_FILE=toolchain_rdk_x3.cmake -DUSE_RDK=ON

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING TRUE)

# Sysroot — pulled from RDK X3 board + X5 SDKs
set(RDK_SYSROOT /home/nvidia/work/rdk/rdk_x3_sysroot CACHE PATH "RDK X3 target sysroot")
set(CMAKE_SYSROOT ${RDK_SYSROOT})

# Cross compiler (same as X5 — both are aarch64)
set(CMAKE_C_COMPILER    aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER  aarch64-linux-gnu-g++)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---- Cameras (same SDKs as X5) ----
set(useORB FALSE CACHE BOOL "Use Orbbec camera")
set(useMVS TRUE CACHE BOOL "Use MVS camera")
set(ORBBEC_SDK_V2_DIR ${RDK_SYSROOT}/opt/OrbbecSDK_v2.7.6)
set(MVS_SDK_DIR ${RDK_SYSROOT}/opt/MVS CACHE PATH "MVS SDK path")

# ---- DNN (Horizon BPU SDK for X3) ----
# X3 puts DNN libs in /usr/lib/hbbpu/ (different from X5 which uses /usr/lib/)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -I${RDK_SYSROOT}/usr/include/dnn -I${RDK_SYSROOT}/opt/MVS/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${RDK_SYSROOT}/usr/include/dnn -I${RDK_SYSROOT}/opt/MVS/include")
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    -L${RDK_SYSROOT}/usr/lib/hbbpu -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${RDK_SYSROOT}/usr/lib/hbbpu -Wl,-unresolved-symbols=ignore-in-shared-libs")

# ---- OpenCV hints ----
set(OpenCV_DIR ${RDK_SYSROOT}/usr/local/lib/cmake/opencv4)

# Avoid picking up host CUDA
set(CUDA_FOUND FALSE CACHE BOOL "Disable host CUDA")
