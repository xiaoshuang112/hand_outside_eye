# CMake Toolchain for RDK X5 (地平线旭日5) Cross-Compilation
# Usage: cmake -B build_rdk -DCMAKE_TOOLCHAIN_FILE=toolchain_rdk_x5.cmake -DUSE_RDK=ON

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING TRUE)

# Sysroot — pulled from RDK X5 board
set(RDK_SYSROOT /home/rb/rdk_x5_sysroot)
set(CMAKE_SYSROOT ${RDK_SYSROOT})

# Cross compiler
set(CMAKE_C_COMPILER    aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER  aarch64-linux-gnu-g++)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---- DNN (Horizon BPU SDK) ----
set(DNN_ROOT /usr)

# ---- OpenCV hints ----
set(OpenCV_DIR ${RDK_SYSROOT}/usr/local/lib/cmake/opencv4)

# ---- Orbbec SDK ----
set(ORBBEC_SDK_V2_DIR ${RDK_SYSROOT}/opt/OrbbecSDK_v2.7.6)

# ---- MVS SDK ----
set(useMVS TRUE CACHE BOOL "Use MVS camera")
set(MVS_SDK_DIR ${RDK_SYSROOT}/opt/MVS CACHE PATH "MVS SDK path")

# ---- Extra compilation flags ----
# DNN and MVS includes — must be full path because --sysroot does not affect -I/-isystem
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -I${RDK_SYSROOT}/usr/include/dnn -I${RDK_SYSROOT}/opt/MVS/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${RDK_SYSROOT}/usr/include/dnn -I${RDK_SYSROOT}/opt/MVS/include")
# Suppress missing transitive deps of shared libs (they are present on the target board)
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-unresolved-symbols=ignore-in-shared-libs")

# Avoid picking up host CUDA
set(CUDA_FOUND FALSE CACHE BOOL "Disable host CUDA")
