#!/usr/bin/env bash
set -euo pipefail
# Usage: tests/run_top_face_offline_checks.sh APP_DIR [00197.jpg 00199.jpg 00218.jpg]
# CPU only: never initializes AI_det, camera, NPU, or robot hardware.
cd "$(dirname "$0")/.."
check_build="$(mktemp -d)"
trap 'rm -rf "$check_build"' EXIT
rdk_headers="${RDK_SYSROOT:-/home/nvidia/work/rdk/rdk_x3_sysroot}/usr/include"
g++ -std=c++17 -O2 -Iinclude -Idet-src tests/top_face_fusion_test.cpp det-src/center_heatmap_infer.cpp $(pkg-config --cflags --libs opencv4) -o "$check_build/fusion"
"$check_build/fusion"
g++ -std=c++17 -O2 -Iinclude -Idet-src tests/top_face_offline_test.cpp det-src/top_face.cpp det-src/center_heatmap_infer.cpp $(pkg-config --cflags --libs opencv4) -o "$check_build/offline"
"$check_build/offline" "$@"
g++ -std=c++17 -O2 -Iinclude -Idet-src tests/top_face_center_mapping_test.cpp det-src/top_face.cpp $(pkg-config --cflags --libs opencv4) -o "$check_build/center-mapping"
if [ "$#" -gt 1 ]; then "$check_build/center-mapping" "$1" "$2"; else "$check_build/center-mapping" "$1"; fi
g++ -std=c++17 -O1 -DUSE_RDK -ffunction-sections -fdata-sections -Iinclude -Idet-src -Idet-src/inih -Idet-src/utils -Idet-src/3rdParty/spdlog/include -I"$rdk_headers" -I"$rdk_headers/dnn" $(pkg-config --cflags opencv4) -c det-src/det.cpp -o "$check_build/det.o"
g++ -std=c++17 -O2 -Wl,--gc-sections -Iinclude -Idet-src tests/top_face_conversion_test.cpp det-src/top_face.cpp det-src/center_heatmap_infer.cpp "$check_build/det.o" $(pkg-config --cflags --libs opencv4) -o "$check_build/conversion"
"$check_build/conversion" "$1"
