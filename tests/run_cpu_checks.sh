#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
check_build="$(mktemp -d)"
trap 'rm -rf "$check_build"' EXIT
read -r -a opencv_flags <<< "$(pkg-config --cflags --libs opencv4)"
common=(-std=c++17 -O1 -Iinclude -Idet-src)
g++ "${common[@]}" tests/top_face_fusion_test.cpp det-src/center_heatmap_infer.cpp "${opencv_flags[@]}" -o "$check_build/fusion"
"$check_build/fusion"
g++ "${common[@]}" tests/top_face_offline_test.cpp det-src/top_face.cpp det-src/center_heatmap_infer.cpp "${opencv_flags[@]}" -o "$check_build/offline"
"$check_build/offline" "$PWD"
g++ "${common[@]}" tests/top_face_center_mapping_test.cpp det-src/top_face.cpp "${opencv_flags[@]}" -o "$check_build/center"
"$check_build/center" "$PWD"
g++ "${common[@]}" tests/top_plane_snapshot_test.cpp det-src/top_face.cpp det-src/center_heatmap_infer.cpp "${opencv_flags[@]}" -o "$check_build/snapshot"
"$check_build/snapshot" "$PWD"
