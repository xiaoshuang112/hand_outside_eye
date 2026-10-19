#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
check_build="$(mktemp -d)"
trap 'rm -rf "$check_build"' EXIT
read -r -a opencv_flags <<< "$(pkg-config --cflags --libs opencv4)"
g++ -std=c++17 -O2 -Iinclude -Idet-src \
  tests/cylinder_top_face_test.cpp det-src/top_face.cpp det-src/center_heatmap_infer.cpp \
  "${opencv_flags[@]}" -o "$check_build/cylinder"
data_dir="${1:-$PWD/data/2026-10-19}"
output_dir="${2:-$PWD/calibration/output/cylinder_detection/results}"
if [[ -d "$data_dir" ]]; then
  "$check_build/cylinder" "$PWD" "$data_dir" "$output_dir"
else
  "$check_build/cylinder" "$PWD"
fi
