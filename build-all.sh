#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jobs="${JOBS:-2}"

cmake -S "$root_dir" -B "$root_dir/build-debug" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$root_dir/build-debug" -j"$jobs"

cmake -S "$root_dir" -B "$root_dir/build-release" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root_dir/build-release" -j"$jobs"
