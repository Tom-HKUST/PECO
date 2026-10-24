#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${NVCC:-}" ]]; then
    CUDA_COMPILER="${NVCC}"
elif [[ -n "${CUDA_HOME:-}" ]]; then
    CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
else
    CUDA_COMPILER=nvcc
fi
CUDA_COMPILER_PATH="$(command -v -- "${CUDA_COMPILER}")"
CUDA_TOOLKIT_ROOT="$(dirname -- "$(dirname -- "$(readlink -f -- "${CUDA_COMPILER_PATH}")")")"
mkdir -p "${PROJECT_ROOT}/build"
"${CUDA_COMPILER}" -O3 -std=c++17 -arch=sm_89 --shared -Xcompiler -fPIC \
    "${PROJECT_ROOT}/peco/kernels/sparse_attention.cu" -o "${PROJECT_ROOT}/build/libpeco_attention.so"
"${CUDA_COMPILER}" -O3 -std=c++17 -arch=sm_89 --shared -Xcompiler -fPIC \
    "${PROJECT_ROOT}/peco/kernels/kmeans.cu" -lcublas \
    -Xlinker -rpath -Xlinker "${CUDA_TOOLKIT_ROOT}/lib64" \
    -o "${PROJECT_ROOT}/build/libpeco_clustering.so"
echo "CUDA libraries built in ${PROJECT_ROOT}/build"
