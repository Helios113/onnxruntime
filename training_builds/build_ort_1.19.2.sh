#!/bin/bash
set -euo pipefail

BUILD_ROOT=/local/scratch/pa511_ort_build
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib

# The login shell's profile sets CPATH/LIBRARY_PATH to /usr/local/cuda-13.0's include/lib
# dirs (for interactive work against that toolkit) -- CPATH is searched by the preprocessor
# ahead of any -I flags build.sh/CMake add for --cuda_home, so nvcc kept resolving
# crt/host_runtime.h from CUDA 13.0 even with --cuda_home/--cuda_version pointed at 12.4.
# That mismatch (13.0's __cudaLaunch macro vs. ORT v1.19.2's older codegen) is what actually
# produced the "__cudaLaunch requires 2 arguments" errors, not the toolkit version itself.
unset CPATH
unset LIBRARY_PATH

echo "=== nvcc check ==="
$CUDACXX --version

echo "=== setting up build venv ==="
# CMake's find_package(Python ...) needs a real Python install with a shared library
# (libpython3.11.so) and dev headers (Python.h) to define the Python::Python imported
# target -- a plain venv only references its base interpreter and doesn't provide this on
# its own if that base interpreter lacks them. The shim/venv-resolved `python3` on PATH here
# is such a case (no libpython3.11.so, no Python.h), which left onnxruntime_training.cmake's
# Python::Python target undefined. pyenv's 3.11.9 was built with --enable-shared and ships
# both, so build the venv from that interpreter directly instead of relying on PATH.
PYENV_PY311=/nfs-share/pa511/.pyenv/versions/3.11.9/bin/python3.11
"$PYENV_PY311" -m venv "$BUILD_ROOT/build_venv"
source "$BUILD_ROOT/build_venv/bin/activate"
pip install --upgrade pip
pip install "cmake<4" ninja packaging numpy wheel nvidia-cudnn-cu12==9.2.1.18

CUDNN_HOME="$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")"
echo "CUDNN_HOME resolved to: $CUDNN_HOME"
ls "$CUDNN_HOME/lib" | head -5
ls "$CUDNN_HOME/include" | head -5

# The nvidia-cudnn-cu12 pip wheel ships only versioned libcudnn.so.9 (no unversioned
# libcudnn.so symlink, which a system -dev package would normally provide) -- the linker's
# -lcudnn needs that unversioned name to exist, so create it ourselves.
if [ ! -e "$CUDNN_HOME/lib/libcudnn.so" ]; then
  ln -s libcudnn.so.9 "$CUDNN_HOME/lib/libcudnn.so"
fi

echo "=== cloning onnxruntime ==="
if [ ! -d onnxruntime ]; then
  git clone --recursive --branch v1.19.2 --depth 1 https://github.com/microsoft/onnxruntime.git  
fi
cd onnxruntime

# ORT's pinned eigen dependency is fetched from a GitLab auto-generated archive whose bytes
# don't reproducibly match the SHA1 ORT's cmake/deps.txt expects (see
# https://gitlab.com/libeigen/eigen/-/issues/2744, referenced directly in deps.txt) --
# cloning the exact pinned commit via git instead and pointing --eigen_path at it sidesteps
# the broken hash check entirely.
EIGEN_COMMIT=e7248b26a1ed53fa030c5c459f7ea095dfd276ac
if [ ! -d "$BUILD_ROOT/eigen" ]; then
  git clone https://gitlab.com/libeigen/eigen.git "$BUILD_ROOT/eigen"
  git -C "$BUILD_ROOT/eigen" checkout "$EIGEN_COMMIT"
fi

echo "=== building (this takes a long time) ==="
./build.sh \
  --config RelWithDebInfo \
  --enable_training \
  --build_wheel \
  --use_cuda \
  --cuda_home /usr/local/cuda-12.8 \
  --cudnn_home "$CUDNN_HOME" \
  --cuda_version=12.8 \
  --skip_tests \
  --parallel \
  --use_preinstalled_eigen \
  --eigen_path "$BUILD_ROOT/eigen" \
  --compile_no_warning_as_error \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 "CMAKE_CUDA_FLAGS=-static-global-template-stub=false" \
  --build_dir "$BUILD_ROOT/ort_build"

echo "=== build finished, wheel(s): ==="
find "$BUILD_ROOT/ort_build" -name "*.whl"
