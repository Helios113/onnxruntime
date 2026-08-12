#!/bin/bash
set -euo pipefail

BUILD_ROOT=/local/scratch/pa511_ort_build_1.20.0
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib

# See build_ort.sh (1.19.2) -- the login shell's profile sets CPATH/LIBRARY_PATH to
# /usr/local/cuda-13.0's include/lib dirs, which nvcc's preprocessor searches ahead of
# --cuda_home's -I flags. Unset both so cuda-12.8 headers are actually the ones used.
unset CPATH
unset LIBRARY_PATH

echo "=== nvcc check ==="
$CUDACXX --version

echo "=== setting up build venv ==="
# Same shared-libpython requirement as the 1.19.2 build: CMake's find_package(Python ...)
# needs libpython3.11.so + Python.h to define Python::Python, which the PATH-resolved
# python3 shim doesn't provide. Use pyenv's --enable-shared 3.11.9 directly.
PYENV_PY311=/nfs-share/pa511/.pyenv/versions/3.11.9/bin/python3.11
"$PYENV_PY311" -m venv "$BUILD_ROOT/build_venv"
source "$BUILD_ROOT/build_venv/bin/activate"
pip install --upgrade pip
pip install "cmake<4" ninja packaging numpy wheel nvidia-cudnn-cu12==9.2.1.18

CUDNN_HOME="$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")"
echo "CUDNN_HOME resolved to: $CUDNN_HOME"
ls "$CUDNN_HOME/lib" | head -5
ls "$CUDNN_HOME/include" | head -5

# nvidia-cudnn-cu12 wheel ships only versioned libcudnn.so.9 -- the linker's -lcudnn needs
# the unversioned name, so create it.
if [ ! -e "$CUDNN_HOME/lib/libcudnn.so" ]; then
  ln -s libcudnn.so.9 "$CUDNN_HOME/lib/libcudnn.so"
fi

echo "=== cloning onnxruntime (v1.20.0) ==="
if [ ! -d onnxruntime ]; then
  git clone --recursive --branch v1.20.0 --depth 1 https://github.com/microsoft/onnxruntime.git
fi
cd onnxruntime

# Same eigen SHA1-mismatch workaround as 1.19.2 -- the pin (commit e7248b2) is unchanged
# in cmake/deps.txt between v1.19.2 and v1.20.0, confirmed via `git diff v1.19.2 v1.20.0 --
# cmake/deps.txt`, so this still applies verbatim.
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
