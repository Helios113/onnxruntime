#!/bin/bash
set -euo pipefail

# On NFS (not /local/scratch) so the staged build output -- setup.py plus the assembled
# onnxruntime/ python package tree with the compiled .so -- is visible from any node for
# editable installs (see pyproject.toml's [tool.uv.sources] in dependent repos), not just
# from ruapehu where the build actually runs.
BUILD_ROOT=/nfs-share/pa511/code_bases/onnxruntime_build_output/1.20.0
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
# sbatch runs a non-login shell, so uv's install dir isn't on PATH by default -- add it
# explicitly rather than relying on the login profile.
export PATH=/usr/local/cuda-12.8/bin:/nfs-share/pa511/uv/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib

# See build_ort.sh (1.19.2) -- the login shell's profile sets CPATH/LIBRARY_PATH to
# /usr/local/cuda-13.0's include/lib dirs, which nvcc's preprocessor searches ahead of
# --cuda_home's -I flags. Unset both so cuda-12.8 headers are actually the ones used.
unset CPATH
unset LIBRARY_PATH

echo "=== nvcc check ==="
$CUDACXX --version

echo "=== setting up build venv ==="
# CMake's find_package(Python ...) needs libpython3.11.so + Python.h to define the
# Python::Python target -- a plain PATH-resolved python3 shim doesn't provide these.
# uv's managed CPython builds (python-build-standalone) do ship both, unlike a bare
# `uv venv` off the system interpreter, so install/use one explicitly.
export UV_PYTHON_INSTALL_DIR=/nfs-share/pa511/uv/python
uv python install 3.11
uv venv --python 3.11 "$BUILD_ROOT/build_venv"
source "$BUILD_ROOT/build_venv/bin/activate"
uv pip install "cmake<4" ninja packaging numpy wheel nvidia-cudnn-cu12==9.2.1.18

CUDNN_HOME="$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")"
echo "CUDNN_HOME resolved to: $CUDNN_HOME"
ls "$CUDNN_HOME/lib" | head -5
ls "$CUDNN_HOME/include" | head -5

# nvidia-cudnn-cu12 wheel ships only versioned libcudnn.so.9 -- the linker's -lcudnn needs
# the unversioned name, so create it.
if [ ! -e "$CUDNN_HOME/lib/libcudnn.so" ]; then
  ln -s libcudnn.so.9 "$CUDNN_HOME/lib/libcudnn.so"
fi

echo "=== using onnxruntime worktree (training/rel-1.20.0) ==="
# Build a dedicated worktree of the training/rel-1.20.0 branch (not the primary checkout
# at /nfs-share/pa511/code_bases/onnxruntime) so this build can run without holding that
# checkout busy -- it stays free for other work (e.g. branch cleanup) while this runs.
ORT_SRC=/nfs-share/pa511/code_bases/onnxruntime_worktrees/training-rel-1.20.0
cd "$ORT_SRC"
git submodule update --init --recursive

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
