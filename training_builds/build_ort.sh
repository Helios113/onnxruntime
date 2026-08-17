#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Site-specific configuration. These are NOT auto-discoverable -- they encode
# where you keep source trees and shared storage, not hardware facts. Set
# via environment variables when invoking the script, e.g.:
#   ORT_SRC=/path/to/onnxruntime BUILD_ROOT=/nfs-share/me/ort_out ./build_ort.sh
#
# BUILD_ROOT must be on shared/network storage if you run this via a job
# scheduler across multiple nodes -- the staged setup.py + compiled .so tree
# needs to be visible from any node for editable installs, not just the node
# that ran the build.
# ---------------------------------------------------------------------------
ORT_SRC="${ORT_SRC:-$(pwd)}"
if [ ! -f "$ORT_SRC/cmake/deps.txt" ]; then
  echo "ERROR: ORT_SRC ($ORT_SRC) doesn't look like an onnxruntime checkout (no cmake/deps.txt)." >&2
  echo "Set ORT_SRC=/path/to/onnxruntime or cd into it before running." >&2
  exit 1
fi

ORT_VERSION="$(cat "$ORT_SRC/VERSION_NUMBER" 2>/dev/null || git -C "$ORT_SRC" describe --tags --always)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDNN_PIP_SPEC="${CUDNN_PIP_SPEC:-nvidia-cudnn-cu12==9.2.1.18}"  # pinned for compatibility with this ORT/CUDA combo, not auto-detected

BUILD_ROOT="${BUILD_ROOT:-$HOME/onnxruntime_build_output/$ORT_VERSION}"
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

# ---------------------------------------------------------------------------
# CUDA toolkit discovery
# ---------------------------------------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
  # Highest-versioned /usr/local/cuda-12.* install (ORT 1.20.0 predates CUDA 13 --
  # its nvcc rejects GCC-style warning flags CMake's compiler feature-detection
  # probes with, e.g. "-Wstrict-aliasing", breaking configure). Excluding cuda-13.*
  # from auto-discovery avoids that; pass CUDA_HOME explicitly to override.
  CUDA_HOME="$(find /usr/local -maxdepth 1 -name 'cuda-12.*' -type d 2>/dev/null | sort -V | tail -1)"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: no nvcc at $CUDA_HOME/bin/nvcc. Set CUDA_HOME explicitly." >&2
  exit 1
fi
export CUDA_HOME
export CUDACXX="$CUDA_HOME/bin/nvcc"
CUDA_VERSION="$("$CUDACXX" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"

# ---------------------------------------------------------------------------
# uv discovery: PATH, then the default per-user install location, then
# install it if it's genuinely absent (sbatch/non-login shells don't source
# the profile that would normally put it on PATH).
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  UV_BIN_DIR="$(dirname "$(command -v uv)")"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN_DIR="$HOME/.local/bin"
else
  echo "=== uv not found, installing to $HOME/.local/bin ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN_DIR="$HOME/.local/bin"
fi
export PATH="$CUDA_HOME/bin:$UV_BIN_DIR:$PATH"

# CUDA's runtime lib directory name depends on CPU arch.
case "$(uname -m)" in
  x86_64)  CUDA_TARGET_DIR="x86_64-linux" ;;
  aarch64) CUDA_TARGET_DIR="sbsa-linux" ;;
  ppc64le) CUDA_TARGET_DIR="ppc64le-linux" ;;
  *)       CUDA_TARGET_DIR="" ;;
esac
if [ -n "$CUDA_TARGET_DIR" ] && [ -d "$CUDA_HOME/targets/$CUDA_TARGET_DIR/lib" ]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/$CUDA_TARGET_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# A login shell's profile can set CPATH/LIBRARY_PATH to a different CUDA
# install than CUDA_HOME above; nvcc's preprocessor searches those ahead of
# --cuda_home's -I flags, so unset both to avoid picking up wrong headers.
unset CPATH
unset LIBRARY_PATH

echo "=== nvcc check ==="
$CUDACXX --version

# ---------------------------------------------------------------------------
# GPU compute capability discovery -> CMAKE_CUDA_ARCHITECTURES
# ---------------------------------------------------------------------------
if [ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=compute_cap --format=csv,noheader >/dev/null 2>&1; then
    CMAKE_CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
      | tr -d ' .' | sort -u | paste -sd ';')"
  fi
  if [ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
    echo "WARNING: no GPU visible on this build host (nvidia-smi missing or empty)." >&2
    echo "Falling back to a broad arch list (70;75;80;86;90 = V100/T4/A100/A6000/H100)." >&2
    echo "Set CMAKE_CUDA_ARCHITECTURES explicitly for a different/narrower target." >&2
    CMAKE_CUDA_ARCHITECTURES="70;75;80;86;90"
  fi
fi
echo "CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"

echo "=== setting up build venv (Python $PYTHON_VERSION) ==="
export UV_PYTHON_INSTALL_DIR="$BUILD_ROOT/uv_python"
uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" "$BUILD_ROOT/build_venv"
source "$BUILD_ROOT/build_venv/bin/activate"

# setuptools isn't preinstalled in a `uv venv` (unlike `python -m venv`, which
# gets it via ensurepip) -- setup.py's `from setuptools import ...` needs it.
uv pip install "cmake<4" ninja packaging numpy wheel setuptools "$CUDNN_PIP_SPEC"
CUDNN_HOME="$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")"
echo "CUDNN_HOME resolved to: $CUDNN_HOME"
ls "$CUDNN_HOME/lib" | head -5
ls "$CUDNN_HOME/include" | head -5

# The nvidia-cudnn-cu12 wheel ships only versioned libcudnn.so.9 -- the
# linker's -lcudnn needs the unversioned name.
if [ ! -e "$CUDNN_HOME/lib/libcudnn.so" ]; then
  ln -s libcudnn.so.9 "$CUDNN_HOME/lib/libcudnn.so"
fi

echo "=== building onnxruntime $ORT_VERSION from $ORT_SRC ==="
cd "$ORT_SRC"
git submodule update --init --recursive

# Eigen commit parsed from this checkout's cmake/deps.txt instead of
# hardcoded, so it stays correct across onnxruntime versions/branches.
EIGEN_COMMIT="$(grep '^eigen;' cmake/deps.txt | grep -oP '[0-9a-f]{40}' | head -1)"
if [ -z "$EIGEN_COMMIT" ]; then
  echo "ERROR: couldn't parse an eigen commit hash from $ORT_SRC/cmake/deps.txt" >&2
  exit 1
fi
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
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --cuda_version="$CUDA_VERSION" \
  --skip_tests \
  --parallel \
  --use_preinstalled_eigen \
  --eigen_path "$BUILD_ROOT/eigen" \
  --compile_no_warning_as_error \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" "CMAKE_CUDA_FLAGS=-static-global-template-stub=false" \
  --build_dir "$BUILD_ROOT/ort_build"

echo "=== build finished, wheel(s): ==="
find "$BUILD_ROOT/ort_build" -name "*.whl"
