#!/bin/bash
#SBATCH --job-name=ort_build_sm86
#SBATCH --partition=ampere
#SBATCH --nodelist=mauao
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10  # cluster caps GPU jobs at 10 CPUs
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-share/pa511/code_bases/onnxruntime/training_builds/logs/build_%j.log

set -euo pipefail

# sm_86-only build of the onnxruntime-training wheel (py3.11 + CUDA 12.6), for
# A40/A6000/RTX 30-series (compute capability 8.6). Runs on mauao (ampere
# partition), the only node in this cluster with sm_86 GPUs -- that node caps
# out at CUDA 12.6 (no 12.8), hence the different CUDA_HOME/wheel cu-tag than
# submit_build.sbatch. Submit with:
#   sbatch training_builds/submit_build_sm86.sbatch
#
# Uses its own BUILD_ROOT (distinct from submit_build.sbatch's) so this never
# touches the 89;90;120 build's ort_build/CMakeCache or its output wheel --
# the two builds are fully isolated from each other. See submit_build.sbatch
# for the BUILD_ROOT/CMakeCache absolute-path caveat this mirrors.
ORT_SRC="/nfs-share/pa511/code_bases/onnxruntime"
ORT_VERSION="$(cat "$ORT_SRC/VERSION_NUMBER" 2>/dev/null || git -C "$ORT_SRC" describe --tags --always)"
# BUILD_ROOT is keyed on the ORT version so a new version never reuses (and reconfigures/overwrites)
# an older version's ort_build, which holds its wheel under ort_build/RelWithDebInfo/dist/.
BUILD_ROOT="/nfs-share/pa511/code_bases/onnxruntime_build_output/${ORT_VERSION}_py311_cuda12_sm86"
PYTHON_VERSION="3.11"
CUDA_HOME="/usr/local/cuda-12.6"  # mauao (ampere partition) tops out at 12.6, unlike ruapehu's 12.8
CUDNN_PIP_SPEC="nvidia-cudnn-cu12==9.2.1.18"  # pinned for compatibility with this ORT/CUDA combo, not auto-detected
CMAKE_CUDA_ARCHITECTURES="86"  # forced, regardless of what nvidia-smi sees on the build node

if [ ! -f "$ORT_SRC/cmake/deps.txt" ]; then
  echo "ERROR: ORT_SRC ($ORT_SRC) doesn't look like an onnxruntime checkout (no cmake/deps.txt)." >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: no nvcc at $CUDA_HOME/bin/nvcc." >&2
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

echo "CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES (forced, sm_86 only)"

echo "=== setting up build venv (Python $PYTHON_VERSION) ==="
export UV_PYTHON_INSTALL_DIR="$BUILD_ROOT/uv_python"
uv python install "$PYTHON_VERSION"
if [ ! -f "$BUILD_ROOT/build_venv/bin/activate" ]; then
  uv venv --python "$PYTHON_VERSION" "$BUILD_ROOT/build_venv"
fi
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

# Eigen: ORT >=1.21 dropped the --use_preinstalled_eigen branch from cmake/external/eigen.cmake and
# always FetchContent-downloads the gitlab archive, whose zip checksum is not stable (SHA1 mismatch
# -> configure fails). FETCHCONTENT_SOURCE_DIR_EIGEN points FetchContent at our pinned clone instead.

# os.cpu_count() (what --parallel with no count relies on) sees the whole
# node's CPUs under SLURM, not the cgroup-limited --cpus-per-task allocation --
# oversubscribing job count vs. reserved CPUs badly thrashes the build. Pin it
# to what SLURM actually gave us, falling back to nproc outside SLURM.
BUILD_PARALLEL_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

echo "=== building (parallel=$BUILD_PARALLEL_JOBS; reusing $BUILD_ROOT/ort_build if present) ==="
# Unlike submit_build.sbatch, no -static-global-template-stub=false in
# CMAKE_CUDA_FLAGS here: that flag was added there for nvcc 12.8 and doesn't
# exist in nvcc 12.6 (this node's toolchain) -- passing it breaks CMake's
# CUDA compiler ABI-detection step outright.
./build.sh \
  --config RelWithDebInfo \
  --enable_training \
  --build_wheel \
  --use_cuda \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --cuda_version="$CUDA_VERSION" \
  --skip_tests \
  --parallel "$BUILD_PARALLEL_JOBS" \
  --compile_no_warning_as_error \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" onnxruntime_BUILD_UNIT_TESTS=OFF "FETCHCONTENT_SOURCE_DIR_EIGEN=$BUILD_ROOT/eigen" \
  --build_dir "$BUILD_ROOT/ort_build"

echo "=== build finished, wheel(s): ==="
find "$BUILD_ROOT/ort_build" -name "*.whl"
