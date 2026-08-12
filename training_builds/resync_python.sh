#!/bin/bash
set -euo pipefail

# Re-copies onnxruntime's Python sources (orttraining/orttraining/python/training/**,
# onnxruntime/python/**, etc.) into the staged CMake build output that dependent repos'
# pyproject.toml [tool.uv.sources] editable installs point at -- without recompiling the
# C++/CUDA core.
#
# The copy is a POST_BUILD custom command on the onnxruntime_pybind11_state CMake target
# (cmake/onnxruntime_python.cmake), not a standalone ninja target: ninja re-runs POST_BUILD
# commands unconditionally whenever the target is requested, even if the .so itself is
# already up to date, so `ninja onnxruntime_pybind11_state` here only recompiles C++/CUDA
# if those sources actually changed -- a Python-only edit makes this just the copy step.
#
# Usage: training_builds/resync_python.sh [1.20.0]

VERSION="${1:-1.20.0}"
BUILD_ROOT="/nfs-share/pa511/code_bases/onnxruntime_build_output/${VERSION}"
BUILD_DIR="$BUILD_ROOT/ort_build/RelWithDebInfo"
BUILD_VENV="$BUILD_ROOT/build_venv"

if [ ! -d "$BUILD_DIR" ]; then
  echo "error: build dir not found: $BUILD_DIR" >&2
  echo "  (run training_builds/submit_build_${VERSION}.sbatch first)" >&2
  exit 1
fi

# ninja/cmake were pip-installed only into the build's own venv (build_ort_${VERSION}.sh),
# not onto the login node's PATH.
source "$BUILD_VENV/bin/activate"

echo "=== re-running ninja onnxruntime_pybind11_state in $BUILD_DIR ==="
ninja -C "$BUILD_DIR" onnxruntime_pybind11_state
echo "=== done -- python sources refreshed, .so only recompiled if C++/CUDA changed ==="
