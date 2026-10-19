#!/bin/bash
#SBATCH --job-name=ort_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs-share/pa511/code_bases/onnxruntime/training_builds/logs/test_%j.log
set -euo pipefail

# Smoke-test the built wheel on a GPU node. Pick the node on the command line, e.g.
#   sbatch --partition=ampere --nodelist=mauao  training_builds/test_wheel.sbatch   # A40, sm_86, CUDA 12.6
#   sbatch --partition=hopper --nodelist=ruapehu training_builds/test_wheel.sbatch  # H100, sm_90, CUDA 12.8
# The node's own /usr/local/cuda-12.x is used as the runtime, which is the cross-toolkit compatibility check.
ORT_SRC="/nfs-share/pa511/code_bases/onnxruntime"
# BUILD_ROOT_OVERRIDE selects another build (e.g. .../1.20.0_py311_cuda12_sm86) instead of the current version's.
BUILD_ROOT="${BUILD_ROOT_OVERRIDE:-/nfs-share/pa511/code_bases/onnxruntime_build_output/$(cat "$ORT_SRC/VERSION_NUMBER")_py311_cuda12}"
WHEEL="$(ls -1 "$BUILD_ROOT"/ort_build/RelWithDebInfo/dist/*.whl | head -1)"
CUDNN_PIP_SPEC="nvidia-cudnn-cu12==9.2.1.18"
TEST_VENV="$BUILD_ROOT/test_venv_$(hostname -s)"

CUDA_HOME="$(ls -d /usr/local/cuda-12.* | sort -V | tail -1)"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset CPATH LIBRARY_PATH
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$BUILD_ROOT/uv_python"

echo "host=$(hostname -s) wheel=$WHEEL cuda_runtime=$CUDA_HOME"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

[ -f "$TEST_VENV/bin/activate" ] || uv venv --python 3.11 "$TEST_VENV"
source "$TEST_VENV/bin/activate"
uv pip install --reinstall-package onnxruntime-training "$WHEEL" numpy ml_dtypes onnx "$CUDNN_PIP_SPEC"
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__))')/lib:$LD_LIBRARY_PATH"

cd "$TEST_VENV"   # not the source tree, so `import onnxruntime` can't pick up the checkout
python "$ORT_SRC/training_builds/${TEST_SCRIPT:-test_wheel.py}"
