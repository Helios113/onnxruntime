#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Submits the full training-wheel build matrix as sbatch jobs:
#   py3.11 + CUDA 12.8
#   py3.12 + CUDA 12.8
#   py3.12 + CUDA 13.1
#
# Each combo gets its own BUILD_ROOT (so build venvs/artifacts don't collide)
# and its own CUDA_HOME (so build_ort.sh's cuda-12.* auto-discovery isn't
# relied on for the versions this script cares about -- pinned explicitly
# instead, since the host may eventually have more than one cuda-12.* dir).
#
# Usage:
#   ORT_SRC=/path/to/onnxruntime_worktrees/training-rel-1.20.0 training_builds/build_all.sh
#
# Jobs are submitted sequentially (not run concurrently) since they share the
# same --nodelist=ruapehu in submit_build.sbatch and would otherwise contend
# for the same GPU/CPU/memory.
# ---------------------------------------------------------------------------
: "${ORT_SRC:?set ORT_SRC to the onnxruntime checkout to build, e.g. ORT_SRC=/path/to/onnxruntime training_builds/build_all.sh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_VERSION="$(cat "$ORT_SRC/VERSION_NUMBER" 2>/dev/null || git -C "$ORT_SRC" describe --tags --always)"
OUTPUT_BASE="${OUTPUT_BASE:-$HOME/onnxruntime_build_output}"

CUDA_12_8_HOME="${CUDA_12_8_HOME:-/usr/local/cuda-12.8}"
CUDA_13_1_HOME="${CUDA_13_1_HOME:-/usr/local/cuda-13.1}"

# python_version:cuda_home:build_root_suffix:cuda_architectures
# cuda_architectures is left blank to fall back to build_ort.sh's own
# nvidia-smi/default discovery, except where a version needs pinning --
# CUDA 13.1 dropped support for compute_70 (Volta/V100), which is still in
# build_ort.sh's no-GPU-visible fallback list, so nvcc rejects it outright
# during CMake's compiler check unless overridden here.
COMBOS=(
  "3.11:$CUDA_12_8_HOME:py311_cuda12:"
  "3.12:$CUDA_12_8_HOME:py312_cuda12:"
  "3.12:$CUDA_13_1_HOME:py312_cuda13:75;80;86;90"
)

for combo in "${COMBOS[@]}"; do
  IFS=: read -r py_version cuda_home suffix cuda_architectures <<<"$combo"

  if [ ! -x "$cuda_home/bin/nvcc" ]; then
    echo "ERROR: no nvcc at $cuda_home/bin/nvcc for combo $suffix -- skipping." >&2
    continue
  fi

  build_root="$OUTPUT_BASE/${ORT_VERSION}_${suffix}"
  echo "=== submitting $suffix: python $py_version, CUDA_HOME=$cuda_home, BUILD_ROOT=$build_root${cuda_architectures:+, CMAKE_CUDA_ARCHITECTURES=$cuda_architectures} ==="

  job_id="$(ORT_SRC="$ORT_SRC" BUILD_ROOT="$build_root" PYTHON_VERSION="$py_version" CUDA_HOME="$cuda_home" \
    ${cuda_architectures:+CMAKE_CUDA_ARCHITECTURES="$cuda_architectures"} \
    sbatch --parsable --export=ALL "$SCRIPT_DIR/submit_build.sbatch")"
  echo "  -> submitted as job $job_id, waiting for it to finish before submitting the next combo..."

  # Sequential: wait for this job to leave the queue (completed or failed)
  # before submitting the next, since all combos target the same node.
  while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
    sleep 30
  done

  if ! sacct -j "$job_id" --format=State --noheader 2>/dev/null | grep -q COMPLETED; then
    echo "WARNING: job $job_id ($suffix) did not complete successfully -- check training_builds/logs/build_${job_id}.log" >&2
  else
    echo "  -> job $job_id ($suffix) completed."
  fi
done

echo "=== all combos submitted ==="
