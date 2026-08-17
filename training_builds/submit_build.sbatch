#!/bin/bash
#SBATCH --job-name=ort_build
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-share/pa511/code_bases/onnxruntime/training_builds/logs/build_%j.log

# Generic submit wrapper for build_ort.sh -- set ORT_SRC (required) and optionally
# BUILD_ROOT/PYTHON_VERSION/CUDA_HOME/etc. as env vars when submitting, e.g.:
#   ORT_SRC=/nfs-share/pa511/code_bases/onnxruntime \
#     sbatch --export=ALL training_builds/submit_build.sbatch
#
# --export=ALL on the sbatch invocation (not this file) is what actually forwards
# ORT_SRC/BUILD_ROOT from the submitting shell into the job -- sbatch does not do
# this by default on every cluster config.

: "${ORT_SRC:?set ORT_SRC to the onnxruntime checkout to build, e.g. ORT_SRC=/path/to/onnxruntime sbatch --export=ALL training_builds/submit_build.sbatch}"

bash "$ORT_SRC/training_builds/build_ort.sh"
