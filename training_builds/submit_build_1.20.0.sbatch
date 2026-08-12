#!/bin/bash
#SBATCH --job-name=ort_build_1.20.0
#SBATCH --partition=hopper
#SBATCH --nodelist=ruapehu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs-share/pa511/code_bases/onnxruntime/training_builds/logs/build_1.20.0_%j.log

bash /nfs-share/pa511/code_bases/onnxruntime/training_builds/build_ort_1.20.0.sh
