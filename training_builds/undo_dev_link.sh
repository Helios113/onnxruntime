#!/bin/bash
set -euo pipefail

# Removes the .pth file written by dev_link.sh, so the venv falls back to whatever
# onnxruntime-training is actually installed via uv (the wheel declared in pyproject.toml).
#
# Usage: training_builds/undo_dev_link.sh <path-to-venv>

VENV="${1:?usage: undo_dev_link.sh <path-to-venv>}"

SITE_PACKAGES="$("$VENV/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
PTH_FILE="$SITE_PACKAGES/_onnxruntime_dev_link.pth"

if [ -f "$PTH_FILE" ]; then
  rm "$PTH_FILE"
  echo "removed $PTH_FILE"
else
  echo "no dev link active ($PTH_FILE not found)"
fi
