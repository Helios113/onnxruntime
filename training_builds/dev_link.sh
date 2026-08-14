#!/bin/bash
set -euo pipefail

# Points a venv's onnxruntime import at the live CMake build output instead of the installed
# wheel, so edits to orttraining/orttraining/python/training/**/*.py (and other python/**
# sources) in the onnxruntime worktree take effect immediately -- no reinstall, no rebuild,
# just re-running resync_python.sh to refresh the copy (or nothing at all, for files CMake
# copies with a symlink rather than a hard copy -- check onnxruntime_python.cmake).
#
# Mechanism: setup.py isn't usable outside build.py's own invocation for an editable/PEP 517
# install (see the comment in pyproject.toml's [tool.uv.sources] for why), so this bypasses
# it entirely with a raw .pth file in site-packages -- the same net effect (this path takes
# priority on sys.path) without going through setup.py at all.
#
# This SHADOWS the uv-managed wheel install (still declared normally in pyproject.toml) --
# a later `uv sync` does not remove the .pth file, but does reinstall the wheel's files
# alongside it; since .pth-added paths are prepended, the worktree version still wins until
# you run undo_dev_link.sh. Re-run this after every `uv sync` if you want to be sure the
# .pth file is still doing anything (uv sync does not touch .pth files it didn't create, but
# double check with `python -c "import onnxruntime; print(onnxruntime.__file__)"`).
#
# Usage: training_builds/dev_link.sh <path-to-venv> [1.20.0]

VENV="${1:?usage: dev_link.sh <path-to-venv> [version]}"
VERSION="${2:-1.20.0}"
BUILD_DIR="/nfs-share/pa511/code_bases/onnxruntime_build_output/${VERSION}/ort_build/RelWithDebInfo"

if [ ! -d "$BUILD_DIR/onnxruntime" ]; then
  echo "error: build output not found: $BUILD_DIR/onnxruntime" >&2
  echo "  (run training_builds/submit_build_${VERSION}.sbatch first)" >&2
  exit 1
fi

SITE_PACKAGES="$("$VENV/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
PTH_FILE="$SITE_PACKAGES/_onnxruntime_dev_link.pth"

echo "$BUILD_DIR" > "$PTH_FILE"
echo "wrote $PTH_FILE -> $BUILD_DIR"
echo "verify with: $VENV/bin/python3 -c \"import onnxruntime; print(onnxruntime.__file__)\""
