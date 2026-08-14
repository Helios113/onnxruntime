# For us

## Clone

```bash
git clone git@github.com:Helios113/onnxruntime.git
cd onnxruntime
git submodule update --init --recursive
```

Upstream (microsoft/onnxruntime) is configured as a second remote for pulling updates:

```bash
git remote add upstream https://github.com/microsoft/onnxruntime.git
```

## Build

Training builds run on the cluster via slurm, using the scripts in [training_builds/](training_builds/):

```bash
sbatch training_builds/submit_build_1.20.0.sbatch
```

This runs [training_builds/build_ort_1.20.0.sh](training_builds/build_ort_1.20.0.sh) on `ruapehu`, which:
- sets up a `uv`-managed Python 3.11 venv with CUDA 12.8
- builds from a dedicated worktree of the current branch (not this checkout) at `/nfs-share/pa511/code_bases/onnxruntime_worktrees/`
- stages build output (including the wheel) to `/nfs-share/pa511/code_bases/onnxruntime_build_output/` on NFS

Logs land in `training_builds/logs/`. See `training_builds/build_ort_1.19.2.sh` for the equivalent 1.19.2 build.

To build locally instead (not via slurm), run `./build.sh` directly — see the flags used in the sbatch scripts above for a training+CUDA config, or run `./build.sh --help` for all options.
