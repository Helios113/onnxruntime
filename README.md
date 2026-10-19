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




issue reported:
fp16/bf16 graphs fail with "nodes assigned to the default CPU EP"

If you export or build an ONNX model at opset 21 and then run it on CUDAExecutionProvider with session.disable_cpu_ep_fallback=1, you'll get:

This session contains graph nodes that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled

Cause: The CUDA EP in ORT 1.20 doesn't register a kernel for Cast-21. Cast-21 is the version that added int4 and other new types. Only Cast up to opset 19 is registered. Any Cast to or from fp16 or bf16 in an opset-21 graph therefore falls back to the CPU EP. Add, Mul, Div and Sqrt in fp16 and bf16 are fine on CUDA. I checked by running each op as a one-node graph.

Fix: Export or build the model with opset 18 (opset_imports=[helper.make_opsetid("", 18)], ir_version=9). The Cast nodes then run on CUDA, and fp16 and bf16 work.

Notes:

With a CPU fallback allowed, the model may still run. The Casts are silently placed on the CPU and add Memcpy nodes (look for the "Memcpy nodes are added" warning). That hurts performance and can hide the problem.
ORT CPU has no native fp16 or bf16 Add/Mul kernels. It upcasts to fp32 and drops the Cast pairs. Don't use CPU results as a reference for fp16 rounding behaviour.
The working setup is in src/fp_error_estimates/rounding.py (run_ort) in fp_error_estimates.
