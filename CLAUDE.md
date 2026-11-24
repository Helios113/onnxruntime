# Notes for Claude (onnxruntime training build)

## Builds: always incremental, never fresh
- Build with `training_builds/submit_build.sbatch` (or the `_matmul_with_error` variant, which only changes the job name,
  log name and BUILD_ROOT). Both reuse `$BUILD_ROOT/ort_build`, so after a source edit only the touched files recompile.
- Never delete or recreate a BUILD_ROOT / `ort_build` to "fix" something; a fresh build costs ~50 min. For a new experiment
  that must not overwrite an existing wheel, copy the sbatch with a different BUILD_ROOT suffix once, then keep reusing it.
- Stale-wheel trap: libs are staged into `ort_build/RelWithDebInfo/onnxruntime/capi/` only when the pybind target relinks, so a
  CUDA-provider-only change used to produce a wheel with the old `libonnxruntime_providers_cuda.so`. The sbatch now refreshes the
  staged libs and repackages (no recompile). If a fix seems to have no effect, compare the .so inside the wheel (zip date/size)
  with `ort_build/RelWithDebInfo/*.so` before debugging the code.
- Test the wheel with `training_builds/test_wheel.sbatch`, e.g.
  `BUILD_ROOT_OVERRIDE=<BUILD_ROOT> TEST_SCRIPT=../onnxruntime/test/python/contrib_ops/test_matmul_with_error.py sbatch --partition=hopper --nodelist=ruapehu training_builds/test_wheel.sbatch`
  (pass the variables via the shell environment, not `--export`, which got the job held with "user env retrieval failed").
- All `*_with_error` op tests: `TEST_SCRIPT=test_with_error_ops.py` (same sbatch, on ruapehu and mauao); needs `ml_dtypes` for bf16 (the test venv installs it).
- The new venv installs reuse the same wheel version string, so `uv` may serve a cached copy; the test script uses `--reinstall-package`.

## Scope
- Stay inside this repo; advise on other repos (e.g. fp_error_estimates, new_jac) instead of editing or running them.
