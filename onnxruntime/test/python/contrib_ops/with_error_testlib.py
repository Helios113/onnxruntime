# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared reference helpers for the *WithError op tests.

All references emulate the kernels' exact operation order (Dot2 / TwoSum with k, or the reduced elements, in row-major
order) so E can be compared bit-for-bit, and compute exact values with Fractions to check E against exact - Y.
"""

import os
from fractions import Fraction

import numpy as np
from onnx import TensorProto

import onnxruntime as ort

try:
    import ml_dtypes

    BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
    BF16 = None

IR_VERSION = 10  # newer onnx packages emit an IR version this ORT build does not accept


def make_session(model, providers):
    """InferenceSession with an explicit intra-op thread count.

    The default pool starts one pinned thread per machine core, and under Slurm/cgroup CPU limits pthread_setaffinity_np
    fails (EINVAL) for every core outside the allowed set, flooding stderr with errors. An explicit count skips pinning.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, len(os.sched_getaffinity(0)))
    return ort.InferenceSession(model, sess_options=so, providers=providers)


def acc_dtype(np_dtype):
    return np.float64 if np_dtype == np.float64 else np.float32


def two_prod_err(a, b, p):
    """Exact error a*b - p of the rounded product p (arrays of the accumulation dtype)."""
    if a.dtype == np.float64:
        f = np.frompyfunc(lambda x, y, z: float(Fraction(float(x)) * Fraction(float(y)) - Fraction(float(z))), 3, 1)
        return f(a, b, p).astype(np.float64)
    # float32: the float64 product of two float32 values is exact, and so is the difference.
    return (a.astype(np.float64) * b.astype(np.float64) - p.astype(np.float64)).astype(np.float32)


def two_sum_step(s, c, p, e, acc):
    t = (s + p).astype(acc)
    bb = (t - s).astype(acc)
    q = ((s - (t - bb).astype(acc)).astype(acc) + (p - bb).astype(acc)).astype(acc)
    return t, (c + (q + e).astype(acc)).astype(acc)


def dot2_state(a, b, acc):
    """(s, c) of the k-ordered Dot2 of a [..., M, K] and b [..., K, N] (broadcasting over leading dims)."""
    a = a.astype(acc)
    b = b.astype(acc)
    s = c = None
    for k in range(a.shape[-1]):
        ak = a[..., :, k : k + 1]
        bk = b[..., k : k + 1, :]
        p = (ak * bk).astype(acc)
        e = two_prod_err(np.broadcast_to(ak, p.shape), np.broadcast_to(bk, p.shape), p)
        if s is None:
            s = np.zeros(p.shape, acc)
            c = np.zeros(p.shape, acc)
        s, c = two_sum_step(s, c, p, e, acc)
    if s is None:  # K == 0
        shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
        s, c = np.zeros(shape, acc), np.zeros(shape, acc)
    return s, c


def finish(t, corr, y, acc):
    """E = (t - y) + corr, NaN where y is not finite."""
    y = y.astype(acc)
    with np.errstate(invalid="ignore", over="ignore"):
        e = ((t - y).astype(acc) + corr).astype(acc)
    return np.where(np.isfinite(y), e, np.array(np.nan, acc)).astype(acc)


def dot2_error(a, b, y, acc):
    s, c = dot2_state(a, b, acc)
    return finish(s, c, y, acc)


def rand_in(rng, shape, dtype, lo=1.0, hi=2.0):
    """Random magnitudes in [lo, hi) with random signs."""
    mag = rng.uniform(lo, hi, size=shape)
    return (mag * rng.choice([-1.0, 1.0], size=shape)).astype(dtype)


def exact_matmul(a, b):
    """Exact Fractions of a [M,K] @ b [K,N] and sum|a*b| (as floats)."""
    M, K = a.shape
    N = b.shape[1]
    af = [[Fraction(float(v)) for v in row] for row in a]
    bf = [[Fraction(float(v)) for v in col] for col in b.T]
    exact = [[None] * N for _ in range(M)]
    mag = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            terms = [af[i][k] * bf[j][k] for k in range(K)]
            exact[i][j] = sum(terms, Fraction(0))
            mag[i, j] = float(sum((abs(t) for t in terms), Fraction(0)))
    return exact, mag


def minus(exact, y):
    """float(exact - y) elementwise for a list-of-lists of Fractions and an array y."""
    out = np.zeros(y.shape)
    for idx in np.ndindex(*y.shape):
        out[idx] = float(exact[idx[0]][idx[1]] - Fraction(float(y[idx])))
    return out


def unit_roundoff(acc):
    return 2.0**-24 if acc == np.float32 else 2.0**-53


def tolerance(target, mag, K, acc):
    """Allowed |E - (exact - Y)|: a few ulps of the result plus the second-order K*u^2*sum|terms| term."""
    u = unit_roundoff(acc)
    tiny = 2.0**-149 if acc == np.float32 else 2.0**-1074
    return 4 * u * np.abs(target) + (K + 4) * 4 * u * u * mag + tiny


ELEM = {
    np.float32: TensorProto.FLOAT,
    np.float64: TensorProto.DOUBLE,
    np.float16: TensorProto.FLOAT16,
}
if BF16 is not None:
    ELEM[BF16] = TensorProto.BFLOAT16
