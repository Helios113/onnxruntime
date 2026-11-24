# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for com.microsoft::MatMulWithError (CUDA: float/double/fp16/bf16, CPU: float/double).

E must be bit-identical to a reference Dot2 (Ogita-Rump-Oishi) accumulation in k order,
E = (s - Y) + c, with Y taken from the op itself, and must track exact - Y.
"""

import unittest

import numpy as np
from onnx import TensorProto, helper
from with_error_testlib import (
    BF16,
    ELEM,
    IR_VERSION,
    acc_dtype,
    dot2_error,
    dot2_state,
    exact_matmul,
    minus,
    rand_in,
    tolerance,
)

from with_error_testlib import make_session

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def make_model(elem_type, a_shape, b_shape, trans_a, trans_b, with_r=False):
    e_type = TensorProto.DOUBLE if elem_type == TensorProto.DOUBLE else TensorProto.FLOAT
    node = helper.make_node(
        "MatMulWithError",
        ["A", "B"],
        ["Y", "E"] + (["R"] if with_r else []),
        domain="com.microsoft",
        transA=int(trans_a),
        transB=int(trans_b),
    )
    graph = helper.make_graph(
        [node],
        "matmul_with_error",
        [helper.make_tensor_value_info("A", elem_type, a_shape), helper.make_tensor_value_info("B", elem_type, b_shape)],
        [helper.make_tensor_value_info("Y", elem_type, None), helper.make_tensor_value_info("E", e_type, None)]
        + ([helper.make_tensor_value_info("R", e_type, None)] if with_r else []),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = IR_VERSION
    return model.SerializeToString()


class MatMulWithErrorCases:
    """Test bodies shared by the CUDA and CPU providers; subclasses set `provider` and `dtypes`."""

    provider = None
    dtypes = ()

    def session(self, elem_type, a_shape, b_shape, trans_a=False, trans_b=False):
        providers = [(CUDA, {"use_tf32": 0}), CPU] if self.provider == CUDA else [CPU]
        sess = make_session(make_model(elem_type, a_shape, b_shape, trans_a, trans_b), providers=providers)
        self.assertEqual(sess.get_providers()[0], self.provider)
        return sess

    def _run(self, np_dtype, a_shape, b_shape, trans_a=False, trans_b=False, seed=0):
        acc = acc_dtype(np_dtype)
        rng = np.random.default_rng(seed)
        a = rand_in(rng, a_shape, np_dtype)
        b = rand_in(rng, b_shape, np_dtype)
        sess = self.session(ELEM[np_dtype], a_shape, b_shape, trans_a, trans_b)
        y, e = sess.run(None, {"A": a, "B": b})
        self.assertEqual(e.dtype, np.dtype(acc))
        self.assertEqual(y.shape, e.shape)
        a_ref = np.swapaxes(a, -1, -2) if trans_a else a
        b_ref = np.swapaxes(b, -1, -2) if trans_b else b
        # rank-1 operands are promoted to a row / column vector and the extra dim is dropped afterwards
        a2 = a_ref[None, :] if a_ref.ndim == 1 else a_ref
        b2 = b_ref[:, None] if b_ref.ndim == 1 else b_ref
        full_shape = np.broadcast_shapes(a2.shape[:-2], b2.shape[:-2]) + (a2.shape[-2], b2.shape[-1])
        expected = dot2_error(a2, b2, y.reshape(full_shape), acc).reshape(y.shape)
        np.testing.assert_array_equal(e, expected)

    def _cases(self, np_dtype):
        self._run(np_dtype, [5, 33], [33, 7])
        self._run(np_dtype, [33, 5], [7, 33], trans_a=True, trans_b=True)
        self._run(np_dtype, [2, 3, 4, 17], [17, 6])  # broadcast B over batch
        self._run(np_dtype, [2, 1, 4, 17], [3, 17, 6])  # broadcast both
        # batched B, with and without transposes (shapes are given before the transpose)
        self._run(np_dtype, [3, 4, 8], [3, 8, 5])
        self._run(np_dtype, [4, 8], [3, 8, 5])
        self._run(np_dtype, [3, 1, 4, 8], [2, 8, 5])
        self._run(np_dtype, [3, 8, 4], [3, 5, 8], trans_a=True, trans_b=True)
        self._run(np_dtype, [8, 4], [3, 5, 8], trans_a=True, trans_b=True)
        self._run(np_dtype, [3, 1, 8, 4], [2, 8, 5], trans_a=True)
        self._run(np_dtype, [1, 4, 8], [1, 5, 8], trans_b=True)  # batch == 1 with batch dims
        self._run(np_dtype, [8], [8, 5])  # rank-1 A
        self._run(np_dtype, [4, 8], [8])  # rank-1 B
        # empty reduction: Y == 0 and E == 0
        sess = self.session(ELEM[np_dtype], [3, 0], [0, 4])
        y, e = sess.run(None, {"A": np.zeros((3, 0), np_dtype), "B": np.zeros((0, 4), np_dtype)})
        np.testing.assert_array_equal(y, np.zeros((3, 4), np_dtype))
        np.testing.assert_array_equal(e, np.zeros((3, 4), acc_dtype(np_dtype)))

    def _check_nonfinite(self, np_dtype, big):
        a = np.full((2, 3), big, np_dtype)
        b = np.full((3, 2), big, np_dtype)
        y, e = self.session(ELEM[np_dtype], [2, 3], [3, 2]).run(None, {"A": a, "B": b})
        self.assertTrue(np.all(np.isinf(y)))
        self.assertTrue(np.all(np.isnan(e)), e)

    def test_float(self):
        self._cases(np.float32)

    def test_double(self):
        self._cases(np.float64)

    def test_overflow_gives_nan_float(self):
        self._check_nonfinite(np.float32, 2.0**70)  # products 2^140 > float max

    def test_overflow_gives_nan_double(self):
        self._check_nonfinite(np.float64, 2.0**600)


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestMatMulWithErrorCuda(MatMulWithErrorCases, unittest.TestCase):
    provider = CUDA

    def test_float16(self):
        self._cases(np.float16)

    @unittest.skipIf(BF16 is None, "ml_dtypes not installed")
    def test_bfloat16(self):
        self._cases(BF16)

    def test_compensated_output(self):
        """The optional third output R is the compensated product s + c."""
        for np_dtype in (np.float32, np.float64, np.float16):
            with self.subTest(dtype=np_dtype):
                acc = acc_dtype(np_dtype)
                rng = np.random.default_rng(1)
                a, b = rand_in(rng, [4, 33], np_dtype), rand_in(rng, [33, 5], np_dtype)
                sess = make_session(
                    make_model(ELEM[np_dtype], [4, 33], [33, 5], False, False, with_r=True),
                    providers=[(CUDA, {"use_tf32": 0})],
                )
                y, e, r = sess.run(None, {"A": a, "B": b})
                s, c = dot2_state(a, b, acc)
                np.testing.assert_array_equal(r, (s + c).astype(acc))
                self.assertEqual(r.dtype, np.dtype(acc))

    def test_overflow_gives_nan_float16(self):
        self._check_nonfinite(np.float16, 300.0)  # products 9e4 > 65504

    @unittest.skipIf(BF16 is None, "ml_dtypes not installed")
    def test_overflow_gives_nan_bfloat16(self):
        self._check_nonfinite(BF16, 2.0**70)  # bf16 has the float exponent range


class TestMatMulWithErrorCpu(MatMulWithErrorCases, unittest.TestCase):
    provider = CPU


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestMatMulWithErrorCpuVsCuda(unittest.TestCase):
    """The k order is fixed, so E must be bit-identical on CPU and CUDA for the same Y. Y itself may differ (MLAS vs
    cuBLAS accumulate differently), so E is compared after re-running Dot2 against the *same* Y: compare Y + E instead,
    which approximates the exact product on both providers."""

    def _both(self, np_dtype, a_shape, b_shape, **kw):
        rng = np.random.default_rng(7)
        a, b = rand_in(rng, a_shape, np_dtype), rand_in(rng, b_shape, np_dtype)
        out = {}
        for name, providers in (("cuda", [(CUDA, {"use_tf32": 0})]), ("cpu", [CPU])):
            sess = make_session(make_model(ELEM[np_dtype], a_shape, b_shape, False, False), providers=providers)
            out[name] = sess.run(None, {"A": a, "B": b})
        return out

    def test_exact_value_agrees(self):
        for np_dtype in (np.float32, np.float64):
            with self.subTest(dtype=np_dtype):
                out = self._both(np_dtype, [6, 257], [257, 5])
                (y_g, e_g), (y_c, e_c) = out["cuda"], out["cpu"]
                # Y + E is the Dot2 estimate of the exact product on each provider: they must agree to ~u^2-level
                # (computed in the accumulation type, so compare in float64 against a loose bound).
                est_g = y_g.astype(np.float64) + e_g.astype(np.float64)
                est_c = y_c.astype(np.float64) + e_c.astype(np.float64)
                u = 2.0**-24 if np_dtype == np.float32 else 2.0**-53
                np.testing.assert_allclose(est_g, est_c, rtol=16 * u * u * 257, atol=0)
                # and where the two Y agree bit-for-bit, E must as well
                same = y_g == y_c
                np.testing.assert_array_equal(e_g[same], e_c[same])


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestMatMulWithErrorHalfRanges(unittest.TestCase):
    """fp16/bf16 inputs widened to fp32: E must track exact - Y (Y being the fp16/bf16-rounded result)."""

    def _check(self, np_dtype, a_exp, b_exp, K, seed):
        rng = np.random.default_rng(seed)
        M, N = 4, 4

        def gen(shape, exps):
            mag = 2.0 ** rng.uniform(exps[0], exps[1], size=shape)
            return (mag * rng.choice([-1.0, 1.0], size=shape)).astype(np_dtype)

        a, b = gen([M, K], a_exp), gen([K, N], b_exp)
        sess = make_session(
            make_model(ELEM[np_dtype], [M, K], [K, N], False, False),
            providers=[(CUDA, {"use_tf32": 0})],
        )
        y, e = sess.run(None, {"A": a, "B": b})
        exact, mag = exact_matmul(a, b)
        target = minus(exact, y)
        # Y is only fp16/bf16 precise, so |target| can be ~2^-9 * |Y|: that is real error, not noise.
        err = np.abs(e.astype(np.float64) - target)
        tol = tolerance(target, mag, K, np.float32)
        self.assertTrue(np.all(err <= tol), f"max err {err.max()} tol at max {tol.flat[err.argmax()]}")

    def _ranges(self, np_dtype, large):
        # (exponent range of A, exponent range of B). `large` is the pairing used for the 2^4..2^8 magnitudes: fp16
        # would overflow with both operands that large (2^16 > 65504), so there A is large and B small.
        for a_exp, b_exp in [((-8, -2), (-8, -2)), large, ((-2, 2), (-2, 2))]:
            for K in (64, 4096):
                with self.subTest(a_exp=a_exp, b_exp=b_exp, K=K):
                    self._check(np_dtype, a_exp, b_exp, K, seed=K + a_exp[0])

    def test_float16_ranges(self):
        self._ranges(np.float16, large=((4, 8), (-8, -4)))

    @unittest.skipIf(BF16 is None, "ml_dtypes not installed")
    def test_bfloat16_ranges(self):
        self._ranges(BF16, large=((4, 8), (4, 8)))


if __name__ == "__main__":
    unittest.main()
