# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for com.microsoft::GemmWithError (CUDA and CPU: float/double/fp16/bf16).

Y = alpha * A' @ B' + beta * C comes from the stock Gemm kernel (CPU fp16/bf16: computed by the op itself). E is checked
  1. bit-for-bit against a numpy emulation of the kernel's operation order, and
  2. against the exact value (Fractions): |E - (exact - Y)| within a few ulps plus a second-order term.
"""

import itertools
import unittest
from fractions import Fraction

import numpy as np
from onnx import TensorProto, helper
from with_error_testlib import (
    BF16,
    ELEM,
    IR_VERSION,
    acc_dtype,
    dot2_state,
    exact_matmul,
    finish,
    rand_in,
    tolerance,
    two_prod_err,
)

from with_error_testlib import make_session

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def make_model(elem_type, a_shape, b_shape, c_shape, trans_a, trans_b, alpha, beta):
    e_type = TensorProto.DOUBLE if elem_type == TensorProto.DOUBLE else TensorProto.FLOAT
    inputs = ["A", "B"] + (["C"] if c_shape is not None else [])
    node = helper.make_node(
        "GemmWithError",
        inputs,
        ["Y", "E"],
        domain="com.microsoft",
        transA=int(trans_a),
        transB=int(trans_b),
        alpha=alpha,
        beta=beta,
    )
    graph_inputs = [
        helper.make_tensor_value_info("A", elem_type, a_shape),
        helper.make_tensor_value_info("B", elem_type, b_shape),
    ]
    if c_shape is not None:
        graph_inputs.append(helper.make_tensor_value_info("C", elem_type, c_shape))
    graph = helper.make_graph(
        [node],
        "gemm_with_error",
        graph_inputs,
        [helper.make_tensor_value_info("Y", elem_type, None), helper.make_tensor_value_info("E", e_type, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = IR_VERSION
    return model.SerializeToString()


def effective(np_dtype, v):
    """The scalar the stock kernel multiplies with: the float attribute converted to the tensor type."""
    return float(np_dtype(np.float32(v)))


def gemm_error_ref(a, b, c, alpha, beta, y, acc):
    """Emulation of the kernel: Dot2 over k, then TwoProd(alpha, s), TwoProd(beta, C), TwoSum."""
    s, c_acc = dot2_state(a, b, acc)
    al = np.full(s.shape, alpha, acc)
    p1 = (al * s).astype(acc)
    e1 = two_prod_err(al, s, p1)
    corr = (e1 + (al * c_acc).astype(acc)).astype(acc)
    hi = p1
    if c is not None:
        be = np.full(s.shape, beta, acc)
        cv = np.broadcast_to(c.astype(acc), s.shape)
        p2 = (be * cv).astype(acc)
        e2 = two_prod_err(be, np.ascontiguousarray(cv), p2)
        hi = (p1 + p2).astype(acc)
        bb = (hi - p1).astype(acc)
        q = ((p1 - (hi - bb).astype(acc)).astype(acc) + (p2 - bb).astype(acc)).astype(acc)
        corr = (corr + (q + e2).astype(acc)).astype(acc)
    return finish(hi, corr, y, acc)


class GemmWithErrorCases:
    """Test bodies shared by the CUDA and CPU providers; subclasses set `provider`."""

    provider = None

    def providers(self):
        return [(CUDA, {"use_tf32": 0}), CPU] if self.provider == CUDA else [CPU]

    def _run(self, np_dtype, M, K, N, c_shape, trans_a, trans_b, alpha, beta, seed=0):
        acc = acc_dtype(np_dtype)
        rng = np.random.default_rng(seed)
        a_shape = [K, M] if trans_a else [M, K]
        b_shape = [N, K] if trans_b else [K, N]
        a = rand_in(rng, a_shape, np_dtype)
        b = rand_in(rng, b_shape, np_dtype)
        c = rand_in(rng, c_shape, np_dtype) if c_shape is not None else None
        sess = make_session(
            make_model(ELEM[np_dtype], a_shape, b_shape, c_shape, trans_a, trans_b, alpha, beta),
            providers=self.providers(),
        )
        self.assertEqual(sess.get_providers()[0], self.provider)
        feeds = {"A": a, "B": b}
        if c is not None:
            feeds["C"] = c
        y, e = sess.run(None, feeds)
        self.assertEqual(e.dtype, np.dtype(acc))
        self.assertEqual(y.shape, (M, N))
        self.assertEqual(e.shape, (M, N))

        a_ref = a.T if trans_a else a
        b_ref = b.T if trans_b else b
        al, be = effective(np_dtype, alpha), effective(np_dtype, beta)
        use_c = c is not None and beta != 0  # the stock op ignores C when beta == 0
        c_ref = c if use_c else None

        # 1. bit-exact vs the kernel-order emulation
        np.testing.assert_array_equal(e, gemm_error_ref(a_ref, b_ref, c_ref, al, be, y, acc))

        # 2. vs the exact value
        exact, mag = exact_matmul(a_ref, b_ref)
        target = np.zeros((M, N))
        total_mag = np.abs(al) * mag
        c_full = None
        if use_c:
            c_full = np.broadcast_to(c.reshape(1, -1) if c.ndim == 1 else c, (M, N))
        for i, j in itertools.product(range(M), range(N)):
            v = Fraction(al) * exact[i][j]
            if use_c:
                cij = float(c_full[i, j])
                v += Fraction(be) * Fraction(cij)
                total_mag[i, j] += abs(be) * abs(cij)
            target[i, j] = float(v - Fraction(float(y[i, j])))
        err = np.abs(e.astype(np.float64) - target)
        tol = tolerance(target, total_mag, K, acc)
        self.assertTrue(np.all(err <= tol), f"max err {err.max()} vs tol {tol.flat[err.argmax()]}")

    def _cases(self, np_dtype):
        M, K, N = 5, 33, 7
        for c_shape in (None, [N], [1, N], [M, 1], [M, N], [1], [1, 1]):
            for trans_a, trans_b in ((False, False), (True, True), (True, False)):
                with self.subTest(c_shape=c_shape, trans_a=trans_a, trans_b=trans_b):
                    self._run(np_dtype, M, K, N, c_shape, trans_a, trans_b, alpha=0.5, beta=2.0)
        for alpha, beta in ((1.0, 1.0), (0.1, 0.3), (1.5, 0.0), (0.25, -0.75), (0.0, 1.0)):
            with self.subTest(alpha=alpha, beta=beta):
                self._run(np_dtype, M, K, N, [M, N], False, True, alpha, beta)
        with self.subTest("K=0"):
            self._run(np_dtype, 3, 0, 4, [3, 4], False, False, alpha=1.0, beta=1.0)

    def test_float(self):
        self._cases(np.float32)

    def test_double(self):
        self._cases(np.float64)

    def test_float16(self):
        self._cases(np.float16)

    @unittest.skipIf(BF16 is None, "ml_dtypes not installed")
    def test_bfloat16(self):
        self._cases(BF16)

    def test_overflow_gives_nan(self):
        big = np.full((2, 3), 300.0, np.float16)
        sess = make_session(
            make_model(TensorProto.FLOAT16, [2, 3], [3, 2], None, False, False, 1.0, 1.0),
            providers=self.providers(),
        )
        y, e = sess.run(None, {"A": big, "B": np.full((3, 2), 300.0, np.float16)})
        self.assertTrue(np.all(np.isinf(y)))
        self.assertTrue(np.all(np.isnan(e)), e)


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestGemmWithErrorCuda(GemmWithErrorCases, unittest.TestCase):
    provider = CUDA


class TestGemmWithErrorCpu(GemmWithErrorCases, unittest.TestCase):
    provider = CPU


if __name__ == "__main__":
    unittest.main()
