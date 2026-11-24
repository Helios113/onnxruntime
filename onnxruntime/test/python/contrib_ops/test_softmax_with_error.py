# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for com.microsoft::SoftmaxWithError (CUDA and CPU: float/double/fp16/bf16).

Y = softmax(X) along `axis`, E the error propagated through x - max, exp, sum and division (the rounding error of exp
itself is not modelled): the chain evaluated exactly on the same x is Y + E. No higher-precision reference is used.
  1. sum_i (Y_i + E_i) = 1 per vector (exact rational sums of the outputs): the chain's exact result is a probability
     vector, so this holds to second order in the unit roundoff (1024 u^2 plus the rounding of E) while sum_i Y_i - 1 is only first order.
  2. Y + E against a numpy emulation of the chain in the accumulation type. The emulation's exp and summation order
     differ from the kernel's by an ulp or so, so the comparison is relative to a few ulps; the propagated subtraction
     error is several ulps for inputs far below the max, so dropping it is detected.
  3. Constant inputs, where exp is exact and E = 1/n - Y to the accumulation precision.
  4. B (bound on the unmodelled exp rounding): non-negative, NaN exactly where E is, covers the difference between the
     kernel's and the emulation's Y + E, and is only a few unit roundoffs relative to Y.
"""

import unittest
from fractions import Fraction

import numpy as np
from onnx import TensorProto, helper
from with_error_testlib import BF16, ELEM, IR_VERSION, acc_dtype, finish, make_session, two_sum_step, unit_roundoff

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def make_model(elem_type, shape, axis):
    e_type = TensorProto.DOUBLE if elem_type == TensorProto.DOUBLE else TensorProto.FLOAT
    node = helper.make_node("SoftmaxWithError", ["X"], ["Y", "E", "B"], domain="com.microsoft", axis=axis)
    graph = helper.make_graph(
        [node],
        "softmax_with_error",
        [helper.make_tensor_value_info("X", elem_type, shape)],
        [helper.make_tensor_value_info("Y", elem_type, None), helper.make_tensor_value_info("E", e_type, None),
         helper.make_tensor_value_info("B", e_type, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = IR_VERSION
    return model.SerializeToString()


def reference(x, axis, np_dtype, acc):
    """(Y, E) of the chain emulated in numpy in the accumulation type, vectors moved to the last axis."""
    xa = np.moveaxis(x, axis, -1).astype(acc)
    with np.errstate(all="ignore"):
        m = xa.max(axis=-1, keepdims=True)
        nm = -m
        t = (xa + nm).astype(acc)
        bb = (t - xa).astype(acc)
        dt = ((xa - (t - bb).astype(acc)).astype(acc) + (nm - bb).astype(acc)).astype(acc)
        dt = np.where(np.isfinite(t), dt, np.array(0, acc)).astype(acc)
        u = np.exp(t).astype(acc)
        eps = (u * dt).astype(acc)
        s = np.zeros(m.shape[:-1], acc)
        c = np.zeros(m.shape[:-1], acc)
        eps_sum = np.zeros(m.shape[:-1], acc)
        for i in range(xa.shape[-1]):
            s, c = two_sum_step(s, c, u[..., i], np.zeros_like(s), acc)
            eps_sum = (eps_sum + eps[..., i]).astype(acc)
        s = s[..., None]
        ds = (c + eps_sum).astype(acc)[..., None]
        y = (u / s).astype(acc)
        # fma(-y, s, u) / s: the float64 product of two float32 values is exact; double uses Fractions
        if acc == np.float32:
            r = ((u.astype(np.float64) - y.astype(np.float64) * s.astype(np.float64)).astype(np.float32) / s).astype(acc)
        else:
            f = np.frompyfunc(
                lambda uu, yy, ss: float(Fraction(float(uu)) - Fraction(float(yy)) * Fraction(float(ss))), 3, 1
            )
            r = (f(u, y, np.broadcast_to(s, y.shape)).astype(np.float64) / s).astype(acc)
        err = (r + ((eps - (y * ds).astype(acc)).astype(acc) / s).astype(acc)).astype(acc)
    yt = y.astype(np_dtype)
    e = finish(y, err, yt.astype(acc), acc)
    return np.moveaxis(yt, -1, axis), np.moveaxis(e, -1, axis)


def frac_sum_plus(y, e):
    """Fraction(y) + Fraction(e) for every element."""
    out = np.empty(y.shape, dtype=object)
    for idx in np.ndindex(*y.shape):
        out[idx] = Fraction(float(y[idx])) + Fraction(float(e[idx]))
    return out


class SoftmaxWithErrorCases:
    """Test bodies shared by the CUDA and CPU providers; subclasses set `provider`."""

    provider = None

    def session(self, elem_type, shape, axis=-1):
        providers = [(CUDA, {"use_tf32": 0}), CPU] if self.provider == CUDA else [CPU]
        sess = make_session(make_model(elem_type, shape, axis), providers=providers)
        self.assertEqual(sess.get_providers()[0], self.provider)
        return sess

    def _run(self, np_dtype, shape, axis, x):
        acc = acc_dtype(np_dtype)
        sess = self.session(ELEM[np_dtype], list(shape), axis)
        y, e, b = sess.run(None, {"X": x})
        self.assertEqual(y.dtype, np.dtype(np_dtype))
        self.assertEqual(e.dtype, np.dtype(acc))
        self.assertEqual(b.dtype, np.dtype(acc))
        self.assertEqual(y.shape, tuple(shape))
        self.assertEqual(e.shape, tuple(shape))
        self.assertEqual(b.shape, tuple(shape))
        self.assertTrue(np.all(y >= 0))
        # B is a non-negative bound, NaN exactly where E is
        np.testing.assert_array_equal(np.isnan(b), np.isnan(e))
        self.assertTrue(np.all(b[np.isfinite(b)] >= 0))
        return y, e, b

    def _check(self, np_dtype, shape, axis, seed=0):
        acc = acc_dtype(np_dtype)
        u = unit_roundoff(acc)
        rng = np.random.default_rng(seed)
        # spread far below the max so that x - max is inexact and the propagated subtraction error is visible
        x = (-rng.uniform(0.0, 60.0, size=shape) - 0.01).astype(np_dtype)
        y, e, b = self._run(np_dtype, shape, axis, x)
        self.assertTrue(np.any(e != 0))
        self.assertTrue(np.all(b > 0))

        total = frac_sum_plus(y, e)
        sums = np.moveaxis(total, axis, -1).sum(axis=-1)
        # E is rounded to the accumulation type: for fp16 / bf16 it is ~2^-8..2^-11 of Y, so that rounding (u |E|) is
        # far above u^2 and enters the bound
        e_abs = np.moveaxis(np.abs(e.astype(np.float64)), axis, -1).sum(axis=-1)
        for v in np.ndindex(*sums.shape):
            tol = 1024 * u * u + 2 * u * float(e_abs[v])
            self.assertLessEqual(abs(float(sums[v] - 1)), tol, (v, float(sums[v] - 1)))

        y_ref, e_ref = reference(x, axis, np_dtype, acc)
        ref_total = frac_sum_plus(y_ref, e_ref)
        for idx in np.ndindex(*shape):
            diff = abs(float(total[idx] - ref_total[idx]))
            self.assertLessEqual(diff, 12 * u * float(ref_total[idx]) + 2.0**-140, (idx, diff))
            # the reference's exp (and the kernel's) is within 1 ulp-ish of correct; B must cover the kernel's own
            self.assertLessEqual(diff, float(b[idx]) + 8 * u * float(ref_total[idx]) + 2.0**-140, (idx, diff, b[idx]))
            # B is a bound on the exp contribution, i.e. a few u relative to y, not orders of magnitude more
            self.assertLessEqual(float(b[idx]), 64 * u * float(ref_total[idx]) + 2.0**-140, (idx, b[idx]))

    def _cases(self, np_dtype):
        for shape, axis in (
            ((5, 7), -1),
            ((3, 4, 5), 1),
            ((3, 4, 5), 0),
            ((2, 300), 1),  # longer than one CUDA block
            ((1, 9), -1),
        ):
            with self.subTest(shape=shape, axis=axis):
                self._check(np_dtype, shape, axis)

    def test_float(self):
        self._cases(np.float32)

    def test_double(self):
        self._cases(np.float64)

    def test_float16(self):
        self._cases(np.float16)

    @unittest.skipIf(BF16 is None, "ml_dtypes not installed")
    def test_bfloat16(self):
        self._cases(BF16)

    def _constant(self, np_dtype, n):
        acc = acc_dtype(np_dtype)
        u = unit_roundoff(acc)
        tiny = 2.0**-149 if acc == np.float32 else 2.0**-1074
        y, e, _ = self._run(np_dtype, (2, n), 1, np.full((2, n), 1.5, np_dtype))
        for idx in np.ndindex(2, n):
            target = float(Fraction(1, n) - Fraction(float(y[idx])))
            self.assertLessEqual(abs(float(e[idx]) - target), 4 * u * abs(target) + 4 * u * u / n + tiny, (idx, e[idx]))

    def test_constant_inputs(self):
        for np_dtype in (np.float32, np.float64, np.float16):
            for n in (1, 3, 7):
                with self.subTest(dtype=np_dtype, n=n):
                    self._constant(np_dtype, n)

    def test_single_element_is_exact(self):
        y, e, b = self._run(np.float32, (4, 1), 1, np.array([[3.0], [-1e3], [0.0], [7.5]], np.float32))
        np.testing.assert_array_equal(y, np.ones((4, 1), np.float32))
        np.testing.assert_array_equal(e, np.zeros((4, 1), np.float32))
        self.assertTrue(np.all(b > 0))  # exp's rounding is bounded even when softmax is trivially 1

    def test_minus_inf_entries(self):
        x = np.array([[-np.inf, 0.0, 1.0]], np.float32)
        y, e, b = self._run(np.float32, (1, 3), 1, x)
        self.assertEqual(y[0, 0], 0.0)
        self.assertEqual(e[0, 0], 0.0)
        self.assertEqual(b[0, 0], 0.0)
        self.assertTrue(np.all(np.isfinite(e)), e)

    def test_all_minus_inf_gives_nan(self):
        sess = self.session(TensorProto.FLOAT, [1, 3])
        y, e, b = sess.run(None, {"X": np.full((1, 3), -np.inf, np.float32)})
        self.assertTrue(np.all(np.isnan(e)), (y, e))
        self.assertTrue(np.all(np.isnan(b)), (y, b))


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestSoftmaxWithErrorCuda(SoftmaxWithErrorCases, unittest.TestCase):
    provider = CUDA


class TestSoftmaxWithErrorCpu(SoftmaxWithErrorCases, unittest.TestCase):
    provider = CPU


if __name__ == "__main__":
    unittest.main()
