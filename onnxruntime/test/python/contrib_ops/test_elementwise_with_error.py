# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for com.microsoft::{Add,Sub,Mul,Div,Sqrt}WithError (CUDA and CPU: float/double/fp16/bf16).

C is the correctly rounded result of the op, E the rounding error of C: exact(op) ~= C + E. E is checked
  1. bit-for-bit against a numpy emulation of the kernels' error-free transformations, and
  2. against the exact value (Fractions): |E - (exact - C)| within a few ulps plus a second-order term.
"""

import math
import unittest
from fractions import Fraction

import numpy as np
from onnx import TensorProto, helper
from with_error_testlib import BF16, ELEM, IR_VERSION, acc_dtype, finish, two_prod_err, unit_roundoff

from with_error_testlib import make_session

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

BINARY = ("Add", "Sub", "Mul", "Div")


def make_model(op, elem_type, a_shape, b_shape):
    e_type = TensorProto.DOUBLE if elem_type == TensorProto.DOUBLE else TensorProto.FLOAT
    unary = op == "Sqrt"
    node = helper.make_node(
        op + "WithError", ["A"] if unary else ["A", "B"], ["C", "E"], domain="com.microsoft"
    )
    graph_inputs = [helper.make_tensor_value_info("A", elem_type, a_shape)]
    if not unary:
        graph_inputs.append(helper.make_tensor_value_info("B", elem_type, b_shape))
    graph = helper.make_graph(
        [node],
        "elementwise_with_error",
        graph_inputs,
        [helper.make_tensor_value_info("C", elem_type, None), helper.make_tensor_value_info("E", e_type, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = IR_VERSION
    return model.SerializeToString()


def exact_remainder(a, b, r, acc, op):
    """Exact a - r*b (Div) or a - r*r (Sqrt) as an array of acc (exactly representable)."""
    if acc == np.float32:
        a64, r64 = a.astype(np.float64), r.astype(np.float64)
        return (a64 - r64 * (b.astype(np.float64) if op == "Div" else r64)).astype(np.float32)
    f = np.frompyfunc(
        lambda x, y, z: float(Fraction(float(x)) - Fraction(float(z)) * Fraction(float(y))),
        3,
        1,
    )
    return f(a, b if op == "Div" else r, r).astype(np.float64)


def reference(op, a, b, np_dtype, acc):
    """(C, E, r_acc, err) emulating the kernels."""
    a = a.astype(acc)
    with np.errstate(all="ignore"):
        if op in ("Add", "Sub"):
            bv = (-b if op == "Sub" else b).astype(acc)
            r = (a + bv).astype(acc)
            bb = (r - a).astype(acc)
            err = ((a - (r - bb).astype(acc)).astype(acc) + (bv - bb).astype(acc)).astype(acc)
        elif op == "Mul":
            b = b.astype(acc)
            r = (a * b).astype(acc)
            err = two_prod_err(a, b, r)
        elif op == "Div":
            b = b.astype(acc)
            r = (a / b).astype(acc)
            err = (exact_remainder(a, b, r, acc, op) / b).astype(acc)
        else:
            r = np.sqrt(a).astype(acc)
            rem = exact_remainder(a, None, r, acc, op)
            err = np.where(r == 0, np.array(0, acc), (rem / (r + r)).astype(acc)).astype(acc)
    c = r.astype(np_dtype)
    return c, finish(r, err, c.astype(acc), acc)


def exact_value(op, a, b):
    fa = Fraction(float(a))
    if op == "Sqrt":
        return None
    fb = Fraction(float(b))
    return {"Add": fa + fb, "Sub": fa - fb, "Mul": fa * fb, "Div": fa / fb if fb != 0 else None}[op]


class ElementwiseWithErrorCases:
    """Test bodies shared by the CUDA and CPU providers; subclasses set `provider`."""

    provider = None

    def session(self, op, elem_type, a_shape, b_shape):
        providers = [(CUDA, {"use_tf32": 0}), CPU] if self.provider == CUDA else [CPU]
        sess = make_session(make_model(op, elem_type, a_shape, b_shape), providers=providers)
        self.assertEqual(sess.get_providers()[0], self.provider)
        return sess

    def _inputs(self, rng, op, np_dtype, a_shape, b_shape):
        # wide dynamic range and mixed signs so the operations really round
        def gen(shape, lo, hi):
            mag = 2.0 ** rng.uniform(lo, hi, size=shape)
            return (mag * rng.choice([-1.0, 1.0], size=shape)).astype(np_dtype)

        lo, hi = {np.float64: (-30, 30), np.float32: (-12, 12), np.float16: (-3, 3)}.get(np_dtype, (-6, 6))
        a = gen(a_shape, lo, hi)
        b = None if op == "Sqrt" else gen(b_shape, lo, hi)
        if op == "Sqrt":
            a = np.abs(a)
        return a, b

    def _run(self, op, np_dtype, a_shape, b_shape, seed=0):
        acc = acc_dtype(np_dtype)
        rng = np.random.default_rng(seed)
        a, b = self._inputs(rng, op, np_dtype, a_shape, b_shape)
        sess = self.session(op, ELEM[np_dtype], a_shape, b_shape)
        feeds = {"A": a} if op == "Sqrt" else {"A": a, "B": b}
        c, e = sess.run(None, feeds)
        out_shape = a_shape if op == "Sqrt" else np.broadcast_shapes(tuple(a_shape), tuple(b_shape))
        self.assertEqual(c.dtype, np.dtype(np_dtype))
        self.assertEqual(e.dtype, np.dtype(acc))
        self.assertEqual(c.shape, tuple(out_shape))
        self.assertEqual(e.shape, tuple(out_shape))

        a_b = np.broadcast_to(a, out_shape)
        b_b = None if b is None else np.broadcast_to(b, out_shape)
        c_ref, e_ref = reference(op, a_b, b_b, np_dtype, acc)
        np.testing.assert_array_equal(c, c_ref)
        np.testing.assert_array_equal(e, e_ref)

        # vs the exact value
        u = unit_roundoff(acc)
        tiny = 2.0**-149 if acc == np.float32 else 2.0**-1074
        for idx in np.ndindex(*out_shape):
            if op == "Sqrt":
                # sqrt(a) - C = (a - C^2) / (sqrt(a) + C), exact numerator; the 1/(2C) first-order form is not accurate
                # enough once C carries fp16/bf16 rounding
                cv = Fraction(float(c[idx]))
                if cv == 0:
                    continue
                av = Fraction(float(a_b[idx]))
                target = float((av - cv * cv) / (Fraction(math.sqrt(float(av))) + cv))
            else:
                ex = exact_value(op, a_b[idx], b_b[idx])
                if ex is None:
                    continue
                target = float(ex - Fraction(float(c[idx])))
            err = abs(float(e[idx]) - target)
            tol = 4 * u * abs(target) + 4 * u * u * abs(float(c[idx])) + tiny
            self.assertLessEqual(err, tol, (idx, err, tol))

    def _cases(self, np_dtype):
        for op in BINARY:
            for a_shape, b_shape in (([7], [7]), ([3, 4, 5], [3, 4, 5]), ([3, 1, 5], [1, 4, 1]), ([4, 5], [5]), ([1], [3, 2])):
                with self.subTest(op=op, a_shape=a_shape, b_shape=b_shape):
                    self._run(op, np_dtype, a_shape, b_shape)
        for shape in ([11], [3, 4, 5]):
            with self.subTest(op="Sqrt", shape=shape):
                self._run("Sqrt", np_dtype, shape, None)

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
        big = np.full((4,), 300.0, np.float16)
        sess = self.session("Mul", TensorProto.FLOAT16, [4], [4])
        c, e = sess.run(None, {"A": big, "B": big})
        self.assertTrue(np.all(np.isinf(c)))
        self.assertTrue(np.all(np.isnan(e)), e)

    def test_division_by_zero_gives_nan(self):
        sess = self.session("Div", TensorProto.FLOAT, [2], [2])
        c, e = sess.run(None, {"A": np.array([1.0, 0.0], np.float32), "B": np.zeros(2, np.float32)})
        self.assertTrue(np.all(np.isnan(e)), (c, e))

    def test_incompatible_shapes_fail(self):
        with self.assertRaises(Exception):
            sess = self.session("Add", TensorProto.FLOAT, [3], [4])
            sess.run(None, {"A": np.zeros(3, np.float32), "B": np.zeros(4, np.float32)})


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestElementwiseWithErrorCuda(ElementwiseWithErrorCases, unittest.TestCase):
    provider = CUDA


class TestElementwiseWithErrorCpu(ElementwiseWithErrorCases, unittest.TestCase):
    provider = CPU


if __name__ == "__main__":
    unittest.main()
