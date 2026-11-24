# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for com.microsoft::ReduceSumWithError (CUDA and CPU: float/double/fp16/bf16).

Y comes from the stock ReduceSum (CPU fp16/bf16: computed by the op itself). E is checked bit-for-bit against a numpy emulation of the kernel (TwoSum over the
reduced elements in row-major order of the reduced axes) and against the exact sum (Fractions).
"""

import unittest
from fractions import Fraction

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from with_error_testlib import BF16, ELEM, IR_VERSION, acc_dtype, finish, tolerance, two_sum_step

from with_error_testlib import make_session

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def make_model(elem_type, x_shape, axes, keepdims, noop_with_empty_axes=False):
    e_type = TensorProto.DOUBLE if elem_type == TensorProto.DOUBLE else TensorProto.FLOAT
    inputs = ["X"]
    initializers = []
    if axes is not None:
        inputs.append("axes")
        initializers.append(numpy_helper.from_array(np.array(axes, np.int64), "axes"))
    node = helper.make_node(
        "ReduceSumWithError",
        inputs,
        ["Y", "E"],
        domain="com.microsoft",
        keepdims=int(keepdims),
        noop_with_empty_axes=int(noop_with_empty_axes),
    )
    graph = helper.make_graph(
        [node],
        "reduce_sum_with_error",
        [helper.make_tensor_value_info("X", elem_type, x_shape)],
        [helper.make_tensor_value_info("Y", elem_type, None), helper.make_tensor_value_info("E", e_type, None)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = IR_VERSION
    return model.SerializeToString()


def reduced_axes(rank, axes):
    return tuple(range(rank)) if not axes else tuple(sorted(a % rank for a in axes))


def sum_error_ref(x, axes, y_full, acc):
    """Emulate the kernel: reduced elements summed in row-major order of the reduced axes."""
    rank = x.ndim
    red = reduced_axes(rank, axes)
    kept = tuple(a for a in range(rank) if a not in red)
    xt = np.transpose(x.astype(acc), kept + red)
    kept_shape = xt.shape[: len(kept)]
    flat = xt.reshape(kept_shape + (-1,))
    s = np.zeros(kept_shape, acc)
    c = np.zeros(kept_shape, acc)
    for r in range(flat.shape[-1]):
        s, c = two_sum_step(s, c, flat[..., r], np.zeros(kept_shape, acc), acc)
    return finish(s, c, y_full.reshape(kept_shape), acc), kept_shape, flat


class ReduceSumWithErrorCases:
    """Test bodies shared by the CUDA and CPU providers; subclasses set `provider`."""

    provider = None

    def session(self, *args, **kwargs):
        providers = [(CUDA, {"use_tf32": 0}), CPU] if self.provider == CUDA else [CPU]
        sess = make_session(make_model(*args, **kwargs), providers=providers)
        self.assertEqual(sess.get_providers()[0], self.provider)
        return sess

    def _run(self, np_dtype, shape, axes, keepdims, exp_range=(-6, 6), seed=0):
        acc = acc_dtype(np_dtype)
        rng = np.random.default_rng(seed)
        # wide dynamic range and mixed signs so the plain sum really rounds
        x = (2.0 ** rng.uniform(*exp_range, size=shape) * rng.choice([-1.0, 1.0], size=shape)).astype(np_dtype)
        sess = self.session(ELEM[np_dtype], list(shape), axes, keepdims)
        y, e = sess.run(None, {"X": x})
        self.assertEqual(e.dtype, np.dtype(acc))
        self.assertEqual(e.shape, y.shape)

        expected, kept_shape, flat = sum_error_ref(x, axes, y, acc)
        np.testing.assert_array_equal(e.reshape(kept_shape), expected)

        # vs the exact sum
        R = flat.shape[-1]
        for idx in np.ndindex(*kept_shape):
            terms = [Fraction(float(v)) for v in flat[idx]]
            exact = sum(terms, Fraction(0))
            mag = float(sum((abs(t) for t in terms), Fraction(0)))
            target = float(exact - Fraction(float(y.reshape(kept_shape)[idx])))
            err = abs(float(e.reshape(kept_shape)[idx]) - target)
            self.assertLessEqual(err, float(tolerance(np.array(target), np.array(mag), R, acc)), (idx, err))

    def _cases(self, np_dtype, exp_range):
        shape = (3, 4, 5)
        for axes in ([1], [0, 2], [-1], [0, 1, 2], None, [2, 0]):
            for keepdims in (True, False):
                with self.subTest(axes=axes, keepdims=keepdims):
                    self._run(np_dtype, shape, axes, keepdims, exp_range)
        with self.subTest("long reduction"):
            self._run(np_dtype, (2, 3000), [1], True, exp_range)
        with self.subTest("rank 1"):
            self._run(np_dtype, (1000,), None, False, exp_range)

    def test_float(self):
        self._cases(np.float32, (-6, 6))

    def test_double(self):
        self._cases(np.float64, (-20, 20))

    def test_float16(self):
        self._cases(np.float16, (-4, 4))

    @unittest.skipIf(BF16 is None, "ml_dtypes not installed")
    def test_bfloat16(self):
        self._cases(BF16, (-6, 6))

    def test_noop_with_empty_axes(self):
        x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
        sess = self.session(TensorProto.FLOAT, [3, 4], [], True, noop_with_empty_axes=True)
        y, e = sess.run(None, {"X": x})
        np.testing.assert_array_equal(y, x)
        np.testing.assert_array_equal(e, np.zeros_like(x))

    def test_overflow_gives_nan(self):
        x = np.full((2, 8), 30000.0, np.float16)  # row sums 240000 > 65504
        sess = self.session(TensorProto.FLOAT16, [2, 8], [1], False)
        y, e = sess.run(None, {"X": x})
        self.assertTrue(np.all(np.isinf(y)))
        self.assertTrue(np.all(np.isnan(e)), e)


@unittest.skipIf(CUDA not in ort.get_available_providers(), "CUDA EP not available")
class TestReduceSumWithErrorCuda(ReduceSumWithErrorCases, unittest.TestCase):
    provider = CUDA


class TestReduceSumWithErrorCpu(ReduceSumWithErrorCases, unittest.TestCase):
    provider = CPU


if __name__ == "__main__":
    unittest.main()
