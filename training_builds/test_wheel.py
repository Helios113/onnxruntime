"""Smoke test for the onnxruntime-training wheel. Run inside a venv with the wheel installed, on a GPU node.

Covers the patches carried on top of upstream: BFloat16/Pow kernels, the Pow-gradient dtype fix, the
dtype-aware loss-gradient seed, plus one of upstream's new gradient builders (Atan).
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper

import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    BF16 = None

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"PASS  {name}", flush=True)
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:400]}", flush=True)


def make_model(nodes, inputs, outputs, opset=17):
    g = helper.make_graph(nodes, "g", inputs, outputs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 9
    return m


def vi(name, elem, shape=(4,)):
    return helper.make_tensor_value_info(name, elem, list(shape))


def cuda_session(model_bytes):
    s = ort.InferenceSession(model_bytes, providers=["CUDAExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
    return s


def t_env():
    print("ort", ort.__version__, "providers", ort.get_available_providers())
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    assert hasattr(C, "GradientGraphBuilder")


def t_pow_fwd(elem, np_dtype, atol):
    def run():
        m = make_model([helper.make_node("Pow", ["x", "y"], ["z"])], [vi("x", elem), vi("y", elem)], [vi("z", elem)])
        x = np.array([0.5, 1.0, 2.0, 3.0]).astype(np_dtype)
        y = np.array([2.0, 2.0, 3.0, 0.5]).astype(np_dtype)
        z = cuda_session(m.SerializeToString()).run(None, {"x": x, "y": y})[0]
        ref = np.power(x.astype(np.float32), y.astype(np.float32))
        np.testing.assert_allclose(z.astype(np.float32), ref, atol=atol, rtol=atol)
    return run


def grad_model(model, y_names, x_names, loss):
    b = C.GradientGraphBuilder(model.SerializeToString(), set(y_names), set(x_names), loss)
    b.build()
    return b.get_model()


def t_pow_grad(x_elem, x_np, y_elem, y_np, atol):
    """dz/dx for Pow(x, y) with base and exponent in *different* float dtypes -- the case the dtype fix targets."""
    def run():
        m = make_model([helper.make_node("Pow", ["x", "y"], ["z"])],
                       [vi("x", x_elem), vi("y", y_elem)], [vi("z", x_elem)])
        gm = grad_model(m, ["z"], ["x"], "z")
        sess = cuda_session(gm)
        x = np.array([0.5, 1.0, 2.0, 3.0]).astype(x_np)
        y = np.array([2.0, 2.0, 3.0, 2.0]).astype(y_np)
        names = [o.name for o in sess.get_outputs()]
        out = dict(zip(names, sess.run(None, {"x": x, "y": y})))
        gname = next(n for n in names if n.startswith("x") and n.endswith("_grad"))
        xf, yf = x.astype(np.float32), y.astype(np.float32)
        ref = yf * np.power(xf, yf - 1)
        np.testing.assert_allclose(out[gname].astype(np.float32), ref, atol=atol, rtol=atol)
    return run


def t_atan_grad():
    m = make_model([helper.make_node("Atan", ["x"], ["z"])], [vi("x", TP.FLOAT)], [vi("z", TP.FLOAT)])
    sess = cuda_session(grad_model(m, ["z"], ["x"], "z"))
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, sess.run(None, {"x": x})))
    gname = next(n for n in names if n.startswith("x") and n.endswith("_grad"))
    np.testing.assert_allclose(out[gname], 1.0 / (1.0 + x * x), atol=1e-5)


check("env: wheel imports, CUDA EP + GradientGraphBuilder present", t_env)
check("Pow fwd float32", t_pow_fwd(TP.FLOAT, np.float32, 1e-5))
check("Pow fwd float16", t_pow_fwd(TP.FLOAT16, np.float16, 1e-2))
if BF16 is not None:
    check("Pow fwd bfloat16 (patch)", t_pow_fwd(TP.BFLOAT16, BF16, 5e-2))
else:
    print("SKIP  Pow fwd bfloat16: ml_dtypes not installed")
check("Pow grad float32 x, float32 y", t_pow_grad(TP.FLOAT, np.float32, TP.FLOAT, np.float32, 1e-4))
check("Pow grad float32 x, float16 y (patch: mixed dtypes)", t_pow_grad(TP.FLOAT, np.float32, TP.FLOAT16, np.float16, 1e-2))
check("Pow grad float16 x, float32 y, fp16 loss seed (patch)", t_pow_grad(TP.FLOAT16, np.float16, TP.FLOAT, np.float32, 5e-2))
check("Atan grad (upstream 1.21 builder)", t_atan_grad)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
