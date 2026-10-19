"""Runtime probe: for each op, does a one-node graph at opset N run on CUDAExecutionProvider with CPU fallback disabled?

Complements cuda_opset_gaps.py (static audit of the kernel registrations). Exit code 1 if any case fails, so it can serve as
the regression test for the registration fixes. Optional argv: opset numbers to test (default 18 19 20 21 22).
"""
import sys

import numpy as np
from onnx import TensorProto as TP, helper, numpy_helper
import onnxruntime as ort

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except ImportError:
    BF16 = None

print("ort", ort.__version__, flush=True)


def vi(n, t, shape):
    return helper.make_tensor_value_info(n, t, list(shape))


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def cases(opset):
    """name -> (nodes, inputs, outputs, initializers, feeds)"""
    f = TP.FLOAT
    out = {}
    out["Cast fp32->fp16"] = ([helper.make_node("Cast", ["x"], ["y"], to=TP.FLOAT16)], [vi("x", f, X.shape)], [vi("y", TP.FLOAT16, X.shape)], [], {"x": X})
    out["Cast fp32->bf16"] = ([helper.make_node("Cast", ["x"], ["y"], to=TP.BFLOAT16)], [vi("x", f, X.shape)], [vi("y", TP.BFLOAT16, X.shape)], [], {"x": X})
    out["Cast fp16->fp32"] = ([helper.make_node("Cast", ["x"], ["y"], to=f)], [vi("x", TP.FLOAT16, X.shape)], [vi("y", f, X.shape)], [], {"x": X.astype(np.float16)})
    out["Cast int64->fp32"] = ([helper.make_node("Cast", ["x"], ["y"], to=f)], [vi("x", TP.INT64, X.shape)], [vi("y", f, X.shape)], [], {"x": X.astype(np.int64)})
    out["Reshape"] = ([helper.make_node("Reshape", ["x", "s"], ["y"])], [vi("x", f, X.shape)], [vi("y", f, (6, 4))], [init("s", np.array([6, 4], np.int64))], {"x": X})
    out["Shape"] = ([helper.make_node("Shape", ["x"], ["y"])], [vi("x", f, X.shape)], [vi("y", TP.INT64, (3,))], [], {"x": X})
    out["Size"] = ([helper.make_node("Size", ["x"], ["y"])], [vi("x", f, X.shape)], [vi("y", TP.INT64, ())], [], {"x": X})
    out["Squeeze"] = ([helper.make_node("Squeeze", ["x", "a"], ["y"])], [vi("x", f, (2, 1, 4))], [vi("y", f, (2, 4))], [init("a", np.array([1], np.int64))], {"x": np.ones((2, 1, 4), np.float32)})
    out["Unsqueeze"] = ([helper.make_node("Unsqueeze", ["x", "a"], ["y"])], [vi("x", f, (2, 4))], [vi("y", f, (2, 1, 4))], [init("a", np.array([1], np.int64))], {"x": np.ones((2, 4), np.float32)})
    out["Transpose"] = ([helper.make_node("Transpose", ["x"], ["y"], perm=[2, 0, 1])], [vi("x", f, X.shape)], [vi("y", f, (4, 2, 3))], [], {"x": X})
    out["Flatten"] = ([helper.make_node("Flatten", ["x"], ["y"], axis=1)], [vi("x", f, X.shape)], [vi("y", f, (2, 12))], [], {"x": X})
    out["Identity"] = ([helper.make_node("Identity", ["x"], ["y"])], [vi("x", f, X.shape)], [vi("y", f, X.shape)], [], {"x": X})
    out["ConstantOfShape"] = ([helper.make_node("ConstantOfShape", ["s"], ["y"], value=numpy_helper.from_array(np.array([1.0], np.float32)))], [], [vi("y", f, (2, 3))], [init("s", np.array([2, 3], np.int64))], {})
    out["Pad"] = ([helper.make_node("Pad", ["x", "p"], ["y"])], [vi("x", f, (2, 3))], [vi("y", f, (4, 5))], [init("p", np.array([1, 1, 1, 1], np.int64))], {"x": np.ones((2, 3), np.float32)})
    out["Equal"] = ([helper.make_node("Equal", ["x", "x"], ["y"])], [vi("x", f, X.shape)], [vi("y", TP.BOOL, X.shape)], [], {"x": X})
    out["ReduceMax"] = ([helper.make_node("ReduceMax", ["x", "a"], ["y"], keepdims=0)], [vi("x", f, X.shape)], [vi("y", f, (2, 4))], [init("a", np.array([1], np.int64))], {"x": X})
    out["ReduceMin"] = ([helper.make_node("ReduceMin", ["x", "a"], ["y"], keepdims=0)], [vi("x", f, X.shape)], [vi("y", f, (2, 4))], [init("a", np.array([1], np.int64))], {"x": X})
    out["AveragePool"] = ([helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[2, 2])], [vi("x", f, (1, 1, 4, 4))], [vi("y", f, (1, 1, 3, 3))], [], {"x": np.ones((1, 1, 4, 4), np.float32)})
    out["QuantizeLinear"] = ([helper.make_node("QuantizeLinear", ["x", "s", "z"], ["y"])], [vi("x", f, X.shape)], [vi("y", TP.INT8, X.shape)], [init("s", np.array(0.5, np.float32)), init("z", np.array(0, np.int8))], {"x": X})
    out["DequantizeLinear"] = ([helper.make_node("DequantizeLinear", ["x", "s", "z"], ["y"])], [vi("x", TP.INT8, X.shape)], [vi("y", f, X.shape)], [init("s", np.array(0.5, np.float32)), init("z", np.array(0, np.int8))], {"x": X.astype(np.int8)})
    out["Resize"] = ([helper.make_node("Resize", ["x", "", "sc"], ["y"], mode="nearest")], [vi("x", f, (1, 1, 2, 2))], [vi("y", f, (1, 1, 4, 4))], [init("sc", np.array([1, 1, 2, 2], np.float32))], {"x": np.ones((1, 1, 2, 2), np.float32)})
    out["GridSample"] = ([helper.make_node("GridSample", ["x", "g"], ["y"])], [vi("x", f, (1, 1, 2, 2)), vi("g", f, (1, 2, 2, 2))], [vi("y", f, (1, 1, 2, 2))], [], {"x": np.ones((1, 1, 2, 2), np.float32), "g": np.zeros((1, 2, 2, 2), np.float32)})
    # If: condition is a graph input; both branches just produce a constant (so only If itself must be on CUDA)
    then_g = helper.make_graph([helper.make_node("Identity", ["x"], ["t"])], "then", [], [vi("t", f, X.shape)])
    else_g = helper.make_graph([helper.make_node("Neg", ["x"], ["e"])], "else", [], [vi("e", f, X.shape)])
    out["If"] = ([helper.make_node("If", ["c"], ["y"], then_branch=then_g, else_branch=else_g)], [vi("c", TP.BOOL, ()), vi("x", f, X.shape)], [vi("y", f, X.shape)], [], {"c": np.array(True), "x": X})
    return out


def run_case(opset, name, spec):
    nodes, ins, outs, inits, feeds = spec
    g = helper.make_graph(nodes, "g", ins, outs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 10
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.log_severity_level = 3
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CUDAExecutionProvider"])
    s.run(None, feeds)


opsets = [int(a) for a in sys.argv[1:]] or [18, 19, 20, 21, 22]
failed = 0
table = {}
for opset in opsets:
    for name, spec in cases(opset).items():
        try:
            run_case(opset, name, spec)
            table.setdefault(name, {})[opset] = "ok"
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0]
            kind = "cpu-fallback" if "default CPU EP" in msg else ("invalid-model" if "INVALID" in msg or "Unsupported model" in msg or "is not a registered" in msg or "Unrecognized" in msg else "other")
            table.setdefault(name, {})[opset] = kind
            failed += 1
            if kind == "other" or kind == "invalid-model":
                print(f"   [{name} @ opset {opset}] {msg[:200]}", flush=True)

print("\n%-22s" % "case" + "".join("%14s" % f"opset {o}" for o in opsets))
for name, row in table.items():
    print("%-22s" % name + "".join("%14s" % row.get(o, "-") for o in opsets))
print(f"\n{failed} failing (case, opset) combinations")
sys.exit(1 if failed else 0)
