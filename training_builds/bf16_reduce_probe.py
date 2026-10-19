"""Probe: do CUDA BFloat16 reductions agree with an fp32 reference? Prints errors; always exits 0."""
import numpy as np
import ml_dtypes
from onnx import TensorProto as TP, helper
import onnxruntime as ort

rng = np.random.default_rng(0)
x32 = rng.standard_normal((64, 1000)).astype(np.float32)
xb = x32.astype(ml_dtypes.bfloat16)
ref_in = xb.astype(np.float32)  # what the kernel actually sees


def run(op, elem, x, opset=18):
    n = helper.make_node(op, ["x", "axes"], ["y"], keepdims=0)
    axes = helper.make_tensor("axes", TP.INT64, [1], [1])
    g = helper.make_graph([n], "g", [helper.make_tensor_value_info("x", elem, list(x.shape))],
                          [helper.make_tensor_value_info("y", elem, [x.shape[0]])], initializer=[axes])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)]); m.ir_version = 9
    s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider"
    return s.run(None, {"x": x})[0].astype(np.float32)


refs = {
    "ReduceSum": ref_in.sum(1),
    "ReduceMean": ref_in.mean(1),
    "ReduceSumSquare": (ref_in ** 2).sum(1),
    "ReduceL2": np.sqrt((ref_in ** 2).sum(1)),
    "ReduceL1": np.abs(ref_in).sum(1),
    "ReduceLogSumExp": np.log(np.exp(ref_in - ref_in.max(1, keepdims=True)).sum(1)) + ref_in.max(1),
    "ReduceMax": ref_in.max(1),
}
for op, ref in refs.items():
    try:
        out = run(op, TP.BFLOAT16, xb)
        rel = np.abs(out - ref) / (np.abs(ref) + 1e-6)
        print(f"{op:16s} max_rel_err={rel.max():.3e} mean_rel_err={rel.mean():.3e}  (bf16 eps ~7.8e-3)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{op:16s} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
