"""Probe: does the CUDA EP place Cast (and Add/Mul/Div/Sqrt) nodes at opset 18/19/21 for fp16/bf16 without CPU fallback?"""
import numpy as np
from onnx import TensorProto as TP, helper
import onnxruntime as ort

print("ort", ort.__version__, flush=True)


def build(nodes, ins, outs, opset):
    g = helper.make_graph(nodes, "g", ins, outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 9 if opset < 21 else 10
    return m.SerializeToString()


def vi(n, t):
    return helper.make_tensor_value_info(n, t, [4])


def try_session(model_bytes):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    s = ort.InferenceSession(model_bytes, so, providers=["CUDAExecutionProvider"])
    return s


def case(name, model_bytes, feeds):
    try:
        s = try_session(model_bytes)
        s.run(None, feeds)
        print(f"OK    {name}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {str(e).splitlines()[0][:230]}", flush=True)


x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
for opset in (18, 19, 21):
    for name, low in (("fp16", TP.FLOAT16), ("bf16", TP.BFLOAT16)):
        # fp32 -> low -> fp32 (two Casts)
        m = build([helper.make_node("Cast", ["x"], ["h"], to=low), helper.make_node("Cast", ["h"], ["y"], to=TP.FLOAT)],
                  [vi("x", TP.FLOAT)], [vi("y", TP.FLOAT)], opset)
        case(f"opset{opset} Cast fp32->{name}->fp32", m, {"x": x})
        # Cast + Add + Mul + Div + Sqrt chain in low precision
        nodes = [helper.make_node("Cast", ["x"], ["h"], to=low),
                 helper.make_node("Add", ["h", "h"], ["a"]), helper.make_node("Mul", ["a", "h"], ["b"]),
                 helper.make_node("Div", ["b", "h"], ["c"]), helper.make_node("Sqrt", ["c"], ["d"]),
                 helper.make_node("Cast", ["d"], ["y"], to=TP.FLOAT)]
        case(f"opset{opset} Cast+Add+Mul+Div+Sqrt+Cast {name}", build(nodes, [vi("x", TP.FLOAT)], [vi("y", TP.FLOAT)], opset), {"x": x})
