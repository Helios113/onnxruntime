"""Static audit: ONNX-domain ops where the CUDA EP has kernels but none covering the schema version in force at
opset N (19..22) -- i.e. a graph exported at that opset silently falls back to the CPU EP.

Usage: python cuda_opset_gaps.py [path/to/cuda_execution_provider.cc]
Needs the `onnx` package (schemas) -- only schema since_versions are used, so any onnx >= 1.16 gives the same answer
for opsets <= 21.
"""
import re
import sys
from collections import defaultdict

import onnx.defs

src = sys.argv[1] if len(sys.argv) > 1 else "onnxruntime/core/providers/cuda/cuda_execution_provider.cc"
text = open(src).read()

# BuildKernelCreateInfo<ONNX_OPERATOR_[VERSIONED_][TYPED_]KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, S[, E][, T], Op)>
pat = re.compile(r"BuildKernelCreateInfo<ONNX_OPERATOR_(VERSIONED_)?(?:TYPED_)?KERNEL_CLASS_NAME\("
                 r"kCudaExecutionProvider,\s*kOnnxDomain,\s*(\d+)(?:,\s*(\d+))?(?:,\s*[\w:]+)?,\s*(\w+)\)>")
ranges = defaultdict(list)  # op -> [(start, end)]
for m in pat.finditer(text):
    versioned, s, e, op = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    if versioned:  # VERSIONED always has start,end
        ranges[op].append((s, int(e)))
    else:          # unversioned macro: exactly that version (what bit Cast)
        ranges[op].append((s, s))
# the TYPED unversioned form has (S, T, Op) so the optional E group can swallow nothing; fine.

schema_versions = defaultdict(set)
for sch in onnx.defs.get_all_schemas_with_history():
    if sch.domain == "":
        schema_versions[sch.name].add(sch.since_version)

rows = []
for op, rs in sorted(ranges.items()):
    vers = sorted(schema_versions.get(op, []))
    for n in (19, 20, 21, 22):
        eff = max((v for v in vers if v <= n), default=None)
        if eff is None:
            continue
        # only flag if the op already has a kernel at some earlier version (otherwise it's just "not implemented")
        if any(s <= eff <= e for s, e in rs):
            continue
        if max(e for _, e in rs) < eff:
            rows.append((op, n, eff, sorted(set(rs))))

if not rows:
    print("no gaps found")
for op, n, eff, rs in rows:
    print(f"opset {n}: {op:24s} schema v{eff:<3d} not covered; CUDA kernels: {rs}")
print(f"\n{len({r[0] for r in rows})} ops affected")
