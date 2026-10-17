#!/usr/bin/env python3
"""Instrument the exact publicly pretrained ViT already hash-checked in CI.

This script reads /tmp/scept-vit-original.onnx saved by the browser compatibility
test. Selects a real mid-network float tensor, exposes it as an ONNX graph
output, validates the graph and writes a reproducibility manifest.
It performs NO training and does not change any learned weight.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = Path("/tmp/scept-vit-original.onnx")
INSTRUMENTED = Path("/tmp/scept-vit-observed.onnx")
EXPECTED_SHA = "32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3"


def main():
    assert ORIGINAL.is_file(), "Pretrained ViT test must run first"
    actual = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    assert actual == EXPECTED_SHA, "Original ViT pretrained weight SHA mismatch"
    model = onnx.load(str(ORIGINAL))
    graph = onnx.shape_inference.infer_shapes(model)
    info = {v.name: v for v in
            list(graph.graph.input) + list(graph.graph.output)
            + list(graph.graph.value_info)}
    original_outputs = {x.name for x in graph.graph.output}
    node_count = len(graph.graph.node)
    choices = []
    for i, node in enumerate(graph.graph.node):
        if i <= node_count * 0.25 or i >= node_count * 0.8:
            continue
        if node.op_type not in ("MatMul", "Add", "LayerNormalization"):
            continue
        for name in node.output:
            if name not in info or name in original_outputs:
                continue
            tp = info[name].type.tensor_type
            if tp.elem_type != onnx.TensorProto.FLOAT:
                continue
            dims = []
            for dim in tp.shape.dim:
                size = dim.dim_value if dim.HasField("dim_value") else 1
                dims.append(size)
            if len(dims) < 2:
                continue
            size = 1
            for d in dims:
                size *= max(1,d)
            if 128 <= size <= 1000000:
                choices.append((abs(i-node_count*.52), name, node.op_type, dims))
    if not choices:
        raise RuntimeError("No suitable ViT intermediate float tensor with inferred shape")
    _, name, op, shape = min(choices)
    subprocess.run([sys.executable, str(ROOT/"scripts/expose_onnx_probes.py"),
                   str(ORIGINAL),str(INSTRUMENTED),"--probe",name],check=True)
    onnx.checker.check_model(str(INSTRUMENTED))
    Path("/tmp/scept-vit-probe-name.txt").write_text(name)
    print(json.dumps({
        "model_sha256":actual,
        "probe":name,
        "producer_operator":op,
        "probe_inferred_dimensions":shape,
        "instrumented_sha256":hashlib.sha256(INSTRUMENTED.read_bytes()).hexdigest(),
        "trained_here":False,
        "purpose":"Genuine hidden-layer SCEPT observability, not clinical/scientific interpretation"
    },indent=2))


if __name__ == "__main__":
    main()
