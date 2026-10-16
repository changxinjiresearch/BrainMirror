#!/usr/bin/env python3
"""Create instrumented pretrained MNIST QA fixture, not for public deployment."""
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.request import urlopen
import onnx
from onnx import TensorProto
from scripts.scept_instrument_onnx import inject

SOURCE = "https://huggingface.co/onnxmodelzoo/mnist-8/resolve/main/mnist-8.onnx"
OUT = Path("scept-studio/fixture.instrumented.onnx")


def main():
    raw = urlopen(SOURCE, timeout=40).read()
    model = onnx.load_from_string(raw)
    inferred = onnx.shape_inference.infer_shapes(model)
    consumers = {i for n in inferred.graph.node for i in n.input}
    info = {i.name: i for i in list(inferred.graph.value_info) + list(inferred.graph.output)}
    choices = []
    for node in inferred.graph.node:
        for name in node.output:
            v = info.get(name)
            if not v or name not in consumers or node.op_type not in {"Relu", "Gemm", "MatMul", "Conv", "Add"}:
                continue
            tt = v.type.tensor_type
            if tt.elem_type != TensorProto.FLOAT:
                continue
            shape = [d.dim_value for d in tt.shape.dim]
            if shape and all(x > 0 for x in shape) and 1 < len(shape) <= 4:
                choices.append((node.op_type, name, shape))
    if not choices:
        raise ValueError("No suitable intermediate float32 tensor with downstream consumers")
    # Last suitable tensor stays modest in size; this is an engineering fixture.
    op, name, shape = choices[-1]
    instrumented, mask = inject(model, [name], name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(instrumented, OUT)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("INSTRUMENTED_PRETRAINED_FIXTURE", "tensor", name, "op", op, "shape", shape, "mask", mask, "sha", sha)
    print("weights unchanged; no training")

if __name__ == "__main__":
    main()
