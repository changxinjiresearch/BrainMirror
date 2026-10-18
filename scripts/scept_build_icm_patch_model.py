#!/usr/bin/env python3
"""Generate a REAL executable ONNX activation-patching model for SCEPT V5 ICM.

The output graph retains its original computation but changes one chosen
existing intermediate tensor into a Where(mask, override, original) expression.
No activation is invented. User must test the unpatched output for exact parity.
Runtime can patch a single token (mask 1 at that token's actual channel values)
or a complete tensor; this builder does NOT infer biology.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def build(source: Path, probe: str, output: Path) -> dict:
    import onnx
    from onnx import TensorProto, helper, shape_inference
    if not source.is_file() or source.stat().st_size > 120_000_000:
        raise ValueError("ONNX must be a real local file <= 120 MB")
    original_bytes = source.read_bytes()
    source_hash = hashlib.sha256(original_bytes).hexdigest()
    model = onnx.load_from_string(original_bytes)
    if any(i.name in ("scept_patch_override", "scept_patch_mask") for i in model.graph.input):
        raise ValueError("Model already defines reserved patch input")
    if probe in {i.name for i in model.graph.input}:
        raise ValueError("Probe must be a computed intermediate tensor, not a model input")
    producers = [(i, n) for i, n in enumerate(model.graph.node) if probe in n.output]
    if len(producers) != 1:
        raise ValueError("Chosen intermediate must have exactly one real ONNX producer")
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception as exc:
        raise ValueError(f"ONNX shape inference failed: {exc}") from exc
    meta = next((v for v in list(inferred.graph.value_info)+list(inferred.graph.output)
                if v.name == probe), None)
    if meta is None or not meta.type.HasField("tensor_type"):
        raise ValueError("Target tensor has no inferred numeric shape")
    shape = meta.type.tensor_type.shape
    dims = [d.dim_value if d.HasField("dim_value") else None for d in shape.dim]
    elem_type = meta.type.tensor_type.elem_type
    if elem_type != TensorProto.FLOAT or not dims or any(d is None or d < 1 for d in dims):
        raise ValueError("V5 patch input requires static float32 tensor dimensions")
    if len(dims) != 3 or dims[0] != 1:
        raise ValueError("First ICM profile supports actual [1,tokens,channels] tensors only")
    if len(model.graph.node) > 200_000:
        raise ValueError("ONNX graph too large")
    idx, node = producers[0]
    shadow = probe + "__scept_original_activation"
    if any(shadow in n.output for n in model.graph.node):
        raise ValueError("Rewriter alias collision")
    node.output[list(node.output).index(probe)] = shadow
    model.graph.input.extend([
        helper.make_tensor_value_info("scept_patch_override", TensorProto.FLOAT, dims),
        helper.make_tensor_value_info("scept_patch_mask", TensorProto.BOOL, dims),
    ])
    replacement = helper.make_node("Where",
        ["scept_patch_mask", "scept_patch_override", shadow], [probe],
        name="scept_v5_genuine_activation_replacement")
    model.graph.node.insert(idx + 1, replacement)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    patch_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    return dict(schema="scept.v5.patch-model.v1",source=str(source),probe=probe,
        shape=dims,source_sha256=source_hash,patch_sha256=patch_hash,
        runtime_inputs=["pixel_values", "scept_patch_override","scept_patch_mask"],
        semantics="Actual ONNX Where mask on genuine internal runtime tensor; requires no-op parity and paired A/B forwards",
        clinical="NOT_ESTABLISHED")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Original model with real exported/inferable internal tensor")
    ap.add_argument("probe", help="Exact ONNX intermediate tensor name")
    ap.add_argument("output", type=Path, help="Destination patched ONNX model")
    args = ap.parse_args()
    r = build(args.source,args.probe,args.output)
    manifest=args.output.with_suffix(".patch.json")
    manifest.write_text(json.dumps(r, indent=2, ensure_ascii=False),encoding="utf-8")
    print(json.dumps(r, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
