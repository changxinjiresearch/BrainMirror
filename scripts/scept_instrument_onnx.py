#!/usr/bin/env python3
"""Make pretrained ONNX models observable and optionally perturbable for SCEPT.

Examples:
    pip install onnx
    python scripts/scept_instrument_onnx.py model.onnx -o model.scept.onnx --list
    python scripts/scept_instrument_onnx.py model.onnx -o model.scept.onnx \
        --expose /encoder/layer.0/output,/encoder/layer.1/output \
        --mask-tensor /encoder/layer.0/output --normalize unit

The original pretrained weights are unchanged. No training is performed.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def inject(model, output_names, mask_tensor=None):
    import onnx
    from onnx import TensorProto, helper

    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    produced = {name: i for i, node in enumerate(graph.node) for name in node.output if name}
    existing = {v.name: v for v in list(graph.value_info) + list(graph.input) + list(graph.output)}
    before_outputs = {v.name for v in graph.output}
    for name in output_names:
        if name not in produced:
            raise ValueError("Tensor not found in graph nodes: " + name)
        if name not in existing:
            raise ValueError("No inferred tensor type/shape for: " + name)
        if name not in before_outputs:
            graph.output.append(existing[name])
    mask_name = None
    if mask_tensor:
        if mask_tensor not in produced:
            raise ValueError("Mask target must be an existing intermediate node output")
        info = existing.get(mask_tensor)
        if not info:
            raise ValueError("Cannot infer mask target shape")
        tensor = info.type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT:
            raise ValueError("Mask target must be float32")
        shape = []
        for dim in tensor.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value < 1:
                raise ValueError("Mask target needs static dimensions")
            shape.append(dim.dim_value)
        safe = re.sub("[^A-Za-z0-9_]+", "_", mask_tensor).strip("_")[-40:]
        mask_name = "__scept_mask__" + safe
        new_name = "__scept_gated__" + safe
        all_names = set(produced) | {v.name for v in graph.input}
        if mask_name in all_names or new_name in all_names:
            raise ValueError("Generated mask name conflicts with existing graph")
        graph.input.append(helper.make_tensor_value_info(mask_name, TensorProto.FLOAT, shape))
        new_node = helper.make_node("Mul", [mask_tensor, mask_name], [new_name], name="SCEPT_Internal_Mask")
        node_at = produced[mask_tensor]
        # Only consumers downstream of the producing node are rewired. The original
        # intermediate output remains accessible as an observation, if requested.
        graph.node.insert(node_at + 1, new_node)
        for node in list(graph.node)[node_at + 2:]:
            for k, val in enumerate(node.input):
                if val == mask_tensor:
                    node.input[k] = new_name
        # If this tensor is already a final model output, there is no downstream
        # consumer to intervene on; refuse to claim a causal ablation.
        if not any(new_name in n.input for n in graph.node):
            raise ValueError("Selected tensor has no downstream consumers; choose an internal activation")
        graph.value_info.append(helper.make_tensor_value_info(new_name, TensorProto.FLOAT, shape))
    onnx.checker.check_model(inferred)
    return inferred, mask_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--list", action="store_true", help="List ONNX node output names and exit")
    p.add_argument("--expose", default="", help="Comma-separated internal tensor names")
    p.add_argument("--mask-tensor", help="Optional internal float32 tensor to gate with a mask input")
    p.add_argument("--normalize", default="unit",
                   choices=["raw", "unit", "signed", "imagenet", "zscore"])
    p.add_argument("--input-name", help="ONNX primary input name (defaults to first)")
    args = p.parse_args()
    try:
        import onnx
    except ImportError as e:
        raise SystemExit("Install offline conversion dependency first: pip install onnx") from e
    model = onnx.load(str(args.model), load_external_data=True)
    if args.list:
        for node in model.graph.node:
            for name in node.output:
                if name:
                    print(name)
        return
    names = [x.strip() for x in args.expose.split(",") if x.strip()]
    if not names and not args.mask_tensor:
        raise SystemExit("Provide --expose or --mask-tensor (or --list)")
    if args.mask_tensor and args.mask_tensor not in names:
        names.append(args.mask_tensor)
    model, mask_name = inject(model, names, args.mask_tensor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    selected = args.input_name or next(
        (x.name for x in model.graph.input if not x.name.startswith("__scept_mask")), None)
    if not selected:
        raise SystemExit("No primary ONNX input detected")
    inp = next(x for x in model.graph.input if x.name == selected)
    shape = []
    for dim in inp.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise SystemExit("Model input shape is dynamic; provide a concrete shape in the website")
        shape.append(dim.dim_value)
    manifest = {
        "schema": "scept.model.v1",
        "name": args.model.stem + " (instrumented, frozen weights)",
        "sha256": sha,
        "input": {"name": selected, "shape": shape, "normalize": args.normalize},
        "observed": names,
        "intervention_mask": mask_name,
        "model_original": str(args.model),
        "training_performed": False,
        "scientific_validation": "NOT_PERFORMED"
    }
    manifest_path = args.output.with_suffix(".scept.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved ONNX:", args.output)
    print("Saved manifest:", manifest_path)
    print("SHA-256:", sha)
    print("Mask input:", mask_name or "none")


if __name__ == "__main__":
    main()
