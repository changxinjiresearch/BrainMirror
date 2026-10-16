#!/usr/bin/env python3
"""Expose ONNX intermediate tensors or insert a runtime activation ablation mask.

NO training and no modification of learned weights.

Usage:
  pip install onnx
  python scripts/expose_onnx_probes.py model.onnx instrumented.onnx --list
  python scripts/expose_onnx_probes.py model.onnx instrumented.onnx --probe attention_weights
  python scripts/expose_onnx_probes.py model.onnx ablatable.onnx --ablate encoder_hidden

The latter adds one input named scept_mask with the same fixed shape as the
selected intermediate tensor. Feed ones for baseline, zeros at selected
elements for exact internal ablation, and compare downstream outputs.
Only a single activation mask is currently supported. The selected value must
be a float32 tensor with fully fixed dimensions. Some ONNX graphs require
export-specific adaptation before this transformation is possible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--ablate", type=str)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    import onnx
    from onnx import TensorProto, checker, helper, shape_inference

    source = args.model.read_bytes()
    model = onnx.load_from_string(source)
    checker.check_model(model)
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception as exc:
        raise SystemExit(f"Shape inference failed: {exc}") from exc

    candidates = {
        entry.name: entry
        for entry in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    if args.list:
        for name in sorted(candidates):
            print(name)
        if not args.probe and not args.ablate:
            return

    selected = list(args.probe)
    if args.ablate:
        selected.append(args.ablate)
    missing = [name for name in selected if name not in candidates]
    if missing:
        raise SystemExit(
            "No known shape for: " + ", ".join(missing)
            + ". Export intermediate tensors with explicit shapes first."
        )

    if args.ablate:
        name = args.ablate
        if any(inp.name == "scept_mask" for inp in model.graph.input):
            raise SystemExit("Reserved graph input scept_mask already exists.")
        vi = candidates[name]
        tensor_type = vi.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT:
            raise SystemExit("Internal ablation currently supports float32 tensors only.")
        dims = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise SystemExit("Ablation tensor dimensions must be fully static.")
            dims.append(int(dim.dim_value))
        producer = None
        for idx, node in enumerate(model.graph.node):
            if name in node.output:
                producer = idx
                break
        if producer is None:
            raise SystemExit("Ablation target must be produced by a graph node.")
        existing_names = set(candidates)
        for node in model.graph.node:
            existing_names.update(node.input)
            existing_names.update(node.output)
        raw_name = name + "__scept_unmasked"
        if raw_name in existing_names:
            raise SystemExit("Internal unmasked tensor naming collision.")
        node = model.graph.node[producer]
        for i, output in enumerate(node.output):
            if output == name:
                node.output[i] = raw_name
        raw_info = copy.deepcopy(vi)
        raw_info.name = raw_name
        model.graph.value_info.append(raw_info)
        model.graph.input.append(
            helper.make_tensor_value_info("scept_mask", TensorProto.FLOAT, dims)
        )
        replacement = helper.make_node(
            "Mul", [raw_name, "scept_mask"], [name],
            name="SCEPT_ActivationMask"
        )
        model.graph.node.insert(producer + 1, replacement)
        if name not in args.probe:
            selected.append(name)

    original_outputs = {entry.name for entry in model.graph.output}
    for name in selected:
        if name not in original_outputs:
            model.graph.output.append(copy.deepcopy(candidates[name]))
            original_outputs.add(name)

    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    artifact = args.output.read_bytes()
    report = {
        "schema": "scept.adapter.v1",
        "source": args.model.name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "instrumented_model": args.output.name,
        "instrumented_sha256": hashlib.sha256(artifact).hexdigest(),
        "observed_outputs": [entry.name for entry in model.graph.output],
        "internal_ablation": (
            {"tensor": args.ablate, "input": "scept_mask",
             "baseline": "all ones", "ablation": "zero chosen indices"}
            if args.ablate else None
        ),
        "model_training_performed": False,
        "note": "Only a controlled internal intervention, not proof of cognitive or biological mechanism."
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
