#!/usr/bin/env python3
"""Export existing ONNX intermediate tensors as additional graph outputs. NO TRAINING.
Usage:
  pip install onnx
  python scripts/expose_onnx_probes.py model.onnx model.observed.onnx --list
  python scripts/expose_onnx_probes.py model.onnx model.observed.onnx --probe /encoder/layer.0/output --probe /encoder/layer.1/output
The chosen probe must exist in inferred value_info or an existing graph output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    import onnx
    from onnx import shape_inference, checker, helper
    source = args.model.read_bytes()
    model = onnx.load_from_string(source)
    checker.check_model(model)
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception as exc:
        raise SystemExit(f"Shape inference failed: {exc}") from exc
    candidates = {entry.name: entry for entry in
                  list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)}
    if args.list:
        for name in sorted(candidates):
            print(name)
        if not args.probe:
            return
    missing = [n for n in args.probe if n not in candidates]
    if missing:
        raise SystemExit("Unknown / uninferred probe(s): " + ", ".join(missing) +
                         ". Export an explicitly instrumented model from the source framework.")
    existing = {o.name for o in model.graph.output}
    for name in args.probe:
        if name not in existing:
            model.graph.output.append(candidates[name])
            existing.add(name)
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    report = {
        "source_file": args.model.name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_file": args.output.name,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "exposed_outputs": list(existing),
        "training_performed": False,
        "note": "Internal tensors are extra graph outputs; this is not causal attribution."
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
