#!/usr/bin/env python3
"""Offline conversion of an existing PyTorch checkpoint to ONNX. NO TRAINING.

Run only with a trusted/local Python model factory and checkpoint. Never execute
untrusted model Python code or allow pickle loading of untrusted weights.

Example:
  python scripts/export_pytorch_to_onnx.py \
    --factory trusted_package.model:MyModel \
    --checkpoint checkpoints/best.pt --shape 1,1,64,64,64 \
    --output exports/model.onnx --kwargs '{"channels":32}'

The Python factory is trusted source code supplied by the researcher; the
checkpoint must be a state_dict or a checkpoint containing state_dict.
Some models (dynamic control flow, custom operators) require custom adapters.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factory", required=True, help="module.path:callable")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--shape", required=True, help="Comma-separated fixed input dimensions")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--kwargs", default="{}", help="JSON kwargs for model constructor")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--input-name", default="input")
    p.add_argument("--output-name", default="output")
    args = p.parse_args()

    import torch
    import onnx

    if ":" not in args.factory:
        raise SystemExit("Factory must be module.path:callable")
    module, function = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), function)
    kwargs = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        raise SystemExit("--kwargs must be a JSON object")
    dims = tuple(int(i) for i in args.shape.split(","))
    if not dims or len(dims) > 6 or any(i < 1 for i in dims):
        raise SystemExit("Invalid input dimensions")
    model = factory(**kwargs)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    elif isinstance(weights, dict) and "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    if not isinstance(weights, dict):
        raise SystemExit("Expected a state_dict of tensors, not a serialized model object")
    model.load_state_dict(weights, strict=True)
    model.to("cpu").eval()
    sample = torch.zeros(dims, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model, sample, str(args.output),
            input_names=[args.input_name],
            output_names=[args.output_name],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print(json.dumps({
        "schema": "scept.export.v1",
        "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "exported_onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model_factory": args.factory,
        "shape": list(dims),
        "opset": args.opset,
        "trained_here": False
    }, indent=2))


if __name__ == "__main__":
    main()
