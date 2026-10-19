"""SCEPT scientific contract primitives shared with the browser's V6 records.

A0: this module carries *observation evidence*, not model importance or causality.
Canonical numeric representation must be verified against JS, not assumed.
"""
from __future__ import annotations
import hashlib
import json
import math
import secrets
from datetime import datetime, timezone
from typing import Any
import numpy as np

SCHEMA = "scept.run-record.v1"
DTYPES = {
    np.dtype("float32"): "float32", np.dtype("float64"): "float64",
    np.dtype("int32"): "int32", np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8", np.dtype("bool"): "bool",
}
TYPE_MAP = {v: k for k, v in DTYPES.items()}
MAX_TENSOR_ELEMENTS = 8_000_000
MAX_CAPTURE_BYTES = 32_000_000

class SCEPTError(ValueError):
    """Fail-closed error with stable machine-readable error code."""
    def __init__(self, code: str, message: str):
        super().__init__(f"SCEPT [{code}]: {message}")
        self.code = code

def fail(code: str, message: str) -> None:
    raise SCEPTError(code, message)

def stable_json(value: Any) -> str:
    """Canonical JSON for this SDK's restricted A0 config types.

    JS V6 stable() has JS-number formatting; to avoid unverified float parity,
    this A0 public input config restricts values to strings, booleans, null,
    and safe integers. Float-bearing preprocess configs require an explicitly
    versioned serializer in A3.
    """
    def valid(x: Any) -> None:
        if x is None or isinstance(x, (str, bool)):
            return
        if isinstance(x, int) and not isinstance(x, bool) and abs(x) <= 2**53 - 1:
            return
        if isinstance(x, list):
            for y in x: valid(y)
            return
        if isinstance(x, dict) and all(isinstance(k, str) for k in x):
            for y in x.values(): valid(y)
            return
        fail("UNSUPPORTED_CANONICAL_VALUE",
             "A0 preprocessing metadata must use strings/bools/null/safe integers only")
    valid(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def tensor_array(source: Any, *, max_elements: int = MAX_TENSOR_ELEMENTS) -> tuple[np.ndarray, str]:
    """Snapshot exact real model tensor into CPU, C-contiguous little-endian values."""
    if hasattr(source, "detach"):  # torch Tensor; optional dependency
        source = source.detach().cpu().numpy()
    raw = np.asarray(source)
    if raw.ndim < 1 or raw.ndim > 8 or any(d <= 0 for d in raw.shape):
        fail("INVALID_SHAPE", "Only real nonempty rank 1..8 dense tensor data")
    if raw.size > max_elements:
        fail("RESOURCE_LIMIT", "Actual tensor exceeds element budget")
    native = raw.dtype.newbyteorder("=")
    if native not in DTYPES:
        fail("UNSUPPORTED_DTYPE", f"Actual dtype {raw.dtype} is not in V6 contract")
    dtype = DTYPES[native]
    # bool in JS contract is Uint8Array; normalize to a binary byte stream.
    if dtype == "bool":
        result = np.array(raw, dtype=np.uint8, order="C", copy=True)
    else:
        result = np.array(raw, dtype=native.newbyteorder("<"), order="C", copy=True)
    if dtype in ("float32", "float64") and not np.isfinite(result).all():
        fail("NONFINITE", "NaN/Inf in measured tensor is not permitted")
    return result, dtype

def tensor_sha(array: np.ndarray, dtype: str) -> str:
    if dtype not in TYPE_MAP: fail("UNSUPPORTED_DTYPE", dtype)
    if array.dtype != (np.dtype("uint8") if dtype == "bool" else TYPE_MAP[dtype]):
        fail("INVALID_DTYPE", "Canonical dtype does not match actual tensor bytes")
    header = stable_json({"type": dtype, "shape": list(array.shape)}).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()

def ensure_hash(value: str, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        fail("INVALID_HASH", description + " must be a verified 64-character SHA-256")
    return value.lower()

def build_record(
    *, model_id: str, model_sha256: str, preprocess_id: str,
    preprocess_config: dict[str, Any], runtime: dict[str, str],
    inputs: dict[str, tuple[np.ndarray, str]], outputs: dict[str, tuple[np.ndarray, str]],
    probes: tuple[str, ...], adapter: str, axis_profile: str,
    availability: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Construct strict V6.0-compatible scientific record from one actual forward.

    The browser V6 record needs the full runtime output signatures; adapter
    adapters must expose (or record) every output they deliberately capture.
    """
    if not model_id or not preprocess_id or not inputs or not outputs:
        fail("INVALID_RECORD", "Model ID, preprocess and named input/output signatures required")
    if not probes or len(probes) > 64 or len(set(probes)) != len(probes):
        fail("INVALID_PROBES", "Require 1..64 unique requested probes")
    pre = {"id": preprocess_id, "config": preprocess_config}
    pre_hash = hashlib.sha256(stable_json(pre).encode("utf-8")).hexdigest()
    meta_in = [{"name": n, "type": dt, "shape": list(a.shape)} for n, (a,dt) in inputs.items()]
    meta_out = [{"name": n, "type": dt, "shape": list(a.shape)} for n, (a,dt) in outputs.items()]
    observations, status = [], []
    for name in probes:
        measured = name in outputs
        # A2 does NOT manufacture NAF / RMS data: full numeric NAF is A3.
        observations.append({
            "tensor": name, "status": "captured_raw" if measured else "not_measured",
            "activity": None,
            "reason": "A3_NAF_NOT_YET_IMPLEMENTED" if measured else "probe_not_observed",
        })
        status.append({"tensor": name,
            "status": "measured" if measured else "not_observed",
            "reason": None if measured else "probe_not_observed"})
    # The A2 record is deliberately distinct: no fake 'measured activity' field
    # that would satisfy the V6 NAF schema without actually computing NAF.
    return {
        "schema": "scept.python-capture.v1", "compatibleBase": SCHEMA,
        "runId": "run-" + secrets.token_hex(16),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "model": {"id": model_id, "sha256": ensure_hash(model_sha256, "model")},
            "inputs": {n: tensor_sha(*inputs[n]) for n in sorted(inputs)},
            "preprocessing": {"id": preprocess_id, "sha256": pre_hash},
            "runtime": {"engine": runtime["engine"], "version": runtime["version"],
                        "provider": runtime["provider"]},
            "source": "actual-frozen-model-forward",
        },
        "signature": {"inputs": meta_in, "outputs": meta_out},
        "outputs": [{"name": n, "status": "measured", "type": dt,
                     "shape": list(a.shape), "sha256": tensor_sha(a, dt),
                     "elements": int(a.size)} for n,(a,dt) in outputs.items()],
        "observations": observations, "availability": status,
        "adapter": adapter, "axisProfile": axis_profile,
        "science": {
            "status": "raw_tensor_capture_only",
            "causal": "NOT_ESTABLISHED", "output_importance": "NOT_ESTABLISHED",
            "stability": "NOT_EVALUATED", "clinical": "NOT_ESTABLISHED",
            "NAF": "NOT_YET_COMPUTED_A3", "comparison_required": False,
        },
    }
