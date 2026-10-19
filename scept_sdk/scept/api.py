"""Public SCEPT Python SDK A0–A2.

A genuine single forward captures named actual model outputs and layer tensors.
NAF mathematical calculation, U10 controlled explanation and V6 ZIP emission
remain A3–A5. They are intentionally NOT silently simulated here.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import PyTorchAdapter, ONNXAdapter
from .contracts import SCEPTError, fail, build_record

@dataclass(frozen=True)
class CapturedRun:
    """Immutable read-only access to one actual inference's private numeric snapshots."""
    _record: dict[str, Any]
    _tensors: dict[str, tuple[np.ndarray, str]]
    _inputs: dict[str, tuple[np.ndarray, str]]
    requested_probes: tuple[str, ...]

    @property
    def run_id(self) -> str:
        return self._record["runId"]

    @property
    def record(self) -> dict[str, Any]:
        """A defensive copy, not a mutable reference into a scientific record."""
        return copy.deepcopy(self._record)

    @property
    def tensors(self) -> tuple[str, ...]:
        """Exactly captured genuine tensor names, not invented layers."""
        return tuple(self._tensors)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self._inputs)

    def tensor(self, name: str) -> np.ndarray:
        """Return complete actual numeric tensor; caller receives its own copy."""
        if name not in self._tensors:
            fail("NOT_OBSERVED", f"Actual tensor not captured: {name}")
        return self._tensors[name][0].copy()

    def input(self, name: str) -> np.ndarray:
        """Return explicitly supplied preprocessed model input, not a NIfTI source."""
        if name not in self._inputs:
            fail("NOT_OBSERVED", f"Input feed is not present: {name}")
        return self._inputs[name][0].copy()

    def fingerprint(self):
        raise NotImplementedError("SCEPT Python NAF belongs to A3. Captured raw tensors are genuine.")

    def export(self, path: str):
        raise NotImplementedError("Cross-language verified scientific ZIP belongs to A5. No fake archive.")

class SCEPT:
    """Reusable independent observer for existing PyTorch or ONNX models.

    Parameters
    ----------
    model:
        Existing torch.nn.Module; ONNX .onnx pathname or serialized ONNX bytes.
    layers:
        Real PyTorch named module names, or real ONNX produced tensor names.
        Select exactly the internal values you want to observe.
    backend:
        'torch' or 'onnx'; can be inferred from model type or .onnx path.
    profile:
        Nonsemantic architecture/coordinate label only. NO output importance.
    """
    def __init__(self, model: Any, layers: list[str] | tuple[str, ...], *,
                 backend: str | None = None, profile: str = "generic",
                 providers: tuple[str, ...] = ("CPUExecutionProvider",),
                 max_capture_bytes: int = 32_000_000):
        if not isinstance(layers,(list,tuple)) or not layers or any(not isinstance(x,str) or not x for x in layers):
            fail("INVALID_PROBES","Specify actual named modules or graph tensors")
        if backend is None:
            backend = "onnx" if isinstance(model,(str,bytes,Path)) else "torch"
        if backend=="torch":
            self._adapter=PyTorchAdapter(model,tuple(layers),max_capture_bytes=max_capture_bytes)
        elif backend=="onnx":
            self._adapter=ONNXAdapter(model,tuple(layers),providers=providers,
                                      max_capture_bytes=max_capture_bytes)
        else:
            fail("INVALID_BACKEND","Only actual torch and onnx backends are supported")
        if not isinstance(profile,str) or not profile:
            fail("INVALID_PROFILE","Explicit nonempty coordinate profile required")
        self.backend=backend
        self.profile=profile
        self.layers=tuple(layers)

    def observe(self, x: Any, *, preprocess_id: str = "identity-v1",
                preprocess_config: dict[str,Any] | None = None,
                model_id: str = "user-supplied-frozen-model") -> CapturedRun:
        """Exactly ONE real forward, no gradient/perturbation/comparison.

        Preprocessing is done by the researcher before this call. The descriptor
        is a declared audit ID, NOT proof of the preprocessing implementation.
        """
        if preprocess_config is None: preprocess_config={"source":"caller-preprocessed"}
        if not isinstance(preprocess_config,dict):
            fail("INVALID_PREPROCESSING","Explicit JSON-safe processing config required")
        ins,outs,model_sha,runtime=self._adapter.observe_once(x)
        if not all(p in outs for p in self.layers):
            fail("PROBE_NOT_EXECUTED","Requested real probes missing")
        record=build_record(
            model_id=model_id,model_sha256=model_sha,preprocess_id=preprocess_id,
            preprocess_config=preprocess_config,runtime=runtime,inputs=ins,outputs=outs,
            probes=self.layers,adapter=self.backend,axis_profile=self.profile
        )
        # The Python A2 record advertises raw capture ONLY; it does not use
        # browser's NAF schema or call missing U10 methods.
        return CapturedRun(record,
             {name:(arr.copy(),dt) for name,(arr,dt) in outs.items()},
             {name:(arr.copy(),dt) for name,(arr,dt) in ins.items()}, self.layers)
