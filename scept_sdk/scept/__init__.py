"""SCEPT Research Python SDK: A0–A2 single-forward real tensor observation.

NAF/controlled XAI/portable V6 archive not yet implemented in Python; see A3–A5.
"""
from .api import SCEPT, CapturedRun
from .contracts import SCEPTError, tensor_sha, tensor_array, stable_json
__version__ = "0.2.0a2"
__all__ = ["SCEPT", "CapturedRun", "SCEPTError", "tensor_sha", "tensor_array", "stable_json"]
