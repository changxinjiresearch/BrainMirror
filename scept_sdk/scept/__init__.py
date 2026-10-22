"""SCEPT Research Python SDK: A0–A6 real NAF and preregistered controlled numeric protocol.

NAF state/pattern and U10 numeric protocol implemented; portable V6 ZIP implemented in A5.
"""
from .api import SCEPT, CapturedRun
from .contracts import SCEPTError, tensor_sha, tensor_array, stable_json
from .protocols import ProtocolRegistry, Mask, U10Plan, run_u10
from .calibration import build_reference, calibrate, empirical, scale
from .archive import ArchiveObservation, load_observation
__version__ = "0.2.0a6"
__all__ = ["SCEPT", "CapturedRun", "SCEPTError", "tensor_sha", "tensor_array", "stable_json", "ProtocolRegistry", "Mask", "U10Plan", "run_u10", "build_reference", "calibrate", "empirical", "scale", "ArchiveObservation", "load_observation"]
