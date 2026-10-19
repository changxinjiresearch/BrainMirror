"""SCEPT A3: NAF from ONLY real A2 tensors, numerically aligned to JS V6.2.

State magnitudes, channel patterns and verified same-forward representation
updates are NOT saliency, causal attribution or scientific/clinical validation.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

import numpy as np

from .contracts import fail, tensor_sha

NAF_SCHEMA = "scept.naf.v1"
PROFILES = {
    "generic": None,
    "vit-token-channel": {2: ("feature", 1), 3: ("token", 1)},
    "cnn-nchw": {2: ("feature", 1), 4: ("channel", 1)},
    "cnn-nhwc": {2: ("feature", 1), 4: ("channel", 3)},
    "cnn-volume-ncdhw": {5: ("channel", 1)},
    "cnn-volume-ndhwc": {5: ("channel", 4)},
    "legacy-vit-197x192": {3: ("token", 1)},
}

def _axis(profile: str, shape: tuple[int, ...]):
    if profile not in PROFILES:
        fail("PROFILE_UNSUPPORTED", f"Unknown NAF architecture profile {profile}")
    if profile == "legacy-vit-197x192" and shape != (1, 197, 192):
        fail("LEGACY_LAYOUT_MISMATCH", "Legacy ViT shape must be exactly [1,197,192]")
    if PROFILES[profile] is None:
        return None
    return PROFILES[profile].get(len(shape))

def _state_units(array: np.ndarray, role: str, axis: int, max_axis_units: int) -> list[dict[str, Any]]:
    # Explicit axis, not inferred from geometry or semantic model importance.
    if array.shape[axis] > max_axis_units:
        return []
    # Float64 aggregation matches JS Number arithmetic semantics.
    values = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)
    n = values.shape[1]
    rows = []
    for index, arr in enumerate(values):
        acc = 0.0; ss = 0.0; pos = 0; neg = 0
        for x in arr:
            v = float(x)
            acc += v; ss += v*v
            if v > 0: pos += 1
            elif v < 0: neg += 1
        rms = math.sqrt(ss/n)
        if not math.isfinite(rms) or not math.isfinite(acc):
            fail("NUMERIC_OVERFLOW", "Overflow while summarizing actual activation")
        rows.append({"index":index, "rms":rms, "mean":acc/n,
                     "positiveFraction":pos/n, "negativeFraction":neg/n,
                     "count":n})
    return rows

def fingerprint(capture: Any, *, max_axis_units: int = 2048) -> dict[str, Any]:
    if not 0 <= max_axis_units <= 16384:
        fail("INVALID_LIMIT", "Axis summary limit must lie in [0,16384]")
    record = capture.record
    if record["schema"] not in ("scept.python-capture.v1", "scept.run-record.v1"):
        fail("INVALID_CAPTURE", "Genuine A2 single-forward Run Record required")
    profile = record["axisProfile"]
    groups = []
    for tensor in capture.requested_probes:
        if tensor not in capture.tensors:
            groups.append({"status":"not_observed","tensor":tensor,
                           "reason":"probe_not_exposed","units":[]})
            continue
        arr = capture.tensor(tensor)
        match = _axis(profile, tuple(arr.shape))
        if match is None:
            groups.append({"status":"unsupported","tensor":tensor,
                           "reason":"no_native_units_for_profile_or_resource_cap","units":[]})
            continue
        role, axis = match
        rows = _state_units(arr,role,axis,max_axis_units)
        if not rows:
            groups.append({"status":"unsupported","tensor":tensor,
                           "reason":"no_native_units_for_profile_or_resource_cap","units":[]})
        else:
            groups.append({"status":"measured","tensor":tensor,
                "architecture":{"vit-token-channel":"vision-transformer",
                   "cnn-nchw":"convolutional","cnn-nhwc":"convolutional",
                   "cnn-volume-ncdhw":"convolutional-volume",
                   "cnn-volume-ndhwc":"convolutional-volume",
                   "legacy-vit-197x192":"legacy-vit-only"}.get(profile,"unknown"),
                "profile":profile, "role":role,"shape":list(arr.shape),
                "units":rows,"scientificMeaning":"numeric internal state magnitude, not importance"})
    return {"schema":NAF_SCHEMA, "version":"6.2.0-python-a3",
        "runId":record["runId"], "provenance":copy.deepcopy(record["provenance"]),
        "profile":profile, "groups":groups, "updates":[],
        "science":{"status":"single_run_observation_only","importance":"NOT_ESTABLISHED",
                   "stability":"NOT_EVALUATED","causality":"NOT_ESTABLISHED",
                   "clinical":"NOT_ESTABLISHED"},
        "sourceRecordSchema":record["schema"],
        "implementation":"python-a3-parity-tested-against-real-JS-V6"}

def pattern(capture: Any, tensor: str, index: int, *, max_values: int = 65536) -> dict[str, Any]:
    if tensor not in capture.tensors:
        fail("UNOBSERVED", "Raw tensor is not a real observed output: "+tensor)
    array = capture.tensor(tensor)
    profile = capture.record["axisProfile"]
    if isinstance(index,bool) or not isinstance(index,int) or index < 0:
        fail("BAD_UNIT", "Unit must be nonnegative integer")
    if array.ndim == 2 and array.shape[0] == 1 and index < array.shape[1]:
        values = array[0, index:index+1].astype(np.float64).tolist()
        coords = {"feature":index}
        provenance = "verified_same_forward_real_output_only"
        interpretation = "model output feature, NOT internal hidden unit"
    else:
        axes = {"vit-token-channel":(3,1),
            "legacy-vit-197x192":(3,1),
            "cnn-nchw":(4,1), "cnn-nhwc":(4,3),
            "cnn-volume-ncdhw":(5,1), "cnn-volume-ndhwc":(5,4)}
        rank_axis = axes.get(profile)
        if rank_axis is None or array.ndim != rank_axis[0] or array.shape[0] != 1:
            fail("BAD_SHAPE","Profile/rank incompatible with actual tensor")
        axis=rank_axis[1]
        if index>=array.shape[axis]:
            fail("BAD_UNIT","Requested unit outside actual tensor dimensions")
        sl=[slice(None)]*array.ndim
        sl[0]=0; sl[axis]=index
        values=array[tuple(sl)].reshape(-1).astype(np.float64).tolist()
        if profile in ("vit-token-channel","legacy-vit-197x192"):
            coords={"token":index,"featureChannels":array.shape[2]}
        elif "volume" in profile:
            spatial=[i for i in range(1,5) if i!=axis]
            coords={"channel":index,"depth":array.shape[spatial[0]],
                    "height":array.shape[spatial[1]],"width":array.shape[spatial[2]]}
        else:
            spatial=[i for i in range(1,4) if i!=axis]
            coords={"channel":index,"height":array.shape[spatial[0]],
                    "width":array.shape[spatial[1]]}
        provenance="verified_same_forward_private_runtime_tensor"
        interpretation="raw channels / spatial feature values only; no semantic labels"
    if len(values)>max_values:
        fail("RESOURCE_LIMIT","Unit pattern exceeds limit")
    if not all(math.isfinite(v) for v in values):
        fail("NONFINITE","Measured pattern contains invalid value")
    return {"schema":"scept.naf.pattern.v1","tensor":tensor,"profile":profile,
            "unit":index,"shape":list(array.shape),"coordinates":coords,
            "values":values,"provenance":provenance,"interpretation":interpretation}

def _connected(graph_edges: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None,
               original: str, final: str) -> bool:
    if not graph_edges: return False
    successors: dict[str, list[str]]={}
    for inputs,outputs in graph_edges:
        for src in inputs:
            successors.setdefault(src,[]).extend(outputs)
    queue=deque([original]); visited=set()
    while queue and len(visited)<50000:
        node=queue.popleft()
        if node==final:return True
        if node in visited:continue
        visited.add(node)
        queue.extend(x for x in successors.get(node,[]) if x not in visited)
    return False

def representation_update(capture: Any, *, input_tensor: str, output_tensor: str,
                          coordinate_semantics: str | None = None,
                          verified_identity: bool = False,
                          declared_by: str | None = None) -> dict[str, Any]:
    """Read ONLY same-forward observed tensors AND real frozen-ONNX graph path.

    PyTorch hooks alone cannot verify a true graph dependency, even if shapes
    happen to agree. We must explicitly return NOT_EVALUABLE in that case.
    """
    def no(reason): return {"schema":"scept.naf.update.v1",
        "inputTensor":input_tensor,"outputTensor":output_tensor,
        "status":"not_evaluable","reason":reason,"units":[]}
    if not input_tensor or not output_tensor:
        return no("no_declared_module_input_output")
    if coordinate_semantics!="same-residual-stream" or verified_identity is not True or not declared_by:
        return no("missing_verified_same_coordinate_semantics")
    if input_tensor not in capture.tensors or output_tensor not in capture.tensors:
        return no("input_or_output_not_observed")
    a=capture.tensor(input_tensor);b=capture.tensor(output_tensor)
    if a.dtype!=np.float32 or b.dtype!=np.float32 or a.shape!=b.shape:
        return no("shape_or_dtype_not_identical")
    if capture.record["adapter"]!="onnx" or not _connected(capture._graph_edges,input_tensor,output_tensor):
        return no("no_real_computation_path")
    profile=capture.record["axisProfile"]
    if profile not in ("vit-token-channel","legacy-vit-197x192") or a.ndim!=3 or a.shape[0]!=1:
        return no("update_axis_not_compatible")
    units=[]
    for i in range(a.shape[1]):
        ss=0.;signed=0.
        for j in range(a.shape[2]):
            d=float(b[0,i,j])-float(a[0,i,j])
            ss+=d*d;signed+=d
        units.append({"index":i,"rmsUpdate":math.sqrt(ss/a.shape[2]),
                      "meanSignedUpdate":signed/a.shape[2]})
    return {"schema":"scept.naf.update.v1","status":"measured",
        "inputTensor":input_tensor,"outputTensor":output_tensor,
        "profile":profile,"role":"token","units":units,
        "declaredBy":declared_by,
        "proof":"same-coordinate-adapter-declaration-and-real_onnx_dependency_path",
        "limitation":"representation update, not FLOPs, importance, or a causal path"}
