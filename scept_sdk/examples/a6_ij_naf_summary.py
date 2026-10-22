"""Summarize real archived NAF states (not attribution or spatial importance)."""
import json
from pathlib import Path
import numpy as np
from scept import load_observation

root = Path("/kaggle/working/a6_ij_private")
records = []
for model in ("S", "J"):
    obs = load_observation(str(root / f"A6_IJ_{model}_local_private.zip"))
    naf = obs.recompute_naf()
    layers = []
    for group in naf["groups"]:
        if group.get("status") != "measured":
            continue
        if group["tensor"] not in ("blocks.0.norm1", "blocks.5.norm1", "blocks.11.norm1"):
            continue
        assert group["shape"] == [1,197,768] and len(group["units"]) == 197
        rms = np.array([unit["rms"] for unit in group["units"]],dtype=float)
        patch = rms[1:]
        layers.append({"layer":group["tensor"],"clsRms":float(rms[0]),
            "patchRmsMean":float(patch.mean()),"patchRmsStd":float(patch.std()),
            "patchRmsMedian":float(np.median(patch))})
    assert len(layers) == 3
    logits = obs.tensor("model.output").reshape(-1)
    records.append({"model":model,"runId":obs.run_id,
        "logitsLggHgg":logits.tolist(),"layers":layers})
result = {"status":"PASS_REAL_NAF_NUMERIC_SUMMARY","nModels":2,
    "nSlices":1,"causalAttribution":"NOT_ESTABLISHED","results":records}
path = root / "a6_ij_naf_numeric_summary.json"
path.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
print("Saved locally:",path)
