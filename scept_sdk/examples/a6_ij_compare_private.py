"""A6-IJ: compare two verified real SCEPT NAF summaries, archive privately.

Descriptive study of ONE validation MRI slice. Not saliency, causal evidence,
patient-level inference, test evaluation, or scientific superiority evidence.
No raw tensors, MRI inputs or medical images are uploaded by this script.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation

REPO = "changxinjiresearch/research-course-brats2020-ijepa-xai"
API = "https://api.github.com/repos/" + REPO
PRIVATE_DIR = "results/scept_a6_ij/first_observation"
LAYER_ORDER = ("blocks.0.norm1", "blocks.5.norm1", "blocks.11.norm1")
CHECKPOINT_SHA = {
    "S": "2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf",
    "J": "719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640",
}

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return sha_bytes(Path(path).read_bytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/kaggle/working/a6_ij_private")
    parser.add_argument("--archive-private", action="store_true",
                        help="Explicitly upload derived summary to verified PRIVATE GitHub research repo")
    args = parser.parse_args()
    root = Path(args.root)
    summary_bytes = (root / "a6_ij_summary.json").read_bytes()
    source = json.loads(summary_bytes)
    if source.get("schema") != "scept.a6-ij.v1" or source.get("status") != "PASS_REAL_OBSERVATION_ONLY":
        raise RuntimeError("Source genuine observation record not accepted")
    entries = source.get("modelSAndJ")
    if not isinstance(entries, list) or len(entries) != 2 or {x.get("model") for x in entries} != {"S","J"}:
        raise RuntimeError("Requires one genuine S and J observation")
    if entries[0]["sourceInputSha256"] != entries[1]["sourceInputSha256"] or entries[0]["selectedSlice"] != entries[1]["selectedSlice"]:
        raise RuntimeError("Source input / slice differ; not a matched pair")
    base = json.loads((root / "a6_ij_naf_numeric_summary.json").read_text())
    if base.get("status") != "PASS_REAL_NAF_NUMERIC_SUMMARY" or base.get("nModels") != 2 or base.get("nSlices") != 1:
        raise RuntimeError("NAF recomputation source report not accepted")
    original = {x["model"]:x for x in entries}
    numeric = {x["model"]:x for x in base["results"]}
    if set(numeric) != {"S","J"}:
        raise RuntimeError("NAF report missing S or J")
    verified = {}
    for model in ("S","J"):
        entry = original[model]
        p = root / f"A6_IJ_{model}_local_private.zip"
        if sha(p) != entry["zipSha256"] or entry["checkpointSha256"] != CHECKPOINT_SHA[model]:
            raise RuntimeError("Source P4 best checkpoint / ZIP provenance invalid")
        obs = load_observation(str(p))
        if obs.run_id != entry["runId"] or numeric[model]["runId"] != obs.run_id:
            raise RuntimeError("Run ID provenance mismatch")
        if obs.manifest["captureScope"]["inputsIncluded"] is not False:
            raise RuntimeError("Refuse input-containing raw observation ZIP")
        # Recompute from the original integrity-checked source archive to avoid
        # trusting a free-standing, editable analysis JSON.
        groups = {g["tensor"]:g for g in obs.recompute_naf()["groups"] if g["status"]=="measured"}
        if set(groups) != set(LAYER_ORDER):
            raise RuntimeError("Original source NAF layer inventory differs")
        layers = {x["layer"]:x for x in numeric[model]["layers"]}
        if set(layers) != set(LAYER_ORDER):
            raise RuntimeError("Three numeric layer summaries required")
        for layer in LAYER_ORDER:
            values = groups[layer]["units"]
            if len(values)!=197 or groups[layer]["shape"]!=[1,197,768]:
                raise RuntimeError("Token/channel geometry mismatch")
            import numpy as np
            rms = np.asarray([v["rms"] for v in values],dtype=np.float64)
            actual = {
                "clsRms":float(rms[0]),
                "patchRmsMean":float(rms[1:].mean()),
                "patchRmsStd":float(rms[1:].std(ddof=0)),
                "patchRmsMedian":float(np.median(rms[1:])),
            }
            for metric,number in actual.items():
                if not np.isfinite(number) or abs(number-float(layers[layer][metric])) > 1e-10:
                    raise RuntimeError("Unexpected numerical modification: " + model + " / " + layer)
        logits = obs.tensor("model.output").reshape(-1).astype(float).tolist()
        if len(logits)!=2 or any(abs(a-b)>1e-6 for a,b in zip(logits,numeric[model]["logitsLggHgg"])):
            raise RuntimeError("Source logits and report differ")
        verified[model] = {"runId":obs.run_id,"zipSha256":sha(p),"layers":layers,
                           "logitsLggHgg":logits}
    comparisons = []
    for layer in LAYER_ORDER:
        a,b=verified["S"]["layers"][layer],verified["J"]["layers"][layer]
        comparisons.append({"layer":layer,"S":a,"J":b,
            "JminusS":{
                key:float(b[key])-float(a[key])
                for key in ("clsRms","patchRmsMean","patchRmsStd","patchRmsMedian")
            }})
    record = {
        "schema":"scept.a6-ij.naf-compare.v1",
        "status":"PASS_DESCRIPTIVE_COMPARISON_ONLY",
        "sourceSummarySha256":sha_bytes(summary_bytes),
        "sourceNumericSummarySha256":sha(root/"a6_ij_naf_numeric_summary.json"),
        "inputIdentity":"same P1 VAL slice; sealed test not accessed",
        "nPatients":1,"nSlices":1,
        "models":{"S":{"runId":verified["S"]["runId"],"zipSha256":verified["S"]["zipSha256"]},
                  "J":{"runId":verified["J"]["runId"],"zipSha256":verified["J"]["zipSha256"]}},
        "layers":comparisons,
        "limitations":{
            "layerNormalization":"norm1 RMS is constrained by LayerNorm parameters; not importance",
            "crossModelChannelAlignment":"NOT_ESTABLISHED",
            "spatialDependence":"NOT_EVALUATED",
            "modelReexecution":"NOT_PERFORMED",
            "causalOrClinicalEvidence":"NOT_ESTABLISHED",
            "populationInference":"NOT_PERMITTED_FOR_SINGLE_SLICE",
        }
    }
    output=root/"a6_ij_naf_comparison.json"
    blob=(json.dumps(record,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode()
    output.write_bytes(blob)
    print("A6-IJ VERIFIED S/J NAF COMPARISON")
    print(f"{'layer':18} {'S patch RMS':>13} {'J patch RMS':>13} {'J-S':>12} {'S std':>11} {'J std':>11}")
    for row in comparisons:
        print(f"{row['layer']:18} {row['S']['patchRmsMean']:13.6f} {row['J']['patchRmsMean']:13.6f} {row['JminusS']['patchRmsMean']:12.6f} {row['S']['patchRmsStd']:11.6f} {row['J']['patchRmsStd']:11.6f}")
    print("Data are descriptive LayerNorm activation statistics, not MRI importance.")
    print("LOCAL COMPARISON:",output)
    if not args.archive_private:
        print("Private GitHub archive not requested; local comparison only.")
        return
    token=UserSecretsClient().get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Kaggle Secret GITHUB_TOKEN unavailable")
    session=requests.Session()
    session.headers.update({"Authorization":"Bearer "+token,
                            "Accept":"application/vnd.github+json",
                            "X-GitHub-Api-Version":"2022-11-28"})
    def request(method,url,**kwargs):
        r=session.request(method,url,timeout=(30,120),**kwargs)
        if r.status_code not in (200,201):
            raise RuntimeError(f"GitHub verification/write failed HTTP {r.status_code}")
        return r.json()
    info=request("GET",API)
    if info.get("private") is not True or info.get("full_name")!=REPO:
        raise RuntimeError("Cannot upload: verified private destination required")
    def github_file(name):
        return request("GET",f"{API}/contents/{PRIVATE_DIR}/{name}")
    remote_original=github_file("summary.json")
    if remote_original.get("encoding")!="base64" or sha_bytes(base64.b64decode(remote_original["content"]))!=sha_bytes(summary_bytes):
        raise RuntimeError("Source summary does not equal archived original in private GitHub")
    target=f"{API}/contents/{PRIVATE_DIR}/naf_comparison.json"
    old=session.get(target,timeout=(30,120))
    if old.status_code==200:
        existing=old.json()
        if existing.get("encoding")!="base64" or base64.b64decode(existing["content"])!=blob:
            raise RuntimeError("Existing comparison differs: immutable record, refusing overwrite")
    elif old.status_code==404:
        request("PUT",target,json={
            "message":"A6-IJ: archive verified descriptive S/J NAF comparison (validation-only)",
            "branch":"main","content":base64.b64encode(blob).decode("ascii")})
    else:
        raise RuntimeError(f"Cannot inspect private archival path HTTP {old.status_code}")
    check=github_file("naf_comparison.json")
    if check.get("encoding")!="base64" or base64.b64decode(check["content"])!=blob:
        raise RuntimeError("Private GitHub readback differs")
    print("A6-IJ PRIVATE NUMERIC COMPARISON ARCHIVE: PASS")
    print("Recorded:",PRIVATE_DIR+"/naf_comparison.json")

if __name__=="__main__":
    main()
