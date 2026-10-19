"""A6-IJ privacy-preserving durable archival for real P4 observation outputs.

Runs in Kaggle AFTER a6_ijepa_p4_observe.py succeeds. Validates both ZIPs
with the production SCEPT SDK, verifies frozen local checkpoints, and uploads
raw tensor ZIPs ONLY to a private research-course GitHub Release. A small
metadata summary and receipt go to that same private repo (never BrainMirror).
Requires Kaggle Secret GITHUB_TOKEN with Contents: write on research repo.

Idempotent, fail-closed. No medical/scientific inference or test-set access.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import os
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation

REPO = "changxinjiresearch/research-course-brats2020-ijepa-xai"
API = "https://api.github.com/repos/" + REPO
TAG = "scept-a6-ij-p4-observation-v1"
LOCAL = Path("/kaggle/working/a6_ij_private")
MODEL_DIR = Path("/kaggle/working/a6_ij_models")
PREFIX = "results/scept_a6_ij/first_observation"
S_SHA = "2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf"
J_SHA = "719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"

def sha(path):
    d=hashlib.sha256()
    with open(path,"rb") as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()

try:
    token=UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception as exc:
    raise RuntimeError("Enable notebook access to existing Kaggle Secret GITHUB_TOKEN") from exc
if not token:
    raise RuntimeError("GITHUB_TOKEN not available in Kaggle Secrets; no upload attempted")
session=requests.Session()
session.headers.update({"Authorization":"Bearer "+token,
                        "Accept":"application/vnd.github+json",
                        "X-GitHub-Api-Version":"2022-11-28"})

def api_json(method, url, **kwargs):
    r=session.request(method,url,timeout=(30,240),**kwargs)
    if r.status_code not in (200,201):
        raise RuntimeError(f"Private GitHub API {method} failed: HTTP {r.status_code} at {url.split('?')[0]}")
    return r.json()

repo_info=api_json("GET",API)
if repo_info.get("private") is not True or repo_info.get("full_name")!=REPO:
    raise RuntimeError("Fail closed: target GitHub repository is not verified PRIVATE")

summary_path=LOCAL/"a6_ij_summary.json"
if not summary_path.is_file():
    raise RuntimeError("First execute A6-IJ real observation; summary file missing")
summary=json.loads(summary_path.read_text())
if summary.get("schema")!="scept.a6-ij.v1" or summary.get("status")!="PASS_REAL_OBSERVATION_ONLY":
    raise RuntimeError("Not a passed A6-IJ source observation")
if summary.get("dataset")!="P1 authorized validation slice; never test":
    raise RuntimeError("Source cohort is not the locked validation smoke cohort")
entries=summary.get("modelSAndJ")
if not isinstance(entries,list) or len(entries)!=2 or {e.get("model") for e in entries}!={"S","J"}:
    raise RuntimeError("Expected exactly paired S/J source observation records")

restore=json.loads((MODEL_DIR/"a6_ij_restore_local.json").read_text())
if restore.get("status")!="READY_FOR_A6_IJ" or restore.get("cohort")!="val" or restore.get("test_set")!="NOT_ACCESSED":
    raise RuntimeError("Cohort/split restriction not established")
val_path=Path(restore["validation_npz"])
if not val_path.is_file():
    raise RuntimeError("Frozen validation input missing")
if any(e.get("sourceInputSha256")!=sha(val_path) for e in entries):
    raise RuntimeError("Archived NPZ source digest differs from local locked validation cache")
for name, expected in (("S",S_SHA),("J",J_SHA)):
    if sha(MODEL_DIR/f"model_{name.lower()}_best.pt")!=expected:
        raise RuntimeError("P4 frozen best checkpoint mismatch for "+name)

archives=[]
for item in sorted(entries,key=lambda e:e["model"]):
    name=item["model"]
    p=LOCAL/f"A6_IJ_{name}_local_private.zip"
    if not p.is_file() or sha(p)!=item.get("zipSha256"):
        raise RuntimeError("Source ZIP missing or full SHA256 mismatch: "+name)
    if item.get("checkpointSha256")!= (S_SHA if name=="S" else J_SHA):
        raise RuntimeError("Wrong checkpoint provenance: "+name)
    imported=load_observation(str(p))
    if imported.run_id!=item["runId"] or len(imported.tensors)!=item["tensorCount"]:
        raise RuntimeError("ZIP source run identity/count mismatch")
    if imported.manifest["captureScope"]["inputsIncluded"] is not False:
        raise RuntimeError("Refuse raw input-containing patient ZIP: "+name)
    naf=imported.recompute_naf()
    if naf["science"]["model_forward_reexecuted"] is not False:
        raise RuntimeError("Recompute incorrectly claimed model rerun")
    measured={g["tensor"] for g in naf["groups"] if g["status"]=="measured"}
    if measured!=set(("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")):
        raise RuntimeError("Frozen three-layer NAF inventory missing")
    archives.append((name,p,sha(p)))

def release_get_or_create():
    response=session.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
    if response.status_code==200:
        obj=response.json()
    elif response.status_code==404:
        obj=api_json("POST",f"{API}/releases",json={
          "tag_name":TAG,"target_commitish":"main",
          "name":"PRIVATE SCEPT A6-IJ P4 first real model observation",
          "body":"Private research course S/J frozen-model activation ZIPs (derived patient data); NOT a test-set result, not scientific attribution proof. DO NOT publish.",
          "draft":False,"prerelease":True})
    else:
        raise RuntimeError(f"Cannot inspect PRIVATE release HTTP {response.status_code}")
    if obj.get("draft") or obj.get("prerelease") is not True:
        raise RuntimeError("Release is not safely identified as a prerelease")
    return obj

def verify_remote_asset(asset, expected_sha, expected_size):
    if int(asset["size"])!=expected_size:
        raise RuntimeError("Remote asset size mismatch: "+asset["name"])
    claimed=asset.get("digest")
    if claimed is not None and claimed!="sha256:"+expected_sha:
        raise RuntimeError("Remote asset SHA256 reported different digest")
    # Download for independently verified remote byte receipt even if metadata digest is absent.
    with session.get(f"{API}/releases/assets/{asset['id']}",
                     headers={"Accept":"application/octet-stream"},allow_redirects=True,
                     stream=True,timeout=(30,600)) as r:
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type","").lower():
            raise RuntimeError("Asset GET returned metadata rather than ZIP")
        d=hashlib.sha256();n=0
        for chunk in r.iter_content(chunk_size=8*1024*1024):
            if chunk:
                d.update(chunk);n+=len(chunk)
    if n!=expected_size or d.hexdigest()!=expected_sha:
        raise RuntimeError("Independent remote ZIP re-download SHA256 differs from source")
    return True

release=release_get_or_create()
uploaded=[]
for name,p,digest in archives:
    target=f"a6_ij_{name.lower()}_{digest[:12]}.zip"
    list_url=f"{API}/releases/{release['id']}/assets?per_page=100"
    current=api_json("GET",list_url)
    existing=[a for a in current if a["name"]==target]
    if len(existing)>1:
        raise RuntimeError("Duplicate named GitHub release assets")
    if existing:
        asset=existing[0]
    else:
        upload_url=release["upload_url"].split("{",1)[0]
        with p.open("rb") as src:
            asset=api_json("POST",upload_url,params={"name":target},
                           headers={"Content-Type":"application/zip",
                                    "Content-Length":str(p.stat().st_size)},
                           data=src)
    verify_remote_asset(asset,digest,p.stat().st_size)
    uploaded.append({"model":name,"asset_name":target,
                     "asset_id":asset["id"],"sha256":digest,
                     "bytes":p.stat().st_size,"remoteRedownloadSha256Verified":True,
                     "releaseUrl":release["html_url"]})
    print("PRIVATE REMOTE ZIP FULL SHA256 PASS:",target,flush=True)

# Same private repository, separate extension path: do not overwrite P4 files.
def commit_private_json(path, data):
    content=(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    url=f"{API}/contents/{path}"
    current=session.get(url,timeout=(30,120))
    if current.status_code==200:
        old=current.json()
        if old.get("encoding")!="base64":
            raise RuntimeError("Unexpected existing GitHub content encoding")
        original=base64.b64decode(old["content"])
        if original==content:
            print("REUSE VERIFIED:",path)
            return
        raise RuntimeError("Existing immutable A6-IJ record differs; refusing overwrite")
    if current.status_code!=404:
        raise RuntimeError("Unable to inspect target record before commit")
    api_json("PUT",url,json={"message":"Archive independent SCEPT A6-IJ real P4 observation integrity record",
                            "content":base64.b64encode(content).decode("ascii"),
                            "branch":"main"})
    check=api_json("GET",url)
    if check.get("encoding")!="base64" or base64.b64decode(check["content"])!=content:
        raise RuntimeError("Remote GitHub record readback mismatch")

# Preserve exact source file from experiment in research PRIVATE repo.
source_summary=json.loads(summary_path.read_text())
receipt={"schema":"scept.a6-ij.private-archive.v1",
         "status":"ARCHIVED_PRIVATE_AND_REDOWNLOADED_SHA256_VERIFIED",
         "originalSummarySha256":sha(summary_path),
         "validationSource":"P1 pinned VAL-only (sealed test NOT ACCESSED)",
         "zipInputsIncluded":False,"realModelReplay":"NOT_PERFORMED",
         "modelOriginIndependentlyAuthenticated":False,
         "clinicalAndSpatialDependence":"NOT_ESTABLISHED",
         "assets":uploaded}
commit_private_json(PREFIX+"/summary.json",source_summary)
commit_private_json(PREFIX+"/archive_receipt.json",receipt)
print("\nA6-IJ PRIVATE ARCHIVE: PASS")
print("Private release:",release["html_url"])
print("Private records:",PREFIX)
print("The complete observation ZIPs contain sensitive derived patient data.")
