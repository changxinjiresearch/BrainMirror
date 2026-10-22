"""A6-IJ-02D-D01: diagnose failed ORIGINAL predeclared 1e-6 replay gate.

IMPORTANT: DIAGNOSTIC ONLY. No weakening existing 02D PASS tolerances.
Identifies input/model/runtime mismatches and quantifies all per-layer raw
tensor errors from independent direct PyTorch Hooks against original 02C ZIPs.
Writes an immutable private GitHub diagnostic report, no patient MRI arrays.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import time
import base64
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import requests
import torch
import timm
from kaggle_secrets import UserSecretsClient
from scept import SCEPT
from scept.contracts import tensor_array,tensor_sha
from a6_ij_02b_measure import manifest_and_zip,x_from_npz,checkpoint_model,get_json,API,REPO
from a6_ij_02d_measure import (COHORT,ROOT,OUT,PROBES,MAX_REPLAY_ABS,
                                private_02c_cases,direct_vit,relative,sha_bytes)

DIAG_FOLDER="results/scept_a6_ij/02d_cross_arch_runtime"
DIAG_FILE="d01_first_replay_mismatch.json"

def private_save_once(session,raw):
 url=f"{API}/contents/{DIAG_FOLDER}/{DIAG_FILE}"
 existing=session.get(url,timeout=(30,120),headers={"Cache-Control":"no-cache"})
 if existing.status_code==200:
  doc=existing.json()
  if doc.get("encoding")!="base64":
   raise RuntimeError("Private diagnostic exists but is not decodable")
  archived=base64.b64decode(doc["content"])
  if archived!=raw:
   print("D01: private diagnostic already exists with different run evidence.")
   print("Existing source is immutable; local diagnosis remains saved. No overwrite.")
   return "EXISTING_DIFFERENT_D01_PRESERVED"
  return "EXISTING_IDENTICAL_D01_VERIFIED"
 if existing.status_code!=404:
  raise RuntimeError(f"Private source diagnostic inspection HTTP {existing.status_code}")
 payload={"branch":"main","message":"Preserve A6-IJ-02D-D01 frozen replay mismatch evidence",
          "content":base64.b64encode(raw).decode("ascii")}
 r=session.put(url,json=payload,timeout=(30,120))
 if r.status_code not in (200,201):
  raise RuntimeError(f"Private GitHub diagnosis write HTTP {r.status_code}")
 commit=r.json()["commit"]["sha"]
 for t in range(8):
  check=session.get(url,params={"ref":commit},timeout=(30,120),
                    headers={"Cache-Control":"no-cache"})
  if check.status_code==404:
   time.sleep(min(2+2*t,8));continue
  check.raise_for_status()
  read=check.json()
  if read.get("encoding")!="base64" or base64.b64decode(read["content"])!=raw:
   raise RuntimeError("Private pinned diagnosis readback mismatch")
  return "PRIVATE_PERSISTED_SHA256_VERIFIED"
 raise RuntimeError("Diagnosis written but readback delayed; keep local report and retry archive later")

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle private read/write secret GITHUB_TOKEN missing")
 session=requests.Session()
 session.headers.update({"Authorization":"Bearer "+token,
   "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 repo=get_json(session,API)
 if repo.get("private") is not True or repo.get("full_name")!=REPO:
  raise RuntimeError("Expected private research repo only")
 cohort,p1archive=manifest_and_zip(session)
 if tuple(r["patient_id"] for r in cohort[:2])!=COHORT:
  raise RuntimeError("The original pinned first two validation patients differ")
 originals,source=private_02c_cases(session)
 report={
  "schema":"scept.a6-ij-02d.d01.v1",
  "status":"DIAGNOSTIC_ONLY_NO_02D_PASS",
  "lockedToleranceMaxAbs":MAX_REPLAY_ABS,
  "source02C":source,
  "host":{"python":platform.python_version(),"torch":torch.__version__,
          "timm":timm.__version__,"os":platform.platform(),
          "torchThreads":torch.get_num_threads(),
          "torchMKLAvailable":bool(torch.backends.mkl.is_available())},
  "results":{},"diagnosticLimitations":[
    "Do not change original 02D replay threshold based on this report.",
    "Different CPU kernels, PyTorch/timm/library builds can change FP32 arithmetic.",
    "Matching input and state fingerprints establishes byte-equivalence of snapshots, not numerical equivalence of executions.",
    "No test set access, no new model training, no clinical or causal inference."]}
 with ZipFile(p1archive) as archive:
  inputs={r["patient_id"]:x_from_npz(archive,r) for r in cohort[:2]}
 models={name:checkpoint_model(name) for name in ("S","J")}
 for name in ("S","J"):
  model=models[name]
  # Independent of SCEPT.observe: model never rerun through the SCEPT adapter.
  state_hash=SCEPT(model,[PROBES[0]],backend="torch",
                   profile="vit-token-channel").model_sha256
  rows={}
  for patient in COHORT:
   orig=originals[patient][name]["observation"]
   x=inputs[patient]
   canonical=tensor_sha(*tensor_array(x))
   old_input=orig.record["provenance"]["inputs"].get("input")
   old_state=orig.record["provenance"]["model"]["sha256"]
   fingerprint={"originalInputSHA256":old_input,"freshInputSHA256":canonical,
                "identicalInputSHA256":bool(old_input==canonical),
                "originalModelStateSHA256":old_state,"freshModelStateSHA256":state_hash,
                "identicalModelStateSHA256":bool(old_state==state_hash)}
   fresh=direct_vit(model,x)
   errors={key:relative(orig.tensor(key),fresh[key])
           for key in (*PROBES,"model.output")}
   fails=[key for key,x in errors.items() if x["maxAbs"]>MAX_REPLAY_ABS]
   print(f"D01 {name} {patient}: INPUT SHA match={fingerprint['identicalInputSHA256']}, "
         f"STATE SHA match={fingerprint['identicalModelStateSHA256']}; "
         f"replay outside original maxAbs={MAX_REPLAY_ABS:g}: {len(fails)} / {len(errors)} tensors",
         flush=True)
   for key in (*PROBES,"model.output"):
    item=errors[key]
    print(f"    {key:20s}: maxAbs={item['maxAbs']:.10g} "
          f"relativeL2={item['relativeL2']:.10g} exact={item['exactEqual']}",
          flush=True)
   rows[patient]={"inputStateFingerprints":fingerprint,
      "originalRuntime":orig.record["provenance"]["runtime"],
      "perTensor":errors,"failedOriginalThreshold":fails}
  report["results"][name]=rows
 # All four comparisons must be recorded even when they fail. This is
 # research-integrity preservation, not a scientific or release pass.
 raw=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 path=OUT/"a6_ij_02d_d01_diagnostic.json"
 if path.exists() and path.read_bytes()!=raw:
  raise RuntimeError("Existing local D01 report differs; refuse silent overwrite")
 path.write_bytes(raw)
 print("D01 local original evidence saved:",path)
 print("D01 private GitHub status:",private_save_once(session,raw))
 print("D01: DONE. No numeric tolerance altered. 02D original scientific engineering gate remains BLOCKED.")

if __name__=="__main__":main()
