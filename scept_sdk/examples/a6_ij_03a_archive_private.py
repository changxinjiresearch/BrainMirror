"""A6-IJ-03A: commit strictly joined original evidence only to PRIVATE research GitHub.

Analysis does not copy original patient raw tensors: source 02B/02C and P4B
raw ZIPs are already private in their independent archived releases.
Verify original 110 exact matched records and SHA-indexed source inventory.
Immutable report+locked plan+remote pinned readback. Do not overwrite.
"""
import base64
import hashlib
import json
import time
from pathlib import Path

import requests
from kaggle_secrets import UserSecretsClient
from a6_ij_03a_pair_source import REPO,API,DEST,PLAN,plan_sha

TARGET="results/scept_a6_ij/03a_exact_first_slice_pairing"
def digest(b):return hashlib.sha256(b).hexdigest()
def safe_get(s,url):
 r=s.get(url,timeout=(30,120));r.raise_for_status();return r.json()
def write_immutable(s,name,content):
 url=f"{API}/contents/{TARGET}/{name}"
 check=s.get(url,timeout=(30,120),headers={"Cache-Control":"no-cache"})
 if check.status_code==200:
  record=check.json()
  if record.get("encoding")!="base64" or base64.b64decode(record["content"])!=content:
   raise RuntimeError("Locked PRIVATE record differs; refusing overwrite "+name)
  return "EXISTING_VERIFIED"
 if check.status_code!=404:raise RuntimeError(f"Private source lookup HTTP {check.status_code}")
 data={"branch":"main","message":"A6-IJ-03A: archive immutable 110 exact SCEPT×P4B first-slice pairs",
       "content":base64.b64encode(content).decode("ascii")}
 put=s.put(url,json=data,timeout=(30,180))
 if put.status_code not in (200,201):
  raise RuntimeError(f"Private source commit failed HTTP {put.status_code}")
 expected=put.json()["content"]["sha"]
 pinned=put.json()["commit"]["sha"]
 for attempt in range(8):
  reply=s.get(url,params={"ref":pinned},
              headers={"Cache-Control":"no-cache"},timeout=(30,120))
  if reply.status_code==404:
   time.sleep(min(2*(attempt+1),8));continue
  reply.raise_for_status()
  raw=reply.json()
  if raw.get("sha")!=expected or raw.get("encoding")!="base64" or base64.b64decode(raw["content"])!=content:
   raise RuntimeError("Pinned private remote readback differs from immutable analysis")
  return "PRIVATE_PINNED_READBACK_SHA256_VERIFIED"
 raise RuntimeError("Private remote readback delayed; retry ARCHIVE only, not analysis")

def main():
 source=DEST/"a6_ij_03a_analysis.json"
 plan_file=DEST/"locked_plan.json"
 raw=source.read_bytes();report=json.loads(raw)
 lock=plan_file.read_bytes();plan=json.loads(lock)
 if (report.get("schema")!="scept.a6-ij-03a.analysis.v1" or
     report.get("status")!="PASS_STRICTLY_PAIRED_55_FIRST_SLICE_VAL" or
     report.get("planSha256")!=plan_sha() or report.get("plan")!=PLAN or
     plan.get("planSha256")!=plan_sha() or plan.get("plan")!=PLAN):
  raise RuntimeError("Not original locked authentic 03A result")
 rows=report.get("exactMatchedPatientRows",[])
 if (len(rows)!=110 or len({(r["patient_id"],r["slice_rank"],r["model"]) for r in rows})!=110
     or len({r["patient_id"] for r in rows})!=55 or
     any(r["slice_rank"]!=1 or r["model"] not in ("S","J") for r in rows)):
  raise RuntimeError("Incomplete original first-slice model/patient paired rows")
 ck=report["sourceChecksums"]
 if len(ck.get("P4BChunks",{}))!=11 or any(x["rows"]!=100 for x in ck["P4BChunks"].values()):
  raise RuntimeError("Original P4B 1100 source slice manifests incomplete")
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle GITHUB_TOKEN must be enabled with private write permission")
 session=requests.Session()
 session.headers.update({"Authorization":"Bearer "+token,
   "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 info=safe_get(session,API)
 if info.get("private") is not True or info.get("full_name")!=REPO:
  raise RuntimeError("Refusing archive outside exactly verified PRIVATE research repo")
 results={}
 results["locked_plan.json"]=write_immutable(session,"locked_plan.json",lock)
 results["analysis.json"]=write_immutable(session,"analysis.json",raw)
 receipt={"schema":"scept.a6-ij-03a.private-archive.v1",
          "status":"PASS_110_EXACT_MATCHED_MODEL_SLICE_RECORDS_PRIVATE",
          "sourcePlanOriginalSHA256":digest(lock),"sourceAnalysisOriginalSHA256":digest(raw),
          "sourceP4BChecksums":ck,"nPatients":55,"nModelPairs":110,
          "patientRawTensorsCopied":False,"sealedTest":"NOT_ACCESSED",
          "scientificScope":"EXPLORATORY_CONCORDANCE_NOT_IMPORTANCE"}
 rec=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
 results["archive_receipt.json"]=write_immutable(session,"archive_receipt.json",rec)
 print("A6-IJ-03A PRIVATE SOURCE-VERIFIED ARCHIVE: PASS")
 print("Archive results:",results)
 print("PRIVATE:",f"https://github.com/{REPO}/blob/main/{TARGET}/analysis.json")
 print("This is 03A only; 03B and 03C not executed.")

if __name__=="__main__":main()
