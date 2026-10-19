"""A6-IJ-02A: save ALL raw observation evidence in PRIVATE research GitHub.

Idempotent upload+remote-re-download SHA256 of 8 SCEPT ZIPs; immutable JSON
report and receipt, regardless of scientific/engineering PASS or FAIL.
No raw MRI/model-input ZIP, no public BrainMirror uploads, no test-set data.
Requires Kaggle GITHUB_TOKEN with Contents read/write on PRIVATE research repo.
"""
import base64
import hashlib
import json
import os
import time
from pathlib import Path
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation

REPO="changxinjiresearch/research-course-brats2020-ijepa-xai"
API="https://api.github.com/repos/"+REPO
TAG="scept-a6-ij-02a-reliability-v1"
PREFIX="results/scept_a6_ij/02a_reliability"
ROOT=Path("/kaggle/working/a6_ij_02a_private")
EXPECTED={"S":"2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf",
          "J":"719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"}
LAYERS={"blocks.0.norm1","blocks.5.norm1","blocks.11.norm1"}
RUNS={"A1","A2","B1","A3"}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def digest(data):return hashlib.sha256(data).hexdigest()

def api(sess,method,url,**kwargs):
 r=sess.request(method,url,timeout=(30,240),**kwargs)
 if r.status_code not in (200,201):
  raise RuntimeError(f"GitHub {method} HTTP {r.status_code}: {url.split('?')[0]}")
 return r.json()

def verify_remote(sess,asset,expected_sha,expected_bytes):
 if asset["size"]!=expected_bytes:raise RuntimeError("Remote asset byte count mismatch")
 if asset.get("digest") and asset["digest"]!="sha256:"+expected_sha:
  raise RuntimeError("Server-reported asset digest mismatch")
 with sess.get(f"{API}/releases/assets/{asset['id']}",
     headers={"Accept":"application/octet-stream"},stream=True,
     allow_redirects=True,timeout=(30,900)) as response:
  response.raise_for_status()
  if "json" in response.headers.get("Content-Type","").lower():
   raise RuntimeError("GitHub returned asset metadata instead of raw ZIP")
  h=hashlib.sha256();total=0
  for block in response.iter_content(chunk_size=8*1024*1024):
   if block:h.update(block);total+=len(block)
 if total!=expected_bytes or h.hexdigest()!=expected_sha:
  raise RuntimeError("Re-downloaded asset fails complete SHA256")
 return True

def put_immutable(sess,name,blob):
 url=f"{API}/contents/{PREFIX}/{name}"
 old=sess.get(url,headers={"Cache-Control":"no-cache"},timeout=(30,120))
 if old.status_code==200:
  obj=old.json()
  if obj.get("encoding")!="base64" or base64.b64decode(obj["content"])!=blob:
   raise RuntimeError("Immutable private archive file already exists with DIFFERENT content")
  return obj.get("sha")
 if old.status_code!=404:raise RuntimeError(f"Cannot inspect remote report HTTP {old.status_code}")
 written=api(sess,"PUT",url,json={"branch":"main",
  "message":"Archive frozen A6-IJ-02A real SCEPT repeatability raw-evidence metadata",
  "content":base64.b64encode(blob).decode("ascii")})
 sha_expected=written["content"]["sha"]
 pinned=written["commit"]["sha"]
 for retry in range(8):
  r=sess.get(url,params={"ref":pinned},headers={"Cache-Control":"no-cache"},timeout=(30,120))
  if r.status_code==404:
   time.sleep(min(2*(retry+1),8))
   continue
  r.raise_for_status()
  obj=r.json()
  if obj.get("encoding")!="base64" or obj.get("sha")!=sha_expected or base64.b64decode(obj["content"])!=blob:
   raise RuntimeError("Pinned GitHub write/readback bytes differ")
  return sha_expected
 raise RuntimeError("Write returned commit but immediate GitHub readback unavailable. Do NOT recompute; retry upload.")

def main():
 source_path=ROOT/"a6_ij_02a_report.json"
 raw=source_path.read_bytes()
 report=json.loads(raw)
 if (report.get("schema")!="scept.a6-ij-02a.report.v1" or
     report.get("cohort")!="P1 VAL ONLY" or report.get("testSet")!="NOT_ACCESSED" or
     report.get("status") not in ("PASS_ENGINEERING_RELIABILITY_AND_INPUT_SENSITIVITY","FAIL_OR_NOT_DETECTED")):
  raise RuntimeError("Unrecognized or non-validation A6-IJ-02A provenance")
 rows=report.get("models")
 if not isinstance(rows,list) or len(rows)!=2 or {x.get("model") for x in rows}!={"S","J"}:
  raise RuntimeError("Expected exactly one S/J pair")
 assets=[]
 for model in rows:
  key=model["model"]
  if model["modelCheckpointSha256"]!=EXPECTED[key]:
   raise RuntimeError("Frozen P4 checkpoint hash mismatch")
  if {r["step"] for r in model["records"]}!=RUNS or len(model["records"])!=4:
   raise RuntimeError("Original four-repeat protocol incomplete")
  for step in model["records"]:
   if step.get("inputAlias") != step["step"][0]:raise RuntimeError("A/B alias mismatch")
   file=ROOT/step["archiveName"]
   if (not file.is_file() or file.name!=f"{key}_{step['step']}_private.zip" or
       file.stat().st_size!=step["archiveBytes"] or sha(file)!=step["archiveSha256"]):
    raise RuntimeError("Missing/altered local original raw observation ZIP")
   obj=load_observation(str(file))
   if obj.run_id!=step["runId"] or obj.manifest["captureScope"]["inputsIncluded"] is not False:
    raise RuntimeError("Incorrect run identity or input privacy declaration")
   naf=obj.recompute_naf()
   if naf["science"]["model_forward_reexecuted"] is not False:
    raise RuntimeError("Archived arithmetic incorrectly claims model inference")
   measured={g["tensor"] for g in naf["groups"] if g["status"]=="measured"}
   if measured!=LAYERS:raise RuntimeError("Not all locked Transformer layer probes captured")
   assets.append({"model":key,"step":step["step"],"path":file,
                  "sha256":sha(file),"bytes":file.stat().st_size})
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle GITHUB_TOKEN secret missing")
 sess=requests.Session()
 sess.headers.update({"Authorization":"Bearer "+token,
   "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 info=api(sess,"GET",API)
 if info.get("full_name")!=REPO or info.get("private") is not True:
  raise RuntimeError("Target must be the exactly named PRIVATE research repository")
 prior=sess.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 if prior.status_code==200:rel=prior.json()
 elif prior.status_code==404:
  rel=api(sess,"POST",f"{API}/releases",json={
   "tag_name":TAG,"target_commitish":"main","prerelease":True,"draft":False,
   "name":"PRIVATE SCEPT A6-IJ-02A reliability and sensitivity",
   "body":"Private original derived-patient activation data; includes failed as well as passing engineering tests. Not patient-level science, not causal attribution. NEVER publish."})
 else:raise RuntimeError(f"Release lookup failed HTTP {prior.status_code}")
 if rel.get("prerelease") is not True:
  raise RuntimeError("Expected dedicated prerelease for sensitive A6-IJ-02A data")
 receipts=[]
 for item in assets:
  filename=f"scept_02a_{item['model'].lower()}_{item['step'].lower()}_{item['sha256'][:12]}.zip"
  existing=api(sess,"GET",f"{API}/releases/{rel['id']}/assets?per_page=100")
  matches=[a for a in existing if a["name"]==filename]
  if len(matches)>1:raise RuntimeError("Duplicate archive artifact name")
  if matches:remote=matches[0]
  else:
   with item["path"].open("rb") as binary:
    remote=api(sess,"POST",rel["upload_url"].split("{",1)[0],
      params={"name":filename},data=binary,
      headers={"Content-Type":"application/zip","Content-Length":str(item["bytes"])})
  verify_remote(sess,remote,item["sha256"],item["bytes"])
  receipts.append({"model":item["model"],"step":item["step"],"assetName":filename,
                   "sha256":item["sha256"],"bytes":item["bytes"],
                   "remoteRedownloadSha256Verified":True})
  print("PRIVATE REMOTE SHA256 PASS:",filename,flush=True)
 summary={"schema":"scept.a6-ij-02a.private-archive.v1",
          "status":"PRIVATE_ARCHIVED_REMOTE_VERIFIED",
          "sourceReportSha256":digest(raw),"sourceStatus":report["status"],
          "testSet":"NOT_ACCESSED","clinicalScientificEvidence":"NOT_ESTABLISHED",
          "releaseUrl":rel["html_url"],"assets":receipts}
 put_immutable(sess,"report.json",raw)
 receipt_raw=(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 put_immutable(sess,"archive_receipt.json",receipt_raw)
 print("\nA6-IJ-02A PRIVATE ARCHIVE: PASS")
 print("Report:",PREFIX+"/report.json")
 print("Private Release:",rel["html_url"])

if __name__=="__main__":main()
