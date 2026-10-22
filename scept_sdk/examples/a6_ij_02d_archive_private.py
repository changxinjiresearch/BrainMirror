"""A6-IJ-02D: archive exactly measured MRI-derived/ResNet engine evidence privately.

Fail-closed: private research repo verification; original output-only ZIP full
SHA256 + model identity; immutable report and plan; remote reread SHA256 of
all 6 original assets, including complete CNN ONNX and Torch weights.
No public file, patient input, test data, or scientific superiority claim.
"""
from __future__ import annotations
import base64
import hashlib
import json
import time
from pathlib import Path
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation
from a6_ij_02b_measure import API,REPO,sha_file
from a6_ij_02d_measure import OUT,PLAN,COHORT,plan_sha

TAG="scept-a6-ij-02d-cross-arch-runtime-v1"
PREFIX="results/scept_a6_ij/02d_cross_arch_runtime"

def h(raw):return hashlib.sha256(raw).hexdigest()
def api(client,method,url,**kwargs):
 result=client.request(method,url,timeout=(30,240),**kwargs)
 if result.status_code not in (200,201):
  raise RuntimeError(f"Private GitHub {method} HTTP {result.status_code} at {url.split('?')[0]}")
 return result.json()

def validated_original_asset(path,metadata,kind,modelsha):
 if path.name!=metadata["name"] or not path.is_file():
  raise RuntimeError("Original private asset absent or path changed")
 if path.stat().st_size!=metadata["bytes"] or sha_file(path)!=metadata["sha256"]:
  raise RuntimeError("Original private asset full bytes SHA256 invalid")
 if kind in ("torch","onnx"):
  obs=load_observation(str(path))
  if obs.manifest["captureScope"]["inputsIncluded"] is not False:
   raise RuntimeError("Refuse original-model-input bearing SCEPT archive")
  if obs.record["provenance"]["model"]["sha256"]!=modelsha:
   raise RuntimeError("Original SCEPT archive model SHA256 provenance mismatch")
  measured={g["tensor"] for g in obs.recompute_naf()["groups"] if g["status"]=="measured"}
  if measured!=({"net.layer1","net.layer3"} if kind=="torch" else {"layer1","layer3"}):
   raise RuntimeError("Original CNN source NAF groups not authentic")
 return path

def verify_remote(client,asset,original):
 if int(asset["size"])!=original.stat().st_size:
  raise RuntimeError("GitHub Release remote byte count mismatch")
 reported=asset.get("digest")
 if reported is not None and reported!="sha256:"+sha_file(original):
  raise RuntimeError("GitHub remote API digest inconsistent with original")
 with client.get(f"{API}/releases/assets/{asset['id']}",
                 headers={"Accept":"application/octet-stream"},stream=True,
                 allow_redirects=True,timeout=(30,900)) as response:
  response.raise_for_status()
  if "json" in response.headers.get("Content-Type","").lower():
   raise RuntimeError("Unexpected GitHub metadata JSON rather than actual file")
  d=hashlib.sha256();n=0
  for block in response.iter_content(chunk_size=4*1024*1024):
   if block:d.update(block);n+=len(block)
 if n!=original.stat().st_size or d.hexdigest()!=sha_file(original):
  raise RuntimeError("Independent private Release complete re-download SHA256 mismatch")

def save_once(client,name,raw):
 endpoint=f"{API}/contents/{PREFIX}/{name}"
 previous=client.get(endpoint,timeout=(30,120),headers={"Cache-Control":"no-cache"})
 if previous.status_code==200:
  record=previous.json()
  if record.get("encoding")!="base64" or base64.b64decode(record["content"])!=raw:
   raise RuntimeError("Existing immutable original 02D private record differs: "+name)
  return
 if previous.status_code!=404:
  raise RuntimeError("Cannot inspect intended private GitHub record")
 made=api(client,"PUT",endpoint,json={
  "branch":"main","message":"Archive immutable 02D real model replay/cross-engine qualification evidence",
  "content":base64.b64encode(raw).decode("ascii")})
 sha=made["content"]["sha"];pinned=made["commit"]["sha"]
 for tries in range(8):
  r=client.get(endpoint,params={"ref":pinned},
               headers={"Cache-Control":"no-cache"},timeout=(30,120))
  if r.status_code==404:
   time.sleep(min(2+2*tries,8))
   continue
  r.raise_for_status()
  doc=r.json()
  if doc.get("sha")!=sha or doc.get("encoding")!="base64" or base64.b64decode(doc["content"])!=raw:
   raise RuntimeError("Pinned original GitHub research artifact readback mismatched")
  return
 raise RuntimeError("Original GitHub write may be delayed; retry archiver only, not model")

def main():
 report_path=OUT/"a6_ij_02d_report.json"
 plan_path=OUT/"locked_plan.json"
 if not report_path.is_file() or not plan_path.is_file():
  raise RuntimeError("Run REAL 02D Kaggle measurement before archival")
 raw=report_path.read_bytes()
 report=json.loads(raw)
 lock=plan_path.read_bytes()
 metadata=json.loads(lock)
 if report.get("status")!="PASS_ENGINEERING_CROSS_ARCHITECTURE_AND_CROSS_ENGINE" or \
   report.get("schema")!="scept.a6-ij-02d.real-mri.v1" or \
   report.get("planSha256")!=plan_sha() or report.get("plan")!=PLAN or \
   metadata.get("plan")!=PLAN or metadata.get("planSha256")!=plan_sha():
  raise RuntimeError("Not genuine complete locked 02D real engineering result")
 if {k for k in report["frozenViTReplay"]}!={"S","J"}:
  raise RuntimeError("Frozen P4 S and J source replay pair missing")
 for name in ("S","J"):
  for patient in COHORT:
   source=report["frozenViTReplay"][name][patient]
   if source.get("status")!="PASS" or len(source["directHookErrors"])!=7:
    raise RuntimeError("Incomplete independent old ViT source run check")
   if any(t["maxAbs"]>PLAN["vitDirectHookMaxAbs"]
          for t in source["directHookErrors"].values()):
    raise RuntimeError("Original frozen ViT reexecution tolerance failed")
 cnn=report["cnn"]
 if cnn["weightTraining"]!="NONE - deterministic random initialization, NOT medically trained":
  raise RuntimeError("Incorrectly declared ResNet architecture weight training")
 saved=[]
 for key in ("torchModelWeights","onnxFile"):
  meta=cnn[key]
  p=OUT/meta["name"]
  validated_original_asset(p,meta,"raw",None)
  saved.append((p,meta))
 state=cnn["stateSha256"]
 onnx_sha=cnn["onnxFile"]["sha256"]
 for patient in COHORT:
  torchrow=cnn["torchObservations"][patient]
  original=torchrow["torchZip"]
  p=OUT/original["name"]
  validated_original_asset(p,original,"torch",state)
  original_run=load_observation(str(p))
  if original_run.run_id!=torchrow["torchRunId"]:
   raise RuntimeError("Original ResNet Torch run ID mismatch")
  saved.append((p,original))
  onnxrow=cnn["onnxObservations"][patient]
  original=onnxrow["archive"]
  p=OUT/original["name"]
  validated_original_asset(p,original,"onnx",onnx_sha)
  if load_observation(str(p)).run_id!=onnxrow["runId"]:
   raise RuntimeError("Original ResNet ONNX run ID mismatch")
  if any(v["maxAbs"]>PLAN["cnnTorchVsONNXMaxAbs"] or
         v["relativeL2"]>PLAN["cnnTorchVsONNXRelativeL2"]
         for v in onnxrow["engineNumericErrors"].values()):
   raise RuntimeError("ONNX original numeric parity tolerance exceeded")
  saved.append((p,original))
 if len(saved)!=6 or len({p.name for p,_ in saved})!=6:
  raise RuntimeError("Not exactly six unique original 02D evidence files")
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle existing PRIVATE GitHub write Secret missing")
 s=requests.Session()
 s.headers.update({"Authorization":"Bearer "+token,
                   "Accept":"application/vnd.github+json",
                   "X-GitHub-Api-Version":"2022-11-28"})
 info=api(s,"GET",API)
 if info.get("full_name")!=REPO or info.get("private") is not True:
  raise RuntimeError("Refuse remote archival outside verified private coursework repository")
 response=s.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 if response.status_code==200:release=response.json()
 elif response.status_code==404:
  release=api(s,"POST",f"{API}/releases",json={
   "tag_name":TAG,"target_commitish":"main","prerelease":True,"draft":False,
   "name":"PRIVATE A6-IJ-02D cross-arch engine source evidence",
   "body":"Private genuine patient-derived ResNet18 Torch/ONNX observations and frozen random CNN graph/weights. Contains no original patient input, no P5 test, no clinical inference."})
 else:raise RuntimeError("Cannot inspect private 02D Release")
 if release.get("prerelease") is not True:
  raise RuntimeError("Expected dedicated private prerelease")
 receipts=[]
 for original,meta in saved:
  items=api(s,"GET",f"{API}/releases/{release['id']}/assets?per_page=100")
  existing=[i for i in items if i["name"]==original.name]
  if len(existing)>1:raise RuntimeError("Duplicate private release source filename")
  if existing:asset=existing[0]
  else:
   with original.open("rb") as f:
    asset=api(s,"POST",release["upload_url"].split("{",1)[0],
      params={"name":original.name},data=f,
      headers={"Content-Type":"application/octet-stream",
               "Content-Length":str(original.stat().st_size)})
  verify_remote(s,asset,original)
  receipts.append({"name":original.name,"sha256":sha_file(original),
                   "bytes":original.stat().st_size,
                   "remoteDownloadSha256Verified":True})
  print("02D PRIVATE ORIGINAL ASSET SHA256 PASS",original.name,flush=True)
 receipt={"schema":"scept.a6-ij-02d.private-archive.v1",
  "status":"PRIVATE_ALL_SIX_ORIGINAL_ASSETS_SHA256_VERIFIED",
  "originalReportSha256":h(raw),"lockedPlanBytesSha256":h(lock),
  "prior02C":report["source02C"],
  "nPatients":2,"nFrozenViTReplayedRuns":4,
  "nResNetTorchPatientRuns":2,"nResNetONNXPatientRuns":2,
  "nPrivateAssets":len(receipts),"assets":receipts,
  "privateRelease":release["html_url"],"sealedTest":"NOT_ACCESSED",
  "originalInputIncludedInSCEPTZip":False,
  "scientificMeaning":"ENGINEERING_ONLY"}
 receipt_bytes=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
 save_once(s,"report.json",raw)
 save_once(s,"locked_plan.json",lock)
 save_once(s,"archive_receipt.json",receipt_bytes)
 print("\nA6-IJ-02D PRIVATE ORIGINAL ASSET ARCHIVE: PASS")
 print("Private Release:",release["html_url"])
 print("Private report:",PREFIX+"/report.json")

if __name__=="__main__":main()
