"""A6-IJ-04A02 private immutable EXPLORATORY post-eligibility pilot evidence.

Require true 10 genuine SCEPT ZIPs (5 frozen conditions × 2 original P4
models), original model state provenance, patient-free original-input
archives, exact report hashes. Upload ONE private Release shard, re-download
full bytes and hash; commit immutable private manifest/report/receipt.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_STORED,ZipFile,ZipInfo

from scept import load_observation
from a6_ij_03b_measure import github,REPO,API
from a6_ij_03b_archive_chunk_private import call,commit_once,remote_sha
from a6_ij_04a02_pilot import OUT,ORDER,PILOT,sha_file,digest,AMENDMENT,amendment_fingerprint,ORIGINAL_SCREEN_GIT_BLOB_SHA
from a6_ij_04a_interventions import PLAN,fingerprint,OPS

TAG="scept-a6-ij-04a02-amended-val-040-input-intervention-v1"
BASE="results/scept_a6_ij/04a02_amended_040_intervention_pilot"
ASSET="scept_04a02_amended_val_040_10_original_observations.zip"

def verify_source(path,expected):
 if path.name!=expected["name"] or path.stat().st_size!=expected["bytes"] or \
    sha_file(path)!=expected["sha256"]:
  raise RuntimeError("Original raw SCEPT evidence bytes changed: "+str(path))
 obs=load_observation(str(path))
 if obs.run_id!=expected["runId"] or \
    obs.record["provenance"]["model"]["sha256"]!=expected["modelStateSHA256"] or \
    obs.manifest["captureScope"]["inputsIncluded"] is not False:
  raise RuntimeError("Original SCEPT ZIP run identity / model / sensitive input archive guard")
 if {g["tensor"] for g in obs.recompute_naf()["groups"] if g["status"]=="measured"}!=set(PLAN["genuineMeasuredLayers"]):
  raise RuntimeError("Original measured genuine SCEPT layer archive incomplete")
 if obs.tensor("model.output").shape!=(1,2):
  raise RuntimeError("Original SCEPT model logits incomplete")
 return True

def archive_files(files,dest):
 hashes={key:{"sha256":sha_file(fp),"bytes":fp.stat().st_size} for key,fp in files}
 if len(files)!=12 or len(hashes)!=12:
  raise RuntimeError("Exactly ten patient-output-only ZIPs, original analysis and original locked plan required")
 if not dest.is_file():
  tmp=dest.with_name(dest.name+".partial")
  with ZipFile(tmp,"w",allowZip64=True) as z:
   for name,path in sorted(files):
    zi=ZipInfo(name,date_time=(1980,1,1,0,0,0))
    zi.compress_type=ZIP_STORED
    zi.external_attr=0o600<<16
    z.writestr(zi,path.read_bytes())
  os.replace(tmp,dest)
 with ZipFile(dest) as z:
  if set(z.namelist())!=set(hashes):
   raise RuntimeError("Original source private Release shard list differs")
  for name,meta in hashes.items():
   if hashlib.sha256(z.read(name)).hexdigest()!=meta["sha256"]:
    raise RuntimeError("Original private Release member SHA256 differs: "+name)
 return hashes

def main():
 report_path=OUT/"a6_ij_04a02_pilot_report.json"
 lock_path=OUT/"locked_plan.json"
 if not report_path.is_file() or not lock_path.is_file():
  raise RuntimeError("Run real frozen P4 model 04A pilot before private archive")
 report_bytes=report_path.read_bytes()
 plan_bytes=lock_path.read_bytes()
 data=json.loads(report_bytes)
 lock=json.loads(plan_bytes)
 if data.get("status")!="PASS_AMENDED_INPUT_INTERVENTION_PILOT_ENGINEERING" or \
    data.get("interventionPlan")!=PLAN or data.get("interventionPlanSha256")!=fingerprint() or \
    data.get("amendmentPlan")!=AMENDMENT or data.get("amendmentPlanSha256")!=amendment_fingerprint() or \
    data.get("patient_id")!=PILOT or data.get("nOriginalSCEPTCapturedForwards")!=10 or \
    data.get("nAdditionalShamSCEPTForwards")!=2 or \
    lock.get("plan")!=PLAN or lock.get("planSha256")!=fingerprint() or \
    lock.get("amendment")!=AMENDMENT or lock.get("amendmentPlanSha256")!=amendment_fingerprint() or \
    lock.get("screenOriginalGitBlobSHA")!=ORIGINAL_SCREEN_GIT_BLOB_SHA or \
    data.get("source",{}).get("originalImageOnlyScreenGitBlobSHA")!=ORIGINAL_SCREEN_GIT_BLOB_SHA or \
    set(data.get("models",{}))!={"S","J"}:
  raise RuntimeError("Original real pilot report/plan does not satisfy preregistered requirements")
 files=[("analysis.json",report_path),("locked_plan.json",lock_path)]
 source_shas=set()
 for model in ("S","J"):
  block=data["models"][model]
  if set(block["rawSCEPTZIPs"])!=set(ORDER) or \
     set(block["shamExactEquality"])!=set(PLAN["genuineMeasuredLayers"]+["model.output"]) or \
     not all(block["shamExactEquality"].values()):
   raise RuntimeError("Original two-model five-condition + sham source incomplete")
  for condition in ORDER:
   item=block["rawSCEPTZIPs"][condition]
   name=f"{PILOT}_{model}_{condition}_private.zip"
   if item["name"]!=name:raise RuntimeError("Raw ZIP name does not match original fixed model/condition")
   fp=OUT/"raw"/name
   verify_source(fp,item)
   if item["sha256"] in source_shas:
    raise RuntimeError("Duplicated original measured run ZIP SHA")
   source_shas.add(item["sha256"])
   files.append(("raw/"+name,fp))
 if len(files)!=12:raise RuntimeError("Not ten genuine model condition raw SCEPT records")
 s=github()
 origin=s.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 if origin.status_code==200:release=origin.json()
 elif origin.status_code==404:
  release=call(s,"POST",f"{API}/releases",json={
   "tag_name":TAG,"target_commitish":"main","prerelease":True,"draft":False,
   "name":"PRIVATE 04A02 amended first-eligible VAL 040 lesion/control frozen S/J observations",
   "body":"Private original patient-derived SCEPT numeric activation ZIPs and ROI coordinates. Engineering pilot only, not clinical/cause-of-decision proof. No original MRI included. Preserves original fixed-patient 022 BLOCKED. This is exploratory post-feasibility selection."})
 else:raise RuntimeError("Could not inspect original private research GitHub release")
 if release.get("prerelease") is not True:
  raise RuntimeError("Sensitive source release is not private prerelease")
 shard=OUT/ASSET
 members=archive_files(files,shard)
 got_sha=sha_file(shard);size=shard.stat().st_size
 items=[a for a in release["assets"] if a["name"]==ASSET]
 if len(items)>1:raise RuntimeError("Duplicated original sensitive source archive")
 if items:
  item=items[0]
 else:
  print(f"Uploading immutable PRIVATE 04A original {size/(1024*1024):.1f} MiB source",flush=True)
  with shard.open("rb") as fp:
   item=call(s,"POST",release["upload_url"].split("{",1)[0],
             params={"name":ASSET},data=fp,
             headers={"Content-Type":"application/zip",
                      "Content-Length":str(size)})
 remote_sha(s,item,got_sha,size)
 receipt={
  "schema":"scept.a6-ij-04a02.private-receipt.v1",
  "status":"PRIVATE_10_REAL_INTERVENTION_OBSERVATIONS_SOURCE_SHA256_VERIFIED",
  "patient":PILOT,"nFrozenP4Models":2,"genuineSourceZIPs":10,
  "separateShamReexecution":2,
  "originalStudyReportSHA256":digest(report_bytes),
  "originalStudyPlanSHA256":digest(plan_bytes),
  "amendmentPlanSha256":amendment_fingerprint(),
  "originalImageOnlyScreenGitBlobSHA":ORIGINAL_SCREEN_GIT_BLOB_SHA,
  "original04Av1BlockedPreserved":True,
  "interventionPlanSha256":fingerprint(),
  "privateRelease":release["html_url"],
  "originalSourceAssetName":ASSET,"originalSourceAssetSHA256":got_sha,
  "originalSourceAssetBytes":size,"sourceMemberHashes":members,
  "originalPatientMRIStoredInSCEPTZips":False,
  "sealedTest":"NOT_ACCESSED","scientificInterpretation":"AMENDED_EXPLORATORY_ENGINEERING_PILOT_ONLY_NO_CAUSAL_OR_CLINICAL_CLAIMS"}
 rec=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 commit_once(s,f"{BASE}/locked_plan.json",plan_bytes)
 commit_once(s,f"{BASE}/analysis.json",report_bytes)
 commit_once(s,f"{BASE}/archive_receipt.json",rec)
 print("A6-IJ-04A02 AMENDED 040 PRIVATE ORIGINAL PATIENT INTERVENTION ARCHIVE: PASS")
 print("10 genuine SCEPT observation ZIPs re-downloaded and verified; report/plan private immutable")
 print(f"Private research report: https://github.com/{REPO}/blob/main/{BASE}/analysis.json")
 print("Scientific evidence dependence: NOT ESTABLISHED by this one-patient technical pilot.")

if __name__=="__main__":main()
