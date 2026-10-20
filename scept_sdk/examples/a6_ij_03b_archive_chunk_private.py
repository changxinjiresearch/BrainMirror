"""A6-IJ-03B: archive one 5-patient chunk of 90 NEW original SCEPT ZIPs.

SHA256 re-download each private Release asset, original 5 patient case
records, immutable full numeric paired table and receipt. Already published
complete chunks are NEVER silently recomputed; archive idempotency supports
Kaggle restarts. Source P4B/02B/03A remain previously archived privately.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_STORED
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation
from a6_ij_03b_measure import (
 OUT,PLAN,psha,sha_file,github,validate_complete,PREFIX,RELEASE_TAG,REPO,API,
 private_contents,finite
)

def digest(raw):return hashlib.sha256(raw).hexdigest()
def call(s,method,url,**kwargs):
 r=s.request(method,url,timeout=(30,240),**kwargs)
 if r.status_code not in (200,201):
  raise RuntimeError(f"Private GitHub {method} HTTP {r.status_code}: {url.split('?')[0]}")
 return r.json()
def commit_once(s,path,raw):
 endpoint=f"{API}/contents/{path}"
 existing=s.get(endpoint,timeout=(30,120),headers={"Cache-Control":"no-cache"})
 if existing.status_code==200:
  item=existing.json()
  if item.get("encoding")!="base64" or base64.b64decode(item["content"])!=raw:
   raise RuntimeError("Different immutable research record already published "+path)
  return "EXISTING_SHA_MATCH"
 if existing.status_code!=404:
  raise RuntimeError("Could not inspect prior immutable private publication")
 obj=call(s,"PUT",endpoint,json={"branch":"main",
  "message":"A6-IJ-03B: immutable original paired patient slice records & SHA256 archive",
  "content":base64.b64encode(raw).decode("ascii")})
 pin=obj["commit"]["sha"]
 expected=obj["content"]["sha"]
 for n in range(8):
  chk=s.get(endpoint,params={"ref":pin},timeout=(30,120),
            headers={"Cache-Control":"no-cache"})
  if chk.status_code==404:
   time.sleep(min(2*(n+1),8))
   continue
  chk.raise_for_status()
  doc=chk.json()
  if doc.get("encoding")!="base64" or doc.get("sha")!=expected or \
     base64.b64decode(doc["content"])!=raw:
   raise RuntimeError("Private pinned readback bytes differ from original")
  return "REMOTE_PINNED_READBACK_MATCH"
 raise RuntimeError("Commit may have propagated late; retry archive only")

def remote_sha(s,item,expected_sha,expected_bytes):
 if item["size"]!=expected_bytes:
  raise RuntimeError("Private Release size differs from expected original chunk")
 if item.get("digest") and item["digest"]!="sha256:"+expected_sha:
  raise RuntimeError("Private Release provider-reported SHA differs")
 with s.get(f"{API}/releases/assets/{item['id']}",
            headers={"Accept":"application/octet-stream"},
            stream=True,allow_redirects=True,timeout=(30,900)) as r:
  r.raise_for_status()
  if "json" in r.headers.get("Content-Type","").lower():
   raise RuntimeError("Expected private original ZIP bytes, obtained metadata")
  d=hashlib.sha256();count=0
  for block in r.iter_content(chunk_size=4*1024*1024):
   if block:d.update(block);count+=len(block)
 if count!=expected_bytes or d.hexdigest()!=expected_sha:
  raise RuntimeError("FULL REMOTE original ZIP SHA256 re-download FAILED")

def remote_status(s,num):
 path=f"{PREFIX}/chunks/chunk_{num:02d}/archive_receipt.json"
 r=s.get(f"{API}/contents/{path}",timeout=(30,120))
 if r.status_code==404:return None
 r.raise_for_status()
 obj=r.json()
 if obj.get("encoding")!="base64":raise RuntimeError("Existing 03B archived chunk receipt corrupt")
 prior=json.loads(base64.b64decode(obj["content"]))
 if (prior.get("schema")!="scept.a6-ij-03b.chunk-receipt.v1" or
     prior.get("status")!="PRIVATE_90_NEW_ZIPS_FULL_REMOTE_SHA256_VERIFIED" or
     prior.get("chunk")!=num or prior.get("planSHA256")!=psha() or
     prior.get("newForwardCount")!=90 or prior.get("original03AReusedCount")!=10):
  raise RuntimeError("Previously archived chunk has incompatible or incomplete protocol")
 return prior

def seal(files,path):
 # Fixed timestamps/order ensure identical archive bytes when re-archived
 # from unchanged original local patient/observation files.
 contents={name:{"sha256":sha_file(fp),"bytes":fp.stat().st_size} for name,fp in files}
 if len(files)!=95 or len(contents)!=95:
  raise RuntimeError("Five case manifests and 90 original ZIPs required")
 if not path.exists():
  temp=path.with_suffix(".zip.partial")
  with ZipFile(temp,"w",allowZip64=True) as z:
   for name,fp in sorted(files,key=lambda t:t[0]):
    info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
    info.compress_type=ZIP_STORED
    info.external_attr=0o600<<16
    z.writestr(info,fp.read_bytes())
  os.replace(temp,path)
 with ZipFile(path) as z:
  if sorted(z.namelist())!=sorted(contents):
   raise RuntimeError("Original ZIP shard incomplete/misindexed")
  for name,meta in contents.items():
   if digest(z.read(name))!=meta["sha256"]:
    raise RuntimeError("Original shard file content SHA mismatches: "+name)
 return contents

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--chunk",type=int,required=True)
 args=parser.parse_args()
 num=args.chunk
 if not 1<=num<=11:raise RuntimeError("Private batch must be 1..11")
 s=github()
 prior=remote_status(s,num)
 release_pre=s.get(f"{API}/releases/tags/{RELEASE_TAG}",timeout=(30,120))
 if prior is not None:
  if release_pre.status_code!=200:
   raise RuntimeError("Existing sealed source chunk receipt has missing Release")
  release=release_pre.json()
  assets=[a for a in release["assets"] if a["name"]==prior["assetName"]]
  if len(assets)!=1 or assets[0]["size"]!=prior["assetBytes"] or \
     (assets[0].get("digest") and assets[0]["digest"]!="sha256:"+prior["assetSha256"]):
   raise RuntimeError("Previously archived private shard missing/corrupt")
  print(f"A6-IJ-03B CHUNK {num:02d}: ALREADY PRIVATE VERIFIED, "
        "no source model reruns",flush=True)
  return
 plan_file=OUT/"locked_plan.json"
 plan=json.loads(plan_file.read_text())
 if plan["plan"]!=PLAN or plan["planSHA256"]!=psha() or len(plan["cohortPatients"])!=55:
  raise RuntimeError("Original study plan not locked for 55-patient cohort")
 ids=plan["cohortPatients"][(num-1)*5:num*5]
 if len(ids)!=5:raise RuntimeError("Original 5-patient chunk incomplete")
 first_case=OUT/"cases"/f"{ids[0]}.json"
 if not first_case.is_file():raise RuntimeError("Original patient observation missing; run 03B measure --chunk")
 source=json.loads(first_case.read_text())["originalSources"]
 allrows=[];originals=[];checked_pairs=0
 for pid in ids:
  case_fp=OUT/"cases"/f"{pid}.json"
  if not case_fp.is_file():
   raise RuntimeError("Patient case not yet measured "+pid)
  data=json.loads(case_fp.read_text())
  if data["originalSources"]!=source:
   raise RuntimeError("A single P4B/03A source chunk must be identical across five patients")
  if validate_complete(case_fp,pid,data["grade"],source) is None:
   raise RuntimeError("Full original patient case not verifiable")
  allrows.append(data)
  originals.append(("cases/"+case_fp.name,case_fp))
  if len(data["slicePairs"])!=10 or [r["slice_rank"] for r in data["slicePairs"]]!=list(range(1,11)):
   raise RuntimeError("Original case rank coverage incomplete")
  for pair in data["slicePairs"]:
   if pair["patient_id"]!=pid or pair["grade"]!=data["grade"] or set(pair["models"])!={"S","J"}:
    raise RuntimeError("Original paired patient and model identity mismatched")
   for model in ("S","J"):
    r=pair["models"][model]
    if pair["slice_rank"]==1:
     if r["SCEPT"]["runSource"]!="REUSED_02B_WITH_03A_SHA_MATCHED_PRIVATE_RAW_ZIP":
      raise RuntimeError("Original first slice not reused correctly")
    else:
     if r["SCEPT"]["runSource"]!="NEW_03B_GENUINE_SCEPT_FORWARD":
      raise RuntimeError("Original 03B new model source not actual")
     meta=r["SCEPT"]["originalZip"]
     fp=OUT/"raw"/meta["name"]
     if fp.name!=f"{pid}_rank{pair['slice_rank']:02d}_{model}_private.zip":
      raise RuntimeError("Original patient-model-rank ZIP name mismatch")
     if sha_file(fp)!=meta["sha256"] or fp.stat().st_size!=meta["bytes"]:
      raise RuntimeError("Original measured tensor source digest mismatch")
     obs=load_observation(str(fp))
     if obs.run_id!=r["SCEPT"]["runId"] or \
        obs.record["provenance"]["model"]["sha256"]!=r["SCEPT"]["modelStateSHA256"] or \
        obs.manifest["captureScope"]["inputsIncluded"] is not False:
      raise RuntimeError("Original SCEPT observation model/run/privacy mismatch")
     if {g["tensor"] for g in obs.recompute_naf()["groups"]
         if g["status"]=="measured"}!=set(PLAN["layers"]):
      raise RuntimeError("Original three-layer measured NAF missing")
     originals.append(("raw/"+fp.name,fp))
    checked_pairs+=1
 if checked_pairs!=100 or len(originals)!=95:
  raise RuntimeError("Expected 100 model-slice pairs with 90 new ZIPs + 5 case JSON files")
 chunk_dir=OUT/"shards";chunk_dir.mkdir(parents=True,exist_ok=True)
 asset_name=f"scept_03b_val_chunk_{num:02d}_90_new_original_observations.zip"
 shard_path=chunk_dir/asset_name
 member_hashes=seal(originals,shard_path)
 digest_value=sha_file(shard_path)
 chunk_summary={
  "schema":"scept.a6-ij-03b.chunk.v1",
  "status":"PASS_5_PATIENTS_100_SOURCE_PAIRED_MODEL_SLICES",
  "planSHA256":psha(),"chunk":num,"patientIds":ids,
  "originalSources":source,"recordCount":100,"reused03aCount":10,
  "newSCEPTForwards":90,"pairedPatientRows":allrows}
 chunk_raw=(json.dumps(chunk_summary,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 if release_pre.status_code==200:release=release_pre.json()
 elif release_pre.status_code==404:
  release=call(s,"POST",f"{API}/releases",json={"tag_name":RELEASE_TAG,
   "target_commitish":"main","prerelease":True,"draft":False,
   "name":"PRIVATE A6-IJ-03B 55x10 paired genuine SCEPT activations",
   "body":"Sensitivity: raw derived-patient frozen-ViT intermediate tensors (990 new VAL runs), not original input MRI, strictly exploratory not causal or clinical."})
 else:raise RuntimeError("Cannot inspect private research Release")
 if release.get("prerelease") is not True:raise RuntimeError("Original sensitive Release must be a prerelease")
 matches=[a for a in release["assets"] if a["name"]==asset_name]
 if len(matches)>1:raise RuntimeError("Duplicate source asset name")
 if matches:asset=matches[0]
 else:
  print(f"Uploading original private chunk {num:02d}; {shard_path.stat().st_size/1024/1024:.1f}MiB",flush=True)
  with shard_path.open("rb") as input_file:
   asset=call(s,"POST",release["upload_url"].split("{",1)[0],
     params={"name":asset_name},data=input_file,
     headers={"Content-Type":"application/zip",
              "Content-Length":str(shard_path.stat().st_size)})
 remote_sha(s,asset,digest_value,shard_path.stat().st_size)
 print(f"03B CHUNK {num:02d}: FULL PRIVATE GITHUB RELEASE REDOWNLOAD SHA256 PASS",flush=True)
 receipt={"schema":"scept.a6-ij-03b.chunk-receipt.v1",
  "status":"PRIVATE_90_NEW_ZIPS_FULL_REMOTE_SHA256_VERIFIED",
  "chunk":num,"planSHA256":psha(),"patientIds":ids,"newForwardCount":90,
  "original03AReusedCount":10,"originalReportSHA256":digest(chunk_raw),
  "assetName":asset_name,"assetSha256":digest_value,
  "assetBytes":shard_path.stat().st_size,"memberSHA256":member_hashes,
  "privateRelease":release["html_url"],"originalP1ImagesIncluded":False,
  "sealedTest":"NOT_ACCESSED","scientificMeaning":"DESCRIPTIVE_ONLY"}
 report_rec=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 target=f"{PREFIX}/chunks/chunk_{num:02d}/"
 commit_once(s,target+"paired_rows.json",chunk_raw)
 commit_once(s,target+"archive_receipt.json",report_rec)
 print(f"A6-IJ-03B CHUNK {num:02d} PRIVATE ARCHIVE: PASS")
 print(f"Five VAL patients, 10 older 02B and 90 NEW measured model slices, "
       f"{len(originals)} original evidence members.")
 print("Original raw ZIPs + patient IDs remain private.")

if __name__=="__main__":main()
