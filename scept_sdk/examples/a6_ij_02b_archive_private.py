"""A6-IJ-02B: durable PRIVATE GitHub release of every raw SCEPT capture.

Archive exactly 55 P0 VAL patient case JSONs + 110 private output-only
full-tensor ZIPs into eleven ZIP_STORED shards, five patients/shard.
GitHub full SHA256 re-download of each shard. Store byte-identical
plan, analysis, index and integrity receipt in the PRIVATE research repo.
No P5 test, no public BrainMirror patient data, no silent overwrite.
"""
import base64
import hashlib
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation
from a6_ij_02b_measure import (STUDY,LAYERS,canonical_plan_hash,sha_file,REPO,API)

ROOT=Path("/kaggle/working/a6_ij_02b_private")
TAG="scept-a6-ij-02b-55val-v1"
PREFIX="results/scept_a6_ij/02b_55_patient_representations"
def digest(content):return hashlib.sha256(content).hexdigest()
def api(s,method,url,**kwargs):
 r=s.request(method,url,timeout=(30,240),**kwargs)
 if r.status_code not in (200,201):
  raise RuntimeError(f"Private GitHub {method} HTTP {r.status_code} ({url.split('?')[0]})")
 return r.json()
def verify_remote_asset(s,asset,expected_digest,expected_bytes):
 if asset["size"]!=expected_bytes:raise RuntimeError("Remote shard size mismatch")
 if asset.get("digest") and asset["digest"]!="sha256:"+expected_digest:
  raise RuntimeError("Remote shard advertised digest mismatch")
 with s.get(f"{API}/releases/assets/{asset['id']}",
            headers={"Accept":"application/octet-stream"},
            stream=True,allow_redirects=True,timeout=(30,900)) as r:
  r.raise_for_status()
  if "json" in r.headers.get("Content-Type","").lower():
   raise RuntimeError("GitHub API returned metadata instead of ZIP bytes")
  h=hashlib.sha256();n=0
  for block in r.iter_content(chunk_size=4*1024*1024):
   if block:h.update(block);n+=len(block)
 if n!=expected_bytes or h.hexdigest()!=expected_digest:
  raise RuntimeError("REMOTE ZIP shard bytes failed independent full SHA256")
def save_immutable(s,name,raw):
 url=f"{API}/contents/{PREFIX}/{name}"
 old=s.get(url,timeout=(30,120),headers={"Cache-Control":"no-cache"})
 if old.status_code==200:
  item=old.json()
  if item.get("encoding")!="base64" or base64.b64decode(item["content"])!=raw:
   raise RuntimeError("Immutable published PRIVATE result differs: "+name)
  return
 if old.status_code!=404:raise RuntimeError("Cannot inspect private target: "+name)
 created=api(s,"PUT",url,json={"branch":"main",
  "message":"Archive locked A6-IJ-02B real 55-validation-patient representation study",
  "content":base64.b64encode(raw).decode("ascii")})
 expected_blob=created["content"]["sha"]
 pin=created["commit"]["sha"]
 for retry in range(8):
  r=s.get(url,params={"ref":pin},timeout=(30,120),
          headers={"Cache-Control":"no-cache"})
  if r.status_code==404:
   time.sleep(min(2+retry*2,8))
   continue
  r.raise_for_status()
  f=r.json()
  if f.get("encoding")!="base64" or f.get("sha")!=expected_blob or base64.b64decode(f["content"])!=raw:
   raise RuntimeError("Pinned private GitHub write/readback differs")
  return
 raise RuntimeError("Private GitHub delayed readback; DO NOT rerun model, retry archiver later")

def main():
 plan_path=ROOT/"locked_plan.json"
 report_path=ROOT/"a6_ij_02b_analysis.json"
 plan_bytes=plan_path.read_bytes()
 data=json.loads(plan_bytes)
 if data["plan"]!=STUDY or data["planSha256"]!=canonical_plan_hash():
  raise RuntimeError("02B plan provenance cannot be verified")
 patients=data["cohortPatients"]
 report_bytes=report_path.read_bytes()
 report=json.loads(report_bytes)
 if (len(patients)!=55 or len(set(patients))!=55 or
     report.get("schema")!="scept.a6-ij-02b.analysis.v1" or
     report.get("status")!="PASS_55_VAL_EXPLORATORY_ANALYSIS" or
     report.get("planSha256")!=canonical_plan_hash() or
     report.get("nPatients")!=55):
  raise RuntimeError("Refusing incomplete/non-preregistered cohort archival")
 if len(report.get("patientRows",[]))!=55:
  raise RuntimeError("Patient-level analysis complete rows missing")
 if {p["patient_id"] for p in report["patientRows"]}!=set(patients):
  raise RuntimeError("Case-level analysis IDs differ from locked plan")
 shards=[]
 shard_dir=ROOT/"private_shards";shard_dir.mkdir(parents=True,exist_ok=True)
 for start in range(0,55,5):
  part=patients[start:start+5]
  files=[]
  for pid in part:
   src=ROOT/"cases"/f"{pid}.json"
   case=json.loads(src.read_text())
   if case["schema"]!="scept.a6-ij-02b.case.v1" or case["patient_id"]!=pid or case["planSha256"]!=canonical_plan_hash():
    raise RuntimeError("Patient case source changed: "+pid)
   files.append(("cases/"+src.name,src))
   for model in ("S","J"):
    row=case["modelResults"][model]
    p=ROOT/"raw"/row["zipName"]
    if p.name!=f"{pid}_{model}_local_private.zip" or sha_file(p)!=row["zipSha256"]:
     raise RuntimeError("Patient raw ZIP sha mismatch")
    obs=load_observation(str(p))
    if obs.run_id!=row["runId"] or obs.manifest["captureScope"]["inputsIncluded"] is not False:
     raise RuntimeError("Source ZIP identity/privacy mismatch")
    if {g["tensor"] for g in obs.recompute_naf()["groups"] if g["status"]=="measured"}!=set(LAYERS):
     raise RuntimeError("Missing actual SCEPT internal NAF")
    files.append(("raw/"+p.name,p))
  shard_name=f"scept_02b_val_{start//5:02d}_patients_{start+1:02d}-{start+len(part):02d}.zip"
  destination=shard_dir/shard_name
  expected={arc:{"sha256":sha_file(fp),"bytes":fp.stat().st_size} for arc,fp in files}
  if destination.exists():
   with ZipFile(destination) as z:
    if sorted(z.namelist())!=sorted(expected):
     raise RuntimeError("Existing private shard has incorrect inventory")
    for name,meta in expected.items():
     if digest(z.read(name))!=meta["sha256"]:
      raise RuntimeError("Existing shard has corrupted member "+name)
  else:
   tmp=destination.with_suffix(".zip.partial")
   with ZipFile(tmp,"w",compression=ZIP_STORED,allowZip64=True) as out:
    for archive_name,local_path in files:
     out.write(local_path,archive_name)
   with ZipFile(tmp) as z:
    if sorted(z.namelist())!=sorted(expected):raise RuntimeError("Shard packing manifest mismatch")
    for name,meta in expected.items():
     if digest(z.read(name))!=meta["sha256"]:
      raise RuntimeError("Packed shard ZIP contains altered output tensor file")
   tmp.replace(destination)
  shards.append({"filename":destination.name,"path":destination,
    "patientCount":len(part),"zipCount":len(files),"bytes":destination.stat().st_size,
    "sha256":sha_file(destination),"memberDigests":expected})
  print("SHARD PACKED / VERIFIED",destination.name,flush=True)
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle private GitHub write Secret GITHUB_TOKEN missing")
 s=requests.Session()
 s.headers.update({"Authorization":"Bearer "+token,
  "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 info=api(s,"GET",API)
 if info.get("private") is not True or info.get("full_name")!=REPO:
  raise RuntimeError("Destination not verified PRIVATE research repo")
 existing=s.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 if existing.status_code==200:release=existing.json()
 elif existing.status_code==404:
  release=api(s,"POST",f"{API}/releases",json={"tag_name":TAG,
   "target_commitish":"main","draft":False,"prerelease":True,
   "name":"PRIVATE A6-IJ-02B 55 VAL patient representations",
   "body":"Sensitive derived-patient model activation data. Research-only; no P5 test, no clinical or causal result. Never publish."})
 else:raise RuntimeError("Cannot inspect intended private GitHub release")
 if release.get("prerelease") is not True:raise RuntimeError("Unexpected release policy")
 receipts=[]
 for shard in shards:
  current=api(s,"GET",f"{API}/releases/{release['id']}/assets?per_page=100")
  found=[a for a in current if a["name"]==shard["filename"]]
  if len(found)>1:raise RuntimeError("Duplicate asset name")
  if found:asset=found[0]
  else:
   with shard["path"].open("rb") as binary:
    asset=api(s,"POST",release["upload_url"].split("{",1)[0],
      params={"name":shard["filename"]},data=binary,
      headers={"Content-Type":"application/zip","Content-Length":str(shard["bytes"])})
  verify_remote_asset(s,asset,shard["sha256"],shard["bytes"])
  receipts.append({k:v for k,v in shard.items() if k!="path"})
  print("PRIVATE RELEASE SHARD FULL SHA256 PASS",shard["filename"],flush=True)
 receipt={"schema":"scept.a6-ij-02b.private-archive.v1",
  "status":"ARCHIVED_55_PATIENTS_110_ZIPS_VERIFIED",
  "sourcePlanSha256":digest(plan_bytes),"sourceAnalysisSha256":digest(report_bytes),
  "lockedProtocolSha256":canonical_plan_hash(),
  "nPatients":55,"nRealModelRuns":110,
  "nShards":11,"release":release["html_url"],
  "rawInputsInArchives":False,"testSet":"NOT_ACCESSED",
  "science":"DESCRIPTIVE_ONLY_NO_CAUSALITY_OR_CLINICAL_EVIDENCE",
  "shards":receipts}
 receipt_raw=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 save_immutable(s,"locked_plan.json",plan_bytes)
 save_immutable(s,"analysis.json",report_bytes)
 save_immutable(s,"archive_receipt.json",receipt_raw)
 print("\nA6-IJ-02B PRIVATE 55-PATIENT ARCHIVE: PASS")
 print("PRIVATE RELEASE:",release["html_url"])
 print("PRIVATE ANALYSIS:",f"{PREFIX}/analysis.json")

if __name__=="__main__":main()
