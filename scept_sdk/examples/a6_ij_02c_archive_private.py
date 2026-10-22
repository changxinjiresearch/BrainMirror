"""A6-IJ-02C: PRIVATE archival of 110 raw real SCEPT pre/post captures + 2 OOD blanks.

Exactly 55 patient records, eleven five-patient raw ZIP shards, and one
blank-control ZIP shard. Immutable plan/analysis/receipt committed to
private research GitHub; each remote shard independently redownloaded SHA256.
No raw patient/model-input data should be published to public BrainMirror.
"""
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED
import requests
from kaggle_secrets import UserSecretsClient
from scept import load_observation
from a6_ij_02b_measure import sha_file,REPO,API
from a6_ij_02c_measure import PLAN,PROBES,plan_sha

ROOT=Path("/kaggle/working/a6_ij_02c_private")
TAG="scept-a6-ij-02c-ln-position-controls-v1"
PREFIX="results/scept_a6_ij/02c_ln_position_controls"

def shabytes(data):return hashlib.sha256(data).hexdigest()

def checked(s,method,url,**kwargs):
 r=s.request(method,url,timeout=(30,240),**kwargs)
 if r.status_code not in (200,201):
  raise RuntimeError(f"Private GitHub {method} HTTP {r.status_code}: {url.split('?')[0]}")
 return r.json()

def source_zip_ok(path,digest,expected_id,model_state):
 if not path.is_file() or sha_file(path)!=digest:
  raise RuntimeError("Source ZIP missing/altered: "+str(path))
 obj=load_observation(str(path))
 if obj.run_id!=expected_id or obj.record["provenance"]["model"]["sha256"]!=model_state:
  raise RuntimeError("Source run/model identity mismatch")
 if obj.manifest["captureScope"]["inputsIncluded"] is not False:
  raise RuntimeError("Refusing original sensitive input-containing ZIP")
 actual={g["tensor"] for g in obj.recompute_naf()["groups"] if g["status"]=="measured"}
 if actual!=set(PROBES):raise RuntimeError("All six true pre/post probes required")
 return obj

def shard(destination,filelist):
 if not filelist:raise RuntimeError("Empty shard")
 wanted={target:{"sha256":sha_file(fp),"bytes":fp.stat().st_size}
         for target,fp in filelist}
 if not destination.exists():
  tmp=destination.with_suffix(".zip.partial")
  with ZipFile(tmp,"w",compression=ZIP_STORED,allowZip64=True) as out:
   for target,fp in filelist:out.write(fp,target)
  os.replace(tmp,destination)
 with ZipFile(destination) as archive:
  if set(archive.namelist())!=set(wanted):
   raise RuntimeError("Shard inventory mismatched")
  for target,meta in wanted.items():
   if shabytes(archive.read(target))!=meta["sha256"]:
    raise RuntimeError("Shard contains altered source file")
 return {"filename":destination.name,"path":destination,"sha256":sha_file(destination),
         "bytes":destination.stat().st_size,"memberHashes":wanted}

def asset_readback(s,asset,sha,expected_size):
 if asset["size"]!=expected_size:raise RuntimeError("GitHub uploaded size mismatch")
 if asset.get("digest") and asset["digest"]!="sha256:"+sha:
  raise RuntimeError("GitHub remote advertised SHA mismatch")
 with s.get(f"{API}/releases/assets/{asset['id']}",headers={"Accept":"application/octet-stream"},
   stream=True,allow_redirects=True,timeout=(30,900)) as reply:
  reply.raise_for_status()
  if "json" in reply.headers.get("Content-Type","").lower():
   raise RuntimeError("Expected private ZIP bytes, got JSON metadata")
  h=hashlib.sha256();count=0
  for block in reply.iter_content(chunk_size=4*1024*1024):
   if block:h.update(block);count+=len(block)
 if count!=expected_size or h.hexdigest()!=sha:
  raise RuntimeError("Re-downloaded private shard full SHA256 mismatch")

def private_immutable(s,name,content):
 endpoint=f"{API}/contents/{PREFIX}/{name}"
 previous=s.get(endpoint,headers={"Cache-Control":"no-cache"},timeout=(30,120))
 if previous.status_code==200:
  data=previous.json()
  if data.get("encoding")!="base64" or base64.b64decode(data["content"])!=content:
   raise RuntimeError("Existing private report differs: REFUSE overwrite "+name)
  return
 if previous.status_code!=404:raise RuntimeError("Cannot inspect private record")
 committed=checked(s,"PUT",endpoint,json={"branch":"main",
   "message":"Archive immutable A6-IJ-02C pre/post LN and positional confound controls",
   "content":base64.b64encode(content).decode("ascii")})
 pin=committed["commit"]["sha"]
 expect=committed["content"]["sha"]
 for i in range(8):
  r=s.get(endpoint,params={"ref":pin},headers={"Cache-Control":"no-cache"},timeout=(30,120))
  if r.status_code==404:
   time.sleep(min(2*(i+1),8))
   continue
  r.raise_for_status()
  payload=r.json()
  if payload.get("sha")!=expect or payload.get("encoding")!="base64" or base64.b64decode(payload["content"])!=content:
   raise RuntimeError("Private pinned-commit GitHub readback changed")
  return
 raise RuntimeError("GitHub eventual consistency delay; do not recompute models, rerun ONLY archiver")

def main():
 plan_raw=(ROOT/"locked_plan.json").read_bytes()
 lock=json.loads(plan_raw)
 if lock.get("plan")!=PLAN or lock.get("planSha256")!=plan_sha():
  raise RuntimeError("02C frozen protocol not accepted")
 ids=lock["cohortPatientIds"]
 if len(ids)!=55 or len(set(ids))!=55:
  raise RuntimeError("Locked source must have 55 unique validation patients")
 report_raw=(ROOT/"a6_ij_02c_analysis.json").read_bytes()
 report=json.loads(report_raw)
 if (report.get("schema")!="scept.a6-ij-02c.analysis.v1" or
     report.get("status")!="PASS_LOCKED_55_PATIENT_CONTROL_ENGINEERING" or
     report.get("nPatients")!=55 or report.get("realPatientCaptures")!=110 or
     report.get("blankOODCaptures")!=2 or report.get("planSha256")!=plan_sha() or
     {p["patient_id"] for p in report["patientRows"]}!=set(ids)):
  raise RuntimeError("Refuse partial or scientifically overstated 02C source report")
 blocks=[]
 dest=ROOT/"private_shards"
 dest.mkdir(exist_ok=True,parents=True)
 count=0
 for start in range(0,55,5):
  listfiles=[]
  for patient in ids[start:start+5]:
   p=ROOT/"cases"/f"{patient}.json"
   case=json.loads(p.read_text())
   if case.get("patient_id")!=patient or case.get("planSha256")!=plan_sha():
    raise RuntimeError("Patient source manifest corrupted")
   listfiles.append(("cases/"+p.name,p))
   for model in ("S","J"):
    row=case["models"][model]
    zp=ROOT/"raw"/row["zipName"]
    if zp.name!=f"{patient}_{model}_private.zip" or zp.stat().st_size!=row["zipBytes"]:
     raise RuntimeError("Invalid original patient/model archive filename/size")
    source_zip_ok(zp,row["zipSha256"],row["runId"],row["modelStateSha256"])
    listfiles.append(("raw/"+zp.name,zp))
    count+=1
  name=f"scept_02c_val_{start//5:02d}_patients_{start+1:02d}-{start+5:02d}.zip"
  output=shard(dest/name,listfiles)
  blocks.append(output)
  print("02C SHARD VERIFIED:",name,flush=True)
 if count!=110:raise RuntimeError("Raw real patient archive count not 110")
 blanks=[]
 for model in ("S","J"):
  meta=ROOT/f"blank_{model}.json"
  row=json.loads(meta.read_text())
  zp=ROOT/"raw"/row["zipName"]
  if zp.name!=f"BLANK_{model}_private.zip" or zp.stat().st_size!=row["zipBytes"]:
   raise RuntimeError("Invalid blank-input control archive")
  source_zip_ok(zp,row["zipSha256"],row["runId"],row["modelStateSha256"])
  blanks.extend([("blank/"+meta.name,meta),("raw/"+zp.name,zp)])
 blocks.append(shard(dest/"scept_02c_blank_ood_controls.zip",blanks))
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle PRIVATE GitHub write Secret GITHUB_TOKEN missing")
 s=requests.Session()
 s.headers.update({"Authorization":"Bearer "+token,
    "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 repo=checked(s,"GET",API)
 if repo.get("private") is not True or repo.get("full_name")!=REPO:
  raise RuntimeError("Destination research repo is not the specified PRIVATE repo")
 response=s.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 if response.status_code==200:release=response.json()
 elif response.status_code==404:
  release=checked(s,"POST",f"{API}/releases",json={
   "tag_name":TAG,"target_commitish":"main",
   "name":"PRIVATE SCEPT A6-IJ-02C 55-person LayerNorm controls",
   "prerelease":True,"draft":False,
   "body":"Patient-derived internal model activations (S/J, 55 validation patients) and synthetic blank controls; private and descriptive ONLY."})
 else:raise RuntimeError("Cannot inspect intended private Release")
 if release.get("prerelease") is not True:raise RuntimeError("Expected a prerelease; existing release policy differs")
 receipts=[]
 for item in blocks:
  assets=checked(s,"GET",f"{API}/releases/{release['id']}/assets?per_page=100")
  matches=[a for a in assets if a["name"]==item["filename"]]
  if len(matches)>1:raise RuntimeError("Duplicate name in destination Release")
  if matches:asset=matches[0]
  else:
   with item["path"].open("rb") as binary:
    asset=checked(s,"POST",release["upload_url"].split("{",1)[0],
      params={"name":item["filename"]},data=binary,
      headers={"Content-Type":"application/zip","Content-Length":str(item["bytes"])})
  asset_readback(s,asset,item["sha256"],item["bytes"])
  receipts.append({k:v for k,v in item.items() if k!="path"})
  print("02C PRIVATE REMOTE FULL SHA256 PASS",item["filename"],flush=True)
 receipt={"schema":"scept.a6-ij-02c.archive-receipt.v1",
   "status":"PRIVATE_55_PATIENT_112_REAL_SCEPT_OBSERVATIONS_ARCHIVED_AND_VERIFIED",
   "nPatients":55,"nValidationModelCaptures":110,"nBlankControlCaptures":2,
   "nRemoteShards":len(blocks),"originalPlanSHA256":shabytes(plan_raw),
   "originalAnalysisSHA256":shabytes(report_raw),
   "privateRelease":release["html_url"],"test":"NOT_ACCESSED",
   "originalMRIIncluded":False,"scientificInterpretation":"DESCRIPTIVE_ONLY",
   "remoteShards":receipts}
 receipt_raw=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 private_immutable(s,"locked_plan.json",plan_raw)
 private_immutable(s,"analysis.json",report_raw)
 private_immutable(s,"archive_receipt.json",receipt_raw)
 print("\nA6-IJ-02C PRIVATE ARCHIVE: PASS")
 print("Release:",release["html_url"])
 print("Private analysis:",PREFIX+"/analysis.json")

if __name__=="__main__":main()
