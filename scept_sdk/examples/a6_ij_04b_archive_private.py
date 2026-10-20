"""04B: immutable private five-person (final three-person) raw SCEPT archive.

Each shard includes all original case reports and 18 genuine output-only
SCEPT ZIPs per person, with complete original SHA256, reread through private
GitHub release. Archive idempotent and never overwrites published source.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from zipfile import ZIP_STORED,ZipFile,ZipInfo
from scept import load_observation
from a6_ij_03b_measure import github,API,REPO
from a6_ij_03b_archive_chunk_private import call,commit_once,remote_sha
from a6_ij_02b_measure import sha_file
from a6_ij_04b_matched_interventions import PLAN,fingerprint04b
from a6_ij_04b_measure import OUT,CONDITIONS,source,canonical,assert_local_case

BASE="results/scept_a6_ij/04b_equal_l2_interventions"
TAG="scept-a6-ij-04b-equal-input-l2-validation-v1"
STATUS="PASS_04B_REAL_ZIP_SOURCE_CHUNK_FULL_SHA256_VERIFIED"

def batch_source(num,eligible):
 if num not in range(1,8):raise RuntimeError("Only seven locked 04B source chunks")
 people=eligible[(num-1)*5:num*5]
 if not people or (num<7 and len(people)!=5) or (num==7 and len(people)!=3):
  raise RuntimeError("04B original source chunk distribution differs")
 return people
def remote_status(session,num,eligible):
 target=f"{BASE}/chunks/chunk_{num:02d}/archive_receipt.json"
 r=session.get(f"{API}/contents/{target}",timeout=(30,120))
 if r.status_code==404:return None
 r.raise_for_status()
 import base64
 result=r.json()
 if result.get("encoding")!="base64":
  raise RuntimeError("Prior 04B archive receipt encoding invalid")
 prior=json.loads(base64.b64decode(result["content"]))
 ids=[x["patient_id"] for x in batch_source(num,eligible)]
 if (prior.get("schema")!="scept.a6-ij-04b.chunk-receipt.v1" or
     prior.get("status")!=STATUS or prior.get("chunk")!=num or
     prior.get("patientIds")!=ids or prior.get("planSHA256")!=fingerprint04b() or
     prior.get("nPatientOriginalRealForwards")!=18*len(ids) or
     prior.get("originalRawZIPCount")!=18*len(ids)):
  raise RuntimeError("Published original 04B archived source receipt conflicts with original scientific design")
 release=session.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 release.raise_for_status()
 assets=[a for a in release.json()["assets"] if a["name"]==prior["assetName"]]
 if len(assets)!=1 or assets[0]["size"]!=prior["assetBytes"] or (
    assets[0].get("digest") and assets[0]["digest"]!="sha256:"+prior["assetSHA256"]):
  raise RuntimeError("Published original 04B private raw source shard unavailable or corrupted")
 return prior

def seal(files,path):
 expected={name:{"sha256":sha_file(fp),"bytes":fp.stat().st_size} for name,fp in files}
 if len(files)!=len(expected):
  raise RuntimeError("04B raw source archive has repeated member names")
 if not path.exists():
  tmp=path.with_suffix(".zip.partial")
  with ZipFile(tmp,"w",allowZip64=True) as z:
   for name,fp in sorted(files):
    info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
    info.compress_type=ZIP_STORED
    info.external_attr=0o600<<16
    z.writestr(info,fp.read_bytes())
  os.replace(tmp,path)
 with ZipFile(path) as z:
  if set(z.namelist())!=set(expected):
   raise RuntimeError("04B original source shard member listing differs")
  import hashlib
  for name,meta in expected.items():
   raw=z.read(name)
   if len(raw)!=meta["bytes"] or hashlib.sha256(raw).hexdigest()!=meta["sha256"]:
    raise RuntimeError("04B inner actual original raw file bytes SHA256 mismatched")
 return expected

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--chunk",type=int,required=True)
 args=parser.parse_args()
 s=github()
 eligible,_,lock=source(s)
 num=args.chunk
 prior=remote_status(s,num,eligible)
 if prior is not None:
  print(f"A6-IJ-04B CHUNK {num:02d} ALREADY PRIVATELY VERIFIED; NO repeated model forward")
  return
 ids=[x["patient_id"] for x in batch_source(num,eligible)]
 file=OUT/"locked_source_plan.json"
 if not file.is_file() or file.read_bytes()!=canonical(lock):
  raise RuntimeError("04B original source selection/plan local lock incomplete")
 originals=[]
 cases=[]
 seen=set()
 for patient in ids:
  folder=OUT/"patients"/patient
  record_path=folder/"case.json"
  if not record_path.is_file():
   raise RuntimeError("Original 04B patient must finish 18 real ZIPs before publishing chunk: "+patient)
  row=json.loads(record_path.read_text())
  if row["patient_id"]!=patient or row["grade"]!=next(x["grade"] for x in eligible if x["patient_id"]==patient) or \
     row["source"]["originalScreenSHA256"]!=lock["originalScreenBytesSHA256"] or \
     row["source"]["03aOriginalSourceSHA256"]!=lock["original03AOriginalSHA256"]:
   raise RuntimeError("Source patient identity / 03A+ROI audit differs")
  assert_local_case(row,folder)
  if row["nGenuineSCEPTForwards"]!=18 or len(row["inputDiagnostics"])!=8:
   raise RuntimeError("04B original patient has less than 18 real forwards")
  cases.append(row)
  originals.append((f"cases/{patient}.json",record_path))
  for model in ("S","J"):
   q=row["models"][model]
   if set(q["originalObservations"])!=set(CONDITIONS):
    raise RuntimeError("Original whole nine-condition SCEPT source missing")
   for condition,meta in q["originalObservations"].items():
    path=folder/"raw"/meta["name"]
    if path.name in seen:raise RuntimeError("Duplicated 04B archived source ZIP name")
    seen.add(path.name)
    if path.stat().st_size!=meta["bytes"] or sha_file(path)!=meta["sha256"]:
     raise RuntimeError("Original source SCEPT ZIP missing SHA256: "+str(path))
    v=load_observation(str(path))
    if (v.run_id!=meta["runId"] or
        v.record["provenance"]["model"]["sha256"]!=meta["modelStateSHA256"] or
        v.manifest["captureScope"]["inputsIncluded"] is not False):
     raise RuntimeError("04B private original SCEPT ZIP provenance mismatch")
    if {g["tensor"] for g in v.recompute_naf()["groups"] if g["status"]=="measured"}!=set(PLAN["layers"]):
     raise RuntimeError("04B actual three named transformer layers not measurable")
    if v.tensor("model.output").shape!=(1,2):
     raise RuntimeError("04B real model output source missing")
    originals.append(("raw/"+path.name,path))
 n=len(ids)
 if len(originals)!=n*19 or len(seen)!=n*18 or len(cases)!=n:
  raise RuntimeError("04B original 18 real-model captures + one patient case each incomplete")
 out=OUT/"shards";out.mkdir(parents=True,exist_ok=True)
 name=f"scept_04b_original_chunk_{num:02d}_{n:02d}patients_{18*n:03d}real.zip"
 full=out/name
 hashes=seal(originals,full)
 source_sha=sha_file(full);size=full.stat().st_size
 endpoint=f"{API}/releases/tags/{TAG}"
 rr=s.get(endpoint,timeout=(30,120))
 if rr.status_code==200:
  release=rr.json()
 elif rr.status_code==404:
  release=call(s,"POST",f"{API}/releases",json={
   "tag_name":TAG,"target_commitish":"main","prerelease":True,
   "draft":False,"name":"PRIVATE SCEPT 04B amplitude-matched glioma input intervention full source tensors",
   "body":"PRIVATE derived patient model internals; 55 VAL source-only screened, 33 eligible; each real frozen P4 S/J source. This does not establish causal clinical fidelity."})
 else:
  raise RuntimeError("Private original 04B release inspection failed")
 if release.get("prerelease") is not True:
  raise RuntimeError("Patient-derived private source release must be prerelease")
 matches=[a for a in release["assets"] if a["name"]==name]
 if len(matches)>1:raise RuntimeError("Multiple assets same original source name")
 if matches:asset=matches[0]
 else:
  print(f"04B uploading original source chunk {num:02d}, {size/1024/1024:.1f} MiB",flush=True)
  with full.open("rb") as fp:
   asset=call(s,"POST",release["upload_url"].split("{",1)[0],
     params={"name":name},data=fp,headers={"Content-Type":"application/zip",
      "Content-Length":str(size)})
 remote_sha(s,asset,source_sha,size)
 print(f"04B original chunk {num:02d}: PRIVATE remote FULL ZIP bytes SHA256 VERIFIED",flush=True)
 chunk={
  "schema":"scept.a6-ij-04b.chunk.v1",
  "status":"PASS_04B_EXACT_MATCHED_ORIGINAL_REAL_INTERVENTION_PATIENT_CHUNK",
  "chunk":num,"patientIds":ids,
  "plan":PLAN,"planSHA256":fingerprint04b(),
  "sourceLock":lock,"patientCases":cases,
  "nPatients":n,"nOriginalGenuineModelForwards":18*n,
  "noModelTraining":True,"sealedTest":"NOT_ACCESSED"}
 body=canonical(chunk)
 from a6_ij_04b_measure import sha
 receipt={
  "schema":"scept.a6-ij-04b.chunk-receipt.v1","status":STATUS,
  "chunk":num,"patientIds":ids,"planSHA256":fingerprint04b(),
  "nPatientOriginalRealForwards":18*n,"originalRawZIPCount":18*n,
  "originalPatientCaseCount":n,
  "originalChunkAnalysisSHA256":sha(body),
  "originalScreenSourceSHA256":lock["originalScreenBytesSHA256"],
  "assetName":name,"assetSHA256":source_sha,
  "assetBytes":size,"allOriginalMemberSHA256":hashes,
  "privateRelease":release["html_url"],"originalMRIIncludedInZIP":False,
  "sealedTest":"NOT_ACCESSED","scientificScope":"EXPLORATORY_EQUAL_PIXEL_L2_OUTPUT_RESPONSE_ONLY"}
 prefix=f"{BASE}/chunks/chunk_{num:02d}"
 commit_once(s,prefix+"/paired_cases.json",body)
 commit_once(s,prefix+"/archive_receipt.json",canonical(receipt))
 print(f"A6-IJ-04B CHUNK {num:02d} PRIVATE ORIGINAL {n} PATIENTS × 18 REAL ZIPs: PASS",flush=True)

if __name__=="__main__":main()
