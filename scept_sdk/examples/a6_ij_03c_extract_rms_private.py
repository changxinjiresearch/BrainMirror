"""A6-IJ-03C-01: calculate exact patch RMS from immutable original 1100 SCEPT ZIPs.

ONE private 5-patient batch at a time; original 02B first-rank + 03B ranks
2..10; verify original release asset SHA256 and every inner ZIP source digest.
NO inference or P5; source observables are private. Archives each 5-patient
RMS table through immutable GitHub content SHA readback.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import requests
from scept import load_observation
from a6_ij_03b_measure import github,private_contents,API,REPO,PLAN as P03B
from a6_ij_03b_archive_chunk_private import remote_status,commit_once

BASE="results/scept_a6_ij/03c_incremental_value"
OLD_BASE="results/scept_a6_ij/02b_55_patient_representations"
OLD_TAG="scept-a6-ij-02b-55val-v1"
NEW_TAG="scept-a6-ij-03b-ten-slice-paired-v1"
LAYERS=("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")
OUT=Path("/kaggle/working/a6_ij_03c_private")

def sha(raw):return hashlib.sha256(raw).hexdigest()
def required_asset(s,tag,name,expected_sha,expected_size):
 r=s.get(f"{API}/releases/tags/{tag}",timeout=(30,120))
 r.raise_for_status()
 release=r.json()
 if release.get("prerelease") is not True:
  raise RuntimeError("Private original source Release policy changed")
 assets=[a for a in release["assets"] if a["name"]==name]
 if len(assets)!=1:raise RuntimeError("Private original Release asset not uniquely found: "+name)
 asset=assets[0]
 if asset["size"]!=expected_size or (asset.get("digest") and asset["digest"]!="sha256:"+expected_sha):
  raise RuntimeError("Private original Release remote source size/digest differs")
 OUT.mkdir(parents=True,exist_ok=True)
 filename=OUT/("source_"+name)
 if filename.exists() and filename.stat().st_size==expected_size:
  digest=hashlib.sha256()
  with filename.open("rb") as fp:
   while True:
    block=fp.read(4*1024*1024)
    if not block:break
    digest.update(block)
  if digest.hexdigest()==expected_sha:return filename
 tmp=OUT/(filename.name+".partial")
 with s.get(f"{API}/releases/assets/{asset['id']}",headers={"Accept":"application/octet-stream"},
            stream=True,allow_redirects=True,timeout=(30,900)) as response:
  response.raise_for_status()
  if "json" in response.headers.get("Content-Type","").lower():
   raise RuntimeError("Source download returned GitHub metadata instead of raw ZIP")
  h=hashlib.sha256();size=0
  with tmp.open("wb") as output:
   for data in response.iter_content(chunk_size=4*1024*1024):
    if data:output.write(data);h.update(data);size+=len(data)
 if size!=expected_size or h.hexdigest()!=expected_sha:
  tmp.unlink(missing_ok=True)
  raise RuntimeError("Downloaded complete ORIGINAL private SCEPT source SHA256 mismatch")
 os.replace(tmp,filename)
 return filename

def patch_rms(source,expected_modelsha=None,expected_run=None):
 if source.manifest["captureScope"]["inputsIncluded"] is not False:
  raise RuntimeError("Original SCEPT source ZIP unexpectedly includes input")
 if expected_modelsha and source.record["provenance"]["model"]["sha256"]!=expected_modelsha:
  raise RuntimeError("Original SCEPT model state provenance altered")
 if expected_run and source.run_id!=expected_run:
  raise RuntimeError("Original SCEPT original run identity altered")
 vals={}
 for layer in LAYERS:
  x=source.tensor(layer)
  if x.shape!=(1,197,768) or x.dtype not in (np.float32,np.float64):
   raise RuntimeError("Original genuine ViT intermediate state dimensions not matched")
  q=x[0,1:,:].astype(np.float64,copy=False)
  v=float(np.sqrt(np.mean(np.square(q))))
  if not np.isfinite(v):raise RuntimeError("Invalid measured amplitude")
  vals[layer]=v
 return vals

def safe_case(source_zip,members,path,digest):
 meta=members.get(path)
 if meta is None or meta["sha256"]!=digest:
  raise RuntimeError("Original inner ZIP digest differs from immutable private receipt: "+path)
 raw=source_zip.read(path)
 if len(raw)!=meta["bytes"] or sha(raw)!=digest:
  raise RuntimeError("Actual raw original SCEPT ZIP SHA256 bytes mismatch: "+path)
 return raw

def compute(s,n):
 receipt=remote_status(s,n)
 if receipt is None:raise RuntimeError("Complete privately verified 03B source chunk missing")
 result_path=f"{BASE}/rms/chunk_{n:02d}.json"
 old=s.get(f"{API}/contents/{result_path}",timeout=(30,120))
 if old.status_code==200:
  prior=json.loads(base64.b64decode(old.json()["content"]))
  if prior.get("status")!="PASS_5_PATIENTS_100_RAW_RMS_VALUES" or prior.get("chunk")!=n or prior.get("03bOriginalShardSHA256")!=receipt["assetSha256"] or len(prior.get("patientRows",[]))!=5:
   raise RuntimeError("Original source RMS chunk exists but does not match locked 03B provenance")
  print(f"03C RMS chunk {n:02d}: ALREADY PRIVATELY ARCHIVED; no new downloads",flush=True)
  return prior
 if old.status_code!=404:raise RuntimeError("Existing private RMS source status unavailable")
 # 03B paired rows are previously committed, immutable per original chunk
 base=f"results/scept_a6_ij/03b_full_slice_pairing/chunks/chunk_{n:02d}"
 orig_raw=private_contents(s,base+"/paired_rows.json")
 if sha(orig_raw)!=receipt["originalReportSHA256"]:
  raise RuntimeError("Authenticated 03B patient paired row source SHA mismatch")
 paired=json.loads(orig_raw)
 if paired.get("status")!="PASS_5_PATIENTS_100_SOURCE_PAIRED_MODEL_SLICES" or paired["patientIds"]!=receipt["patientIds"]:
  raise RuntimeError("Original source paired patients changed")
 old_receipt=json.loads(private_contents(s,OLD_BASE+"/archive_receipt.json"))
 if old_receipt.get("status")!="ARCHIVED_55_PATIENTS_110_ZIPS_VERIFIED" or len(old_receipt["shards"])!=11:
  raise RuntimeError("Original 02B 110 source ZIP receipt not valid")
 old_item=old_receipt["shards"][n-1]
 if old_item["patientCount"]!=5:raise RuntimeError("Old 02B original chunk identity missing")
 a=required_asset(s,OLD_TAG,old_item["filename"],old_item["sha256"],old_item["bytes"])
 b=required_asset(s,NEW_TAG,receipt["assetName"],receipt["assetSha256"],receipt["assetBytes"])
 patientRows=[]
 try:
  with ZipFile(a) as early,ZipFile(b) as current:
   old_members=old_item["memberDigests"]
   new_members=receipt["memberSHA256"]
   for case in paired["pairedPatientRows"]:
    pid=case["patient_id"]
    if pid not in receipt["patientIds"] or len(case["slicePairs"])!=10:
     raise RuntimeError("Original source patient identity/ten ranks invalid")
    rows=[]
    for r in case["slicePairs"]:
     rank=r["slice_rank"]
     if not 1<=rank<=10:raise RuntimeError("Original slice rank out of bounds")
     models={}
     for model in ("S","J"):
      e=r["models"][model]["SCEPT"]
      if rank==1:
       if e["runSource"]!="REUSED_02B_WITH_03A_SHA_MATCHED_PRIVATE_RAW_ZIP":
        raise RuntimeError("Original first-slice source provenance invalid")
       key=f"raw/{pid}_{model}_local_private.zip"
       rec=old_members.get(key)
       if not rec:raise RuntimeError("02B first-slice raw source missing")
       raw=safe_case(early,old_members,key,rec["sha256"])
       obs=load_observation(raw)
       if obs.manifest["captureScope"]["inputsIncluded"] is not False:
        raise RuntimeError("Patient source ZIP contains raw MRI")
       modelsha=obs.record["provenance"]["model"]["sha256"]
      else:
       if e["runSource"]!="NEW_03B_GENUINE_SCEPT_FORWARD":
        raise RuntimeError("Not measured original 03B forward")
       entry=e["originalZip"];key="raw/"+entry["name"]
       raw=safe_case(current,new_members,key,entry["sha256"])
       if len(raw)!=entry["bytes"]:
        raise RuntimeError("03B inner actual SCEPT bytes mismatch")
       obs=load_observation(raw)
       modelsha=e["modelStateSHA256"]
      models[model]={"patchRMS":patch_rms(obs,modelsha,e.get("runId")),
                      "originalZipSHA256":sha(raw),"sourceRelease":"02B" if rank==1 else "03B"}
     rows.append({"slice_rank":rank,"models":models})
    if [row["slice_rank"] for row in rows]!=list(range(1,11)):
     raise RuntimeError("Original 10-slice ordering changed")
    patientRows.append({"patient_id":pid,"grade":case["grade"],"sliceRows":rows})
 finally:
  # Preserve disk capacity; each 03B raw shard ~170MB but original Release is durable.
  if a.exists():a.unlink()
  if b.exists():b.unlink()
 if len(patientRows)!=5 or sum(len(r["sliceRows"])*2 for r in patientRows)!=100:
  raise RuntimeError("Private RMS derivation missing original patient/model/rank")
 report={"schema":"scept.a6-ij-03c.rms-chunk.v1",
   "status":"PASS_5_PATIENTS_100_RAW_RMS_VALUES","chunk":n,
   "patientIds":receipt["patientIds"],"nPatients":5,"nModelSliceValues":100,
   "nOriginalRawLayerMeasurements":300,
   "03bOriginalShardSHA256":receipt["assetSha256"],
   "02bOriginalShardSHA256":old_item["sha256"],
   "03bPairedSourceSHA256":sha(orig_raw),
   "patchRMSFormula":"sqrt(mean(float64(original_tensor[0,1:,:])^2))",
   "noNewModelInference":True,"sealedP5":"NOT_ACCESSED",
   "patientRows":patientRows}
 serialized=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 commit_once(s,result_path,serialized)
 print(f"A6-IJ-03C RMS PRIVATE CHUNK {n:02d}: PASS; 5 patients; 100 original sources × 3 layers",flush=True)
 return report

def main():
 p=argparse.ArgumentParser()
 g=p.add_mutually_exclusive_group(required=True)
 g.add_argument("--chunk",type=int)
 g.add_argument("--all",action="store_true")
 args=p.parse_args()
 if args.chunk is not None and not 1<=args.chunk<=11:
  raise RuntimeError("Select 03B original chunk 1..11")
 s=github()
 chunks=range(1,12) if args.all else (args.chunk,)
 for n in chunks:compute(s,n)
 print("A6-IJ-03C RMS SOURCE DERIVATION: COMPLETE for requested chunks",flush=True)

if __name__=="__main__":main()
