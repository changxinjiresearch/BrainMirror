"""04C02 independent FULL original SCEPT 197x768 tensor audit, no models/MRI.

Reads 7 original private 04B raw 90/90/90/90/90/90/54 observation ZIP shards,
verifies full remote SHA and every nested ZIP's SHA against original receipts,
then reopens authentic clean/perturbed 3 named layer tensors. Computes target
token energy share, distal energy, CLS and patch signal propagation across
blocks 0/5/11. No new model training, no new medical input, no attribution
or causal-faithfulness claim. Kaggle CPU and private GITHUB_TOKEN needed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scept import load_observation
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_04b_archive_private import TAG as SRC_TAG,BASE as SOURCE_B
from a6_ij_04c_association_private import load,canonical,sha,L,BASE

ROOT=Path("/kaggle/working/a6_ij_04c02_original_tensor_private")
KERNEL_LIMIT=1e-12
STATUS="PASS_04C02_FULL_PRIVATE_ORIGINAL_ACTIVATION_TENSOR_REPLAY"

def hashfile(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for part in iter(lambda:f.read(4*1024*1024),b""):h.update(part)
 return h.hexdigest()

def download(session,asset,local,receipt):
 """Read-only original private Release full-byte SHA-256 validation."""
 if local.is_file():
  if local.stat().st_size==receipt["assetBytes"] and hashfile(local)==receipt["assetSHA256"]:
   print("04C02 original private source shard reuse SHA-verified",local.name,flush=True)
   return
  raise RuntimeError("Existing local 04B raw shard bytes different; do not overwrite")
 local.parent.mkdir(parents=True,exist_ok=True)
 tmp=local.with_suffix(".zip.partial")
 if tmp.exists():tmp.unlink()
 with session.get(f"{API}/releases/assets/{asset['id']}",
      headers={"Accept":"application/octet-stream"},
      stream=True,allow_redirects=True,timeout=(30,900)) as r:
  r.raise_for_status()
  if "json" in r.headers.get("Content-Type","").lower():
   raise RuntimeError("04C02 private GitHub Release asset is JSON, not ZIP")
  with tmp.open("wb") as w:
   for chunk in r.iter_content(chunk_size=4*1024*1024):
    if chunk:w.write(chunk)
 if tmp.stat().st_size!=receipt["assetBytes"] or hashfile(tmp)!=receipt["assetSHA256"]:
  raise RuntimeError("04C02 ORIGINAL PRIVATE COMPLETE RELEASE BYTES SHA256 MISMATCH")
 os.replace(tmp,local)

def layer_metrics(clean,changed,grid,reference):
 c=np.asarray(clean,dtype=np.float64)
 x=np.asarray(changed,dtype=np.float64)
 if c.shape!=(1,197,768) or x.shape!=c.shape or \
    not np.isfinite(c).all() or not np.isfinite(x).all():
  raise RuntimeError("04C02 original three-layer authentic source shape/nonfinite")
 row,col=grid
 if not (0<=row<14 and 0<=col<14):
  raise RuntimeError("04C02 original ViT token mapping invalid")
 idx=1+14*row+col
 delta=x[0]-c[0]
 token_energy=np.sum(delta*delta,axis=1)
 patch_energy=float(np.sum(token_energy[1:]))
 target=float(token_energy[idx])
 relative=float(np.sqrt(patch_energy)/(np.linalg.norm(c[0,1:])+KERNEL_LIMIT))
 direct=float(np.sqrt(target))
 cls=float(np.sqrt(token_energy[0]))
 expected=reference
 if (abs(relative-float(expected["patchRelativeL2"]))>1e-5 or
     abs(direct-float(expected["directTargetedTokenDeltaL2"]))>1e-5 or
     abs(cls-float(expected["clsTokenDeltaL2"]))>1e-5):
  raise RuntimeError("04C02 original full authentic tensor contradicts previously sealed 04B summary")
 return {
  "directTokenDeltaL2":direct,
  "patchRelativeL2":relative,
  "clsDeltaL2":cls,
  "targetTokenChangedEnergyFraction":target/(patch_energy+KERNEL_LIMIT),
  "distalPatchEnergyFraction":max(0.,patch_energy-target)/(patch_energy+KERNEL_LIMIT),
  "totalPatchChangeEnergy":patch_energy,
  "patchTokenCount":196,"targetTokenIndex":idx,
  "sourceSummaryNumericallyReverified":True}

def verify_original_run(raw,meta):
 with tempfile.NamedTemporaryFile(prefix="scept_04c02_",suffix=".zip") as tmp:
  tmp.write(raw);tmp.flush()
  rec=load_observation(tmp.name)
  if (rec.run_id!=meta["runId"] or
      rec.record["provenance"]["model"]["sha256"]!=meta["modelStateSHA256"] or
      rec.manifest["captureScope"]["inputsIncluded"] is not False):
   raise RuntimeError("04C02 original 04B observed run provenance differs")
  logits=np.asarray(rec.tensor("model.output"),dtype=np.float64).reshape(-1)
  if logits.shape!=(2,) or not np.allclose(logits,meta["output"]["logits"],atol=2e-6,rtol=0):
   raise RuntimeError("04C02 original prediction logits changed on full SCEPT ZIP readback")
  result={k:np.asarray(rec.tensor(k)).copy() for k in L}
  if any(t.shape!=(1,197,768) for t in result.values()):
   raise RuntimeError("04C02 original physical 197x768 hidden activations malformed")
  return result

def chunk_audit(session,num,case_rows,release):
 receipt=json.loads(private_contents(session,
   f"{SOURCE_B}/chunks/chunk_{num:02d}/archive_receipt.json"))
 allcase=case_rows[(num-1)*5:num*5]
 target=f"{BASE}/full_tensor/chunk_{num:02d}_verified.json"
 get=session.get(f"{API}/contents/{target}",timeout=(30,120))
 if get.status_code==200:
  old=json.loads(private_contents(session,target))
  if (old.get("status")!=STATUS or old.get("sourceAssetSHA256")!=receipt["assetSHA256"] or
      old.get("patientIds")!=[r["patient_id"] for r in allcase] or
      old.get("nOriginalSourceRunTensorSets")!=18*len(allcase)):
   raise RuntimeError("04C02 older private chunk conflicts with original immutable study")
  print(f"04C02 original tensor chunk {num:02d} previously persisted: PASS",flush=True)
  return old
 if get.status_code!=404:raise RuntimeError("04C02 prior private original chunk unreadable")
 possible=[a for a in release["assets"] if a["name"]==receipt["assetName"]]
 if len(possible)!=1:raise RuntimeError("04C02 original source Release asset missing")
 shard=ROOT/(f"chunk_{num:02d}.zip")
 download(session,possible[0],shard,receipt)
 summaries=[]
 checked=0
 with ZipFile(shard,"r") as full:
  actual=set(full.namelist())
  expected=set(receipt["allOriginalMemberSHA256"])
  if actual!=expected or len(expected)!=19*len(allcase):
   raise RuntimeError("04C02 original private shard member count/names changed")
  def nested(name):
   raw=full.read(name)
   rec=receipt["allOriginalMemberSHA256"][name]
   if len(raw)!=rec["bytes"] or sha(raw)!=rec["sha256"]:
    raise RuntimeError("04C02 exact original private nested member bytes SHA changed")
   return raw
  for patient in allcase:
   pid=patient["patient_id"]
   old_case=json.loads(nested("cases/"+pid+".json"))
   if old_case["patient_id"]!=pid or old_case["planSHA256"]!=patient["planSHA256"]:
    raise RuntimeError("04C02 original per-patient case numerical source mismatch")
   patient_report={"patient_id":pid,"grade":patient["grade"],
                   "modelResults":{},"nOriginalRunTensorSets":0}
   for model in ("S","J"):
    original=patient["models"][model]
    cleanmeta=original["originalObservations"]["clean"]
    clean=verify_original_run(nested("raw/"+cleanmeta["name"]),cleanmeta)
    checked+=1;patient_report["nOriginalRunTensorSets"]+=1
    res={}
    for condition,meta in original["originalObservations"].items():
     if condition=="clean":continue
     parts=condition.rsplit("_",1)
     arm=parts[-1]
     if arm not in ("lesion","control"):
      raise RuntimeError("04C02 condition target region invalid")
     metric=original["realInterventionSummaries"][condition]
     newer=verify_original_run(nested("raw/"+meta["name"]),meta)
     checked+=1;patient_report["nOriginalRunTensorSets"]+=1
     region=patient["patientFixedROIs"][arm]
     coord=(int(region["row"]),int(region["col"]))
     res[condition]={
      "originalTrueGradeSignedMarginDelta":metric["signedTrueGradeMarginDelta"],
      "inputMagnitude":patient["inputDiagnostics"][condition]["actualPixelL2"],
      "threeOriginalFullTensorLayers":{
        k:layer_metrics(clean[k],newer[k],coord,metric["layerStateDelta"][k])
        for k in L}}
    if len(res)!=8:raise RuntimeError("04C02 missing paired operator condition")
    patient_report["modelResults"][model]=res
   if patient_report["nOriginalRunTensorSets"]!=18:
    raise RuntimeError("04C02 original model × condition output count broken")
   summaries.append(patient_report)
 if checked!=18*len(allcase):raise RuntimeError("04C02 original 594 run complete source gate broken")
 body={
   "schema":"scept.a6-ij-04c02.independently-verified-original-full-tensors.v1",
   "status":STATUS,"chunk":num,"nOriginalSourceRunTensorSets":checked,
   "patientIds":[r["patient_id"] for r in allcase],
   "sourceAssetSHA256":receipt["assetSHA256"],
   "original04BChunkSHA256":receipt["originalChunkAnalysisSHA256"],
   "patientSummaries":summaries,
   "originalMRIInputsInArchivedSCEPTZips":False,
   "sealedP5":"NOT_ACCESSED",
   "scientificVerdict":"INTERNAL_TENSOR_SPREAD_AND_ASSOCIATION_ONLY; NOT CAUSAL OR CLINICAL FIDELITY"}
 commit_once(session,target,canonical(body))
 # Source Release already rechecked on Kaggle; local shard can now be discarded.
 shard.unlink()
 print(f"A6-IJ-04C02 CHUNK {num:02d} ALL ORIGINAL CLEAN/PERTURBED 197x768 TENSORS VERIFIED: PASS",flush=True)
 return body

def main():
 cli=argparse.ArgumentParser()
 cli.add_argument("--chunk",type=int,choices=range(1,8),default=None)
 args=cli.parse_args()
 session=github()
 rows,originalsha,study,rec=load(session)
 response=session.get(f"{API}/releases/tags/{SRC_TAG}",timeout=(30,120))
 response.raise_for_status();release=response.json()
 batches=range(1,8) if args.chunk is None else [args.chunk]
 for k in batches:
  chunk_audit(session,k,rows,release)
 print("A6-IJ-04C02 REQUESTED ORIGINAL REAL FULL TENSOR AUDIT CHUNKS COMPLETE",flush=True)

if __name__=="__main__":
 main()
