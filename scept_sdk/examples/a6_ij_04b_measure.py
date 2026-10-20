"""04B: source-locked 33 VAL first-slice real frozen S/J SCEPT intervention runs.

Use --pilot for the first fixed eligible patient only (NOT a completed chunk),
then --chunk 1..7 for each 5/5/5/5/5/5/3 patients. Never train, access P5,
read sealed test slices, or choose outcomes/ROIs post hoc. Original full tensor
ZIPs and patient reports are local until explicit PRIVATE archive succeeds.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from scept import SCEPT,load_observation
from a6_ij_02b_measure import (manifest_and_zip,checkpoint_model,sha_file,
                                LAYERS as SOURCE_LAYERS,ROOT)
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_04a_interventions import choose,fingerprint as roi_sha,score_tensors,strict_image
from a6_ij_04b_matched_interventions import (
 PLAN,DOSES,OPS,LAYERS,matched_inputs,condition_name,
 fingerprint04b,real_output,contrast_from_outputs
)

OUT=ROOT/"a6_ij_04b_private"
SCREEN="results/scept_a6_ij/04a02_original_mask_only_feasibility/full_val_first_slice_screen.json"
SCREEN_GIT_BLOB_SHA="fe39ebcc9bc323aa260c6cec4fd497287971d06c"
OLD_03A="results/scept_a6_ij/03a_exact_first_slice_pairing/analysis.json"
STATUS="ELIGIBLE_UNDER_UNCHANGED_V1_RULE"
CONDITIONS=("clean",)+tuple(condition_name(op,dose,arm)
    for op in OPS for dose in DOSES for arm in ("lesion","control"))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(o):return (json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def check_mask_selection(chosen,record):
 old=record["regionSelection"]
 if old is None or abs(float(old["matchScore"])-float(chosen["matchScore"]))>1e-9:
  raise RuntimeError("Image-only original WT/control match changed: source not locked")
 for arm in ("lesion","control"):
  for k in ("row","col"):
   if int(chosen[arm][k])!=int(old[arm][k]):
    raise RuntimeError("Source-original lesion/control ROI changed before model inference")
 if float(chosen["matchScore"])>2.0 or chosen["control"]["dilatedWTFraction"]!=0:
  raise RuntimeError("Locked 04A-v1 source eligibility predicate changed")
 return True
def source(session):
 r=session.get(f"{API}/contents/{SCREEN}",params={"ref":"main"},timeout=(30,120))
 r.raise_for_status()
 if r.json().get("sha")!=SCREEN_GIT_BLOB_SHA:
  raise RuntimeError("Original unchanged mask-only eligibility audit Git blob hash differs")
 raw=private_contents(session,SCREEN)
 report=json.loads(raw)
 rows=report.get("patientRecords",[])
 yes=[r for r in rows if r.get("status")==STATUS]
 no=[r for r in rows if r.get("status")!=STATUS]
 if (report.get("status")!="PASS_COMPLETE_MASK_ONLY_SCREEN" or
     report.get("nScreened")!=55 or report.get("nEligible")!=33 or
     report.get("nIneligible")!=22 or len(rows)!=55 or len(yes)!=33 or len(no)!=22 or
     len({r["patient_id"] for r in rows})!=55 or
     rows!=sorted(rows,key=lambda q:q["patient_id"]) or
     sum(r["grade"]=="HGG" for r in yes)!=26 or
     sum(r["grade"]=="LGG" for r in yes)!=7 or
     report.get("original04APlanSHA256")!=roi_sha() or
     report.get("sealedP5")!="NOT_ACCESSED"):
  raise RuntimeError("Original immutable full-55 source-only screening scope changed")
 if report["exploratoryNextPilot"]["patient_id"]!=yes[0]["patient_id"]:
  raise RuntimeError("Amended first eligible source candidate identity disagrees")
 # Old original 04A FAILED and distinct new 04A02 PASSED; never rewrite.
 failure=json.loads(private_contents(session,
    "results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json"))
 if failure.get("status")!="BLOCKED_FIXED_PATIENT_ROI_ELIGIBILITY":
  raise RuntimeError("Original old v1 BLOCKED state missing")
 c=report.get("original04Av1FailureSHA256")
 old=private_contents(session,
    "results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json")
 if sha(old)!=c:
  raise RuntimeError("Original old v1 failed case source SHA changed")
 orig_raw=private_contents(session,OLD_03A)
 orig=json.loads(orig_raw)
 if (orig.get("status")!="PASS_STRICTLY_PAIRED_55_FIRST_SLICE_VAL" or
     orig.get("cohort",{}).get("nMatchedModelSliceRecords")!=110):
  raise RuntimeError("Original actual P4/03A first-slice baseline invalid")
 old_model={(r["patient_id"],r["model"]):r for r in orig["exactMatchedPatientRows"]}
 if len(old_model)!=110 or any((r["patient_id"],m) not in old_model for r in yes for m in ("S","J")):
  raise RuntimeError("Original 03A 55 first-slice predicted model sources incomplete")
 lock={"schema":"scept.a6-ij-04b.sources.v1",
       "plan":PLAN,"planSHA256":fingerprint04b(),
       "originalScreenGitBlobSHA":SCREEN_GIT_BLOB_SHA,
       "originalScreenBytesSHA256":sha(raw),
       "original03AOriginalSHA256":sha(orig_raw),
       "cohortSelectedPatients":[r["patient_id"] for r in yes],
       "nQualified33":33,"nOriginalExcluded":22,
       "sealedP5":"NOT_ACCESSED"}
 return yes,old_model,lock

def unpack_val_zip(session):
 """Use original full P1 classification ZIP; download with SHA if Kaggle reset."""
 try:
  return manifest_and_zip(session)
 except RuntimeError as exc:
  if "SHA-verified P1 ZIP missing" not in str(exc):raise
 from a6_ij_02b_measure import get_json
 release=get_json(session,f"{API}/releases/tags/p0-p2-2026-10-13")
 assets=[a for a in release["assets"] if a["name"].startswith("p1_classification_npz-") and a["name"].endswith(".zip")]
 if len(assets)!=1:raise RuntimeError("Original P1 archive missing from private research Release")
 item=assets[0]
 basename=item["name"].rsplit("-",1)[-1][:-4]
 loc=ROOT/"a6_ij_models"/"p1_classification_npz.zip"
 loc.parent.mkdir(parents=True,exist_ok=True)
 tmp=loc.with_suffix(".zip.partial")
 with session.get(f"{API}/releases/assets/{item['id']}",
         headers={"Accept":"application/octet-stream"},
         stream=True,allow_redirects=True,timeout=(30,900)) as r:
  r.raise_for_status()
  if "json" in r.headers.get("Content-Type","").lower():
   raise RuntimeError("P1 original archive private asset response is JSON")
  h=hashlib.sha256();size=0
  with tmp.open("wb") as fp:
   for c in r.iter_content(chunk_size=4*1024*1024):
    if c:fp.write(c);h.update(c);size+=len(c)
 if size!=item["size"] or not h.hexdigest().startswith(basename) or (
     item.get("digest") and item["digest"]!="sha256:"+h.hexdigest()):
  raise RuntimeError("Original complete P1 SHA256 asset invalid")
 os.replace(tmp,loc)
 return manifest_and_zip(session)

def pure_input(image,grade):
 if image.shape!=(224,224) or image.dtype!=np.float32:
  raise RuntimeError("Original frozen P1 input shape/dtype mismatched")
 return torch.from_numpy(image.copy()).view(1,1,224,224).repeat(1,3,1,1).contiguous()

def capture(observer,im,grade,patient,model,condition,p1_sha,stage):
 x=pure_input(im,grade)
 run=observer.observe(x,model_id="P4_original_frozen_"+model,
   preprocess_id="P1_FLAIR_224_locked_v1",
   preprocess_config={"sourceNPZSha256":p1_sha,"patient_id":patient,
     "sliceIndex":0,"split":"val","condition":condition,
     "planSha256":fingerprint04b(),"explicitlyNoTraining":True})
 if ({g["tensor"] for g in run.fingerprint()["groups"] if g["status"]=="measured"}!=set(LAYERS) or
     any(run.tensor(k).shape!=(1,197,768) for k in LAYERS) or
     run.tensor("model.output").shape!=(1,2) or
     not all(np.isfinite(run.tensor(k)).all() for k in (*LAYERS,"model.output"))):
  raise RuntimeError("Real SCEPT 3 authentic layer/2 logit output capture incomplete")
 output=real_output(run.tensor("model.output")[0],grade)
 tmp=stage/(f"{patient}_{model}_{condition}.partial.zip")
 run.export(str(tmp),include_inputs=False)
 again=load_observation(str(tmp))
 if (again.run_id!=run.run_id or
     again.record["provenance"]["model"]["sha256"]!=observer.model_sha256 or
     again.manifest["captureScope"]["inputsIncluded"] is not False):
  raise RuntimeError("Untrusted original complete SCEPT output-only archive")
 for key in (*LAYERS,"model.output"):
  np.testing.assert_array_equal(run.tensor(key),again.tensor(key))
 name=f"{patient}_{model}_{condition}_private.zip"
 final=stage/name
 os.replace(tmp,final)
 return run,{"name":name,"sha256":sha_file(final),"bytes":final.stat().st_size,
             "runId":run.run_id,"modelStateSHA256":observer.model_sha256,
             "output":output}

def assert_local_case(case,fp):
 if case.get("status")!="PASS_ORIGINAL_04B_PATIENT_18_REAL_MODEL_SOURCE_ZIPS" or \
    case.get("planSHA256")!=fingerprint04b() or \
    set(case.get("models",{}))!={"S","J"}:
  raise RuntimeError("Existing locally stored 04B case not original protocol")
 if len(case.get("inputDiagnostics",{}))!=8:raise RuntimeError("Original 04B input diagnostics incomplete")
 for model in ("S","J"):
  obs=case["models"][model]["originalObservations"]
  if set(obs)!=set(CONDITIONS):
   raise RuntimeError("Original 9 named frozen conditions for model missing")
  for condition,meta in obs.items():
   zipped=fp/"raw"/meta["name"]
   if zipped.stat().st_size!=meta["bytes"] or sha_file(zipped)!=meta["sha256"]:
    raise RuntimeError("Local patient full numeric ZIP failed SHA256 replay")
 return True

def one_patient(item,zipfile,models,observers,source_model,screen,lock):
 patient=item["patient_id"]
 final=OUT/"patients"/patient
 if final.is_dir():
  r=json.loads((final/"case.json").read_text())
  if r["patient_id"]!=patient or r["source"]["P1NPZSHA256"]!=item["p1_sha256"]:
   raise RuntimeError("Original completed patient data source differs")
  assert_local_case(r,final)
  print("04B REUSE locally original SHA verified patient:",patient,flush=True)
  return r
 stage=OUT/"inprogress"/patient
 if stage.exists():
  # Only incomplete, unpublished, local *inprogress* files are discarded.
  shutil.rmtree(stage)
 stage.mkdir(parents=True)
 raw=zipfile.read(item["zip_member"])
 if sha(raw)!=item["p1_sha256"] or item["p1_sha256"]!=screen["originalFullP1NPZSHA256"]:
  raise RuntimeError("Original P1 patient full original SHA256 differs from private mask-only 55 screen")
 with np.load(io.BytesIO(raw),allow_pickle=False) as data:
  if not {"flair","wt","foreground","axial_indices"}.issubset(data.files) or \
      data["flair"].shape!=(10,224,224) or data["wt"].shape!=(10,224,224) or \
      data["foreground"].shape!=(10,224,224):
   raise RuntimeError("Original frozen P1 ten image/mask slices changed")
  image=np.asarray(data["flair"][0],dtype=np.float32).copy()
  wt=np.asarray(data["wt"][0]).copy()
  fg=np.asarray(data["foreground"][0]).copy()
  axial=int(data["axial_indices"][0])
 if axial!=int(screen["axialIndex"]):
  raise RuntimeError("Original P1 axial index differs from eligibility screen")
 strict_image(image,wt,fg)
 selection=choose(image,wt,fg)
 check_mask_selection(selection,screen)
 images,inputs=matched_inputs(image,wt,fg,selection)
 record={
  "schema":"scept.a6-ij-04b.patient.v1",
  "status":"INCOMPLETE","patient_id":patient,"grade":item["grade"],
  "planSHA256":fingerprint04b(),"source":{
    "P1NPZSHA256":item["p1_sha256"],"originalScreenSHA256":lock["originalScreenBytesSHA256"],
    "03aOriginalSourceSHA256":lock["original03AOriginalSHA256"],"sealedP5":"NOT_ACCESSED"},
  "sliceRank":1,"axialIndex":axial,
  "patientFixedROIs":selection,"inputDiagnostics":inputs,
  "models":{},"nGenuineSCEPTForwards":0,
  "scientificVerdict":"NOT_ESTABLISHED; requires 33 patients and intervention-artifact controls"}
 for model in ("S","J"):
  observer=observers[model]
  runs={};report={};summaries={}
  for condition in CONDITIONS:
   run,meta=capture(observer,images[condition],item["grade"],patient,model,condition,item["p1_sha256"],stage)
   if condition=="clean":
    source=source_model[(patient,model)]
    diff=abs(meta["output"]["pHgg"]-float(source["pHggSCEPT"]))
    if diff>0.03:
     raise RuntimeError("Locked 03A first-slice frozen model output mismatch")
    clean_diff=diff
   else:
    operator=next(op for op in OPS if condition.startswith(op+"_dose"))
    dose=next(d for d in DOSES if f"_dose{int(d*100):02d}_" in condition)
    arm=condition.rsplit("_",1)[-1]
    baseline=runs["clean"]
    pos=selection[arm]
    summaries[condition]={
      "output":meta["output"],
      "input":inputs[condition],
      "signedTrueGradeMarginDelta":meta["output"]["trueGradeMargin"]-
                                    report["clean"]["output"]["trueGradeMargin"],
      "layerStateDelta":{k:score_tensors(baseline.tensor(k),run.tensor(k),
                 (pos["row"],pos["col"])) for k in LAYERS}
    }
   # Only the original clean run is held for later pair contrasts.
   if condition=="clean":runs["clean"]=run
   report[condition]=meta
   record["nGenuineSCEPTForwards"]+=1
   if condition!="clean":del run
  paired={}
  for operator in OPS:
   for dose in DOSES:
    pair={arm:condition_name(operator,dose,arm) for arm in ("lesion","control")}
    left=summaries[pair["lesion"]]["output"];right=summaries[pair["control"]]["output"]
    paired[f"{operator}_dose{int(dose*100):02d}"]=contrast_from_outputs(
      report["clean"]["output"],{"lesion":left,"control":right},
      inputs[pair["lesion"]],inputs[pair["control"]])
  record["models"][model]={
    "originalFrozenModelSHA256":observer.model_sha256,
    "clean03AProbabilityAbsoluteError":clean_diff,
    "originalObservations":report,
    "realInterventionSummaries":summaries,"pairedOutputContrasts":paired}
  del runs
 if record["nGenuineSCEPTForwards"]!=18:
  raise RuntimeError("04B original S/J forwards incomplete")
 (stage/"raw").mkdir()
 for path in stage.glob("*_private.zip"):
  os.replace(path,stage/"raw"/path.name)
 record["status"]="PASS_ORIGINAL_04B_PATIENT_18_REAL_MODEL_SOURCE_ZIPS"
 tmp=stage/"case.json.partial"
 tmp.write_bytes(canonical(record));os.replace(tmp,stage/"case.json")
 assert_local_case(record,stage)
 final.parent.mkdir(parents=True,exist_ok=True)
 if final.exists():raise RuntimeError("Original archived 04B patient destination unexpectedly exists")
 os.replace(stage,final)
 print("A6-IJ-04B PATIENT ORIGINAL 18 S/J ZIPs: PASS",patient,flush=True)
 return record

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--chunk",type=int,required=True)
 p.add_argument("--pilot",action="store_true")
 args=p.parse_args()
 if args.chunk not in range(1,8) or (args.pilot and args.chunk!=1):
  raise RuntimeError("Locked 04B private patient source chunk 1..7, pilot only chunk1")
 torch.set_num_threads(min(4,torch.get_num_threads()))
 s=github()
 selection,baseline,lock=source(s)
 # Mandatory all-33 ORIGINAL PRE-MODEL physical input feasibility archive
 # before any frozen ViT model is even constructed.
 preflight_path="results/scept_a6_ij/04b_equal_l2_interventions/premodel_input_feasibility.json"
 previous=json.loads(private_contents(s,preflight_path))
 if (previous.get("status")!="PASS_33_OF_33_PATIENTS_4_PAIRED_INPUT_L2_DOSES_PREMODEL" or
     previous.get("planSHA256")!=fingerprint04b() or
     previous.get("sourceLock")!=lock or
     previous.get("nNewMatchedPass")!=33 or previous.get("nNewMatchedBlocked")!=0 or
     len(previous.get("patientRecords",[]))!=33):
  raise RuntimeError("04B exact original source-only all-33 equal-L2 feasibility gate NOT PASS; NO model inference")
 source_rows,zip_path=unpack_val_zip(s)
 by_id={r["patient_id"]:r for r in source_rows}
 if len(by_id)!=55:raise RuntimeError("P0 original VAL whole cohort missing")
 chosen=selection[(args.chunk-1)*5:args.chunk*5]
 if not chosen or (len(chosen)!=5 and args.chunk!=7) or \
     (args.chunk==7 and len(chosen)!=3):
  raise RuntimeError("Original eligible ordered 5/5/.../3 chunk partition differs")
 for r in chosen:
  if r["patient_id"] not in by_id or r["grade"]!=by_id[r["patient_id"]]["grade"]:
   raise RuntimeError("Original 04B mask-only first-slice eligible cohort changed")
 OUT.mkdir(parents=True,exist_ok=True)
 lock_file=OUT/"locked_source_plan.json"
 if lock_file.exists():
  if lock_file.read_bytes()!=canonical(lock):
   raise RuntimeError("Original 04B local immutable source/plan changed")
 else:
  lock_file.write_bytes(canonical(lock))
 names=["S","J"]
 models={k:checkpoint_model(k) for k in names}
 observers={k:SCEPT(models[k],list(LAYERS),backend="torch",
               profile="vit-token-channel",max_capture_bytes=16_000_000) for k in names}
 todo=chosen[:1] if args.pilot else chosen
 with ZipFile(zip_path) as z:
  for i,r in enumerate(todo,1):
   one_patient(by_id[r["patient_id"]],z,models,observers,baseline,r,lock)
   print(f"04B {i}/{len(todo)} original patient case validated in selected chunk {args.chunk}",flush=True)
 print(("A6-IJ-04B ONE-PATIENT OPERATIONAL PILOT: PASS; NOT FULL CHUNK"
        if args.pilot else f"A6-IJ-04B CHUNK {args.chunk:02d} ALL PATIENTS MEASURED: PASS; PRIVATE ARCHIVE NEXT"),flush=True)

if __name__=="__main__":main()
