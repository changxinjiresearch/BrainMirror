"""A6-IJ-04A genuine P1 VAL MRI/S+J frozen ViT evidence-perturbation PILOT.

Fixed person BraTS20_Training_022 rank1. Two pre-output-selected ViT-aligned
patch ROIs (tumor vs healthy), two fixed operators, same area. Exactly ten
full original SCEPT observations, plus two clean sham stability forwards.
This is *input intervention response*, NOT neuronal causal mediation.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import platform
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import timm
from scept import SCEPT,load_observation
from a6_ij_02b_measure import (
 manifest_and_zip,checkpoint_model,private_contents,sha_file,LAYERS,ROOT
)
from a6_ij_03b_measure import github
from a6_ij_04a_interventions import (
 PLAN,OPS,fingerprint,choose,mutate,score_tensors,strict_image
)

OUT=ROOT/"a6_ij_04a_private"
PILOT="BraTS20_Training_022"
SOURCE="results/scept_a6_ij/03a_exact_first_slice_pairing/analysis.json"
ORDER=("clean",)+tuple(f"{op}_{arm}" for op in OPS for arm in ("lesion","control"))

def digest(raw):return hashlib.sha256(raw).hexdigest()
def write_fixed(path,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=raw:
   raise RuntimeError("Existing local original pilot report differs; never overwrite!")
  return
 tmp=path.with_name(path.name+".partial");tmp.write_bytes(raw);os.replace(tmp,path)
def torch_input(x):
 if x.shape!=(224,224) or x.dtype!=np.float32:
  raise RuntimeError("Original P1 input must remain float32 [224,224]")
 return torch.from_numpy(x.copy()).view(1,1,224,224).repeat(1,3,1,1).contiguous()
def original_run(observer,image,patient,model,condition,source_sha):
 x=torch_input(image)
 run=observer.observe(x,model_id="P4_original_frozen_"+model,
  preprocess_id="P1_FLAIR_224_locked_v1",
  preprocess_config={"sourceNPZSha256":source_sha,"patient_id":patient,
   "sliceIndex":0,"split":"val","condition":condition,
   "study":"a6-ij-04a","explicitlyNoTraining":True})
 expected=set(LAYERS)
 measured={g["tensor"] for g in run.fingerprint()["groups"] if g["status"]=="measured"}
 if measured!=expected or any(run.tensor(k).shape!=(1,197,768) for k in LAYERS) or \
    run.tensor("model.output").shape!=(1,2):
  raise RuntimeError("Failed genuine measured P4 ViT named layers/logit shapes")
 if not all(np.isfinite(run.tensor(key)).all() for key in (*LAYERS,"model.output")):
  raise RuntimeError("Measured intermediate/logit contains NaN or Inf")
 OUT.mkdir(parents=True,exist_ok=True)
 raw_dir=OUT/"raw";raw_dir.mkdir(exist_ok=True)
 name=f"{patient}_{model}_{condition}_private.zip"
 fp=raw_dir/name
 tmp=raw_dir/(name+".partial")
 run.export(str(tmp),include_inputs=False)
 v=load_observation(str(tmp))
 if v.run_id!=run.run_id or v.manifest["captureScope"]["inputsIncluded"] is not False or \
    v.record["provenance"]["model"]["sha256"]!=observer.model_sha256:
  raise RuntimeError("ZIP does not preserve exact source model, run or privacy")
 for layer in (*LAYERS,"model.output"):
  np.testing.assert_array_equal(v.tensor(layer),run.tensor(layer))
 if v.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
  raise RuntimeError("Offline numeric rederivation must not pretend model forward")
 if fp.exists():
  if sha_file(fp)!=sha_file(tmp):
   raise RuntimeError("Existing output-only original raw ZIP differs: "+name)
  tmp.unlink()
 else:os.replace(tmp,fp)
 return run,{"name":name,"sha256":sha_file(fp),"bytes":fp.stat().st_size,
             "runId":run.run_id,"modelStateSHA256":observer.model_sha256}
def logits(run,grade):
 values=np.asarray(run.tensor("model.output")[0],dtype=np.float64)
 y=1 if grade=="HGG" else 0
 p=float(1/(1+np.exp(np.clip(values[0]-values[1],-500,500))))
 return {"logits":values.tolist(),"trueGradeLogit":float(values[y]),
         "trueGradeMargin":float(values[y]-values[1-y]),"pHgg":p}
def main():
 torch.set_num_threads(min(4,torch.get_num_threads()))
 OUT.mkdir(parents=True,exist_ok=True)
 result_path=OUT/"a6_ij_04a_pilot_report.json"
 if result_path.exists():
  source=json.loads(result_path.read_text())
  if source.get("status")=="PASS_INPUT_INTERVENTION_PILOT_ENGINEERING" and \
     source.get("interventionPlanSha256")==fingerprint():
   print("A6-IJ-04A old ORIGINAL local pilot exists; run archiver, do not rerun",flush=True)
   return
  raise RuntimeError("Existing incompatible 04A local record cannot be overwritten")
 s=github()
 cohort,zip_path=manifest_and_zip(s)
 case=next((x for x in cohort if x["patient_id"]==PILOT),None)
 if case is None or cohort[0]["patient_id"]!=PILOT or case["grade"]!="HGG":
  raise RuntimeError("Preregistered original first VAL patient/grade no longer matches")
 source_bytes=private_contents(s,SOURCE)
 archive=json.loads(source_bytes)
 if archive.get("status")!="PASS_STRICTLY_PAIRED_55_FIRST_SLICE_VAL" or \
    archive.get("cohort",{}).get("nMatchedModelSliceRecords")!=110:
  raise RuntimeError("Original private 03A first-slice identity not verified")
 old={(v["patient_id"],v["model"]):v for v in archive["exactMatchedPatientRows"]}
 if (PILOT,"S") not in old or (PILOT,"J") not in old:
  raise RuntimeError("Original strict SCEPT/Grad-CAM source model absent")
 with ZipFile(zip_path) as zipfile:
  raw=zipfile.read(case["zip_member"])
  if digest(raw)!=case["p1_sha256"]:
   raise RuntimeError("Original P1 patient NPZ full SHA256 changed")
  with np.load(io.BytesIO(raw),allow_pickle=False) as npz:
   image=np.asarray(npz["flair"][0],dtype=np.float32).copy()
   wt=np.asarray(npz["wt"][0]).copy()
   fg=np.asarray(npz["foreground"][0]).copy()
   axes=np.asarray(npz["axial_indices"])
 if axes.shape!=(10,) or int(axes[0])!=old[(PILOT,"S")]["axial_index"] or \
    int(axes[0])!=old[(PILOT,"J")]["axial_index"]:
  raise RuntimeError("Original P1 rank1 axial index/source mask incompatible")
 strict_image(image,wt,fg)
 try:
  selection=choose(image,wt,fg)
 except RuntimeError as exc:
  # Predeclared region-eligibility rejection is a meaningful negative protocol
  # result. Persist privately; never reselect patient/ROI to manufacture PASS.
  from a6_ij_03b_archive_chunk_private import commit_once
  blocked={"schema":"scept.a6-ij-04a.eligibility-failure.v1",
   "status":"BLOCKED_FIXED_PATIENT_ROI_ELIGIBILITY",
   "patient_id":PILOT,"grade":"HGG","sliceRank":1,
   "interventionPlanSha256":fingerprint(),
   "originalP1PatientNPZFullSHA256":digest(raw),
   "failedFixedCriterion":str(exc),"notRerunOnAlternativePatient":True,
   "originalTest":"NOT_ACCESSED","scientificVerdict":"NOT_DETERMINED"}
  local=OUT/"eligibility_failure.json"
  write_fixed(local,blocked)
  commit_once(s,"results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json",
              local.read_bytes())
  print("A6-IJ-04A LOCKED ROI ELIGIBILITY BLOCKED; FAILURE ARCHIVED PRIVATELY",flush=True)
  raise
 inputs={"clean":image}
 mutation={}
 for op in OPS:
  for arm in ("lesion","control"):
   key=f"{op}_{arm}"
   changed,details=mutate(image,wt,fg,selection,arm,op)
   inputs[key]=changed
   mutation[key]=details
   print("04A region",key,"source delta L2=",round(details["inputROIChangeL2"],6),flush=True)
 if tuple(inputs)!=ORDER:raise RuntimeError("04A exact paired operation order unexpectedly changed")
 original_sha=digest(raw)
 report={
  "schema":"scept.a6-ij-04a.pilot.v1",
  "status":"INCOMPLETE",
  "scope":"ENGINEERING_INPUT_INTERVENTION_ONLY; NOT NEURON_CAUSAL_OR_CLINICAL",
  "interventionPlan":PLAN,"interventionPlanSha256":fingerprint(),
  "patient_id":PILOT,"grade":"HGG","split":"val","sliceRank":1,
  "axial_index":int(axes[0]),
  "source":{"originalP1NPZFullSHA256":original_sha,
      "03aOriginalAnalysisSHA256":digest(source_bytes),
      "lockedOriginalFrozenModels":["S","J"],
      "P5test":"NOT_ACCESSED","P6external":"NOT_ACCESSED"},
  "maskAndROISelection":selection,
  "perturbationInputDiagnostics":mutation,
  "models":{},"nOriginalSCEPTCapturedForwards":0,
  "nAdditionalShamSCEPTForwards":0,
  "nReusedGradCAMObservations":0,
  "numericScientificConclusions":"NOT_ESTABLISHED"
 }
 for name in ("S","J"):
  model=checkpoint_model(name)
  observer=SCEPT(model,list(LAYERS),backend="torch",
                 profile="vit-token-channel",max_capture_bytes=16_000_000)
  old_row=old[(PILOT,name)]
  runs={}; metas={}
  for label in ORDER:
   run,meta=original_run(observer,inputs[label],PILOT,name,label,original_sha)
   runs[label]=run;metas[label]=meta
  original=logits(runs["clean"],"HGG")
  error=abs(original["pHgg"]-float(old_row["pHggSCEPT"]))
  if error>0.03:
   raise RuntimeError("Original clean 04A source output differs from old original validated 03A P(HGG)")
  # Zero/no-op sham: genuine second forward on the bitwise-same original
  # input, exactly the same original frozen computation (not counted as an
  # additional unique experimental input or archived condition).
  sham=observer.observe(torch_input(inputs["clean"]),model_id="P4_original_frozen_"+name,
    preprocess_id="P1_FLAIR_224_locked_v1",
    preprocess_config={"sourceNPZSha256":original_sha,"patient_id":PILOT,
                       "sliceIndex":0,"split":"val","condition":"sham_noop"})
  equality={}
  for target in (*LAYERS,"model.output"):
   aa=runs["clean"].tensor(target)
   bb=sham.tensor(target)
   equality[target]=bool(np.array_equal(aa,bb))
   if not equality[target]:
    raise RuntimeError("04A genuine duplicate no-op sham model output not numerically reproducible")
  effects={}
  for op in OPS:
   for arm in ("lesion","control"):
    label=f"{op}_{arm}"
    got=logits(runs[label],"HGG")
    roi=selection[arm]
    bylayer={layer:score_tensors(runs["clean"].tensor(layer),
               runs[label].tensor(layer),(roi["row"],roi["col"])) for layer in LAYERS}
    effects[label]={"perturbedOutput":got,
      "signedTrueGradeLogitDelta":got["trueGradeLogit"]-original["trueGradeLogit"],
      "absoluteTrueGradeLogitDelta":abs(got["trueGradeLogit"]-original["trueGradeLogit"]),
      "trueGradeMarginDelta":got["trueGradeMargin"]-original["trueGradeMargin"],
      "pHggDelta":got["pHgg"]-original["pHgg"],
      "inputROIChangeL2":mutation[label]["inputROIChangeL2"],
      "measuredLayers":bylayer}
  paired={}
  for op in OPS:
   a=effects[f"{op}_lesion"];b=effects[f"{op}_control"]
   paired[op]={
    "absoluteTrueGradeLogitDelta_LminusControl":
      a["absoluteTrueGradeLogitDelta"]-b["absoluteTrueGradeLogitDelta"],
    "signedTrueGradeLogitDelta_LminusControl":
      a["signedTrueGradeLogitDelta"]-b["signedTrueGradeLogitDelta"],
    "inputROIL2_LoverControl":a["inputROIChangeL2"]/max(b["inputROIChangeL2"],1e-12),
    "patchRelativeL2_LminusControlByLayer":{
      layer:a["measuredLayers"][layer]["patchRelativeL2"]-
            b["measuredLayers"][layer]["patchRelativeL2"] for layer in LAYERS}}
  report["models"][name]={
    "originalModelSha256":observer.model_sha256,
    "originalCleanOutput":original,
    "original03aPredictionMatchAbsoluteError":error,
    "shamExactEquality":equality,
    "rawSCEPTZIPs":metas,"effects":effects,"matchedPairedComparisons":paired}
  report["nOriginalSCEPTCapturedForwards"]+=5
  report["nAdditionalShamSCEPTForwards"]+=1
  del model,observer,runs,sham
  print(f"04A genuine frozen {name}: 5 original SCEPT ZIPs + 1 no-op sham PASS",flush=True)
 if report["nOriginalSCEPTCapturedForwards"]!=10 or \
    report["nAdditionalShamSCEPTForwards"]!=2:
  raise RuntimeError("Preregistered original 04A pilot model counts incomplete")
 report["status"]="PASS_INPUT_INTERVENTION_PILOT_ENGINEERING"
 write_fixed(OUT/"locked_plan.json",{"plan":PLAN,"planSha256":fingerprint(),
                                     "sourcePatientNPZSha256":original_sha})
 write_fixed(result_path,report)
 print("A6-IJ-04A GENUINE 2-MODEL LESION/CONTROL INPUT-INTERVENTION PILOT: PASS")
 for model,details in report["models"].items():
  print(" MODEL",model,"original frozen clean p(HGG)=",round(details["originalCleanOutput"]["pHgg"],6))
  for op in OPS:
   print("  ",op,"lesion-minus-control |true-grade logit change|=",
         round(details["matchedPairedComparisons"][op]["absoluteTrueGradeLogitDelta_LminusControl"],8))
 print("Scientifically valid evidence dependence:",report["numericScientificConclusions"])
 print("PRIVATE LOCAL:",result_path)
 print("IMPORTANT: archive before the Kaggle session ends.")

if __name__=="__main__":main()
