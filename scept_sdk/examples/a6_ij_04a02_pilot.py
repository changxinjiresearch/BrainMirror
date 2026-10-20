"""A6-IJ-04A02 EXPLORATORY new-patient P1 VAL MRI/S+J frozen ViT pilot.

Original v1 022 BLOCKED preserved. Fixed NEW 040 chosen by the locked full-55 input-only screen, NEVER by model outcome. Original rank1. Two pre-output-selected ViT-aligned
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
from a6_ij_03b_measure import github,API
from a6_ij_04a_interventions import (
 PLAN,OPS,fingerprint,choose,mutate,score_tensors,strict_image
)

OUT=ROOT/"a6_ij_04a02_private"
PILOT="BraTS20_Training_040"
SOURCE="results/scept_a6_ij/03a_exact_first_slice_pairing/analysis.json"
SCREEN="results/scept_a6_ij/04a02_original_mask_only_feasibility/full_val_first_slice_screen.json"
OLD_BLOCKED="results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json"
ORIGINAL_SCREEN_GIT_BLOB_SHA="fe39ebcc9bc323aa260c6cec4fd497287971d06c"
PILOT_SOURCE_P1_SHA256="b6737737a34cccfb466f8091b823c98952e7c473f638e3376de93c9ec67e177a"
AMENDMENT={
 "schema":"scept.a6-ij-04a02.amended-exploratory-input-intervention.v1",
 "original04Av1Outcome":"BLOCKED_MATCHING_SCORE_3.14177_GT_LOCKED_2.0",
 "prior04A03aResultMutations":"FORBIDDEN",
 "selection":"first originally eligible P0 VAL first P1 slice by patient ID in full 55 mask-only original-v1 screen",
 "screenOriginalGitBlobSHA":ORIGINAL_SCREEN_GIT_BLOB_SHA,
 "screenOriginalPatients":55,"originalScreenEligible":33,
 "newPilot":PILOT,"originalP1FullSHA256":PILOT_SOURCE_P1_SHA256,
 "sameOriginalROIAndInterventionRuleSha256":fingerprint(),
 "originalROIMatchScoreUpperLimit":2.0,
 "test":"NOT_ACCESSED","external":"NOT_ACCESSED",
 "scientificStatus":"exploratory technical input intervention, not independent test, neuron causal proof or validated clinical evidence dependence"
}
def amendment_fingerprint():
 return digest(json.dumps(AMENDMENT,sort_keys=True,separators=(",",":")).encode())
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
   "study":"a6-ij-04a02","amendmentSha256":amendment_fingerprint(),"explicitlyNoTraining":True})
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
 result_path=OUT/"a6_ij_04a02_pilot_report.json"
 if result_path.exists():
  source=json.loads(result_path.read_text())
  if source.get("status")=="PASS_AMENDED_INPUT_INTERVENTION_PILOT_ENGINEERING" and \
     source.get("amendmentPlanSha256")==amendment_fingerprint():
   print("A6-IJ-04A02 new candidate already complete locally; archive now",flush=True)
   return
  raise RuntimeError("Existing incompatible 04A local record cannot be overwritten")
 s=github()
 # Pin exact original full-55, pre-model screen to its original immutable Git blob.
 screen_response=s.get(f"{API}/contents/{SCREEN}",params={"ref":"main"},timeout=(30,120))
 screen_response.raise_for_status()
 if screen_response.json().get("sha")!=ORIGINAL_SCREEN_GIT_BLOB_SHA:
  raise RuntimeError("Amendment original eligibility screen Git blob SHA was changed")
 screen_bytes=private_contents(s,SCREEN)
 screen=json.loads(screen_bytes)
 if (screen.get("status")!="PASS_COMPLETE_MASK_ONLY_SCREEN" or
     screen.get("nScreened")!=55 or screen.get("nEligible")!=33 or
     screen.get("original04APlanSHA256")!=fingerprint() or
     screen.get("exploratoryNextPilot",{}).get("patient_id")!=PILOT or
     screen.get("exploratoryNextPilot",{}).get("sourceP1FullSHA256")!=PILOT_SOURCE_P1_SHA256):
  raise RuntimeError("Amended 04A02 candidate not determined by original image-only screening")
 all_patients=screen["patientRecords"]
 if (len(all_patients)!=55 or
     len({q["patient_id"] for q in all_patients})!=55 or
     next(q["patient_id"] for q in all_patients
          if q["status"]=="ELIGIBLE_UNDER_UNCHANGED_V1_RULE")!=PILOT):
  raise RuntimeError("Original sorted first eligible person proof differs")
 previous=json.loads(private_contents(s,OLD_BLOCKED))
 if (previous.get("status")!="BLOCKED_FIXED_PATIENT_ROI_ELIGIBILITY" or
     previous.get("patient_id")!="BraTS20_Training_022" or
     previous.get("failedFixedCriterion")!="04A_MATCHING_SCORE_EXCEEDED: score=3.14177 threshold=2.0"):
  raise RuntimeError("Original 04A fixed original 022 failure cannot be rewritten")
 screen_case=next(q for q in all_patients if q["patient_id"]==PILOT)
 if (screen_case["status"]!="ELIGIBLE_UNDER_UNCHANGED_V1_RULE" or
     screen_case["originalFullP1NPZSHA256"]!=PILOT_SOURCE_P1_SHA256 or
     screen_case["grade"]!="HGG" or screen_case["sliceRank"]!=1):
  raise RuntimeError("Exact original 040 first-slice image-only eligibility missing")
 cohort,zip_path=manifest_and_zip(s)
 case=next((x for x in cohort if x["patient_id"]==PILOT),None)
 if (case is None or len(cohort)!=55 or case["grade"]!="HGG" or
     case["p1_sha256"]!=PILOT_SOURCE_P1_SHA256 or
     [q["patient_id"] for q in cohort]!=[q["patient_id"] for q in all_patients]):
  raise RuntimeError("Original VAL patient count/cohort/same 040 source differs")
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
 selection=choose(image,wt,fg)
 # No cherry picking: a new result must reproduce the exact ORIGINAL
 # image/mask-only ROI selected before seeing any S/J network responses.
 prior_roi=screen_case["regionSelection"]
 for arm in ("lesion","control"):
  for key in ("row","col"):
   if int(selection[arm][key])!=int(prior_roi[arm][key]):
    raise RuntimeError("Original image-only paired ROI selection differs")
 if (abs(float(selection["matchScore"])-float(prior_roi["matchScore"]))>1e-9 or
     selection["matchScore"]>2.0 or
     selection["lesion"]["wtFraction"]<.25 or
     selection["control"]["dilatedWTFraction"]!=0):
  raise RuntimeError("Original 04A-v1 input-only matching scientific constraints changed")
 inputs={"clean":image}
 mutation={}
 for op in OPS:
  for arm in ("lesion","control"):
   key=f"{op}_{arm}"
   changed,details=mutate(image,wt,fg,selection,arm,op)
   inputs[key]=changed
   mutation[key]=details
   print("04A02 region",key,"source delta L2=",round(details["inputROIChangeL2"],6),flush=True)
 if tuple(inputs)!=ORDER:raise RuntimeError("04A exact paired operation order unexpectedly changed")
 original_sha=digest(raw)
 report={
  "schema":"scept.a6-ij-04a02.pilot.v1",
  "status":"INCOMPLETE",
  "scope":"EXPLORATORY_AMENDED_ENGINEERING_INPUT_INTERVENTION_ONLY; NOT NEURON_CAUSAL_OR_CLINICAL",
  "interventionPlan":PLAN,"interventionPlanSha256":fingerprint(),
  "amendmentPlan":AMENDMENT,"amendmentPlanSha256":amendment_fingerprint(),
  "patient_id":PILOT,"grade":"HGG","split":"val","sliceRank":1,
  "axial_index":int(axes[0]),
  "source":{"originalP1NPZFullSHA256":original_sha,
      "03aOriginalAnalysisSHA256":digest(source_bytes),
      "originalImageOnlyScreenSHA256":digest(screen_bytes),
      "originalImageOnlyScreenGitBlobSHA":ORIGINAL_SCREEN_GIT_BLOB_SHA,
      "original04Av1BlockedRetained":True,
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
   raise RuntimeError("Original clean 04A02 source output differs from old original validated 03A P(HGG)")
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
    raise RuntimeError("04A02 genuine duplicate no-op sham model output not numerically reproducible")
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
  print(f"04A02 genuine frozen {name}: 5 original SCEPT ZIPs + 1 no-op sham PASS",flush=True)
 if report["nOriginalSCEPTCapturedForwards"]!=10 or \
    report["nAdditionalShamSCEPTForwards"]!=2:
  raise RuntimeError("Preregistered original 04A pilot model counts incomplete")
 report["status"]="PASS_AMENDED_INPUT_INTERVENTION_PILOT_ENGINEERING"
 write_fixed(OUT/"locked_plan.json",{"plan":PLAN,"planSha256":fingerprint(),
                                     "amendment":AMENDMENT,"amendmentPlanSha256":amendment_fingerprint(),
                                     "screenSha256":digest(screen_bytes),
                                     "screenOriginalGitBlobSHA":ORIGINAL_SCREEN_GIT_BLOB_SHA,
                                     "sourcePatientNPZSha256":original_sha})
 write_fixed(result_path,report)
 print("A6-IJ-04A02 AMENDED 040 INPUT-INTERVENTION PILOT ENGINEERING: PASS")
 for model,details in report["models"].items():
  print(" MODEL",model,"original frozen clean p(HGG)=",round(details["originalCleanOutput"]["pHgg"],6))
  for op in OPS:
   print("  ",op,"lesion-minus-control |true-grade logit change|=",
         round(details["matchedPairedComparisons"][op]["absoluteTrueGradeLogitDelta_LminusControl"],8))
 print("Scientifically valid evidence dependence:",report["numericScientificConclusions"])
 print("PRIVATE LOCAL:",result_path)
 print("IMPORTANT: archive before the Kaggle session ends.")

if __name__=="__main__":main()
