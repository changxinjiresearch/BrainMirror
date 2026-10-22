"""A6-IJ-04D true frozen ViT post-block internal intervention, one original patient or one chunk.

Exactly 29 *actual* PyTorch inferences per frozen model/patient. Per-patient
source full block output donor NPZ and every intervened genuine logit are
stored in local private scratch; source checksum-validated 04B/04C is never
changed. All model computations in eval/inference_mode; no training.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from a6_ij_02b_measure import checkpoint_model,sha_file
from a6_ij_03b_measure import github,private_contents
from a6_ij_04a_interventions import choose,strict_image
from a6_ij_04b_measure import unpack_val_zip,check_mask_selection,pure_input,canonical,sha
from a6_ij_04b_matched_interventions import matched_inputs,fingerprint04b
from a6_ij_04c_association_private import load as original04b_load,BASE as CBASE
from a6_ij_04d_hook import (
 LAYERS,ARMS,SELECTORS,DIRECTIONS,PLAN,plan_sha,TOL,OPERATOR,DOSE,
 indices_for_roi,capture_original,patched_forward,assert_logits,
 truth_margin,bounded_fraction
)

OUT=Path("/kaggle/working/a6_ij_04d_private")
BASE="results/scept_a6_ij/04d_post_residual_causal_interventions"
STATUS="PASS_04D_ORIGINAL_ONE_PATIENT_58_REAL_FROZEN_COMPUTATION_FORWARDS"
def state_digest(model):
 """Exact read-only frozen in-memory PyTorch weights, including buffers."""
 h=hashlib.sha256()
 for name,t in model.state_dict().items():
  a=t.detach().cpu().contiguous().numpy()
  h.update(name.encode());h.update(str(a.shape).encode())
  h.update(str(a.dtype).encode());h.update(a.tobytes(order="C"))
 return h.hexdigest()
def save_npz(file,arrays):
 if len(arrays)!=12:raise RuntimeError("04D complete clean/lesion/control × S/J × two layers source states required")
 temp=file.with_suffix(".npz.partial")
 with temp.open("wb") as dest:
  np.savez_compressed(dest,**arrays)
 os.replace(temp,file)
 with np.load(file,allow_pickle=False) as f:
  if set(f.files)!=set(arrays):raise RuntimeError("04D original source donor state NPZ member set differs")
  for key in arrays:
   if not np.array_equal(f[key],arrays[key]):
    raise RuntimeError("04D original donor postblock tensor save/read changed")
 return sha_file(file)
def norm_values(result):
 return {"logits":list(map(float,result)),
         "pHgg":float(1/(1+np.exp(np.clip(float(result[0]-result[1]),-500,500))))}

def verify_inputs(selected,source_row,p1):
 if selected["originalFullP1NPZSHA256"]!=p1["p1_sha256"] or \
    source_row["source"]["P1NPZSHA256"]!=p1["p1_sha256"]:
  raise RuntimeError("04D selected source patient full NPZ SHA256 changed")
def patient_measure(item,original,zipf,models):
 patient=item["patient_id"]
 local=OUT/"patients"/patient
 if local.is_dir():
  case=json.loads((local/"case.json").read_text())
  if (case.get("status")!=STATUS or case.get("patient_id")!=patient or
      case.get("planSHA256")!=plan_sha() or
      case["source"]["originalP1NPZSHA256"]!=item["p1_sha256"] or
      set(case.get("models",{}))!={"S","J"} or
      case.get("realForwardCount")!=58 or
      sha_file(local/"postblock_donor_activations.npz")!=case["donorStatesNPZSha256"]):
   raise RuntimeError("04D previously measured private patient is inconsistent")
  print("04D locally original patient reused with donor SHA:",patient,flush=True)
  return case
 work=OUT/"inprogress"/patient
 if work.exists():shutil.rmtree(work)
 work.mkdir(parents=True,exist_ok=True)
 raw=zipf.read(item["zip_member"])
 if sha(raw)!=item["p1_sha256"]:raise RuntimeError("04D original P1 NPZ full SHA invalid")
 with np.load(io.BytesIO(raw),allow_pickle=False) as npz:
  if not {"flair","wt","foreground","axial_indices"}.issubset(npz.files):
   raise RuntimeError("04D original P1 MRI/mask source incomplete")
  image=np.asarray(npz["flair"][0],dtype=np.float32).copy()
  wt=np.asarray(npz["wt"][0]).copy();fg=np.asarray(npz["foreground"][0]).copy()
  axis=int(npz["axial_indices"][0])
 if axis!=int(original["axialIndex"]):
  raise RuntimeError("04D original P1 first-slice axial index differs")
 strict_image(image,wt,fg)
 selection=choose(image,wt,fg)
 check_mask_selection(selection,original)
 orig04b=original["patientFixedROIs"]
 for arm in ARMS:
  for coord in ("row","col"):
   if selection[arm][coord]!=orig04b[arm][coord]:
    raise RuntimeError("04D attempted to select a different tumor/control ROI")
 inp,diagnostics=matched_inputs(image,wt,fg,selection)
 chosen={arm:f"{OPERATOR}_dose50_{arm}" for arm in ARMS}
 if any(abs(diagnostics[chosen[arm]]["actualPixelL2"]-
            original["inputDiagnostics"][chosen[arm]]["actualPixelL2"])>1e-5 for arm in ARMS):
  raise RuntimeError("04D original 04B physical pixel-L2 perturbation mismatch")
 tensors={condition:pure_input(inp[condition],item["grade"])
          for condition in ("clean",*chosen.values())}
 case={
  "schema":"scept.a6-ij-04d.patient.v1",
  "status":"INCOMPLETE",
  "patient_id":patient,"grade":item["grade"],"axialIndex":axis,
  "planSHA256":plan_sha(),"source":{
    "originalP1NPZSHA256":item["p1_sha256"],
    "original04BPlanSHA256":original["planSHA256"],
    "original04BStudyReference":"04b original verified case",
    "original04CFullTensorStatus":"VERIFIED_PRIOR_TO_04D",
    "sealedP5":"NOT_ACCESSED"},
  "origROI":selection,
  "actualSourceInputEqualL2":{
    arm:diagnostics[chosen[arm]] for arm in ARMS},
  "models":{},"realForwardCount":0,
  "interpretation":"causal effect of an engineered hidden activation substitution only, not tumor medical causality"
 }
 donated={}
 for model_name in ("S","J"):
  model=models[model_name]
  if model.training:raise RuntimeError("04D frozen model must remain eval")
  baseline={}
  activations={}
  for condition in ("clean",*chosen.values()):
   actual,states=capture_original(model,tensors[condition])
   source_logits=original["models"][model_name]["originalObservations"][condition]["output"]["logits"]
   assert_logits(actual,source_logits,"same-frozen-model "+condition)
   baseline[condition]=norm_values(actual)
   activations[condition]=states
   for layer in LAYERS:
    a=states[layer].cpu().numpy()
    if a.shape!=(1,197,768) or not np.isfinite(a).all():
     raise RuntimeError("04D authentic postblock donor states malformed")
    donated[f"{model_name}_{condition}_{layer.replace('.','_')}"]=a.copy()
  case["realForwardCount"]+=3
  results={}
  selectors={arm:indices_for_roi(selection,arm) for arm in ARMS}
  for arm in ARMS:
   condition=chosen[arm]
   clean_margin=truth_margin(baseline["clean"]["logits"],item["grade"])
   edited_margin=truth_margin(baseline[condition]["logits"],item["grade"])
   for layer in LAYERS:
    for selector in SELECTORS:
     indices=selectors[arm][selector]
     effects={}
     for direction in DIRECTIONS:
      if direction=="rescue":
       recipient=tensors[condition]; donor=activations["clean"][layer]
       original_recipient=baseline[condition]["logits"]
      else:
       recipient=tensors["clean"];donor=activations[condition][layer]
       original_recipient=baseline["clean"]["logits"]
      result=patched_forward(model,recipient,layer,donor,indices)
      actual_margin=truth_margin(result,item["grade"])
      frac=bounded_fraction(clean_margin,edited_margin,actual_margin,direction)
      if selector=="all":
       expected=(baseline["clean"]["logits"] if direction=="rescue"
                 else baseline[condition]["logits"])
       absolute_error=assert_logits(result,expected,
           "post-block full donor recovery/insertion "+direction)
      else:
       absolute_error=None
      effects[direction]={
        "patchedLogits":norm_values(result),
        "signedTrueGradeMargin":actual_margin,
        "deltaMarginFromRecipient":float(actual_margin-truth_margin(original_recipient,item["grade"])),
        "sourceTokenIndices":list(indices) if selector!="all" else "ALL_197",
        "fractionOfOriginalMarginDifference":frac,
        "fullPositiveControlLogitAbsoluteError":absolute_error
      }
      case["realForwardCount"]+=1
     results[f"{arm}|{layer}|{selector}"]=effects
  shams={}
  for layer in LAYERS:
   idx=selectors["lesion"]["target"]
   sham=patched_forward(model,tensors["clean"],layer,activations["clean"][layer],idx)
   error=assert_logits(sham,baseline["clean"]["logits"],"same-source self-patch sham "+layer)
   shams[layer]={"maxAbsoluteLogitError":error,"isSameOriginalCleanSource":True}
   case["realForwardCount"]+=1
  if len(results)!=2*len(LAYERS)*len(SELECTORS) or \
     case["realForwardCount"]!=29*(1 if model_name=="S" else 2):
   raise RuntimeError("04D original bidirectional graph patch forward coverage incomplete")
  case["models"][model_name]={
    "originalUnpatchedSourceLogits":baseline,
    "bidirectionalPostBlockInterventions":results,
    "sameSourceSelfPatchShams":shams,
    "originalFrozenWeightsVerified":"checkpoint hash at loading and per-complete-chunk model memory digest",
    "nActualFwd":29}
 npz=work/"postblock_donor_activations.npz"
 case["donorStatesNPZSha256"]=save_npz(npz,donated)
 case["donorStateTensorCount"]=len(donated)
 if case["realForwardCount"]!=58:
  raise RuntimeError("04D no patient may pass without 58 physical model forwards")
 case["status"]=STATUS
 (work/"case.json").write_bytes(canonical(case))
 local.parent.mkdir(parents=True,exist_ok=True)
 os.replace(work,local)
 print("A6-IJ-04D REAL 58 FROZEN S/J DIRECT RESIDUAL GRAPH FORWARDS: PASS",patient,flush=True)
 return case

def prepare():
 s=github()
 patients,original04bsha,final04b,_=original04b_load(s)
 if len(patients)!=33:raise RuntimeError("04D 33 original source patient cohort incomplete")
 prior=json.loads(private_contents(s,f"{CBASE}/final/analysis.json"))
 receipt=json.loads(private_contents(s,f"{CBASE}/final/archive_receipt.json"))
 if (prior.get("status")!="PASS_A6_IJ_04C_33_PATIENT_594_GENUINE_ORIGINAL_FULL_TENSOR_OBSERVATIONAL_AUDIT"
     or prior.get("nIndependentOriginalP0ValidationPatients")!=33
     or prior.get("nAuthenticOriginalViTOutputFullTensorReplays")!=594
     or receipt.get("complete04COriginalReportFullSHA256")!=sha(canonical(prior))):
  raise RuntimeError("04D full independently verified 04C source not completed")
 return s,patients,original04bsha,final04b

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--chunk",type=int,choices=range(1,8),required=True)
 p.add_argument("--pilot",action="store_true")
 args=p.parse_args()
 if args.pilot and args.chunk!=1:raise RuntimeError("04D pilot may only use original first source patient")
 torch.set_num_threads(min(4,torch.get_num_threads()))
 s,people,studysha,study=prepare()
 val,zip_path=unpack_val_zip(s)
 known={v["patient_id"]:v for v in val}
 selected=people[(args.chunk-1)*5:args.chunk*5]
 if len(selected)!=(5 if args.chunk<7 else 3):
  raise RuntimeError("04D old eligible 5+...+3 partition unexpectedly changed")
 for r in selected:
  item=known.get(r["patient_id"])
  if item is None or item["grade"]!=r["grade"] or item["p1_sha256"]!=r["source"]["P1NPZSHA256"]:
   raise RuntimeError("04D original P0/P1 source lock changed")
 locked={"schema":"scept.a6-ij-04d.sources.v1","plan":PLAN,
         "planSHA256":plan_sha(),"04BOriginalAnalysisSHA256":studysha,
         "04COriginalPatientIDs":[q["patient_id"] for q in people],
         "sealedTest":"NOT_ACCESSED"}
 OUT.mkdir(parents=True,exist_ok=True)
 save=OUT/"locked_plan.json"
 if save.exists():
  if save.read_bytes()!=canonical(locked):
   raise RuntimeError("Original source frozen 04D local lock differs")
 else:save.write_bytes(canonical(locked))
 models={m:checkpoint_model(m) for m in ("S","J")}
 weight_hashes={k:state_digest(v) for k,v in models.items()}
 with ZipFile(zip_path) as z:
  for idx,original in enumerate(selected[:1] if args.pilot else selected,1):
   patient_measure(known[original["patient_id"]],original,z,models)
   print(f"A6-IJ-04D CHUNK {args.chunk:02d} LOCAL patient {idx}/{1 if args.pilot else len(selected)} source SHA and causal controls PASS",flush=True)
 for k,m in models.items():
  if state_digest(m)!=weight_hashes[k]:
   raise RuntimeError("04D original frozen ViT in-memory parameters CHANGED; reject outputs")
 print("A6-IJ-04D "+("ONE-PATIENT PILOT ONLY" if args.pilot else
     f"CHUNK {args.chunk:02d} ALL 58-FORWARD REAL CASES MEASURED; ARCHIVE NEXT"),flush=True)

if __name__=="__main__":main()
