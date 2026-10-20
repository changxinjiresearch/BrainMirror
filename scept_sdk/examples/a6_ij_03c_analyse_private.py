"""A6-IJ-03C: locked 55-patient leave-one-out incremental association experiment.

Three FIXED non-tuned ridge arms B0, B1 (including exact 10-slice raw RMS),
F (add 9 real SCEPT structural variables). All 55 patient records train/hold
out by PATIENT, not slice. 100 within-grade permuted negative controls,
2000 paired patient fixed-OOF bootstrap resamples, immutable private report.
No neural inference, no Grad-CAM recomputation, no causal/clinical claims.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from a6_ij_03b_measure import github,private_contents,LAYERS,PLAN as P03B
from a6_ij_03c_extract_rms_private import BASE,sha,commit_once

ALPHA=10.0
PERMUTATIONS=100
BOOTSTRAPS=2000
SEED=240216
PREFIX="results/scept_a6_ij/03b_full_slice_pairing"
FEATURES={
 "B0":["hgg","meanLesionReferenceFraction","meanPHgg_S","meanPHgg_J"],
 "B1_RMS":["rmsJminusS_Block0","rmsJminusS_Block5","rmsJminusS_Block11"],
 "F_STRUCTURE":[f"{layer}_{k}" for layer in LAYERS
               for k in ("patchMeanOffDiagonalCosine","patchEffectiveRankNormalized","crossModelCKA")]
}
PLAN={
 "schema":"scept.a6-ij-03c.analytic-plan.v1",
 "patients":55,"grades":{"HGG":44,"LGG":11},"source03b":"PASS_EXACT_55_PATIENTS_1100_PAIRED_MODELSLICE_RECORDS",
 "sourceRaw":"11 original 02B/03B source-SHA-verified private RMS batches, 1100 real SCEPT original tensor archives",
 "aggregation":"mean 10 slices within same person and model, then J-S where appropriate",
 "primaryEndpoint":"patient-mean-IoU10-J-minus-S","secondaryEndpoint":"patient-mean-TAP-J-minus-S",
 "regression":"ridge regression, training-fold-only z scores and mean-centered target, intercept unpenalised",
 "alpha":ALPHA,"folds":55,"foldStrategy":"leave exactly one complete person out",
 "baselineB0":FEATURES["B0"],
 "baselineB1additional":FEATURES["B1_RMS"],
 "structuralFullAdditional":FEATURES["F_STRUCTURE"],
 "primaryIncrement":"MAE(B1)-MAE(F); higher means lower held-out prediction error",
 "patientBootstrapResamples":BOOTSTRAPS,"withinGradePermutationCount":PERMUTATIONS,
 "seed":SEED,
 "negativeControl":"permute 9-column person-level SCEPT structural rows together within grade; B1/y unchanged",
 "sourceData":"VALIDATION only, no independence/causal interpretation",
 "sealedP5":"NOT_ACCESSED","externalP6":"NOT_ACCESSED"
}
def plan_sha():return sha(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode())
def numeric(value):
 x=float(value)
 if not np.isfinite(x):raise RuntimeError("Nonfinite original study measurement")
 return x
def ridge_oof(x,y,alpha=ALPHA):
 x=np.asarray(x,dtype=np.float64)
 y=np.asarray(y,dtype=np.float64)
 n,p=x.shape
 if n!=55 or y.shape!=(55,) or p<1 or not np.isfinite(x).all() or not np.isfinite(y).all():
  raise RuntimeError("Only complete original 55 patients permitted; no missing-value imputation")
 ret=np.empty(n,dtype=np.float64)
 for test in range(n):
  tr=np.arange(n)!=test
  xt=x[tr]
  mean=xt.mean(axis=0)
  std=xt.std(axis=0)
  std=np.where(std<1e-12,1.0,std)
  z=(xt-mean)/std
  k=(x[test]-mean)/std
  yc=y[tr]-y[tr].mean()
  beta=np.linalg.solve(z.T@z+alpha*np.eye(p),z.T@yc)
  ret[test]=y[tr].mean()+k@beta
 if not np.isfinite(ret).all():raise RuntimeError("Nonfinite held-out patient predictions")
 return ret
def skill(y,pred):
 y=np.asarray(y,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
 residual=y-pred
 rss=float(np.sum(residual**2))
 tss=float(np.sum((y-y.mean())**2))
 return {"MAE":float(np.mean(np.abs(residual))),
         "RMSE":float(np.sqrt(np.mean(residual**2))),
         "R2AgainstOverallMean":float(1-rss/tss) if tss>1e-18 else None}
def bootstrap_residual_improvement(y,p_b1,p_f,rng):
 # Fixed original OOF predictions; resamples patients but DOES NOT refit ridge
 diff=np.abs(y-p_b1)-np.abs(y-p_f)
 idx=rng.integers(0,55,size=(BOOTSTRAPS,55))
 means=diff[idx].mean(axis=1)
 return {"meanImprovementMAE":float(diff.mean()),
         "patientBootstrap95":list(map(float,np.quantile(means,[.025,.975]))),
         "method":"paired patient bootstrap of fixed out-of-fold residuals, models not refit"}
def records(s):
 final_raw=private_contents(s,PREFIX+"/final/analysis.json")
 result=json.loads(final_raw)
 if result.get("status")!="PASS_EXACT_55_PATIENTS_1100_PAIRED_MODELSLICE_RECORDS" or \
    result.get("nPatients")!=55 or result.get("nMatchedModelSliceRecords")!=1100 or \
    result.get("new03BRealSCEPTModelRuns")!=990 or result.get("originalFirstSlice02BReusedModelRuns")!=110:
  raise RuntimeError("Not the authoritative complete 03B private source")
 r3b=json.loads(private_contents(s,PREFIX+"/final/archive_receipt.json"))
 if r3b.get("status")!="PRIVATE_COMPLETE_1100_PAIRED_SLICES_990_RAW_VERIFIED" or \
    r3b.get("analysisSHA256")!=sha(final_raw) or r3b.get("nVerifiedRemotePrivateShards")!=11:
  raise RuntimeError("Original 03B SHA-verified immutable final receipt mismatch")
 by_id={}
 rms_sha={}
 for n in range(1,12):
  original=private_contents(s,f"{PREFIX}/chunks/chunk_{n:02d}/paired_rows.json")
  receipt=json.loads(private_contents(s,f"{PREFIX}/chunks/chunk_{n:02d}/archive_receipt.json"))
  if sha(original)!=receipt["originalReportSHA256"]:
   raise RuntimeError("03B original model-slice matched records SHA mismatch")
  paired=json.loads(original)
  rows_raw=private_contents(s,f"{BASE}/rms/chunk_{n:02d}.json")
  rms=json.loads(rows_raw)
  if rms.get("status")!="PASS_5_PATIENTS_100_RAW_RMS_VALUES" or rms.get("chunk")!=n or \
     rms.get("03bOriginalShardSHA256")!=receipt["assetSha256"] or \
     rms.get("03bPairedSourceSHA256")!=sha(original) or \
     rms.get("nModelSliceValues")!=100:
   raise RuntimeError("RMS data not from original 03B patient records/raw ZIPs")
  c={r["patient_id"]:r for r in paired["pairedPatientRows"]}
  v={r["patient_id"]:r for r in rms["patientRows"]}
  if set(c)!=set(v) or set(c)!=set(receipt["patientIds"]) or len(c)!=5:
   raise RuntimeError("A source five-person chunk changed patient identities")
  for pid in receipt["patientIds"]:
   orig=c[pid];x=v[pid]
   if (pid in by_id or orig["grade"]!=x["grade"] or
       len(orig["slicePairs"])!=10 or len(x["sliceRows"])!=10):
    raise RuntimeError("03C patient/correspondence mismatch")
   for k in range(10):
    src=orig["slicePairs"][k];amps=x["sliceRows"][k]
    if src["slice_rank"]!=k+1 or amps["slice_rank"]!=k+1:
     raise RuntimeError("Raw output rank does not correspond to Grad-CAM source rank")
    for model in ("S","J"):
     fp=amps["models"][model];node=src["models"][model]["SCEPT"]
     if set(fp["patchRMS"])!=set(LAYERS):
      raise RuntimeError("Original source RMS layer set altered")
     if k>0 and fp["originalZipSHA256"]!=node["originalZip"]["sha256"]:
      raise RuntimeError("Measured source rms derived from DIFFERENT original patient ZIP")
   by_id[pid]={"paired":orig,"rms":x}
  rms_sha[f"chunk_{n:02d}"]=sha(rows_raw)
 ids=sorted(by_id)
 if len(ids)!=55 or len(result["privatePatientAggregates"])!=55 or \
     set(ids)!=set(result["privatePatientAggregates"]):
  raise RuntimeError("Original 55-person final study patient list not matched")
 rows=[]
 for pid in ids:
  c=by_id[pid]["paired"]
  r=by_id[pid]["rms"]
  p=c["slicePairs"];q=r["sliceRows"]
  base={"hgg":float(c["grade"]=="HGG"),
        "meanLesionReferenceFraction":float(np.mean([
            x["models"]["S"]["GradCAM"]["referenceFraction"] for x in p])),
        "meanPHgg_S":float(np.mean([x["models"]["S"]["GradCAM"]["pHgg"] for x in p])),
        "meanPHgg_J":float(np.mean([x["models"]["J"]["GradCAM"]["pHgg"] for x in p]))}
  for x in p:
   if abs(x["models"]["S"]["GradCAM"]["referenceFraction"]-
          x["models"]["J"]["GradCAM"]["referenceFraction"])>1e-10:
    raise RuntimeError("Original pathology reference differs across model for same source image")
  for i,layer in enumerate(LAYERS):
   suffix=("Block0","Block5","Block11")[i]
   base["rmsJminusS_"+suffix]=float(np.mean([
    x["models"]["J"]["patchRMS"][layer]-x["models"]["S"]["patchRMS"][layer]
    for x in q]))
  from_final=result["privatePatientAggregates"][pid]
  for layer in LAYERS:
   for metric in ("patchMeanOffDiagonalCosine","patchEffectiveRankNormalized"):
    feature=layer+"_"+metric
    base[feature]=float(np.mean([
       x["models"]["J"]["SCEPT"]["layers"][layer][metric]-
       x["models"]["S"]["SCEPT"]["layers"][layer][metric] for x in p]))
    if abs(base[feature]-from_final[feature])>1e-10:
     raise RuntimeError("Original 03B registered patient S-J structural feature recomputation differs")
   feature=layer+"_crossModelCKA"
   base[feature]=float(np.mean([x["crossModelCKA"][layer] for x in p]))
   if abs(base[feature]-from_final[feature])>1e-10:
    raise RuntimeError("Original cross-model CKA patient mean differs")
  targets={}
  for column,source in (("iou10","iou10"),("tap","tap")):
   targets[column]=float(np.mean([
     x["models"]["J"]["GradCAM"][source]-x["models"]["S"]["GradCAM"][source]
     for x in p]))
   if abs(targets[column]-from_final[column])>1e-10:
    raise RuntimeError("Original 03B Grad-CAM patient aggregated target differs")
  allvars=base
  if set(allvars)!=set(FEATURES["B0"]+FEATURES["B1_RMS"]+FEATURES["F_STRUCTURE"]):
   raise RuntimeError("One or more preregistered patient variables missing")
  if not np.isfinite(list(allvars.values())).all() or not np.isfinite(list(targets.values())).all():
   raise RuntimeError("Missing or non-finite model covariates/outcomes")
  rows.append({"patient_id":pid,"grade":c["grade"],"features":base,"outcomes":targets})
 return rows,{"03bFinalOriginalSHA256":sha(final_raw),
              "03bFinalReceiptSHA256":sha(private_contents(s,PREFIX+"/final/archive_receipt.json")),
              "03cRMSChunksOriginalSHA256":rms_sha}

def analyse(rows,source):
 ids=[r["patient_id"] for r in rows]
 if len(rows)!=55 or len(set(ids))!=55:
  raise RuntimeError("Population changed; independent sample is original 55 patients")
 feature_order=FEATURES["B0"]+FEATURES["B1_RMS"]+FEATURES["F_STRUCTURE"]
 matrix=np.array([[r["features"][name] for name in feature_order] for r in rows],dtype=np.float64)
 b0=matrix[:,:4];b1=matrix[:,:7];full=matrix
 grades=np.array([r["grade"] for r in rows])
 if np.count_nonzero(grades=="HGG")!=44 or np.count_nonzero(grades=="LGG")!=11:
  raise RuntimeError("Locked grade distribution changed")
 results={}
 for outcome_idx,key in enumerate(("iou10","tap")):
  y=np.array([r["outcomes"][key] for r in rows],dtype=np.float64)
  pred={"B0":ridge_oof(b0,y),"B1":ridge_oof(b1,y),"F":ridge_oof(full,y)}
  rng=np.random.default_rng(SEED+outcome_idx)
  metrics={k:skill(y,v) for k,v in pred.items()}
  main=bootstrap_residual_improvement(y,pred["B1"],pred["F"],rng)
  secondary={"B0minusB1":metrics["B0"]["MAE"]-metrics["B1"]["MAE"],
             "B0minusF":metrics["B0"]["MAE"]-metrics["F"]["MAE"]}
  perm_rng=np.random.default_rng(SEED+1000+outcome_idx)
  permdeltas=[]
  for _ in range(PERMUTATIONS):
   shuffled=full.copy()
   for grade in ("HGG","LGG"):
    ix=np.flatnonzero(grades==grade)
    shuffled[ix,7:]=full[perm_rng.permutation(ix),7:]
   f_pred=ridge_oof(shuffled,y)
   permdeltas.append(float(metrics["B1"]["MAE"]-skill(y,f_pred)["MAE"]))
  permdeltas=np.asarray(permdeltas)
  compare=float(main["meanImprovementMAE"])
  negative={"permutations":PERMUTATIONS,
      "source":"within each grade, shuffle entire 9-dimensional SCEPT structural vector by patient; preserve B1+target",
      "meanShuffledIncrement":float(permdeltas.mean()),
      "shuffled95Quantile":list(map(float,np.quantile(permdeltas,[.025,.975]))),
      "fractionShuffledGreaterEqualOriginal":float(np.mean(permdeltas>=compare)),
      "exploratoryOnly":"Not a calibrated or multiplicity-adjusted hypothesis test"}
  strat={}
  for grade in ("HGG","LGG"):
   mask=grades==grade
   dif=np.abs(y[mask]-pred["B1"][mask])-np.abs(y[mask]-pred["F"][mask])
   strat[grade]={"nPatients":int(np.sum(mask)),"meanHeldOutAbsoluteErrorGain":float(dif.mean()),
                 "MAE_B1":float(np.mean(np.abs(y[mask]-pred["B1"][mask]))),
                 "MAE_F":float(np.mean(np.abs(y[mask]-pred["F"][mask])))}
  predictions=[{"patient_id":pid,"grade":r["grade"],"observed":float(y[i]),
                "B0":float(pred["B0"][i]),"B1":float(pred["B1"][i]),"F":float(pred["F"][i]),
                "originalFixedOOFErrorGain":float(abs(y[i]-pred["B1"][i])-abs(y[i]-pred["F"][i]))}
               for i,(pid,r) in enumerate(zip(ids,rows))]
  results[key]={"status":"EXECUTED_UNCONDITIONALLY","LOPO":metrics,
    "primaryB1minusF":main,"secondary":secondary,
    "withinGradeNegativeControls":negative,"gradeStratification":strat,
    "originalPatientHeldOutPredictions":predictions}
 return {"schema":"scept.a6-ij-03c.analysis.v1",
    "status":"PASS_FIXED_55_PATIENT_LOPO_RMS_CONTROL_AND_SHUFFLE_EXECUTED",
    "plan":PLAN,"planSHA256":plan_sha(),
    "sourceOriginalSHA256":source,"nPatients":55,
    "nFullMatchedGradCAMModelSlices":1100,"nRMSOriginalSCEPTSourceRuns":1100,
    "dataRows":rows,"endpoints":results,
    "limits":["Validation cohort and true-grade-logit Grad-CAM target, not clinical or causal fidelity.",
    "With 55 patients and 16 predictors, LOPO fit uncertainty may be substantial.",
    "Bootstrap resamples fixed held-out residuals, not model re-training uncertainty.",
    "Grade-constrained negative controls and two endpoints are exploratory, not corrected p-values.",
    "Same model outputs/probabilities and lesion fraction may be mutually related to target.",
    "No P5/P6 or new model inference or automatically established superiority."]}

def main():
 s=github()
 original_path=f"{BASE}/analysis.json"
 already=s.get("https://api.github.com/repos/changxinjiresearch/research-course-brats2020-ijepa-xai/contents/"+original_path,
               timeout=(30,120))
 if already.status_code==200:
  original=json.loads(private_contents(s,original_path))
  if original.get("status")=="PASS_FIXED_55_PATIENT_LOPO_RMS_CONTROL_AND_SHUFFLE_EXECUTED" and \
      original.get("planSHA256")==plan_sha():
   print("A6-IJ-03C PRIVATE ORIGINAL 55-PATIENT ANALYSIS ALREADY COMPLETE",flush=True)
   return
  raise RuntimeError("Existing 03C research original is incompatible; refuse overwrite")
 if already.status_code!=404:
  raise RuntimeError("Cannot safely inspect prior private 03C analysis")
 rows,sources=records(s)
 result=analyse(rows,sources)
 raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 commit_once(s,original_path,raw)
 print("A6-IJ-03C 55-PATIENT INCREMENTAL + NEGATIVE CONTROLS: PASS",flush=True)
 print("Full 11 source RMS chunks, 1100 real old SCEPT ZIPs / no new neural model inference",flush=True)
 for key in ("iou10","tap"):
  test=result["endpoints"][key]
  delta=test["primaryB1minusF"]
  print(f"{key} original held-out B1 MAE={test['LOPO']['B1']['MAE']:.7f}, "
        f"F MAE={test['LOPO']['F']['MAE']:.7f}, "
        f"MAE improvement B1-F={delta['meanImprovementMAE']:+.7f}, "
        f"95% paired fixed-OOF bootstrap={delta['patientBootstrap95']}",flush=True)
  print(f"{key} prelocked 100 grade-shuffled controls; fraction >= observed: "
        f"{test['withinGradeNegativeControls']['fractionShuffledGreaterEqualOriginal']:.3f}",flush=True)
 print("PRIVATE ORIGINAL:",f"https://github.com/changxinjiresearch/research-course-brats2020-ijepa-xai/blob/main/{original_path}")

if __name__=="__main__":main()
