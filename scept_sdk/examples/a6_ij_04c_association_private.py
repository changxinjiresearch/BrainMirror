"""A6-IJ-04C-01: pre-locked 33-person SCEPT internal/decision association.

Reads ONLY pre-existing SHA-validated PRIVATE A6-IJ-04B summary batches.
No MRI or neural model inference, no new Grad-CAM, no patient public upload.
04C-01 is observational. Its pass is NOT causal explanation fidelity.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.stats import rankdata
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_03b_archive_chunk_private import commit_once

B="results/scept_a6_ij/04b_equal_l2_interventions"
BASE="results/scept_a6_ij/04c_representation_decision_association"
L=("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")
OP=("gaussian_local_sigma3","healthy_pool_median_fill")
DOSE=(25,50)
N=33
R=2000
SEED=240216
PRIMARY="gaussian_local_sigma3_dose50"
TARGET_LAYER="blocks.11.norm1"
FEATURES=("directTargetedTokenDeltaL2","patchRelativeL2","patchDeltaRMS",
          "directTargetedTokenCosineCleanVsPerturbed")
STATUS="PASS_A6_IJ_04C01_33_PATIENT_ORIGINAL_SUMMARY_ASSOCIATION"
def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(data):return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def corr(a,b,kind="spearman"):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():
  raise RuntimeError("Original 04C independent case/finite shape broken")
 if kind=="spearman":a=rankdata(a);b=rankdata(b)
 a=a-a.mean();b=b-b.mean()
 denom=float(np.linalg.norm(a)*np.linalg.norm(b))
 return None if denom<=1e-14 else float(np.dot(a,b)/denom)
def boot_and_shuffle(a,b,seed=SEED):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 observed=corr(a,b)
 if observed is None:return {"observedSpearman":None,"reason":"ZERO_VARIANCE_OR_UNIDENTIFIABLE"}
 rng=np.random.default_rng(seed)
 draws=[]
 invalid=0
 for _ in range(R):
  ids=rng.integers(0,len(a),size=len(a))
  v=corr(a[ids],b[ids])
  if v is None:invalid+=1
  else:draws.append(v)
 if not draws:raise RuntimeError("All predeclared 04C patient bootstraps degenerate")
 rng=np.random.default_rng(seed)
 perm=[]
 for _ in range(R):
  v=corr(a[rng.permutation(len(a))],b)
  if v is None:raise RuntimeError("One original 04C patient shuffle invalid")
  perm.append(v)
 return {"observedSpearman":observed,
         "patientBootstrap95":np.quantile(draws,[.025,.975]).tolist(),
         "bootstrapValid":len(draws),"bootstrapDegenerate":invalid,
         "patientShuffleTwoSidedTailPlusOne":float((1+sum(abs(x)>=abs(observed)-1e-15 for x in perm))/(R+1)),
         "negativeShuffleMean":float(np.mean(perm)),
         "patientCount":len(a),"permutations":R}
def loocv(x,y,alpha=1.0):
 X=np.asarray(x,dtype=float);Y=np.asarray(y,dtype=float)
 if X.ndim!=2 or X.shape[0]!=len(Y) or len(Y)!=N or not np.isfinite(X).all():
  raise RuntimeError("Original 04C LOPO matrix incomplete")
 predictions=[]
 for i in range(N):
  mask=np.ones(N,dtype=bool);mask[i]=False
  train=X[mask];test=X[i];train_y=Y[mask]
  ymean=float(np.mean(train_y))
  if X.shape[1]==0:
   predictions.append(ymean);continue
  mean=train.mean(axis=0)
  sd=train.std(axis=0)
  sd=np.where(sd<1e-12,1.0,sd)
  z=(train-mean)/sd
  zz=(test-mean)/sd
  beta=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(train_y-ymean))
  predictions.append(float(ymean+zz@beta))
 return np.array(predictions)
def predict_metrics(pred,y):
 err=np.asarray(pred)-np.asarray(y)
 se=float(np.sum((y-np.mean(y))**2))
 return {"MAE":float(np.mean(abs(err))),"RMSE":float(np.sqrt(np.mean(err**2))),
         "R2":None if se<1e-20 else float(1-np.sum(err**2)/se)}
def feature(row,model,key,arm,layer,metric):
 summary=row["models"][model]["realInterventionSummaries"][key+"_"+arm]
 v=summary["layerStateDelta"][layer][metric]
 if metric=="directTargetedTokenCosineCleanVsPerturbed":
  return float(1-v)
 return float(np.log1p(v)) if metric=="directTargetedTokenDeltaL2" else float(v)

def load(session):
 finalraw=private_contents(session,f"{B}/final/analysis.json")
 receipt=json.loads(private_contents(session,f"{B}/final/archive_receipt.json"))
 report=json.loads(finalraw)
 if (sha(finalraw)!=receipt["original33PatientAnalysisSHA256"] or
     receipt["nOriginalPatientRecords"]!=33 or receipt["nGenuineFrozenSCEPTForwards"]!=594 or
     receipt["nRawSourceSHA256VerifiedPrivateChunks"]!=7 or
     report["nActualGenuineSCEPTSourceRuns"]!=594):
  raise RuntimeError("Original complete 04B private study checksum/manifest changed")
 rows=[]
 for ch in report["sourceChunkSha"]:
  num=ch["chunk"]
  src=private_contents(session,f"{B}/chunks/chunk_{num:02d}/paired_cases.json")
  rec=json.loads(private_contents(session,f"{B}/chunks/chunk_{num:02d}/archive_receipt.json"))
  if (sha(src)!=ch["sha256"] or sha(src)!=rec["originalChunkAnalysisSHA256"] or
      rec["originalRawZIPCount"]!=ch["originalZIPs"] or
      rec["assetSHA256"]!=ch["assetSHA256"] or
      rec["status"]!="PASS_04B_REAL_ZIP_SOURCE_CHUNK_FULL_SHA256_VERIFIED"):
   raise RuntimeError("04C original 04B private chunk SHA or raw archive receipt mismatch")
  body=json.loads(src)
  if (body["patientIds"]!=ch["patients"] or
      body["nOriginalGenuineModelForwards"]!=ch["originalZIPs"] or
      len(body["patientCases"])!=len(ch["patients"])):
   raise RuntimeError("04C original batch people/count inconsistent")
  rows+=body["patientCases"]
 if (len(rows)!=N or len(set(r["patient_id"] for r in rows))!=N or
     sum(r["grade"]=="HGG" for r in rows)!=26 or
     sum(r["grade"]=="LGG" for r in rows)!=7 or
     len(report["sourceChunkSha"])!=7 or
     [r["patient_id"] for r in rows]!=
       [x["patient_id"] for x in report["privateOriginalPatientModelPairedRecords"][PRIMARY]]):
  raise RuntimeError("04C original eligible patient independent-unit denominator changed")
 for row in rows:
  if row["nGenuineSCEPTForwards"]!=18 or len(row["inputDiagnostics"])!=8 or set(row["models"])!={"S","J"}:
   raise RuntimeError("04C original 18 genuine observation source data missing")
  for model in ("S","J"):
   m=row["models"][model]
   if len(m["originalObservations"])!=9 or len(m["realInterventionSummaries"])!=8:
    raise RuntimeError("04C original condition and output source incomplete")
   for operator in OP:
    for dose in DOSE:
     key=f"{operator}_dose{dose}"
     for arm in ("lesion","control"):
      cond=key+"_"+arm
      a=m["realInterventionSummaries"][cond]
      if set(a["layerStateDelta"])!=set(L) or \
         any(set(a["layerStateDelta"][k])!=set(FEATURES)|{"clsTokenDeltaL2"} for k in L):
       raise RuntimeError("04C all original measured layer shape/summary metrics required")
      nums=list(a["layerStateDelta"][L[0]].values())
      if not np.isfinite(nums).all():
       raise RuntimeError("04C original real observation metrics nonfinite")
  # Strict primary 04B decision response reproduction to avoid outcome relabel.
  original=next(x for x in report["privateOriginalPatientModelPairedRecords"][PRIMARY]
                if x["patient_id"]==row["patient_id"])
  for model in ("S","J"):
   before=row["models"][model]["pairedOutputContrasts"][PRIMARY]["absMarginSpecificityLesionMinusControl"]
   if abs(before-original[model]["absMarginSpecificityLesionMinusControl"])>1e-9:
    raise RuntimeError("04C original 04B patient paired actual margin outcome changed")
 return rows,sha(finalraw),report,receipt

def summarize(rows,source_sha):
 summaries={}
 cohort={}
 for operator in OP:
  for dose in DOSE:
   key=f"{operator}_dose{dose}"
   Y=[];T=[];inputs=[];all_features={metric:{l:[] for l in L} for metric in FEATURES}
   permodel={m:{"Y":[],"T":[]} for m in ("S","J")}
   for row in rows:
    effect=[];internal=[]
    for model in ("S","J"):
     margin=float(row["models"][model]["pairedOutputContrasts"][key]["absMarginSpecificityLesionMinusControl"])
     direct=(feature(row,model,key,"lesion",TARGET_LAYER,FEATURES[0])-
             feature(row,model,key,"control",TARGET_LAYER,FEATURES[0]))
     effect.append(margin);internal.append(direct)
     permodel[model]["Y"].append(margin);permodel[model]["T"].append(direct)
    Y.append(float(np.mean(effect)));T.append(float(np.mean(internal)))
    for metric in FEATURES:
     for layer in L:
      all_features[metric][layer].append(float(np.mean([
       feature(row,m,key,"lesion",layer,metric)-
       feature(row,m,key,"control",layer,metric) for m in ("S","J")])))
    d=row["inputDiagnostics"]
    l=d[key+"_lesion"];c=d[key+"_control"]
    delta=abs(l["actualPixelL2"]-c["actualPixelL2"])
    if delta>max(1e-5,1e-5*l["commonSourceMaxL2"]):
     raise RuntimeError("Original 04B source actual same-input-L2 requirement broken")
    inputs.append([float(l["inputROIAbsMeanShift"]-c["inputROIAbsMeanShift"]),
                   float(np.log1p(l["sourceFullReplacementL2"])-np.log1p(c["sourceFullReplacementL2"]))])
   Y=np.asarray(Y);T=np.asarray(T);inputs=np.array(inputs)
   nonnegative=int(np.sum(Y>0));negative=int(np.sum(Y<0))
   paired={
    "nPatients":N,"meanPrimaryOutputMarginContrast":float(np.mean(Y)),
    "meanSCEPTTargetedTokenLogL2Contrast":float(np.mean(T)),
    "positiveOutputPatientCount":nonnegative,"negativeOutputPatientCount":negative,
    "primaryRepresentationAssociation":boot_and_shuffle(T,Y),
    "pearsonSensitivity":corr(T,Y,"pearson"),
    "inputOnlyIndividualSpearman":{
     "sourceROIAbsoluteMeanShiftContrast":corr(inputs[:,0],Y),
     "sourceFullReplacementL2Contrast":corr(inputs[:,1],Y)
    },
    "exploratoryOtherMetrics":{
      metric:{layer:corr(np.array(v),Y) for layer,v in lay.items()}
      for metric,lay in all_features.items()},
    "modelStratified":{
      model:{"nPatients":33,"spearman":corr(permodel[model]["T"],permodel[model]["Y"]),
             "meanOutputContrast":float(np.mean(permodel[model]["Y"]))}
      for model in ("S","J")}
   }
   if key==PRIMARY:
    y=Y
    pred0=loocv(np.empty((N,0)),y)
    pred1=loocv(inputs,y)
    pred2=loocv(np.column_stack([inputs,T]),y)
    rng=np.random.default_rng(SEED)
    diff=np.abs(y-pred1)-np.abs(y-pred2)
    boots=[float(np.mean(diff[rng.integers(0,N,size=N)])) for _ in range(R)]
    paired["LOPOFixedRidge"]={
      "patientFoldCount":33,"alpha":1.0,"trainedOnlyOnOtherPatients":True,
      "interceptOnly":predict_metrics(pred0,y),
      "twoInputProxies":predict_metrics(pred1,y),
      "twoInputProxiesPlusTargetedToken":predict_metrics(pred2,y),
      "patientMeanInputMAEMinusAugmentedMAE":float(np.mean(diff)),
      "incrementalMAEImprovementBootstrap95":np.quantile(boots,[.025,.975]).tolist(),
      "noModelRetuningOrHyperparameterSearch":True}
   summaries[key]=paired
   cohort[key]={"patientPairs":[{"patient_id":r["patient_id"],"grade":r["grade"],
      "meanTwoModelsOutputContrast":float(Y[i]),
      "meanTwoModelsSCEPTTargetedTokenContrast":float(T[i]),
      "inputProxyAbsoluteMeanShiftContrast":float(inputs[i,0]),
      "inputProxySourceFullL2Contrast":float(inputs[i,1])}
      for i,r in enumerate(rows)]}
 return summaries,cohort

def main():
 s=github()
 private=f"{BASE}/analysis.json"
 rr=s.get(f"{API}/contents/{private}",timeout=(30,120))
 if rr.status_code==200:
  x=json.loads(private_contents(s,private))
  if x.get("status")==STATUS:
   print("04C01 original science already privately saved; no changing results")
   return
  raise RuntimeError("Existing 04C original source must not be overwritten")
 if rr.status_code!=404:raise RuntimeError("Could not read 04C private scientific path")
 rows,original_sha,report,receipt=load(s)
 summary,pairs=summarize(rows,original_sha)
 primary=summary[PRIMARY]
 out={
   "schema":"scept.a6-ij-04c01.internal-decision-association.v1",
   "status":STATUS,"scope":"PREDECLARED_SECONDARY_EXPLORATORY_OBSERVATION_ONLY",
   "original04BAnalysisFullSHA256":original_sha,
   "original04BPlanSHA256":report["planSHA256"],
   "original04BReceiptSHA256":sha(canonical(receipt)),
   "nPatientIndependentUnits":33,"nOriginalRealSCEPTObservations":594,
   "nOriginalPrivatelyVerifiedBatches":7,"sealedP5":"NOT_ACCESSED",
   "externalP6":"NOT_ACCESSED",
   "primary":"gaussian_local_sigma3_dose50, directTargetedTokenDeltaL2 log1p lesion minus control at blocks.11.norm1",
   "originalBootstrapResamples":R,"originalPatientShuffles":R,
   "primaryResult":primary,"allPrespecifiedConditions":summary,
   "privatePatientLevelOutcomeAndRepresentationRecords":pairs,
   "limitations":[
      "Actual named intermediate tensors were captured in 04B and SHA-archived; 04C01 uses 04B ORIGINAL SUMMARY METRICS and has not re-examined full raw 197x768 tensors.",
      "Spearman and ridge performance are associations, not evidence of hidden unit causal mediation.",
      "LOPO is same-source conditional exploratory internal validation on the same 33 selected VAL patients; independent external model/data/generalization NOT demonstrated.",
      "MRI source intensity, patch anatomy, OOD artifacts, tumor localization and patient eligibility selection remain confounds beyond exact input pixel L2.",
      "All secondary layer/operator/model results descriptive only; no post hoc declare positive medical attribution fidelity.",
      "04D direct activation intervention, model-independent causal fidelity, head-to-head Grad-CAM and independent external validation remain unexecuted."
   ]}
 raw=canonical(out)
 commit_once(s,private,raw)
 receip={
   "schema":"scept.a6-ij-04c01.verified-analysis-receipt.v1",
   "status":"PRIVATE_04C01_33_PATIENT_ASSOCIATION_AND_NEGATIVE_CONTROL_PERSISTED",
   "original04BAnalysisSHA256":original_sha,
   "original04C01AnalysisSHA256":sha(raw),
   "nOriginalPatients":33,"nAuthentic04BObservations":594,
   "sevenOriginalRawSourceChunks":7,"nOriginalBootstrap":R,
   "nOriginalPermutationNegatives":R,
   "nOriginalLOPOFolds":33,
   "scientificVerdict":"ASSOCIATION_ONLY_NOT_CAUSAL_ATTRIBUTION_OR_CLINICAL_EVIDENCE"}
 commit_once(s,f"{BASE}/archive_receipt.json",canonical(receip))
 print("A6-IJ-04C01 33-PATIENT INTERNAL STATE / DECISION ASSOCIATION: PASS",flush=True)
 print("04C PRIMARY SPEARMAN:",primary["primaryRepresentationAssociation"]["observedSpearman"])
 print("04C PRIMARY 95% PATIENT BOOTSTRAP:",primary["primaryRepresentationAssociation"]["patientBootstrap95"])
 print("04C NEGATIVE PATIENT-SHUFFLE TAIL:",primary["primaryRepresentationAssociation"]["patientShuffleTwoSidedTailPlusOne"])
 print("04C LOPO INPUT VS INPUT+SCEPT:",primary["LOPOFixedRidge"]["twoInputProxies"]["MAE"],
       primary["LOPOFixedRidge"]["twoInputProxiesPlusTargetedToken"]["MAE"])
 print("04C01 actual raw layer-tensor reinspection remains 04C02 PENDING.")
 print("PRIVATE result: https://github.com/changxinjiresearch/research-course-brats2020-ijepa-xai/blob/main/"+private)

if __name__=="__main__":
 main()
