"""04D final fail-closed original 33-person bounded causal graph-intervention result.

Primary analysis must not depend on improving frozen classifier performance.
Each person's S/J paired original output is a repeated measurement, not 66
independent patient samples. Keeps negative, zero, indefinite effects.
"""
from __future__ import annotations
import json
import numpy as np
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_04b_measure import canonical,sha
from a6_ij_04d_hook import PLAN,plan_sha,LAYERS,ARMS,SELECTORS,DIRECTIONS,MIN_MARGIN_DELTA
from a6_ij_04d_measure import BASE,STATUS,prepare
from a6_ij_04d_archive_private import prior_receipt,subset

FINAL="PASS_04D_COMPLETE_33_PATIENT_1914_REAL_FROZEN_POSTBLOCK_CAUSAL_INTERVENTION_AUDIT"
RESAMPLES=2000
SEED=240216

def summary(values,rng):
 arr=np.asarray(values,dtype=np.float64)
 if arr.size==0:
  return {"nValidPatients":0,"mean":None,"median":None,
          "patientBootstrap95":None,"reason":"ALL_INDETERMINATE"}
 if not np.isfinite(arr).all():raise RuntimeError("04D true model margin effect nonfinite")
 simulated=np.mean(arr[rng.integers(0,len(arr),size=(RESAMPLES,len(arr)))],axis=1)
 return {"nValidPatients":len(arr),"mean":float(np.mean(arr)),
         "median":float(np.median(arr)),
         "patientBootstrap95":list(map(float,np.quantile(simulated,[.025,.975])))}
def main():
 s=github()
 cases,original04bsha,original04b=prepare()[1:]
 rows=[];source=[]
 if len(cases)!=33:raise RuntimeError("04D original eligibility cohort changed")
 for chunk in range(1,8):
  receipt=prior_receipt(s,chunk,cases)
  if receipt is None:
   raise RuntimeError(f"Original 04D full source shard {chunk} NOT archived; no final")
  p=f"{BASE}/chunks/chunk_{chunk:02d}/original_cases.json"
  data=private_contents(s,p)
  if sha(data)!=receipt["originalCaseBatchFullSHA256"]:
   raise RuntimeError("04D existing private original case report bytes differ")
  body=json.loads(data)
  expected=[r["patient_id"] for r in subset(chunk,cases)]
  if (body.get("status")!="PASS_04D_ALL_ORIGINAL_DIRECT_FROZEN_POSTBLOCK_INTERVENTIONS" or
      body.get("chunk")!=chunk or
      body.get("patientIds")!=expected or
      body.get("planSHA256")!=plan_sha() or
      body.get("original04BFullAnalysisSHA256")!=original04bsha or
      len(body.get("patientCases",[]))!=len(expected) or
      body.get("realFrozenComputationalForwards")!=58*len(expected)):
   raise RuntimeError("04D original frozen graph source cohort/protocol changed")
  rows+=body["patientCases"]
  source.append({"chunk":chunk,"patientIds":expected,
                 "originalCaseJSONSHA256":sha(data),
                 "originalDonorStateReleaseZIP_SHA256":receipt["assetSHA256"],
                 "nOriginalRealForwards":receipt["nGenuineDirectGraphForwards"]})
 if (len(rows)!=33 or [q["patient_id"] for q in rows]!=[q["patient_id"] for q in cases] or
     sum(q["grade"]=="HGG" for q in rows)!=26 or
     sum(q["grade"]=="LGG" for q in rows)!=7 or
     sum(q["realForwardCount"] for q in rows)!=1914):
  raise RuntimeError("04D all true 1914 frozen model forward source records incomplete")
 full_ratio={}
 effects={}
 for arm in ARMS:
  for layer in LAYERS:
   label=f"{arm}|{layer}"
   for direction in DIRECTIONS:
    signed=[];normalized=[];eligible_count=0;not_identifiable=0
    original_patient_rows=[]
    for case in rows:
     scores={}
     for model in ("S","J"):
      entries=case["models"][model]["bidirectionalPostBlockInterventions"]
      target=entries[f"{arm}|{layer}|target"][direction]
      placebo=entries[f"{arm}|{layer}|placebo"][direction]
      full=entries[f"{arm}|{layer}|all"][direction]
      if full["fullPositiveControlLogitAbsoluteError"] is None or \
         full["fullPositiveControlLogitAbsoluteError"]>2e-4:
       raise RuntimeError("04D genuine causal graph full-donor positive validation no longer passed")
      margin=float(target["signedTrueGradeMargin"]-placebo["signedTrueGradeMargin"])
      if not np.isfinite(margin):raise RuntimeError("04D intervention margin missing")
      tfrac=target["fractionOfOriginalMarginDifference"]
      pfrac=placebo["fractionOfOriginalMarginDifference"]
      if tfrac["status"]!=pfrac["status"]:
       raise RuntimeError("04D target and placebo comparison denominator gate changed")
      if tfrac["signedFraction"] is None:
       if pfrac["signedFraction"] is not None:
        raise RuntimeError("04D original negative denominator unexpectedly estimated")
       norm=None
      else:
       norm=float(tfrac["signedFraction"]-pfrac["signedFraction"])
      scores[model]={"directTargetMinusPlaceboSignedMargin":margin,
                     "normalizedTargetMinusPlaceboFraction":norm,
                     "originalContrastGateStatus":tfrac["status"]}
     # Source patient is SINGLE independent unit; do not count S/J as 2 patients.
     signed_effect=float(np.mean([scores[k]["directTargetMinusPlaceboSignedMargin"] for k in ("S","J")]))
     signed.append(signed_effect)
     two_full=all(scores[k]["normalizedTargetMinusPlaceboFraction"] is not None for k in ("S","J"))
     if two_full:
      frac=float(np.mean([scores[k]["normalizedTargetMinusPlaceboFraction"] for k in ("S","J")]))
      normalized.append(frac);eligible_count+=1
     else:not_identifiable+=1;frac=None
     original_patient_rows.append({"patient_id":case["patient_id"],"grade":case["grade"],
                                  "twoModelSignedMarginTargetMinusPlacebo":signed_effect,
                                  "twoModelNormalizedFractionIfBothMeasurable":frac,
                                  "originalS":scores["S"],"originalJ":scores["J"]})
    raw_key=label+"|"+direction
    rng=np.random.default_rng(SEED)
    effects[raw_key]={
      "original33PatientPairedSignedMargin":summary(signed,rng),
      "bothModelsMeasurableFractionOnly":summary(normalized,np.random.default_rng(SEED)),
      "nOriginalBothModelsMeasurable":eligible_count,
      "nOriginalPatientFractionIndeterminate":not_identifiable,
      "originalPatientLevelSAndJPairedResults":original_patient_rows,
      "scientificInterpretation":"within frozen model, intervention-specific output effect, NOT tumor causal attribution"}
 primary=effects["lesion|blocks.5|rescue"]
 output={
  "schema":"scept.a6-ij-04d.complete-bounded-causal-graph-intervention.v1",
  "status":FINAL,
  "plan":PLAN,"planSHA256":plan_sha(),
  "nIndependentOriginalPatients":33,
  "nTrueFrozenModelPyTorchForwards":1914,
  "nOriginalRealPrivateDonorAndLogitChunks":7,
  "nOriginalP0Screened":55,"nPriorEligibilityExcluded":22,
  "grades":{"HGG":26,"LGG":7},"sourceOriginal04BSHA256":original04bsha,
  "sevenAuthenticatedOriginalInterventionSourceChunks":source,
  "primaryPrespecifiedLesionBlock5RescueTargetVsPlacebo":primary,
  "allEightOriginalLayerArmDirectionResults":effects,
  "sealedP5Test":"NOT_ACCESSED","independentP6":"NOT_ACCESSED",
  "scientificVerdict":"DIRECT_POST_BLOCK_SUBSTITUTION_CHANGES_FROZEN_COMPUTATION_WITH_BOUNDED_CAUSAL_INTERVENTION_MEANING; NO MEDICAL_CAUSAL_FIDELITY_CLAIM",
  "conclusionsBoundedTo":[
    "Interventions act on actual post-residual ViT outputs and downstream computation.",
    "Same-patient/ROI Gaussian 50% edited versus clean branch, frozen S/J only.",
    "Full donor controls and clean-to-clean placebo must pass on every original patient.",
    "Near-zero clean-edited true grade decision difference is genuinely indeterminate for normalized mediation fraction.",
    "A target-vs-other-ROI result is not biologically or semantically matched placebo, and not a universal faithful explanation.",
    "No comparison to CAAP/Grad-CAM or independent held-out external data made in 04D."
  ]}
 path=f"{BASE}/final/analysis.json"
 response=s.get(f"{API}/contents/{path}",timeout=(30,120))
 raw=canonical(output)
 if response.status_code==200:
  if private_contents(s,path)!=raw:
   raise RuntimeError("Previously archived 04D final is immutable and differs")
 elif response.status_code==404:
  commit_once(s,path,raw)
 else:raise RuntimeError("04D private final source path unreadable")
 receipt={"schema":"scept.a6-ij-04d.final-private-receipt.v1",
  "status":FINAL,"original33PatientSignedAndIndeterminateReportSHA256":sha(raw),
  "nIndependentPatients":33,"nTrueFrozenModelForwards":1914,
  "nPrivateOriginalDonorStateReleaseAssets":7,
  "sevenOriginalBatchSHA256":source,
  "noImprovementToModelAccuracyClaimed":True,
  "scope":"BOUNDED_GRAPH_CAUSAL_INTERVENTION_NOT_TUMOR_MEDICAL_CAUSALITY"}
 commit_once(s,f"{BASE}/final/archive_receipt.json",canonical(receipt))
 print("A6-IJ-04D FULL 33-PATIENT 1914-REAL-FROZEN-INTERNAL-PATCH STUDY: PASS",flush=True)
 print("PRIMARY lesion block5 rescue targeted-minus-placebo mean:",
       primary["original33PatientPairedSignedMargin"]["mean"],flush=True)
 print("Fraction indeterminate due to true model original output near-constant:",
       primary["nOriginalPatientFractionIndeterminate"],"/33",flush=True)
 print("This is bounded causal graph intervention of a frozen model, not clinical attribution fidelity.")
 print("PRIVATE:",path)

if __name__=="__main__":main()
