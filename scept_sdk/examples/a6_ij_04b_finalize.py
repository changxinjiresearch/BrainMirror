"""04B strict 33-person source-gated evaluation (NO rerun/retune).

Primary: patient mean S/J absolute true-grade margin response lesion minus
healthy under equal original pixel L2, Gaussian at 50% shared achievable
strength. Always archive negative/null effects and all original patient
contrasts. 2000 patient-level bootstrap + 2000 patient sign-flip controls.
"""
from __future__ import annotations
import json
import numpy as np
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_04b_measure import source,canonical,sha,CONDITIONS
from a6_ij_04b_matched_interventions import PLAN,OPS,DOSES,condition_name,fingerprint04b
from a6_ij_04b_archive_private import BASE,TAG,remote_status,batch_source

SEED=240216
NBOOT=2000
NFLIP=2000
STATUS="PASS_COMPLETE_33_PATIENT_EQUAL_PIXEL_L2_PAIRED_INTERVENTION_ANALYSIS"
def boot(x,rng):
 v=np.asarray(x,dtype=np.float64)
 n=len(v)
 if n==0 or not np.isfinite(v).all():
  raise RuntimeError("No finite independent patient outcomes")
 draws=v[rng.integers(0,n,size=(NBOOT,n))].mean(axis=1)
 return {"nIndependentPatients":n,
         "mean":float(v.mean()),"median":float(np.median(v)),
         "patientBootstrap95":list(map(float,np.quantile(draws,[.025,.975])))}
def inspect(session):
 eligible,_,lock=source(session)
 if len(eligible)!=33:raise RuntimeError("Old original eligible cohort incomplete")
 release_resp=session.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 release_resp.raise_for_status()
 release=release_resp.json()
 all_cases=[]
 chunks=[]
 for num in range(1,8):
  receipt=remote_status(session,num,eligible)
  if receipt is None:
   raise RuntimeError(f"Original 04B private 7-chunk archive missing chunk {num:02d}")
  path=f"{BASE}/chunks/chunk_{num:02d}/paired_cases.json"
  raw=private_contents(session,path)
  if sha(raw)!=receipt["originalChunkAnalysisSHA256"]:
   raise RuntimeError("Original private 04B batch source SHA256 mismatch")
  body=json.loads(raw)
  expected=[r["patient_id"] for r in batch_source(num,eligible)]
  if (body.get("status")!="PASS_04B_EXACT_MATCHED_ORIGINAL_REAL_INTERVENTION_PATIENT_CHUNK" or
      body.get("patientIds")!=expected or body.get("planSHA256")!=fingerprint04b() or
      body.get("plan")!=PLAN or body.get("sourceLock")!=lock or
      len(body.get("patientCases",[]))!=len(expected) or
      body.get("nOriginalGenuineModelForwards")!=18*len(expected)):
   raise RuntimeError("04B original batch metadata/protocol no longer complete")
  for i,row in enumerate(body["patientCases"]):
   patient=expected[i]
   if (row["patient_id"]!=patient or
       row["grade"]!=next(q["grade"] for q in eligible if q["patient_id"]==patient) or
       row["planSHA256"]!=fingerprint04b() or row["status"]!="PASS_ORIGINAL_04B_PATIENT_18_REAL_MODEL_SOURCE_ZIPS" or
       row["nGenuineSCEPTForwards"]!=18 or len(row["inputDiagnostics"])!=8 or
       set(row["models"])!={"S","J"} or
       row["source"]["P1NPZSHA256"]!=next(q["originalFullP1NPZSHA256"] for q in eligible if q["patient_id"]==patient)):
    raise RuntimeError("04B individual original patient SHA-verified paired results differ")
   for model in ("S","J"):
    m=row["models"][model]
    if len(m["originalObservations"])!=9 or set(m["originalObservations"])!=set(CONDITIONS) or \
       len(m["realInterventionSummaries"])!=8 or len(m["pairedOutputContrasts"])!=4:
     raise RuntimeError("Original model has incomplete 04B input dose and original output observations")
    for op in OPS:
     for dose in DOSES:
      name=f"{op}_dose{int(dose*100):02d}"
      effects=m["pairedOutputContrasts"][name]
      l=effects["lesionActualPixelL2"];c=effects["controlActualPixelL2"]
      orig=row["inputDiagnostics"][condition_name(op,dose,"lesion")]
      common=orig["commonSourceMaxL2"]
      tol=max(PLAN["l2AbsoluteTolerance"],PLAN["l2RelativeTolerance"]*common)
      if (abs(l-c)>tol or abs(l-common*dose)>tol or not np.isfinite(list(effects.values())).all()):
       raise RuntimeError("Original 04B equal ACTUAL input L2 paired dose gate FAILED")
      true_margin=m["originalObservations"]["clean"]["output"]["trueGradeMargin"]
      a=m["originalObservations"][condition_name(op,dose,"lesion")]["output"]["trueGradeMargin"]
      b=m["originalObservations"][condition_name(op,dose,"control")]["output"]["trueGradeMargin"]
      expected_effect=abs(a-true_margin)-abs(b-true_margin)
      if abs(expected_effect-effects["absMarginSpecificityLesionMinusControl"])>1e-9:
       raise RuntimeError("Original paired margin effect does not equal recorded genuine logits")
   all_cases.append(row)
  chunks.append({"chunk":num,"sha256":sha(raw),"assetSHA256":receipt["assetSHA256"],
                 "patients":expected,"originalZIPs":18*len(expected)})
 ids=[r["patient_id"] for r in all_cases]
 if ids!=[p["patient_id"] for p in eligible] or len(ids)!=33 or \
    len(set(ids))!=33 or sum(r["grade"]=="HGG" for r in all_cases)!=26 or \
    sum(r["grade"]=="LGG" for r in all_cases)!=7 or sum(c["originalZIPs"] for c in chunks)!=594:
  raise RuntimeError("04B original patient selection/594 real observations incomplete")
 return all_cases,chunks,lock

def study(all_cases,chunks,lock):
 rng=np.random.default_rng(SEED)
 levels=[(o,d) for o in OPS for d in DOSES]
 summaries={}
 originals={}
 for operator,dose in levels:
  label=f"{operator}_dose{int(dose*100):02d}"
  by_patient=[]
  for row in all_cases:
   permodel={m:row["models"][m]["pairedOutputContrasts"][label] for m in ("S","J")}
   mean=float(np.mean([v["absMarginSpecificityLesionMinusControl"] for v in permodel.values()]))
   by_patient.append({"patient_id":row["patient_id"],"grade":row["grade"],
        "S":permodel["S"],"J":permodel["J"],"twoFrozenModelPatientMean":mean})
  originals[label]=by_patient
  v=np.asarray([r["twoFrozenModelPatientMean"] for r in by_patient],dtype=np.float64)
  gr=np.asarray([r["grade"] for r in by_patient])
  normal=boot(v,rng)
  models={m:boot([r[m]["absMarginSpecificityLesionMinusControl"] for r in by_patient],rng)
          for m in ("S","J")}
  group={g:boot(v[gr==g],rng) for g in ("HGG","LGG")}
  # Two-sided joint patient-sign flip null; ONE common sign for a patient, so
  # correlated same-patient S and J observations are not falsely independent.
  signs=rng.choice([-1,1],size=(NFLIP,len(v)))
  flipped=np.mean(signs*v[None,:],axis=1)
  null={"nOriginalPatientSignFlips":NFLIP,
        "tailFractionAbsNullMeanGreaterEqualAbsOriginal":float(np.mean(np.abs(flipped)>=abs(v.mean()))),
        "null95":list(map(float,np.quantile(flipped,[.025,.975]))),
        "allPrespecifiedSignedNullMeans":flipped.tolist(),
        "scope":"descriptive patient-level sign-flips, not clinical fidelity hypothesis testing"}
  summaries[label]={"status":"EXECUTED_UNCONDITIONALLY","pairedTwoModel":normal,
                     "separateModels":models,"gradeStratified":group,
                     "jointPatientSwapNegativeControl":null}
 primary="gaussian_local_sigma3_dose50"
 report={
   "schema":"scept.a6-ij-04b.analysis.v1","status":STATUS,
   "plan":PLAN,"planSHA256":fingerprint04b(),
   "sourceOriginal":lock,"nOriginalVALScreened":55,"nOriginalVALExcluded":22,
   "nOriginalFirstSliceEligiblePatients":33,"nFrozenModels":2,
   "nActualGenuineSCEPTSourceRuns":594,"nPrivateShards":7,
   "sourceChunkSha":chunks,
   "primaryEndpoint":primary,"primaryObserved":summaries[primary],
   "allFourPrespecifiedOperatorDoseOutcomes":summaries,
   "privateOriginalPatientModelPairedRecords":originals,
   "scientificInterpretation":[
    "Quantifies selected-validation-patient input-edit magnitude-matched frozen-model responses only.",
    "No claim that an induced margin shift or observed internal activation change identifies pathologic decision evidence.",
    "The 22 original ineligible cases are excluded by input-only rule and remain in the original screening source.",
    "Model S/J are correlated same-patient observations, not 66 independent participants.",
    "Input-pixel L2 is matched, but anatomical area/position, local image statistics and out-of-distribution artifacts need separate controls.",
    "The original 04A02 first-case exploratory observation was in the eligibility screen; this is not independent sealed test validation.",
    "No P5 sealed TEST, P6 independent external data, training, or causal activation-patching procedure."
   ]}
 return report

def main():
 s=github()
 full_path=f"{BASE}/final/analysis.json"
 check=s.get(f"{API}/contents/{full_path}",timeout=(30,120))
 if check.status_code==200:
  original=json.loads(private_contents(s,full_path))
  if original.get("status")==STATUS and original.get("planSHA256")==fingerprint04b():
   print("A6-IJ-04B ORIGINAL PATIENT-LEVEL RESEARCH ALREADY PRIVATE; NO recomputation",flush=True)
   return
  raise RuntimeError("Existing archived private 04B study conflicts with frozen protocol")
 if check.status_code!=404:raise RuntimeError("Cannot examine private final original 04B report")
 people,chunks,lock=inspect(s)
 report=study(people,chunks,lock)
 raw=canonical(report)
 commit_once(s,full_path,raw)
 receipt={"schema":"scept.a6-ij-04b.final-archive.v1",
    "status":"PRIVATE_33_PATIENTS_594_SOURCE_ZIPS_AND_PAIRED_L2_ANALYSIS_COMPLETE",
    "planSHA256":fingerprint04b(),
    "original33PatientAnalysisSHA256":sha(raw),
    "nOriginalPatientRecords":33,"nGenuineFrozenSCEPTForwards":594,
    "nRawSourceSHA256VerifiedPrivateChunks":7,
    "sevenOriginalPrivateSourceChunkSHAs":chunks,
    "prespecifiedNegativesPerOperatorDose":NFLIP,
    "patientBootstrapResamples":NBOOT,
    "resultSignCannotGateEngineeringPass":True,
    "originalP5test":"NOT_ACCESSED","independentP6":"NOT_ACCESSED",
    "scope":"MATCHED_PIXEL_L2_EXPLORATORY_RESPONSE_NOT_CAUSAL_CLINICAL_EXPLANATION"}
 commit_once(s,f"{BASE}/final/archive_receipt.json",canonical(receipt))
 primary=report["primaryObserved"]["pairedTwoModel"]
 print("A6-IJ-04B COMPLETE 33-PATIENT 594-REAL-FORWARD MATCHED-L2 ANALYSIS: PASS",flush=True)
 print("Primary patient mean abs true-grade margin lesion-control:",primary["mean"])
 print("Original patient-bootstrap 95%:",primary["patientBootstrap95"])
 print("Both signs valid; NOT a claim of tumor-evidence causality.")
 print("PRIVATE original final:",f"https://github.com/changxinjiresearch/research-course-brats2020-ijepa-xai/blob/main/{full_path}")

if __name__=="__main__":main()
