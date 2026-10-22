"""A6-IJ-03C: immutable research archive receipt for fixed 55-person analyses.

Complete patient-derived real-RMS data remain PRIVATE in 11 SHA-proven
chunk records; original per-fold patient predictions also PRIVATE.
No model execution and no claim of improved/clinical explanation value.
"""
import base64
import hashlib
import json
from a6_ij_03b_measure import github,private_contents,API,REPO
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_03c_analyse_private import BASE,PLAN,plan_sha
from a6_ij_03c_extract_rms_private import sha

def main():
 s=github()
 path=f"{BASE}/analysis.json"
 original=private_contents(s,path)
 report=json.loads(original)
 if (report.get("schema")!="scept.a6-ij-03c.analysis.v1" or
     report.get("status")!="PASS_FIXED_55_PATIENT_LOPO_RMS_CONTROL_AND_SHUFFLE_EXECUTED" or
     report.get("planSHA256")!=plan_sha() or report.get("plan")!=PLAN or
     report.get("nPatients")!=55 or
     report.get("nRMSOriginalSCEPTSourceRuns")!=1100 or
     len(report.get("dataRows",[]))!=55 or
     set(report.get("endpoints",{}))!={"iou10","tap"}):
  raise RuntimeError("03C original locked complete analysis unavailable/incompatible")
 rows=report["dataRows"]
 if len({r["patient_id"] for r in rows})!=55 or \
    sum(r["grade"]=="HGG" for r in rows)!=44 or sum(r["grade"]=="LGG" for r in rows)!=11:
  raise RuntimeError("03C private result patient identities/grades changed")
 for y in ("iou10","tap"):
  o=report["endpoints"][y]
  if len(o["originalPatientHeldOutPredictions"])!=55 or \
     o["withinGradeNegativeControls"]["permutations"]!=100 or \
     o["primaryB1minusF"]["method"]!="paired patient bootstrap of fixed out-of-fold residuals, models not refit":
   raise RuntimeError("03C expected all held-out paired predictions and 100 negative controls")
 source=report["sourceOriginalSHA256"]["03cRMSChunksOriginalSHA256"]
 if len(source)!=11:
  raise RuntimeError("Eleven independently archived source RMS private chunks required")
 for n in range(1,12):
  p=f"{BASE}/rms/chunk_{n:02d}.json"
  raw=private_contents(s,p)
  if sha(raw)!=source[f"chunk_{n:02d}"]:
   raise RuntimeError("Stored original patient RMS chunk bytes SHA changed")
 receipt={"schema":"scept.a6-ij-03c.private-closeout.v1",
  "status":"PRIVATE_55_PATIENT_RMS_BASELINE_AND_INCREMENTAL_NEGATIVES_ARCHIVED",
  "originalAnalysisSHA256":sha(original),
  "originalRMSChunkSHA256":source,
  "planSHA256":plan_sha(),
  "patients":55,"originalModelSliceSourceRuns":1100,
  "independentOutOfPatientFolds":55,
  "patientBootstrapResamples":2000,"withinGradeShuffles":100,
  "resultsNotGatedOnPositiveImprovement":True,
  "testSet":"NOT_ACCESSED","externalData":"NOT_ACCESSED",
  "scientificScope":"ASSOCIATION_AND_OUT_OF_PATIENT_GRADCAM_METRIC_PREDICTABILITY_ONLY"}
 doc=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
 commit_once(s,f"{BASE}/archive_receipt.json",doc)
 print("A6-IJ-03C PRIVATE SCIENTIFIC-ANALYSIS ARCHIVE: PASS")
 print("Original private results:",f"https://github.com/{REPO}/blob/main/{BASE}/analysis.json")
 print("Original status:",report["status"])
 print("This is not lesion evidence dependence or medically validated causal XAI.")

if __name__=="__main__":main()
