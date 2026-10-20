"""04A-02 amendment: FULL P0 validation first-slice MASK-ONLY eligibility audit.

Preserve old v1 fixed-patient BLOCKED unchanged. Test all 55 original locked
P1 VAL MRI+WT+brain masks with ORIGINAL 04A ROI rule/threshold. No model or
Grad-CAM invocation; choose first eligible only by frozen patient ID order.
Every failure and exact source checksum stays PRIVATE in research repository.
"""
from __future__ import annotations
import hashlib
import io
import json
from zipfile import ZipFile
import numpy as np

from a6_ij_02b_measure import manifest_and_zip,private_contents
from a6_ij_03b_measure import github
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_04a_interventions import PLAN,fingerprint,choose,strict_image

BASE="results/scept_a6_ij/04a02_original_mask_only_feasibility"
PREVIOUS="results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json"
STATUS_ELIGIBLE="ELIGIBLE_UNDER_UNCHANGED_V1_RULE"
STATUS_INELIGIBLE="INELIGIBLE_UNDER_UNCHANGED_V1_RULE"
PILOT="BraTS20_Training_022"

def sha(raw):
 return hashlib.sha256(raw).hexdigest()

def canonical(data):
 return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")

def screen_record(patient,grade,source_sha,axial,x,wt,fg):
 # Pure original P1 image/mask QA. Keep UNMODIFIED WT/foreground arrays;
 # a failed original-v1 containment precondition is a separately reported
 # ineligibility, NOT blanket corrupted original MRI and NOT a valid match.
 record={"patient_id":patient,"grade":grade,"sliceRank":1,
         "axialIndex":int(axial),"originalFullP1NPZSHA256":source_sha,
         "original04APlanSHA256":fingerprint(),
         "modelsRun":0}
 # Record diagnostic overlap, but do NOT intersect/dilate/change original masks.
 # The strict_image() routine below remains the single original-v1 gate.
 if np.asarray(wt).shape!=(224,224) or np.asarray(fg).shape!=(224,224):
  raise RuntimeError("Original P1 WT/foreground 224x224 shape corrupt")
 original_wt=np.asarray(wt)>0
 original_fg=np.asarray(fg)>0
 outside=int(np.count_nonzero(original_wt & ~original_fg))
 total=int(np.count_nonzero(original_wt))
 record["originalMaskGeometry"]={
  "wtPixels":total,
  "foregroundPixels":int(np.count_nonzero(original_fg)),
  "wtPixelsOutsideForeground":outside,
  "wtOutsideForegroundFraction":float(outside/max(total,1)),
  "masksLeftUnchanged":True}
 try:
  x,wt,fg=strict_image(x,wt,fg)
  chosen=choose(x,wt,fg)
 except RuntimeError as exc:
  message=str(exc)
  expected=("04A_FIXED_LESION_PATCH_INELIGIBLE",
            "04A_NO_MATCHED_BRAIN_CONTROL",
            "04A_MATCHING_SCORE_EXCEEDED",
            "04A_HEALTHY_DONOR_POOL_INELIGIBLE")
  if message=="Original WT is not contained within original brain foreground":
   if outside<=0:
    raise RuntimeError("Containment failure without measured WT/foreground mismatch")
   record.update({
    "status":"INELIGIBLE_ORIGINAL_V1_MASK_CONTAINMENT",
    "ineligibilityReason":message,
    "regionSelection":None,
    "scientificInterpretation":"Original input mask containment precondition fails; no claim original P1 scan is corrupted"})
  elif message=="Original P1 WT or brain mask unexpectedly empty":
   record.update({
    "status":"INELIGIBLE_ORIGINAL_V1_EMPTY_WT_OR_TOO_SMALL_FOREGROUND",
    "ineligibilityReason":message,
    "regionSelection":None,
    "scientificInterpretation":"Exact original-v1 eligibility precondition fails, with no source file edits"})
  elif message.startswith(expected):
   record.update({"status":STATUS_INELIGIBLE,
                  "ineligibilityReason":message,
                  "regionSelection":None})
  else:
   # Bad encoding, non-finite MRI or unexpected input errors are source
   # integrity problems and still abort rather than become eligibility labels.
   raise
 else:
  if outside!=0:raise RuntimeError("Invalid original-v1 accepted ROI geometry")
  record.update({"status":STATUS_ELIGIBLE,
                 "ineligibilityReason":None,
                 "regionSelection":chosen})
 return record

def validate_cohort(records,prior):
 if len(records)!=55 or [r["patient_id"] for r in records]!=sorted(r["patient_id"] for r in records):
  raise RuntimeError("04A02 must screen exactly original 55 sorted validation patients")
 if len({r["patient_id"] for r in records})!=55:
  raise RuntimeError("04A02 duplicate original patient IDs")
 if sum(r["grade"]=="HGG" for r in records)!=44 or sum(r["grade"]=="LGG" for r in records)!=11:
  raise RuntimeError("04A02 P0 original validation grade distribution changed")
 original=next((r for r in records if r["patient_id"]==PILOT),None)
 if original is None or original["status"]!=STATUS_INELIGIBLE:
  raise RuntimeError("04A02 must preserve prior fixed-patient ROI ineligibility")
 if not original["ineligibilityReason"].startswith("04A_MATCHING_SCORE_EXCEEDED:"):
  raise RuntimeError("04A02 original prior fixed patient rejection rule changed")
 if (prior.get("patient_id")!=PILOT or prior.get("sliceRank")!=1 or
     prior.get("interventionPlanSha256")!=fingerprint() or
     prior.get("status")!="BLOCKED_FIXED_PATIENT_ROI_ELIGIBILITY" or
     prior.get("originalP1PatientNPZFullSHA256")!=original["originalFullP1NPZSHA256"] or
     prior.get("failedFixedCriterion")!=original["ineligibilityReason"]):
  raise RuntimeError("04A02 would rewrite old v1 fixed-patient BLOCKED outcome")
 return True

def main():
 session=github()
 prior_raw=private_contents(session,PREVIOUS)
 prior=json.loads(prior_raw)
 # Never run 04A02 on an unrecorded first failure.
 if (prior.get("patient_id")!=PILOT or
     prior.get("status")!="BLOCKED_FIXED_PATIENT_ROI_ELIGIBILITY" or
     prior.get("interventionPlanSha256")!=fingerprint()):
  raise RuntimeError("Original 04A fixed-patient failure not anchored/unchanged")
 locked,zip_path=manifest_and_zip(session)
 if len(locked)!=55 or locked[0]["patient_id"]!=PILOT:
  raise RuntimeError("Locked P0 original validation cohort has changed")
 records=[]
 with ZipFile(zip_path) as z:
  for idx,item in enumerate(locked,1):
   # Check full original SHA256 for each patient NPZ, not only ZIP prefix.
   raw=z.read(item["zip_member"])
   original_sha=sha(raw)
   if original_sha!=item["p1_sha256"]:
    raise RuntimeError("Original P1 validation NPZ SHA mismatch: "+item["patient_id"])
   with np.load(io.BytesIO(raw),allow_pickle=False) as data:
    needed={"flair","wt","foreground","axial_indices"}
    if not needed.issubset(set(data.files)):
     raise RuntimeError("Original P1 NPZ key is missing, source corrupt")
    if (data["flair"].shape!=(10,224,224) or
        data["wt"].shape!=(10,224,224) or
        data["foreground"].shape!=(10,224,224) or
        data["axial_indices"].shape!=(10,)):
     raise RuntimeError("Original P1 frozen slice/mask geometry differs")
    im=np.asarray(data["flair"][0],dtype=np.float32).copy()
    wt=np.asarray(data["wt"][0]).copy()
    fg=np.asarray(data["foreground"][0]).copy()
    axial=int(data["axial_indices"][0])
   report=screen_record(item["patient_id"],item["grade"],original_sha,axial,im,wt,fg)
   records.append(report)
   print(f"04A02 input-only {idx:02d}/55 {item['patient_id']} "
         f"{report['status']}"
         +(f" matched-score={report['regionSelection']['matchScore']:.6f}"
           if report["status"]==STATUS_ELIGIBLE else
           f" reason={report['ineligibilityReason']}"),flush=True)
 validate_cohort(records,prior)
 eligible=[r for r in records if r["status"]==STATUS_ELIGIBLE]
 summary={
  "schema":"scept.a6-ij-04a02.input-only-feasibility.v1",
  "status":"PASS_COMPLETE_MASK_ONLY_SCREEN" if eligible else "BLOCKED_NO_ELIGIBLE_FIRST_SLICES",
  "protocol":"04A-02 prospective cohort-screening amendment AFTER original 04A fixed-case BLOCKED",
  "original04Av1FailureSHA256":sha(prior_raw),
  "original04APlan":PLAN,
  "original04APlanSHA256":fingerprint(),
  "cohort":"original P0 55 VAL original first P1 FLAIR slices only",
  "nScreened":55,"nEligible":len(eligible),"nIneligible":55-len(eligible),
  "nOriginalV1MaskContainmentFailures":sum(r["status"]=="INELIGIBLE_ORIGINAL_V1_MASK_CONTAINMENT" for r in records),
  "nOriginalV1EmptyMaskOrSmallForegroundFailures":sum(r["status"]=="INELIGIBLE_ORIGINAL_V1_EMPTY_WT_OR_TOO_SMALL_FOREGROUND" for r in records),
  "totalOriginalWTPixelsOutsideForeground":sum(r["originalMaskGeometry"]["wtPixelsOutsideForeground"] for r in records),
  "selectionRule":"First eligible by original ascending P0 VAL patient_id, matching unchanged original 04A input-only rules; never select by model output",
  "exploratoryNextPilot":{"patient_id":eligible[0]["patient_id"],
                          "grade":eligible[0]["grade"],"sliceRank":1,
                          "sourceP1FullSHA256":eligible[0]["originalFullP1NPZSHA256"]}
                          if eligible else None,
  "patientRecords":records,
  "frozenModelsRun":0,"gradcamCalls":0,"sceptCalls":0,
  "sealedP5":"NOT_ACCESSED","externalP6":"NOT_ACCESSED",
  "scientificConclusion":"NONE: original v1 mask containment failures are separately counted as ineligible; no patient mask altered; no decision evidence dependence"
 }
 orig=canonical(summary)
 target=f"{BASE}/full_val_first_slice_screen.json"
 commit_once(session,target,orig)
 # A separate source pointer, no public patient metadata.
 print("A6-IJ-04A02 COMPLETE FIXED-THRESHOLD VAL INPUT-ONLY FEASIBILITY:",summary["status"],flush=True)
 print(f"Screened 55, eligible={len(eligible)}, ineligible={55-len(eligible)}",flush=True)
 print("Original fixed 04A BLOCKED preserved; exact original rule unchanged",flush=True)
 if eligible:
  print("Next exploratory first eligible original patient (image-only, no models):",eligible[0]["patient_id"],flush=True)
 print("PRIVATE original input eligibility:",
       f"https://github.com/changxinjiresearch/research-course-brats2020-ijepa-xai/blob/main/{target}",flush=True)

if __name__=="__main__":
 main()
