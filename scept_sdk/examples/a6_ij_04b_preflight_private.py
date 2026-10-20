"""A6-IJ-04B: mask/image-only ALL 33 prespecified equivalent-L2 dose preflight.

No model forward or Grad-CAM. Authenticates original exact 55-patient
pre-model screening and all original P1 SHA256. Every original eligible
case receives the same 2-operator x 2-dose test. Private report is immutable
and must be a complete 33/33 pass before any new 04B neural inference.
"""
import io
import json
from zipfile import ZipFile
import numpy as np
from a6_ij_03b_measure import github,API
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_04b_measure import source,unpack_val_zip,canonical,sha,check_mask_selection
from a6_ij_04a_interventions import choose,strict_image
from a6_ij_04b_matched_interventions import matched_inputs,fingerprint04b
from a6_ij_04b_archive_private import BASE

STATUS="PASS_33_OF_33_PATIENTS_4_PAIRED_INPUT_L2_DOSES_PREMODEL"
def main():
 s=github()
 people,_,lock=source(s)
 vals,fullzip=unpack_val_zip(s)
 original={r["patient_id"]:r for r in vals}
 rows=[]
 with ZipFile(fullzip) as z:
  for idx,record in enumerate(people,1):
   pid=record["patient_id"]
   item=original.get(pid)
   if item is None or item["p1_sha256"]!=record["originalFullP1NPZSHA256"]:
    raise RuntimeError("All 33 original source NPZ identities must match original eligibility scan")
   raw=z.read(item["zip_member"])
   if sha(raw)!=item["p1_sha256"]:
    raise RuntimeError("Original patient source full NPZ SHA256 mismatch")
   with np.load(io.BytesIO(raw),allow_pickle=False) as data:
    image=np.asarray(data["flair"][0],dtype=np.float32).copy()
    wt=np.asarray(data["wt"][0]).copy()
    fg=np.asarray(data["foreground"][0]).copy()
    axis=int(data["axial_indices"][0])
   if axis!=record["axialIndex"]:
    raise RuntimeError("Unchanged original P1 first MRI slice index changed")
   base={"patient_id":pid,"grade":record["grade"],
         "originalP1NPZSHA256":item["p1_sha256"],"axialIndex":axis,
         "origScreenGitBlobSHA":lock["originalScreenGitBlobSHA"],
         "preOutcome":True}
   try:
    strict_image(image,wt,fg)
    select=choose(image,wt,fg)
    check_mask_selection(select,record)
    _,delta=matched_inputs(image,wt,fg,select)
   except RuntimeError as error:
    base.update({"status":"BLOCKED_EQUAL_INPUT_DOSE_CONDITION",
                 "error":str(error),"inputMatchedConditions":None})
   else:
    base.update({"status":"PASS_EQUAL_INPUT_DOSE_CONDITION",
                 "error":None,"inputMatchedConditions":delta})
   rows.append(base)
   print(f"04B ORIGINAL IMAGE-ONLY PREFLIGHT {idx:02d}/33 "
         f"{pid} {base['status']}",flush=True)
 if len(rows)!=33 or len({r["patient_id"] for r in rows})!=33:
  raise RuntimeError("04B pre-effect input-only patient cohort integrity failure")
 blocked=[r for r in rows if r["status"]!="PASS_EQUAL_INPUT_DOSE_CONDITION"]
 report={"schema":"scept.a6-ij-04b.premodel-input-feasibility.v1",
         "status":STATUS if not blocked else "BLOCKED_AT_LEAST_ONE_ORIGINAL_EQUAL_L2_PATIENT",
         "planSHA256":fingerprint04b(),"sourceLock":lock,
         "nAllOriginalVALScreened":55,"nOriginalEligible":33,"nOriginalExcluded":22,
         "nNewMatchedPass":33-len(blocked),"nNewMatchedBlocked":len(blocked),
         "nNewModelForward":0,"sourceP5Test":"NOT_ACCESSED",
         "patientRecords":rows}
 prefix=f"{BASE}/premodel_input_feasibility.json"
 commit_once(s,prefix,canonical(report))
 print("A6-IJ-04B 33-PERSON EQUAL-L2 PRE-MODEL FEASIBILITY:",report["status"],flush=True)
 print("Original 33 eligible, complete matched:",33-len(blocked),"blocked:",len(blocked),flush=True)
 print("PRIVATE:",prefix)
 if blocked:
  raise RuntimeError("Original 04B dose feasibility BLOCKED; no model allowed until separately preregistered amendment")

if __name__=="__main__":main()
