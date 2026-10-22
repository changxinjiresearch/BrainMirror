"""A6-IJ-04B pre-outcome equal-L2 pairwise dose algebra (NO model, NO patient files).

Scientific protocol is in docs/SCEPT_A6_IJ_04B_LOCKED_PROTOCOL.md.
The old 04A source patch selection and operators remain unchanged;
only new interpolation doses are created. All evaluation is per patient.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from a6_ij_04a_interventions import PLAN as ROI_PLAN,OPS,choose,mutate,rect,fingerprint

DOSES=(0.25,0.50)
LAYERS=("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")
PLAN={
 "schema":"scept.a6-ij-04b.plan.v1",
 "original04Av1ROISHA256":fingerprint(),
 "cohort":"33 image-only eligible of 55 locked P0 VAL first P1 slices",
 "nEligible":33,"nExcluded":22,"gradeCounts":{"HGG":26,"LGG":7},
 "models":["S","J"],"originalOperators":list(OPS),"fractionalDoses":list(DOSES),
 "roi":"old original WT-positive vs intensity matched tumor-free 16x16 ViT block",
 "matching":"for each operator/patient common=min(original lesion/control full ROI L2); lambda=dose*common/ownFullL2",
 "l2AbsoluteTolerance":1e-5,"l2RelativeTolerance":1e-5,"minimumSourceL2":1e-6,
 "forwardsPerPatient":18,"totalRealSCEPTSourceForwards":594,
 "layers":list(LAYERS),
 "primaryEndpoint":"mean over S/J of [abs(delta true-grade logit MARGIN) lesion - control], gaussian_local_sigma3, dose 0.50, patient first",
 "statisticalUnit":"person; 33 independent patient mean paired contrasts",
 "bootstrapResamples":2000,"signFlipNegatives":2000,"randomSeed":240216,
 "originalTest":"NOT_ACCESSED","externalP6":"NOT_ACCESSED",
 "interpretation":"input perturbation response only; no causal neuronal mediation or clinical fidelity claim",
}
def fingerprint04b():
 return hashlib.sha256(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def condition_name(operator,dose,arm):
 if operator not in OPS or dose not in DOSES or arm not in ("lesion","control"):
  raise RuntimeError("Only predeclared intervention/operator/dose/ROI")
 return f"{operator}_dose{int(dose*100):02d}_{arm}"
def matched_inputs(image,wt,foreground,selection):
 """Return 8 fixed float32 images and exact original input intervention report.

 Raises RuntimeError on any violated physical-input matching invariant;
 does NOT choose other patients or adjust dose.
 """
 original=np.asarray(image)
 if original.shape!=(224,224) or original.dtype!=np.float32 or not np.isfinite(original).all():
  raise RuntimeError("Expected exact unchanged P1 first FLAIR source")
 ims={"clean":original.copy()}
 meta={}
 for operator in OPS:
  proposals={}
  full={}
  for arm in ("lesion","control"):
   target,details=mutate(original,wt,foreground,selection,arm,operator)
   proposals[arm]=target
   full[arm]=float(details["inputROIChangeL2"])
  common=min(full.values())
  if common<=PLAN["minimumSourceL2"]:
   raise RuntimeError("LOCKED_04B_UNREACHABLE_EQUAL_INPUT_L2: natural pair dose is zero")
  for dose in DOSES:
   pair={}
   for arm in ("lesion","control"):
    key=condition_name(operator,dose,arm)
    alpha=float(dose*common/full[arm])
    if not (0<alpha<=1):
     raise RuntimeError("Original 04B fractional intervention strength impossible")
    r=selection[arm]["row"];c=selection[arm]["col"]
    sl=rect(r,c)
    x=original.copy()
    diff=proposals[arm][sl].astype(np.float64)-original[sl].astype(np.float64)
    x[sl]=(original[sl].astype(np.float64)+alpha*diff).astype(np.float32)
    outside=np.ones((224,224),dtype=bool);outside[sl]=False
    if not np.array_equal(x[outside],original[outside]):
     raise RuntimeError("4B changed original source OUTSIDE the preregistered ROI")
    observed_l2=float(np.linalg.norm(x[sl].astype(np.float64)-original[sl].astype(np.float64)))
    if observed_l2<=1e-10:
     raise RuntimeError("4B equal-L2 perturbation rounded to zero in float32")
    ims[key]=x
    pair[arm]={"operator":operator,"dose":dose,"arm":arm,
      "position":[int(r),int(c)],"roiPixels":256,
      "sourceFullReplacementL2":full[arm],"commonSourceMaxL2":common,
      "fractionalInterpolationAlpha":alpha,
      "targetPixelL2":dose*common,"actualPixelL2":observed_l2,
      "changedPixelCount":int(np.count_nonzero(x[sl]!=original[sl])),
      "outsideROIBitwiseSame":True,
      "inputROIAbsMeanShift":float(np.mean(np.abs(x[sl].astype(np.float64)-original[sl].astype(np.float64)))),
      "lesionWTFraction":float(selection[arm]["wtFraction"]),
      "originalP1IntensityMean":float(selection[arm]["imageMean"])}
   l=pair["lesion"]["actualPixelL2"];c=pair["control"]["actualPixelL2"]
   maxerr=max(PLAN["l2AbsoluteTolerance"],PLAN["l2RelativeTolerance"]*common)
   if abs(l-c)>maxerr or abs(l-dose*common)>maxerr or abs(c-dose*common)>maxerr:
    raise RuntimeError(f"LOCKED_04B_ACTUAL_INPUT_MAGNITUDE_MISMATCH: delta={abs(l-c):.8g} tolerance={maxerr:.8g}")
   for arm in ("lesion","control"):
    meta[condition_name(operator,dose,arm)]=pair[arm]
 if len(ims)!=9 or len(meta)!=8:
  raise RuntimeError("Exactly 1 clean and eight dose-matched original source images expected")
 return ims,meta

def real_output(logits,grade):
 v=np.asarray(logits,dtype=np.float64).reshape(-1)
 if v.shape!=(2,) or not np.isfinite(v).all() or grade not in ("HGG","LGG"):
  raise RuntimeError("Only genuine finite frozen P4 binary logits with P0 grade")
 label=1 if grade=="HGG" else 0
 p=float(1/(1+np.exp(np.clip(v[0]-v[1],-500,500))))
 return {"logits":v.tolist(),"pHgg":p,"trueGradeLogit":float(v[label]),
         "trueGradeMargin":float(v[label]-v[1-label])}
def contrast_from_outputs(clean,other,lesion_meta,control_meta):
 for x in (clean,*other.values()):
  if set(x)!={"logits","pHgg","trueGradeLogit","trueGradeMargin"}:
   raise RuntimeError("Original model output analysis schema incomplete")
 return {
  "signedMarginChangeLesion":float(other["lesion"]["trueGradeMargin"]-clean["trueGradeMargin"]),
  "signedMarginChangeControl":float(other["control"]["trueGradeMargin"]-clean["trueGradeMargin"]),
  "absMarginSpecificityLesionMinusControl":float(abs(other["lesion"]["trueGradeMargin"]-clean["trueGradeMargin"])-
                                             abs(other["control"]["trueGradeMargin"]-clean["trueGradeMargin"])),
  "absTrueGradeLogitSpecificityLesionMinusControl":float(abs(other["lesion"]["trueGradeLogit"]-clean["trueGradeLogit"])-
                                              abs(other["control"]["trueGradeLogit"]-clean["trueGradeLogit"])),
  "lesionActualPixelL2":float(lesion_meta["actualPixelL2"]),
  "controlActualPixelL2":float(control_meta["actualPixelL2"])
 }
