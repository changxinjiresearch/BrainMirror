"""04A strict, synthetic-testable frozen 16×16 paired MRI intervention algebra.

Does not load external/private data, does not execute/replace internal layers.
All region selections depend ONLY on P1 image/segmentation, not on model output,
Grad-CAM or SCEPT difference. First-patient real experiment imports this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

SIDE=224
PATCH=16
GRID=14
OPS=("gaussian_local_sigma3","healthy_pool_median_fill")
PLAN={
 "schema":"scept.a6-ij-04a.intervention.v1",
 "pilotPatient":"BraTS20_Training_022",
 "sliceRank":1,
 "frozenModels":["S","J"],
 "roiGeometry":"one exactly ViT-aligned 16x16 square on original 224x224 FLAIR",
 "selection":"max WT occupancy among >=0.85-brain patches; raster-tie",
 "lesionWTMin":0.25,
 "foregroundOccupancyMin":0.85,
 "controlSelection":"WT-dilated-3 zero and >=0.85 brain; minimize intensity mean+std standardized gap; raster-tie",
 "controlMaxIntensityMatchScore":2.0,
 "healthyPool":"foreground and outside WT 3pixel dilation, >=512 donor pixels",
 "operators":list(OPS),
 "sigmaPixels":3.0,
 "fullRawSCEPTZip":True,
 "genuineMeasuredLayers":["blocks.0.norm1","blocks.5.norm1","blocks.11.norm1"],
 "classifierTarget":"fixed source P0 true-grade binary logit, not output-selected class",
 "interpretation":"Input perturbation response engineering. Does NOT prove neuron causality, natural counterfactual or medically faithful explanations.",
 "sourceP5":"NOT_ACCESSED",
 "sourceP6":"NOT_ACCESSED"
}

def fingerprint():
 return hashlib.sha256(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def strict_image(image,wt,foreground):
 x=np.asarray(image)
 a=np.asarray(wt)
 f=np.asarray(foreground)
 if x.shape!=(SIDE,SIDE) or x.dtype!=np.float32 or not np.isfinite(x).all():
  raise RuntimeError("P1 original FLAIR 224x224 float32 finite required")
 if a.shape!=x.shape or f.shape!=x.shape:
  raise RuntimeError("P1 original segmentation/brain geometry differs")
 if not np.isin(a,[0,1]).all() or not np.isin(f,[0,1]).all():
  raise RuntimeError("WT and foreground masks must be binary")
 a=a.astype(bool,copy=False)
 f=f.astype(bool,copy=False)
 if np.any(a & ~f):
  raise RuntimeError("Original WT is not contained within original brain foreground")
 if not a.any() or np.count_nonzero(f)<1000:
  raise RuntimeError("Original P1 WT or brain mask unexpectedly empty")
 return x,a,f

def rect(r,c):
 return (slice(r*PATCH,(r+1)*PATCH),slice(c*PATCH,(c+1)*PATCH))

def choose(image,wt,foreground):
 x,wt,fg=strict_image(image,wt,foreground)
 blocked=binary_dilation(wt,iterations=3)
 candidates=[];norm=np.std(x[fg].astype(np.float64))
 if norm<=1e-8:raise RuntimeError("Near constant brain input cannot support matched control")
 for r in range(GRID):
  for c in range(GRID):
   idx=rect(r,c)
   brain_fraction=float(np.mean(fg[idx]))
   frac=float(np.mean(wt[idx]))
   block=x[idx].astype(np.float64)
   obj={"row":r,"col":c,"foregroundFraction":brain_fraction,
        "wtFraction":frac,"imageMean":float(np.mean(block)),
        "imageStd":float(np.std(block)),
        "dilatedWTFraction":float(np.mean(blocked[idx]))}
   candidates.append(obj)
 tumors=[q for q in candidates if q["foregroundFraction"]>=PLAN["foregroundOccupancyMin"] and
         q["wtFraction"]>=PLAN["lesionWTMin"]]
 if not tumors:raise RuntimeError("04A_FIXED_LESION_PATCH_INELIGIBLE: cannot relax preregistered WT occupancy")
 lesion=sorted(tumors,key=lambda q:(-q["wtFraction"],q["row"],q["col"]))[0]
 healthy=[]
 for q in candidates:
  if q["foregroundFraction"]<PLAN["foregroundOccupancyMin"] or q["dilatedWTFraction"]!=0:
   continue
  score=(abs(q["imageMean"]-lesion["imageMean"])+
         abs(q["imageStd"]-lesion["imageStd"]))/(norm+1e-6)
  healthy.append((score,q["row"],q["col"],q))
 if not healthy:raise RuntimeError("04A_NO_MATCHED_BRAIN_CONTROL: no separate tumor-free ViT patch")
 score,_,_,control=min(healthy,key=lambda z:(z[0],z[1],z[2]))
 if score>PLAN["controlMaxIntensityMatchScore"]:
  raise RuntimeError(f"04A_MATCHING_SCORE_EXCEEDED: score={score:.6g} threshold={PLAN['controlMaxIntensityMatchScore']}")
 pool=fg & ~blocked
 if int(np.count_nonzero(pool))<512:
  raise RuntimeError("04A_HEALTHY_DONOR_POOL_INELIGIBLE: fewer than 512 original brain pixels")
 donor=float(np.median(x[pool].astype(np.float64)))
 if not np.isfinite(donor):raise RuntimeError("Healthy donor pool median nonfinite")
 return {"lesion":lesion,"control":control,"matchScore":float(score),
         "globalForegroundStd":float(norm),"healthyDonorPixels":int(pool.sum()),
         "healthyDonorMedian":donor}

def mutate(image,wt,foreground,selection,arm,operator):
 x,wt,fg=strict_image(image,wt,foreground)
 if arm not in ("lesion","control") or operator not in OPS:
  raise RuntimeError("Only preregistered paired ROI/operator arms supported")
 info=selection[arm]
 r,c=int(info["row"]),int(info["col"])
 if r not in range(GRID) or c not in range(GRID):
  raise RuntimeError("Preregistered ROI out of range")
 idx=rect(r,c)
 if arm=="control" and (wt[idx].any() or binary_dilation(wt,iterations=3)[idx].any()):
  raise RuntimeError("Control ROI overlaps actual WT or 3px tumor safety margin")
 if arm=="lesion" and float(np.mean(wt[idx]))<PLAN["lesionWTMin"]:
  raise RuntimeError("Lesion ROI is no longer tumor-targeted")
 shifted=x.copy()
 if operator=="gaussian_local_sigma3":
  smooth=gaussian_filter(x,sigma=PLAN["sigmaPixels"],mode="reflect")
  shifted[idx]=smooth[idx]
 else:
  shifted[idx]=np.float32(selection["healthyDonorMedian"])
 if shifted.dtype!=np.float32 or not np.isfinite(shifted).all():
  raise RuntimeError("Perturbation violated original tensor format")
 untouched=np.ones_like(wt,dtype=bool);untouched[idx]=False
 if not np.array_equal(x[untouched],shifted[untouched]):
  raise RuntimeError("Changed original source pixels OUTSIDE declared 16x16 intervention ROI")
 change=shifted[idx].astype(np.float64)-x[idx].astype(np.float64)
 inorm=float(np.linalg.norm(change))
 if inorm<=1e-8:raise RuntimeError("Perturbation does not actually change ROI pixel data")
 return shifted,{"arm":arm,"operator":operator,"position":[r,c],
                 "roiPixels":256,"changedPixelCount":int(np.count_nonzero(change)),
                 "inputROIChangeL2":inorm,
                 "inputROIChangeMeanAbsolute":float(np.mean(np.abs(change))),
                 "outsideROIBitwiseSame":True,
                 "wtFraction":float(np.mean(wt[idx])),
                 "foregroundFraction":float(np.mean(fg[idx]))}

def score_tensors(original,changed,selected_rc):
 old=np.asarray(original,dtype=np.float64)
 new=np.asarray(changed,dtype=np.float64)
 if old.shape!=(1,197,768) or new.shape!=old.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
  raise RuntimeError("Actual ViT original/intervention tensor malformed")
 # Patch-only model-state aggregate, keeping CLS separate for transparency.
 clean=old[0,1:,:];other=new[0,1:,:]
 diff=other-clean
 position=selected_rc[0]*GRID+selected_rc[1]
 a=clean[position];b=other[position]
 cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>1e-12 and np.linalg.norm(b)>1e-12 else None
 return {"patchDeltaRMS":float(np.sqrt(np.mean(diff**2))),
         "patchRelativeL2":float(np.linalg.norm(diff)/max(np.linalg.norm(clean),1e-12)),
         "directTargetedTokenDeltaL2":float(np.linalg.norm(b-a)),
         "directTargetedTokenCosineCleanVsPerturbed":cosine,
         "clsTokenDeltaL2":float(np.linalg.norm(new[0,0,:]-old[0,0,:]))}
