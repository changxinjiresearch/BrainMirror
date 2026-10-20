# A6-IJ-04A02 — respond to locked ROI matching failure without threshold tuning

Original **04A fixed patient** `BraTS20_Training_022` VAL P1 first slice was formally ineligible with **matching score 3.14177 > locked max 2.0**. Original failure was committed privately at `results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json`. It is **NOT** a neural experiment PASS and must never be overwritten or quietly dropped.

A6-IJ-04A02 is a new **post-failure, explicitly documented, pre-model-effect feasibility amendment**. It audits every one of the original **55 P0 VAL** first P1 slices, using the **same** locked ROI and intensity-matching rule and threshold. It does not invoke ViT/SCEPT/Grad-CAM. All 55 patient selection successes/failures, exact original P1 NPZ SHA256 and original blocked case linkage are saved ONLY to the private research repository. The output may produce a candidate for a distinctly labeled exploratory 04A02 pilot by first eligible patient ID. Such a pilot will **not** change the original 04A-v1 BLOCKED result and is not independent or confirmatory testing.

## Single Kaggle cell

Use your existing Notebook and existing authorized private `GITHUB_TOKEN` Secret. The full source P1 patient ZIP is 189 MiB and the restorer will download frozen models if not already in the working directory.

```python
import os,sys,subprocess
from pathlib import Path
repo=Path("/kaggle/working/BrainMirror")
branch="phase/scept-sdk-a6-m5-release-qualification"
if (repo/".git").is_dir():
    subprocess.run(["git","-C",str(repo),"pull","--ff-only","origin",branch],check=True)
else:
    subprocess.run(["git","clone","--depth","1","--branch",branch,
                    "https://github.com/changxinjiresearch/BrainMirror.git",str(repo)],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","numpy","scipy","requests","timm"],check=True)
env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OPENBLAS_NUM_THREADS"]="1"
env["MKL_NUM_THREADS"]="1"
env["OMP_NUM_THREADS"]="4"
EX=repo/"scept_sdk"/"examples"
# Original verified P1+frozen P4 source restoration (no model forward).
subprocess.run([sys.executable,str(EX/"a6_ij_restore_p4.py")],env=env,check=True)
# Locked first-slice mask-only VAL eligibility audit.
subprocess.run([sys.executable,str(EX/"a6_ij_04a02_screen_val_private.py")],env=env,check=True)
```

## Accept / interpret

Look for either:
- `A6-IJ-04A02 COMPLETE FIXED-THRESHOLD VAL INPUT-ONLY FEASIBILITY: PASS_COMPLETE_MASK_ONLY_SCREEN` with exact eligible count, or
- `A6-IJ-04A02 COMPLETE FIXED-THRESHOLD VAL INPUT-ONLY FEASIBILITY: BLOCKED_NO_ELIGIBLE_FIRST_SLICES`.

A completed feasibility report is a legitimate scientific artifact even if no first slice is eligible. The source script uploads every one of 55 screening records, including rejected original fixed case, to private `results/scept_a6_ij/04a02_original_mask_only_feasibility/full_val_first_slice_screen.json`. It will fail rather than silently broaden the score cutoff, rechoose the original patient based on model effects, ignore source SHA problems, or access original P5 sealed TEST.

No scientific evidence-dependence result is reported at this stage. If the screen finds an eligible image-only patient, the next task is to create a distinct **04A02 revised pilot** with permanently pinned selected original patient ID, unchanged intervention operators and unique private artifacts, then run frozen S/J and audit magnitude/specificity. If no eligible patient exists, do **not** brute-force a PASS: develop a separately preregistered control design appropriate to real glioma intensity and OOD/perturbation magnitude.
