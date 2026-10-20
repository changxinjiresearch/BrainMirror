# A6-IJ-03C — Kaggle run instructions

A6-IJ-03B is complete (55 people, ten original slices per person, frozen model S/J). A6-IJ-03C reuses the 1100 original SCEPT ZIP archives: 110 from 02B and 990 from 03B. It does not rerun models, Grad-CAM, or sealed P5 test.

## Execution

Run in the existing Kaggle Notebook after enabling the existing authorized private GitHub Secret.

```python
import os, sys, subprocess
from pathlib import Path
repo=Path("/kaggle/working/BrainMirror")
branch="phase/scept-sdk-a6-m5-release-qualification"
if (repo/".git").is_dir():
    subprocess.run(["git","-C",str(repo),"pull","--ff-only","origin",branch],check=True)
else:
    subprocess.run(["git","clone","--depth","1","--branch",branch,
                    "https://github.com/changxinjiresearch/BrainMirror.git",str(repo)],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","numpy","scipy","timm","requests"],check=True)
env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OPENBLAS_NUM_THREADS"]="1"
env["MKL_NUM_THREADS"]="1"
env["OMP_NUM_THREADS"]="4"
subprocess.run([sys.executable,str(repo/"scept_sdk"/"examples"/"a6_ij_03c_run.py")],
               env=env,check=True)
```

## Actions and evidence

The pipeline first downloads, authenticates, and re-analyzes original patient ZIP archives a private five-person batch at a time to calculate exactly matched ten-slice patch RMS. Each completed five-person RMS batch is committed independently into the private research repository. After all eleven RMS chunks are complete, the program executes deterministic leave-one-patient-out Ridge B0/B1/F analysis on IoU10 and TAP differences, 2,000 paired patient bootstrap draws and 100 within-grade shuffled negative controls. All patient values are stored privately. No GPU or extra paid model service is required.

Successful outputs:
- `A6-IJ-03C RMS PRIVATE CHUNK XX: PASS` for the 11 new RMS batches.
- `A6-IJ-03C 55-PATIENT INCREMENTAL + NEGATIVE CONTROLS: PASS`.
- `A6-IJ-03C PRIVATE SCIENTIFIC-ANALYSIS ARCHIVE: PASS`.
- `A6-IJ-03C FULL LOCKED ANALYSIS PIPELINE COMPLETED`.

Final results: private `results/scept_a6_ij/03c_incremental_value/analysis.json` and `archive_receipt.json`; RMS chunks under `rms/chunk_XX.json`. Retrying the driver skips already private-archived RMS chunks. Total download volume is roughly 2 GB across the original release assets, and the existing Kaggle internet connection is required.

**Interpretation:** Success means correct execution, not necessarily positive incremental predictive utility. Even positive improvement is exploratory within the original validation cohort and is NOT proof of lesion evidence dependence, superior clinical explanation, or advantage over Grad-CAM.
