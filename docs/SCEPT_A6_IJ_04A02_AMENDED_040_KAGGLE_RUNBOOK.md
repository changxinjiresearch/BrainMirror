# A6-IJ-04A02 amended VAL 040 exploratory intervention — Kaggle operator

Original `BraTS20_Training_022` fixed 04A-v1 pilot was **BLOCKED** by preregistered healthy-patch matching score 3.14177 > 2.0. Keep that failure immutable. Later, the original **55-patient first-slice source-only** screen found 33 eligible (21 score>2 rejected, one WT/foreground geometry rejected). Its original private Git blob is `fe39ebcc9bc323aa260c6cec4fd497287971d06c`. Original first eligible by ascending patient ID is **`BraTS20_Training_040`** at slice rank1, matching score ~1.799854. This NEW 04A02 pilot is exploratory, not confirmation or a replacement of the originally failed case.

Code is in public BrainMirror for audit, medical numeric sources and all experiment outputs in private research GitHub. Original unchanged tumor ROI 16x16 grid [5,3], control [10,7]. Source NPZ SHA256 `b6737737a34cccfb466f8091b823c98952e7c473f638e3376de93c9ec67e177a`. Frozen P4 Models S/J; identical original 04A input intervention operators and <2.0 source match rule. The new pilot checks original scan Git blob, source SHA, deterministic selection and original 04A failure before running any model.

## Run original frozen-model amended first eligible case and private archive

In existing Kaggle Notebook with authorized private research `GITHUB_TOKEN` Secret enabled (no paid service, CPU sufficient):

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
scripts=repo/"scept_sdk"/"examples"
for task in ("a6_ij_restore_p4.py","a6_ij_04a02_pilot.py","a6_ij_04a02_archive_private.py"):
    print("RUNNING",task,flush=True)
    subprocess.run([sys.executable,str(scripts/task)],env=env,check=True)
print("A6-IJ-04A02 NEW PILOT/PRIVATE ARCHIVE PIPELINE COMPLETED")
```

All original 04A-v1 patient files remain untouched by these two **distinctly named** new programs:
- new local output: `/kaggle/working/a6_ij_04a02_private/`
- new private GitHub report: `results/scept_a6_ij/04a02_amended_040_intervention_pilot/analysis.json`
- new private original ZIP Release: `scept-a6-ij-04a02-amended-val-040-input-intervention-v1`

Expected key log labels:

```text
A6-IJ-04A02 AMENDED 040 INPUT-INTERVENTION PILOT ENGINEERING: PASS
A6-IJ-04A02 AMENDED 040 PRIVATE ORIGINAL PATIENT INTERVENTION ARCHIVE: PASS
```

**Two-model case counts:** 2 models × (clean + lesion/control Gaussian + lesion/control healthy-pool median) = 10 original full-tensor SCEPT ZIPs, plus 2 genuine clean sham-repeat forwards. Each ZIP is reloaded and numeric data consistency checked. The remote private ZIP shard is re-downloaded and whole SHA256 checked.

**Interpretation remains limited**: passing the engineering pilot demonstrates controlled same-size input changes and authentic internal/logit changes in a fixed, screened frozen-model patient. It cannot establish lesion evidence dependence or direct neuronal causal mediation. The selected lesion patch and healthy patch have different original mean image intensity (2.224 vs 0.069) despite the original normalized match-score threshold passing; **input L2 ratios** and operator artifacts must be examined, not treated as comparable exposure magnitudes by default. This result must not be advertised as SCEPT outperforming Grad-CAM.

If GitHub or Kaggle interruptions occur, do not delete old private source archives or modify source patients; report the exact error for diagnosis.
