# A6-IJ-04D — Real frozen-model internal computational intervention: Kaggle procedure

**Status:** scientific protocol and executable code prepared in SCEPT Research SDK branch; real Kaggle 04D patient intervention data are **not yet produced**. 04C is already completed (33 patients, 594 full original tensor replays). Do not call this 04D scientific study complete until the private original `final/analysis.json` plus `final/archive_receipt.json` pass.

**Research question:** What is the effect on original frozen ViT S/J decisions of replacing an *actual post-residual hidden activation* with the corresponding same-patient original clean/edited activation? This is bounded causal intervention of the **model computation**; not training, model-accuracy improvement, clinical glioma attribution or novel top-tier paper validation.

Original cohort and exact scan/ROI: unchanged previously screened 33 first P1 VAL FLAIR slices (26 HGG/7 LGG), 22 pre-existing input-only exclusions unchanged, original 04B exact same pixel-L2 `gaussian_local_sigma3_dose50` only. Two actual residual output layers `blocks.5` and `blocks.11`; 3 selectors (target patch, opposite ROI placebo, whole 197-token positive control), both source-to-recipient directions (clean donor into edited and edited donor into clean), plus two clean self-patching shams. Every case: 29 genuine frozen-model forwards per model × 2 models = **58**, **1,914** for 33 subjects. All original donor block tensors and actual patched output logits archived in SHA-verified separate PRIVATE release chunks.

## Kaggle environment: one code cell

The Kaggle notebook must have a private GitHub `GITHUB_TOKEN` secret with access to the existing private research repo. Use existing Kaggle session; if it has reset, this cell will re-clone the source and reinstall the package. No GPU is required; CPU float32 is used to keep original source computations comparable. No user API bill.

```python
import os,sys,subprocess
from pathlib import Path
repo=Path("/kaggle/working/BrainMirror")
branch="phase/scept-sdk-a6-m5-release-qualification"
if (repo/".git").is_dir():
    subprocess.run(["git","-C",str(repo),"pull","--ff-only","origin",branch],check=True)
else:
    if repo.exists():raise RuntimeError("Check existing non-Git BrainMirror directory; do not delete patient data")
    subprocess.run(["git","clone","--depth","1","--branch",branch,
        "https://github.com/changxinjiresearch/BrainMirror.git",str(repo)],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q",
                "-e",str(repo/"scept_sdk"),"timm","scipy","requests"],check=True)
env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OPENBLAS_NUM_THREADS"]="1"
env["MKL_NUM_THREADS"]="1"
env["OMP_NUM_THREADS"]="4"
runner=repo/"scept_sdk"/"examples"/"a6_ij_04d_run.py"
subprocess.run([sys.executable,str(runner),"--status"],env=env,check=True)
```

This `--status` performs private source integrity checks only, zero new neural model forwards, and prints the previously completed and verified 04D private chunk indexes.

## Step 1. Exactly one real-patient computational patch pilot

```python
subprocess.run([sys.executable,str(runner),"--pilot"],env=env,check=True)
```

The source-only pilot of the first prespecified original eligible validation patient measures 58 actual S/J frozen graph forwards and validates all donor-source and full-replacement/sham controls; original source donor state NPZ and paired genuine patched logits remain **Kaggle-local** until first whole original 5-patient batch is archived. Does **NOT** count as a permanent experiment chunk.

Expect:
```text
A6-IJ-04D REAL 58 FROZEN S/J DIRECT RESIDUAL GRAPH FORWARDS: PASS ...
04D ONE-PATIENT GRAPH-PATCH PILOT COMPLETE LOCALLY
```

## Step 2. First 5 patients, privately archive complete original source

```python
subprocess.run([sys.executable,str(runner),"--chunk","1"],env=env,check=True)
```

It reuses the first verified local case and computes the remaining original 4 subjects (290 genuine original model forwards for the full 5-patient chunk, including reused pilot), then privately uploads original donor-state NPZ and patch case reports, re-downloads the full Release asset and verifies exact entire SHA256.

Expect:
```text
A6-IJ-04D CHUNK 01 PRIVATE REAL 5 PATIENTS × 58 FROZEN GRAPH FORWARDS: PASS
```

The first case is only reliably reusable after the private full chunk passes; a Kaggle reset before that requires reconstructing any original unarchived cases.

## Step 3. Full original 33 patients (7 private batches), descriptive patient-clustered science

```python
subprocess.run([sys.executable,str(runner),"--all"],env=env,check=True)
```

The runner verifies and skips previously archived complete original private chunks, computes only remaining patients, persists complete 5+5+5+5+5+5+3 study batches and rereads all remote original full SHA256 assets, then runs full patient-clustered descriptive analysis and saves private final.

It will print:
```text
A6-IJ-04D FULL 33-PATIENT 1914-REAL-FROZEN-INTERNAL-PATCH STUDY: PASS
04D ALL 33 PATIENT ORIGINAL 1914-REAL-FORWARD GRAPH INTERVENTION AND PRIVATE AUDIT COMPLETE
```

Output authority — *ONLY private* `changxinjiresearch/research-course-brats2020-ijepa-xai`:
- Source chunks: `results/scept_a6_ij/04d_post_residual_causal_interventions/chunks/chunk_01` … `chunk_07` each `original_cases.json` and `archive_receipt.json`.
- Seven original complete GitHub private Release ZIPs holding patient-specific original postblock donor-state NPZs and real patched output logs, with remote full SHA256 verification.
- Final original report: `results/scept_a6_ij/04d_post_residual_causal_interventions/final/analysis.json`.
- Immutable final receipt: `results/scept_a6_ij/04d_post_residual_causal_interventions/final/archive_receipt.json`.

**Interpretation:** Passing means SCEPT's research intervention plugin executed authentic clean→edited and edited→clean postblock hidden-state manipulations with all authentic graph positive/sham controls and archived all outputs. There is no criterion requiring prediction accuracy improvements, positive tumor bias, or a nonzero difference. Near-zero original clean-versus-edited outputs make *normalized* recovery/insertion fractions scientifically indeterminate, and those patients remain in the original cohort. Neither positive nor null intervention response establishes medical tumor attribution truth, pathology-specific clinical evidence, or superiority to Grad-CAM. Those questions belong to a **later separately designed study**, not A6-IJ-04D.

**Resource planning:** The full 04D is 1,914 real ViT forwards rather than the previous 04B's 594, plus private ZIP storage and SHA re-downloads. Start with pilot and first archived chunk before choosing `--all`, keep Kaggle session alive, and account for available free CPU quota. Rerun after session interruptions using fresh setup; fully archived batches skip safely.

**No top-tier paper experiment design is included here, as requested.**
