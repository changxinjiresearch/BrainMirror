# A6-IJ-04B — Kaggle operation for real equal-input-L2 SCEPT intervention research

**Development status:** experiment prepared, no 04B genuine model inference or 04B private patient results claimed until the Kaggle phases below actually PASS. Code CI uses fabricated images only.

## Locked scientific basis

Unchanged original validation screening: 55 original first-slice P1 FLAIR MRI+WT/brain masks; 33 eligible patients including 26 HGG and 7 LGG, and 22 old-source ineligible cases remain permanently documented in private GitHub. The first 04A fixed patient was ineligible; 04A02 produced only one separately labeled exploratory engineering pilot. Those original records and 03C null incremental-prediction result are not overwritten.

**04B is an entirely new locked study of frozen-model input-response**, not another attempt to improve classifier accuracy, predict Grad-CAM, or prove lesion causal fidelity. For each eligible person it holds the original 16×16 ViT tumor/non-tumor ROI selected strictly from the **original prior image/mask-only screen**. Two operators (Gaussian sigma 3 and healthy-median-fill **directions**) × two doses (25% and 50% of the common attainable L2) × two target ROIs, with identical *actual float32 ROI pixel-space L2* for lesion and healthy control. Each S/J model observes clean and eight matched interventions, giving 18 authentic output-only SCEPT ZIPs per patient, **594 real forwards across 33 people**. Two-model within-patient primary endpoint: absolute true-grade logit-margin response lesion minus control at 50% Gaussian. Report all endpoints including negative values, 2000 patient bootstrap and 2000 patient sign-flip negative controls. Even a positive effect cannot prove tumor-causal dependence without additional OOD, operator, spatial and downstream-neuron intervention checks.

All patient values, exact ROIs, inputs' hashes, original raw SCEPT output ZIPs, model predictions and every source archive receipt stay **PRIVATE**, not in the public software code repository. The SCEPT observation ZIPs contain no original MRI input pixels.

## 0. Kaggle environment and the mandatory pre-model 33-person feasibility audit

In your existing Kaggle Notebook, enable the existing authorized private research `GITHUB_TOKEN` Secret. Start with:

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
subprocess.run([sys.executable,"-m","pip","install","-q",
                "numpy","scipy","timm","requests"],check=True)
env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OPENBLAS_NUM_THREADS"]="1"
env["MKL_NUM_THREADS"]="1"
env["OMP_NUM_THREADS"]="4"
runner=repo/"scept_sdk"/"examples"/"a6_ij_04b_run.py"
subprocess.run([sys.executable,str(runner),"--preflight-only"],env=env,check=True)
```

This source-only audit checks **all 33 eligible original MRI+mask cases** for actual intensity-matched pixel-space L2 feasibility under 2 operators × 2 fixed doses. It performs **zero** frozen-model forwards. The audit is permanently archived in the private research repo as `results/scept_a6_ij/04b_equal_l2_interventions/premodel_input_feasibility.json`; existing 22 screen exclusions are not changed.

Expected:

```text
A6-IJ-04B 33-PERSON EQUAL-L2 PRE-MODEL FEASIBILITY: PASS_33_OF_33_PATIENTS_4_PAIRED_INPUT_L2_DOSES_PREMODEL
A6-IJ-04B COMPLETE 33/33 PRE-MODEL INPUT L2 SOURCE AUDIT: PASS
```

**If any one of the 33 cannot achieve equal L2**, fail closed, preserve the negative input-only record, and STOP; do not secretly reduce/raise dose, alter control ROI, change medical masks, exclude that person or claim whole-cohort science. A separate future amendment would need to precede any revised model work.

## 1. One-patient real-model operational pilot, **not** published as full chunk

After all-33 input preflight PASS, use:

```python
subprocess.run([sys.executable,str(runner),"--pilot"],env=env,check=True)
```

This runs the **first image-only eligible original patient** from the locked prior source list for 18 genuine full-tensor real S/J forwards (2 models × nine conditions) and keeps the complete output locally under `/kaggle/working/a6_ij_04b_private/`. This tests real Kaggle performance and scientific output source matching. This one-patient pilot is not a new scientific finding and is not a durable private first-five-person archive until step 2 finishes.

## 2. First chunk (5 people) and private SHA-256 archival

```python
subprocess.run([sys.executable,str(runner),"--chunk","1"],env=env,check=True)
```

The pilot's locally complete original patient is source-SHA verified and reused; the other four fixed patients are measured. Ninety actual full-tensor original source ZIPs and five original patient case reports are packed in one distinct private Release asset; its complete bytes are re-downloaded from private GitHub and verified by SHA256. Immutable private source case and receipt records are then committed. Expect:

```text
A6-IJ-04B CHUNK 01 PRIVATE ORIGINAL 5 PATIENTS × 18 REAL ZIPs: PASS
```

## 3. All seven patient chunks and true patient-level scientific result

```python
subprocess.run([sys.executable,str(runner),"--all"],env=env,check=True)
```

This **skips** any prior verified private 04B chunks, executes the remaining chunks in original source order 5+5+5+5+5+5+3, and SHA-verifies each complete private Release asset before counting a chunk as complete. It then verifies all 594 source model ZIP original source receipts, all 33 matched people and all original inputs, performs the locked patient-level bootstrap/sign-flip uncertainty and saves the private final result.

Success output:

```text
A6-IJ-04B COMPLETE 33-PATIENT 594-REAL-FORWARD MATCHED-L2 ANALYSIS: PASS
A6-IJ-04B FULL 33-PATIENT 594-REAL-SCEPT MATCHED-INPUT SCIENTIFIC ANALYSIS PIPELINE COMPLETED
```

Private output locations:
- Original all-eligible pre-model source audit: `results/scept_a6_ij/04b_equal_l2_interventions/premodel_input_feasibility.json`;
- Seven independently verified original private source Release shards, each with original `chunks/chunk_XX/paired_cases.json` and `archive_receipt.json`;
- Complete original private patient-level science: `results/scept_a6_ij/04b_equal_l2_interventions/final/analysis.json` and `final/archive_receipt.json`.

**Resource note:** 33 people × 18 genuine ViT forwards = 594 actual neural model observations plus private source ZIP upload/remote hash validation. This is a real Kaggle compute-and-network workload, not an instant query. Expect run time to vary; use free Kaggle resource quota and only run long `--all` after pilot and first chunk verify. Local incomplete patient outputs may be lost across a Kaggle reset, but each **completed private GitHub source chunk** remains resumable. Rerun setup and `--all` after interruptions; do not delete private archived results.
