# Kaggle operator instructions — SCEPT A6-IJ-03B

**Current status: code + static CI accepted; real Kaggle pilot and full SCEPT observation PENDING.** Do not label the 03B 1,100-pair research analysis as PASS without the private `final/archive_receipt.json`.

## What this does

Original P4B 1,100 validation Grad-CAM records and original 02B/03A 110 first-slice SCEPT captures already exist and have SHA-256-certified evidence in the private research repo. This phase adds exactly **990 actual frozen S/J ViT-B/16 observations for slice ranks 2–10**, across 55 patients, and tests same-patient/same-slice/same-model matching. It does not rerun Grad-CAM, re-train, or touch P5 TEST/P6 external data.

Each original five-patient chunk contains **100** exact P4B/SCEPT model-slice matches: ten previous slice-rank-1 source observations (private 02B Release), plus **90 new full-tensor SCEPT ZIPs**. After an entire five-patient chunk succeeds, the original 90 new ZIPs and patient records are uploaded as one private Release asset, independently re-downloaded, and FULL SHA256 verified. An immutable receipt in private GitHub authorizes subsequent sessions to skip that entire chunk. There are 11 chunks.

## One cell for the original single-patient operational pilot

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
subprocess.run([sys.executable,"-m","pip","install","-q","timm","scipy"],check=True)
env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OPENBLAS_NUM_THREADS"]="1"
env["MKL_NUM_THREADS"]="1"
env["OMP_NUM_THREADS"]="4"
script=repo/"scept_sdk"/"examples"/"a6_ij_03b_run.py"
subprocess.run([sys.executable,str(script),"--pilot"],env=env,check=True)
```

The pilot runs **one** authentic patient with nine new slices × two frozen models = **18 new forwards**, validates exact original P1/P4B axial ranks and probability agreement, exports/reads each original private full-tensor ZIP, and prints elapsed processing time. **A pilot is not a private five-person archive**; execute the next step as soon as pilot PASS.

## Finish first 5 patients and immediately archive

In the same notebook, re-use `script`, `env` from above:

```python
subprocess.run([sys.executable,str(script),"--chunk","1"],env=env,check=True)
```

This rechecks the first pilot patient and skips its 18 original saved observations; runs the other four patients and uploads all 90 genuine original ZIPs as a private SHA-verified first shard. Look for `A6-IJ-03B CHUNK 01 PRIVATE ARCHIVE: PASS`.

## Resume all remaining chunks and finalize automatically

```python
subprocess.run([sys.executable,str(script),"--all"],env=env,check=True)
```

It inspects all 11 immutable PRIVATE chunk receipts first, skips any fully verified previously published chunks, restores source P4 checkpoints/P1 ZIP once if any work remains, processes and uploads one original patient chunk at a time, then runs the final statistical analysis. Within any Kaggle notebook reset, rerun the initial cell to re-clone the public code and initialize `env`; then run `--all` directly (or manually `--chunk N`). Original 02B/02C/P4B and already archived 03B chunks remain safe in PRIVATE GitHub even when local Kaggle working data disappears.

## Final requirements and honest limits

Final exact success string: `A6-IJ-03B FULL 55 × 10 × TWO FROZEN MODELS: PASS`.

- 55 P0 validation persons, 10 original P1 FLAIR slices each, 2 actual frozen P4 models per slice.
- Full source evidence: 110 previously observed first-slice SCEPT model captures reused and **990 NEW real SCEPT forwards** with original full tensor ZIPs; 1,100 exact patient/slice/model matched source rows and 11 source-GitHub SHA256-verified private ZIP shards.
- Cross-model token CKA, patch cosine/rank, exact original Grad-CAM TAP/IoU10 and classified output source with numerical provenance. Patient-level (not independent slice) bootstrap seed 240216, 2,000 replicates.
- No causal claim, output evidence dependence, medical explanation fidelity, independent external validation, or baseline superiority is established by paired observational correlation.
- First-patient pilot proves neither complete calculation time nor capacity for all 990 forwards. Monitor Kaggle free CPU runtime, private GitHub token permissions, quota and availability. If interrupted midway through a *not yet archived five-person chunk*, some local work can be lost if the Kaggle session resets; do not falsely claim those patient observations persisted.
