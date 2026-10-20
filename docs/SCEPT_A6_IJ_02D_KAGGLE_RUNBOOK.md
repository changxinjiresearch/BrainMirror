# A6-IJ-02D — exact Kaggle execution and interpretation

**Status:** Code and synthetic fresh-runner CI available; original patient model reexecution is still waiting for actual Kaggle CPU session. DO NOT claim 02D completion from CI alone.

## Purpose

Finish the first *observational tooling* evidence stage: (1) new-process independent direct-hook replay of the original frozen **P4 ViT S/J** true six pre/post LayerNorm tensors from private original 02C ZIPs on two pinned P0 VAL MRI slices, (2) authentic **ResNet-18** CNN architecture adapter on two real P1 VAL inputs using **random, untrained** deterministic weights (engineering only), and (3) real PyTorch-to-ONNXRuntime CPU tensor parity on a single frozen CNN graph/weights. No TEST data or causal explanation is inspected.

For independent *host environment* checks, the separate public [02D clean-runner GitHub Actions](../.github/workflows/scept_a6_ij_02d_clean_runners.yml) launches Python 3.10/3.11/3.12 on isolated fresh Linux runners, with **synthetic** deterministic convolutional and token-transformer data, not patient MRI. These are reproducibility checks by different software execution contexts, **not by an independently operated external lab**.

## Kaggle prerequisite

Previous A6-IJ-02C Kaggle notebook and existing Kaggle `GITHUB_TOKEN` Secret have authorized read/write on PRIVATE `changxinjiresearch/research-course-brats2020-ijepa-xai`. Token MUST stay in Kaggle Secrets. Kaggle Internet and local free CPU quota must be available. CPU-only is sufficient. No training, new API charges, or paid infrastructure.

The first two lexicographic pinned validation patients are `BraTS20_Training_022` and `BraTS20_Training_040`. Source P0 SHA, P1 patient NPZ full SHA, P4 best S/J source checkpoint full SHA and P4 epoch/protocol metadata are reverified. The original 02C source first-five Release shard is downloaded and hashed, and original nested ZIPs are SHA verified.

## Run in one Kaggle Python notebook cell

```python
import os, sys, subprocess
from pathlib import Path

repo=Path("/kaggle/working/BrainMirror")
branch="phase/scept-sdk-a6-m5-release-qualification"
if repo.is_dir():
    subprocess.run(["git","-C",str(repo),"pull","--ff-only","origin",branch],check=True)
else:
    subprocess.run(["git","clone","--depth","1","--branch",branch,
                    "https://github.com/changxinjiresearch/BrainMirror.git",str(repo)],check=True)

# Use existing free Kaggle Python install; avoid changing frozen clinical methods.
subprocess.run([sys.executable,"-m","pip","install","-q",
                "timm","onnx>=1.15","onnxruntime>=1.17","onnxscript>=0.1"],check=True)

env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OMP_NUM_THREADS"]="4"
env["MKL_NUM_THREADS"]="4"
env["OPENBLAS_NUM_THREADS"]="1"

scripts=repo/"scept_sdk"/"examples"
for task in ("a6_ij_restore_p4.py",
             "a6_ij_02d_measure.py",
             "a6_ij_02d_archive_private.py"):
    print("RUNNING",task,flush=True)
    subprocess.run([sys.executable,str(scripts/task)],env=env,check=True)
```

## Success gates and artifacts

1. `A6-IJ-02D REAL MRI CROSS-ARCH/ENGINE REPLAY: PASS` means both 02C ViT architectures (S and J) independently reexecute and match original patient ZIPs for **all six actual tensors + logits**, and genuine timm ResNet-18 Torch/direct-hook/ONNXRuntime captures are within locked 1e-3 maximum absolute AND 1e-3 relative L2 numeric engine-parity tolerances.
2. `A6-IJ-02D PRIVATE ORIGINAL ASSET ARCHIVE: PASS` means four original CNN MRI-derived output-only ZIPs plus exactly pinned random CNN PyTorch state and exported ONNX model **(six original Release assets)** have all been remotely reread and SHA-256 checked, and an immutable original analysis + manifest + receipt live in PRIVATE GitHub `results/scept_a6_ij/02d_cross_arch_runtime` and dedicated private Release tag `scept-a6-ij-02d-cross-arch-runtime-v1`.
3. The GitHub 02D **three fresh-runner Python matrix jobs** must independently PASS, including mandatory ONNX parity, to claim cross-environment engineering scope. Generic A0–A6 regressions are independent of optional ONNX exporting and may omit it; they do not satisfy the dedicated 02D gate alone.

## Failure/interpretation

Never change numeric thresholds after observing results; preserve actual tracebacks. If GitHub private archive upload fails after local PASS, rerun **the archive script only**. If fresh-environment runner fails, repair tooling/version incompatibility and rerun CI, without altering the scientific P4/P1 definitions. No result here demonstrates clinical validity, model causality, saliency, superiority over Grad-CAM, generality for all neural networks, or an independent human-lab replication. SCEPT SDK M5 is still gated by independently run external operator/licensing release requirements; U10/U11/M4 science remains separately unqualified.
