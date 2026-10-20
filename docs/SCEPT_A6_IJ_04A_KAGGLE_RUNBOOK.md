# A6-IJ-04A real frozen-P4 lesion and non-lesion input intervention pilot

**Status: initial executable pilot; original Kaggle patient experiment must finish before any scientific/engineering PASS is claimed.**

This stage asks a **new** question: what happens to genuine model internal activation tensors and fixed target logits when a frozen original P1 MRI is subject to the same controlled input perturbation in a tumor-overlapping patch versus an intensity-matched non-tumor brain patch?

**Not** an experiment to improve model accuracy or to predict Grad-CAM TAP/IoU10; the negative 03C finding is preserved. **Not** proof of pathological evidence dependence, neuron causal mediation, superiority to Grad-CAM, or clinical validity.

## Locked first patient and interventions

- Original P0 VAL patient `BraTS20_Training_022`, original first P1 FLAIR slice with original nearest-neighbor WT and original foreground masks, and exact S/J frozen P4 ViT-B/16 weights. Sources and original 03A classification probability are SHA-audited in a PRIVATE GitHub repo.
- Select the maximal WT-occupancy ViT-aligned 16x16 grid patch (with preregistered brain fraction and minimum tumor occupancy); independently match an equal-area tumor-free patch with >=3px clearance using an original MRI patch mean/std score. **Source input mask selection NEVER uses the original model's output or the Grad-CAM image**.
- Two **identical operators** on each patch: local Gaussian replacement sigma 3, and original healthy-brain donor-pool median. Keep all original source pixels **outside** the declared patch *bitwise identical*. A no-op sham reexecutes original source to guard forward repeatability.
- Per frozen model: original clean, original lesion Gaussian, healthy Gaussian, lesion median, healthy median = **five complete genuine SCEPT archives**, + a separate no-op sham forward for equality audit. Two models = ten original archives plus two sham forwards.
- Each archive contains full original output tensors for three named blocks, actual model logits, exact checkpoint state SHA256 and SHA-verified ZIP, but **never original MRI input**. Keep ROI positions, patient IDs, input SHA, model logits and report PRIVATE.
- Acceptance is deterministic source identity, same-size and source-limited input perturbations, genuine multi-layer observation + no-op sham, original 03A clean-output agreement and remote immutable SHA256; **no numerical directional effect gate**.
- If fixed eligibility fails (too little lesion patch WT, no valid matched healthy patch, or donor coverage inadequate), report the actual error and keep the original protocol blocked. Do not switch patient or choose a patch based on logits/desired conclusion after observing results.

## Single Kaggle cell for 04A (Kaggle Secret GITHUB_TOKEN must be enabled)

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
                "numpy","scipy","requests","timm"],check=True)

env=os.environ.copy()
env["PYTHONPATH"]=str(repo/"scept_sdk")
env["OPENBLAS_NUM_THREADS"]="1"
env["MKL_NUM_THREADS"]="1"
env["OMP_NUM_THREADS"]="4"

scripts=repo/"scept_sdk"/"examples"
for task in ("a6_ij_restore_p4.py",
             "a6_ij_04a_pilot.py",
             "a6_ij_04a_archive_private.py"):
    print("04A START",task,flush=True)
    subprocess.run([sys.executable,str(scripts/task)],env=env,check=True)
```

The original `a6_ij_restore_p4.py` is idempotent and restores/verifies the two pretrained original frozen P4 checkpoints and source P1 original classification ZIP. It can require a download of about 844 MiB if the Kaggle working directory has reset.

## Success outputs

- `A6-IJ-04A GENUINE 2-MODEL LESION/CONTROL INPUT-INTERVENTION PILOT: PASS` — ten measured real-model condition ZIPs, independent duplicate no-op sham passes.
- `A6-IJ-04A PRIVATE ORIGINAL PATIENT INTERVENTION ARCHIVE: PASS` — ten full original SCEPT ZIPs re-downloaded and full SHA256 checked from PRIVATE Release, source report/plan and audit receipt immutable in PRIVATE research GitHub.

Private report: `results/scept_a6_ij/04a_controlled_input_pilot/analysis.json`. Original raw source archive: private Release `scept-a6-ij-04a-locked-input-intervention-pilot-v1`; immutable receipt at `results/scept_a6_ij/04a_controlled_input_pilot/archive_receipt.json`.

## Next gate, not automatically authorized by technical pilot

04B: register a larger patient/slice intervention cohort with matched non-lesion controls and no ineligible-case cherry-picking. 04C: test explanation specificity using model-output causality and alternative explanation baselines including original Grad-CAM under matched MRI masks. 04D: separate real hidden-activation patching/ablation to test direct internal computational mediation, with scientific controls. All retain sealed P5 TEST and independent P6 boundary.
