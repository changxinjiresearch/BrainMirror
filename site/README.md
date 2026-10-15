# BrainMirror v0.4 — Anatomy × Model Observatory

A dual-view interface where **the left may load real anatomical MRI volume data** (NIfTI-1 or a reviewed compact private preview) and **the right displays real recorded PyTorch computation from a synthetic-model experiment**. These are *not linked*. No anatomically trained model or 581-person IXI mean is claimed.

## New website features
- Three anatomy modes: group reference, individual T1, and controlled-experiment workspace (editing remains gated until validated ROI/warps).
- Try to download an independent **MNI152Lin 2 mm reference** from TemplateFlow CDN. If unavailable (e.g., CORS), the UI reports failure; it never substitutes synthetic anatomy silently.
- Import local NIfTI-1 `.nii`/`.nii.gz` (with browser gzip support). Raw voxels stay entirely in the browser; no upload API. Browser preview downsamples to at most 96 voxels on the longest side and applies within-volume display normalization; orientation uses original index axes pending a fully registered anatomical viewer.
- Rotate and zoom an approximate real-voxel-derived point-cloud surface, inspect axial/coronal/sagittal slices, adjust *display-only* threshold, and select the anatomy ROI label when a valid private preview label map is supplied.
- Explicit model selector for BrainAE (available *offline recorded synthetic traces*) and VAE/MAE/ViT (unavailable until separately trained). User cannot mistake BrainAE for another architecture.
- Existing six recorded genuine PyTorch synthetic-input runs, 512 real conv3 tensor units, fixed-unit 216 incoming actual convolution coefficients, ablation record and JSON export retained.

## IXI template pipeline (code prepared, **not run on Kaggle data**)
1. Audit 581 actual IXI T1 files privately, acquire/version corresponding skull-stripped brain masks, review acquisition license and space/subject linkage.
2. Lock subject-level train/dev/test; use **train only** to build a population template. Never use held-out subjects to construct model-training reference.
3. Supply a human-reviewed CSV with `subject_id,image_path,mask_path,split,qc_approved` to `scripts/ixi_template_pipeline.py`, together with a reviewed reference and reference mask. It runs SimpleITK affine registration, checks reference overlap, streams voxelwise mean/SD/coverage, saves source-hashed private QC receipts. This is an **affine exploratory mean**, not a fully nonlinear unbiased template.
4. Independently inspect quality, alignments, statistical coverage, normalcy and licensing. Do not publish actual MRI or individualized data to public Git.
5. Convert an approved brain-only template into a compact **PRIVATE preview** with `scripts/export_anatomy_preview.py` and import the JSON via browser.

```
python scripts/ixi_template_pipeline.py --approved-manifest /private/approved_manifest.csv \
 --reference /private/reference.nii.gz --reference-mask /private/reference_brain_mask.nii.gz \
 --output-dir /private/ixi-template-candidate
python scripts/export_anatomy_preview.py --image /private/ixi-template-candidate/train_affine_mean.nii.gz \
 --brain-mask /private/reference_brain_mask.nii.gz --output /private/ixi-mean-preview.json \
 --kind ixi_affine_group_candidate --source 'IXI train-only affine-aligned candidate'
```

Dependencies for optional template scripts: `numpy`, `scipy`, `SimpleITK`, `nibabel`.

## Current scientific status
- IXI real data: uploaded to user's private Kaggle input; 581 filenames validated by user. Actual voxel QA not complete.
- IXI population-average template: **NOT YET BUILT**.
- MNI152 reference: public external reference import attempted on demand; never confused with IXI.
- Network traces: real **recorded** PyTorch activations, trained only on 8 synthetic phantom samples for 3 epochs; no real anatomy representation claim.
- Live MRI→trained-model linkage: **NOT AVAILABLE**.
- No clinical validation, diagnosis or mechanistic conclusions.