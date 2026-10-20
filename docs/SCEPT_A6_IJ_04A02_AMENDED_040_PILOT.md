# A6-IJ-04A-02 — amended first-eligible original-VAL exploratory intervention pilot

**Version:** independent post-screen amendment, not a restart/repair of the original 04A pilot.  
**Status:** code prepared, first real Kaggle ViT observations **not yet executed**.

## Pre-model source screening, immutable outcomes

Original 04A pilot fixed patient `BraTS20_Training_022`, first P1 VAL slice. The original independent matching rule failed with matching score `3.14177` exceeding preregistered `2.0`. The original failed-case evidence remains at PRIVATE `results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json`.

04A-02 performed an explicit all-55 mask-and-FLAIR-only pre-effect feasibility scan, stored at PRIVATE `results/scept_a6_ij/04a02_original_mask_only_feasibility/full_val_first_slice_screen.json`, original **Git blob SHA** `fe39ebcc9bc323aa260c6cec4fd497287971d06c`. The initial 55 cases were never selected by model output or Grad-CAM:

- 55 locked P0 validation patients screened, original first P1 FLAIR slice only.
- 33 eligible (26 HGG, 7 LGG); 22 ineligible, of whom **21** had original intensity-match score exceeding 2.0, and **1** had original WT not completely within original brain-foreground (65 original WT pixels outside). Unmodified source masks, no clinical claim of corrupted data.
- The **first eligible patient in pre-locked ascending original P0 patient ID order** is `BraTS20_Training_040` (HGG), first slice `axial_index=55`. Source original P1 full NPZ SHA256 `b6737737a34cccfb466f8091b823c98952e7c473f638e3376de93c9ec67e177a`.
- Original pre-model `040` lesion ROI (16×16 ViT patch, zero-based grid) = `row=5,col=3`, WT occupancy 1.0, brain-foreground 1.0; matched nonlesion ROI `row=10,col=7`, WT occupancy zero and 3px dilated WT overlap zero, brain fraction 0.859375. The original intensity matching score is **1.799853600294234**, below unchanged locked cutoff 2.0; donor pool 11,754 original healthy brain pixels.
- **Scientific warning:** matching score below 2.0 does not mean the two ROIs have equal original MRI intensity or induce equal out-of-distribution shifts. At this selected patient, original lesion mean was ~2.224 and control mean ~0.0687 on the source P1 normalized FLAIR; the same operator may create different input L2 magnitudes and artifacts. Record those, avoid attributing simple output contrast to tumor evidence.

## Distinct exploratory source-locked pilot

The second pilot must use **the same original 04A-v1** 16×16 grid, lesion WT fraction >=0.25, brain fraction >=0.85, healthy control WT-dilated-3 excluded, mean/std score <=2.0, original `gaussian_local_sigma3` and `healthy_pool_median_fill`, exactly unmodified original frozen P4 ViT S and J and P1 VAL first-slice FLAIR with genuine original masks. Original `a6_ij_04a_interventions.py` remains unchanged; the old `PLAN` still records original v1 fixed patient `022`, and a **separate amendment plan** explicitly identifies `040` as first eligible from the later screen. Never falsely claim 040 was the originally locked v1 pilot.

New source code `scept_sdk/examples/a6_ij_04a02_pilot.py` must verify the original Git blob SHA, all-55 counts and candidate, original source NPZ SHA, old patient blocked failure, exact masks/ROI selection and original 03A frozen clean output. **Source cohort, patient, lesion ROI and control ROI are checked before any new model forward.**

Run both frozen models on clean, lesion/control Gaussian blur and lesion/control healthy-pool median fill (10 true SCEPT original output archives) plus an extra no-op repeated clean inference for each model to check exact repeatability. Report actual true-grade logits, margin, patch-token state changes, ROI L2 relative magnitude and patch-layer differences. Patient-derived output ZIPs and all patient report data remain in private GitHub; they never get committed to public BrainMirror or as original MRI input inside SCEPT archives. Archive original output ZIPs as a **new independently tagged private Release** `scept-a6-ij-04a02-amended-val-040-input-intervention-v1`, with SHA256 re-download check and `analysis.json` + immutable source provenance and receipt at `results/scept_a6_ij/04a02_amended_040_intervention_pilot/`.

**Engineering PASS** requires identical source conditions and genuine verifiable computations. **Scientific verdict remains UNDETERMINED**, whether or not lesion perturbation produces more internal/logit change than the nominal healthy control. 04B/04C must address input perturbation magnitude and artificial/OOD shifts, no patient-level causal assertions from one exploratory subject. No sealed P5 TEST or P6 external validation is opened; original 03C negative predictive utility result remains valid.
