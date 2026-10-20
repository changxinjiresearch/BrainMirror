# A6-IJ-04 — SCEPT controlled-input perturbation and evidence-dependence research

**Project state: protocol + 04A code development, not yet real Kaggle PASS.** This study is a *new* registered research question. A6-IJ-03C's null/negative result for predicting Grad-CAM TAP/IoU10 statistics remains immutable and must not be rewritten as a positive result.

## Scientific claim hierarchy

1. **Observational authenticity:** SCEPT reads genuine named intermediate model tensors. Previously engineering-qualified on frozen S/J P4 ViT and other tested architectures.
2. **Input-intervention response:** changing a defined MRI region can produce an authentic, measured internal-tensor difference and a counterfactual logit difference. This is an intervention on the **input**, not proof of causal mediation by a particular internal unit.
3. **Evidence specificity and faithfulness:** only partially supported if lesion-targeted interventions consistently induce greater or more relevant output changes than *well-matched non-lesion controls*, across patient clusters, distinct intervention operators and artifact/OOD controls. Failed or reversed findings must be preserved.
4. **Mechanistic role of particular internal states:** further requires preregistered direct internal activation patching/ablation experiments under controlled computation; purely observing a hidden-unit difference does not identify causal mediation.
5. **Clinical/external validity:** separate sealed P5 and independent P6 governance. Not attempted at 04A.

SCEPT **does not retrain or improve the original classifier**. The primary outcome is frozen-model response to controlled evidence perturbation, not change in classifier accuracy and not ability to predict Grad-CAM.

## A6-IJ-04A — input perturbation implementation and first patient qualification

**Goal:** prove the protocol runs on original P0 VAL P1 FLAIR and can hold the *identity, location, shape, area, and preprocessing* of intervention/healthy controls auditable, with actual SCEPT full raw outputs and model logits. This is NOT scientific confirmation of disease evidence dependence.

- Cohort pilot **one fixed original first validation patient** `BraTS20_Training_022`, **P1 slice_rank=1** (zero-based slice index 0, original axial_index from private P1 source). Both exact pinned frozen Model S and J, with original `p0_split_sha256`, original patient NPZ full SHA256 and original checkpoint full SHA256 and protocol metadata.
- Geometry: original `flair` float32, `wt` integer whole-tumour and `foreground` boolean masks on `[10,224,224]`. Each intervention ROI is **a single ViT-aligned 16×16 patch** in 14×14 lattice (196 grid patches). No resize of masks or redefinition of input.
- **Lesion patch selection independent of model outputs and Grad-CAM:** among original brain-containing blocks with at least **85% brain-foreground occupancy** and nonzero WT overlap, choose maximum WT pixel **occupancy**; lexicographically first row/column breaks exact ties. Require WT occupancy at least **25%**, else pilot **BLOCKED / ineligible** without selecting an alternative patient post hoc.
- **Matched non-lesion patch:** 16×16 patch with >=85% original brain foreground occupancy, zero WT overlap, no overlap with a **3-pixel-dilated WT mask**. Choose minimum frozen score `abs(mean_intensity−lesion_mean)/(global_foreground_std+1e-6) + abs(std_intensity−lesion_std)/(global_foreground_std+1e-6)`; raster-order ties. Require score <=1.0 and >=1 eligible control, else pilot **BLOCKED**. This is a heuristic intensity-matched control, not perfect pathological or anatomical matching.
- Interventions: (i) `gaussian_local_sigma3`, replace ONLY the original 16×16 ROI with globally Gaussian-smoothed sigma=3 MRI values at identical original coordinates; (ii) `ring_median_fill`, replace ONLY original 16×16 ROI with the median original intensity from a **5-pixel ring around that ROI restricted to brain foreground AND outside dilated WT**, require >=32 eligible original ring pixels, else **BLOCKED**. Both operators are evaluated for lesion and control. No copying global preprocessing statistics, no new scaling or renormalization. Keep source pixels outside ROI **bitwise unchanged** and all perturbations float32.
- For each frozen model: 5 real forward observations (clean, 2 intervention operators × lesion/control) = **10 new SCEPT observations** in total. Capture full authentic `blocks.0.norm1`, `blocks.5.norm1`, `blocks.11.norm1` tensors plus actual logits, and save full output-only original observation ZIP. Full source exact tensors and masks derived from medical patient inputs stay exclusively PRIVATE.
- Clean model predicted `P(HGG)` must match original 03A source patient/model/slice to <=0.03 absolute (same original source, not a scientific effect limit). The scientific logits are **original true-grade logit** (grade-dependent class 0/1), logit-margin and P(HGG); never silently choose the best-looking class or flip the target to observed top-1.
- For each intervention and each model compute **signed** true-grade-logit change, absolute change, target-logit margin change, and input perturbation Euclidean norm restricted to ROI. For each source layer compute patch-token change `||H_intervened−H_clean||_F/sqrt(H.size)` and normalized relative L2, plus cosine shift of the **directly targeted 16×16 patch token**. Exclude CLS from full-patch aggregate if calculating patch-specific metrics; separately preserve original full tensor ZIPs to support later calculations.
- For each operator, define preregistered paired contrast `absolute_logit_change(lesion) − absolute_logit_change(matched_nonlesion)` and `patch_relative_change(lesion) − patch_relative_change(control)`. **No required positive result**; faithful mechanisms are not established by a one-person pilot.
- Artifact controls: exact ROI area 256 pixels for both arms; exact outside-ROI original pixel equality; mask overlaps, foreground fraction, original intensity means/stds, input-shift L2, and intervention-specific ring sample counts must be recorded. `gaussian_local_sigma3` and `ring_median_fill` differ in artifact risks; do not call ring filling medically plausible inpainting. A zero/no-op sham must leave both input and output exactly unchanged, included as source reproducibility sanity rather than another explanation.
- A4 pilot artifacts: a local report + **10 original raw ZIPs** and exact local plan; run the private archiver **only after full successful pilot**, upload a single immutable original private ZIP shard, remotely reread/check its full SHA256, commit immutable report and receipt into private coursework repository. Any failures recorded explicitly and must not trigger threshold tuning.
- **Engineering PASS** means authentic model source and old-output identity, all paired ROI/perturbation invariants, logit and hidden-state measurement and original remote immutable SHA256. **Scientific verdict remains UNDETERMINED** regardless of the sign.

## A6-IJ-04B — controlled cohort expansion

Only after the fixed 04A pilot passes, register expansion to all 55 locked P0 VAL patients × same 10 original P1 slice ranks × frozen S/J, using **both** preregistered perturbations and *at least two independent matched nonlesion controls where eligible*. Report control eligibility, failures, segmented lesion coverage, patient-clustered uncertainty and operator-stratified estimates; do not treat slices as independent patients. Do **not** silently delete patients or pick easier slices if fixed eligibility rules fail. Patient and source audit shards stay private.

## A6-IJ-04C — comparative, falsifiable explanatory validity

Compare SCEPT's intervention-state and output association to: (i) original frozen Grad-CAM where correctly co-registered, (ii) simple original logit/probability changes, and (iii) simple patch activation RMS or randomly initialized / output-insensitive probes when scientifically justified. Verify whether changes are decision-relevant after controlling for input perturbation magnitude and OOD/artifact signatures. Use within-patient model-paired comparisons and negative controls, preregister endpoint hierarchy and patient bootstrap. A significant association is still not proof of causal mediation by internal units.

## A6-IJ-04D — direct computation intervention (only if technically and scientifically justified)

Using stable hookable models, consider *activation patching* with matched clean/counterfactual internal tensors and targeted named module output replacement, capturing the downstream effect on fixed target logits. This is the design required to support bounded claims that an internal state mediates a decision, and needs its own intervention validity/sanity controls and model consistency checks. Avoid automatically claiming a true clinical biological mechanism.

## Scope restrictions

- All experiments run on original frozen P4 checkpoints and original P1 validation MRI/masks. Reuse the same original P0 split. **No P5 sealed TEST and no P6 independent external cohort** without explicitly separate authorized governance; no retraining.
- Reject out-of-range/ambiguous slice model or mask geometry, missing input source checksums, failed repeatability, insufficient healthy patch controls, non-finite logits/tensors, or patient-derived artifacts outside private repo.
- Negative, null, OOD, and incomplete results must be retained. Prespecified acceptance is **valid execution and auditable source provenance**, not beneficial lesion response or SCEPT superiority to Grad-CAM.
