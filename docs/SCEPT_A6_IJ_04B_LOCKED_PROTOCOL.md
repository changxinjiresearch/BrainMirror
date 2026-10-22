# A6-IJ-04B — prespecified **input-magnitude-matched**, patient-paired SCEPT response study

**Status: preregistered engineering implementation, not a completed Kaggle experiment, not clinical/causal XAI validation.** The old fixed 04A failure and later amended 04A02 one-patient pilot are immutable, and 03C's negative incremental Grad-CAM-prediction finding remains unchanged.

## Selection and provenance (fixed before 04B model outcomes)

- Source: original P0 **55 validation patients**, each original P1 first FLAIR slice, WT mask, brain-foreground, frozen P4 best S/J ViT-B/16 checkpoint. No retraining. Keep P5 sealed TEST and independent P6 untouched.
- Source screening: originally archived, independent pre-model-effect `04a02_original_mask_only_feasibility/full_val_first_slice_screen.json` in the **private** research repo. Verify the exact immutable Git blob and original 04A v1 ROI algorithm SHA. Screen has 55 total, **33** first-slice eligible (**26 HGG, 7 LGG**) and 22 excluded (21 original score >2.0, 1 original mask containment). The 22 never contribute denominators for conditional-included estimates; **retain them explicitly in the full selection audit**. Never choose alternative slices, relax eligibility, or select based on a model output.
- Per eligible patient, original tumor and healthy ViT-aligned 16×16 ROI coordinates and selection scores must match original screen exactly, and original full NPZ must match private SHA256 inventory. The original upper-bound heuristic score **2.0** remains unchanged, even though that criterion does not establish matched anatomy.
- Original `04a02` one-case pilot is *exploratory*, and its case is part of the current eligible cohort; disclose lack of independent confirmation, do not call 04B a sealed validation. All endpoint analyses are descriptive within the same validation cohort.

## Input magnitude-matching intervention design

Same two operator **directions** as 04A: `gaussian_local_sigma3` (original full-slice Gaussian replacement inside the 16×16 grid block) and `healthy_pool_median_fill` (original fixed healthy-brain median donor value replacing that block). For each person and operator, precompute original *full* 16×16 lesion/control replacement deltas `D_l, D_c`, in the original P1 input pixel coordinates, without any neural model inference.

**Unlike 04A's unequal original full replacement magnitudes**, create equal-strength fractional interventions via the deterministic interpolation

\[
D^* = \min(\lVert D_l\rVert_2,\lVert D_c\rVert_2),\quad
I'_{a,o,d}=I + \alpha_{a,o,d}D_{a,o},
\quad\alpha_{a,o,d}=d\,D^*/\lVert D_{a,o}\rVert_2,
\quad d\in\{0.25,0.50\}.
\]

Values are stored in float32; match the **actual** computed float64 L2 deltas of the final inputs with `abs(L2_l − L2_c) <= max(1e-5,1e-5*D^*)`, abort a paired condition if unreachable. `D^*<=1e-6` is **ineligible-for-dose** (not a zero-effect evidence result); report explicit patient/operator status rather than cherry-pick. A 04B median-fill condition is a *partial-dose interpolation towards* the original median-fill direction; **not** the full replacement or medically realistic tumor removal. Original pixels outside ROI remain bitwise unchanged. Repeatable operators, no normalization or resizing after interpolation, exact area 256 each, independent input-shift L2, source image SHA retained privately.

**8 perturbed inputs per patient** = 2 operators × 2 arms × 2 doses. Frozen S/J each get one clean plus eight interventions ⇒ **18 genuine SCEPT model forwards and 18 original full-tensor ZIPs per patient**, with model logits, three named ViT layer tensors, source provenance, original mask/intervention/dose data. For 33 patients = **594 genuine new SCEPT forward archives**. Clean source p(HGG) must correspond to original 03A first-slice same-model output within fixed absolute 0.03; sham exact reproducibility is qualified in 04A02 and not redundantly counted in 04B.

## Predeclared scientific endpoints

For frozen model M, each patient i, original true-grade class y, compare clean vs intervention **true-grade logit margin** \(m=z_y-z_{1-y}\). Retain *signed* margin change; **primary specificity contrast** for original Gaussian operator at dose 0.50:

\[
\Delta_i=\frac{1}{2}\sum_{M\in\{S,J\}}
\left(\lvert m_{i,M,\mathrm{lesion}}-m_{i,M,\mathrm{clean}}\rvert-
      \lvert m_{i,M,\mathrm{control}}-m_{i,M,\mathrm{clean}}\rvert\right).
\]

Positive means **greater output response to lesion intervention at equal pixel-space L2**, a *descriptive within-patient intervention specificity statistic*, **not proof the original medical tumor evidence was causally decisive**. Two model-specific primary contrasts and other pre-locked doses/operators are secondary. Report per-model full absolute/signed true-grade margin and original logit differences, actual input-shift magnitudes, proportion of operator/dose infeasibility, and original patient exclusions, regardless of sign. Record patch-only three-layer relative L2 and direct ROI-token change *separately* from output contrasts; do not manufacture a blended SCEPT "importance" measure post hoc.

Independent inference unit = **patient (n=33 eligible first slices)**, not 66 model patients, 132 operator/dose pairs or 594 model forwards. Aggregate S/J within patient; 2,000 fixed-seed (240216) patient bootstrap mean 95% intervals (two-sided percentile); 2,000 fixed-seed patient-wise **swap lesion/control sign-flip negatives** (same flip applied to both models per person) for original primary contrast, with descriptive null tail fraction. HGG (n=26) and LGG (n=7) secondary estimates. Outcomes with unmatched source, missing paired condition, nonfinite tensor, mask or checksum fail **scientific result completion** and cannot be silently dropped.

## Research validity limits and future 04C gate

- L2 equalizes **input energy, not the distribution of artifacts**, anatomical location, shape of alteration, actual pathology plausibility or semantic content. Gaussian smooth/median interpolation may still create OOD MRI artifacts; this experiment only qualifies selected *model responses to these particular input edits*.
- The two spatial blocks may occupy different attention/positional contexts. A matched-patch conditional result is not a statement about all tumor tissue or entire brain.
- A layer-state shift is an observation, NOT evidence of the particular neuron/circuit causally mediating the output. Direct activation patching is reserved for a separately preregistered 04D.
- Negative or null primary result is a **valid result**; never tune dose, outcome or case inclusion after observing it. A source integrity or paired-L2 mismatch is not scientific negative, but an engineering failure.
- All patient-specific real numerical results, source checksums, ROI coordinates, tensors, logs, manifests and patient reports must be persisted **only in the PRIVATE authoritative** `changxinjiresearch/research-course-brats2020-ijepa-xai` repo or its private release assets. BrainMirror public code is for reproducible method, source-only/synthetic QA, and this preregistered protocol (no per-patient source metrics).
- For fail-safe persistence: fixed eligible list is partitioned into **seven** ordered chunks of at most five patients (5+5+5+5+5+5+3). Each completed chunk must store every original model ZIP, the five/three original case reports and a SHA256-verified full private GitHub Release asset; immutable private per-chunk receipts permit reruns to skip previously archived chunks. Complete analysis must require seven receipts, 33 fully paired people, 594 actual original output ZIPs and private final analysis/receipt. The runtime must never report completed before actual Kaggle real inference and full remote SHA verification.

**Scientific milestone:** A6-IJ-04B qualifies controlled, equal-input-L2 observations and transparent patient-clustered evidence-specificity contrasts. It does not by itself establish medical fidelity, direct neural causal mediation, or superiority to Grad-CAM.
