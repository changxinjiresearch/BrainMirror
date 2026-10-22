# A6-IJ-03C — preregistered incremental representation utility and negative controls

**Phase 03C is analysis, not a new MRI or classifier inference.** The 03B paired frozen-model observational stage passed with 55 validation patients, 10 exact frozen P1 FLAIR slices each, original true-grade-logit P4B Grad-CAM source and 1,100 S/J SCEPT internal layer captures (990 new and 110 reused). This phase must not claim causal XAI faithfulness or clinical superiority.

## Locked goal and estimand (before 03C outcome execution)

Question: **do SCEPT's internal structural metrics have conditional, out-of-patient descriptive predictive value for previously measured P4B Grad-CAM TAP/IoU10 differences**, above ordinary case covariates and same-layer activation magnitude?

Unit of prediction and statistical inference: **55 independent patient records**, not 550 slices or 1,100 pseudo-independent model slices. Within each patient, first compute the 10-slice mean of a feature/outcome per frozen model, then take **Model J minus Model S** for S/J-comparable targets. All 55 patients remain validation-cohort exploratory, not held-out independent training/test.

Primary descriptive endpoint: patient-average `IoU10(J−S)`. Secondary descriptive endpoint: patient-average `TAP(J−S)`. Report both regardless of sign. P4B measures attribution to the original **true grade logit**, not actual diagnostic prediction fidelity. The 03C regression target is Grad-CAM's **numeric observed statistic**, NOT tumor causality.

## Data provenance and amplitude control

- Authoritative 03B final `results/scept_a6_ij/03b_full_slice_pairing/final/analysis.json` and receipt; verify bytes and content against immutable 11 per-chunk reports/receipts and 11 P4B original source CSV SHA256 manifest records. Require 55×10×2 exact pair records and original 03B stored patient means, source float and exact patient keys.
- For meaningful RMS control, read **existing original SCEPT actual raw tensors** from the private 02B first-slice release (110 original ZIPs) and 03B 11-chunk raw ZIP release (990 original ZIPs). Check full outer asset SHA256 and each inner ZIP SHA256 against original immutable receipts, run IDs/provenance, `include_inputs=False`. Compute patch-only FP64 `sqrt(mean(float64(tensor[0,1:,:])**2))` at `blocks.0.norm1`, `blocks.5.norm1`, `blocks.11.norm1` for every matching patient/rank/model. **No neural-network forward, no Grad-CAM recomputation.** Save each 5-patient RMS cohort chunk independently to private GitHub as immutable JSON (not public); each full raw shard already resides privately, not copied anew. A Kaggle reset must be able to skip source-verified already archived RMS chunks.
- On RMS completion, write final 55-patient numerical matrix, cryptographic source provenance and original receipt in private GitHub. Verify existing analysis integrity/uniqueness, mismatch fails closed. RMS is an amplitude-control **derived from the exact 1,100 SCEPT raw ZIP sources**, not a selectively sampled 02B-first-slice proxy.
- Underlying 03B source patient statistics remain unchanged: `patchMeanOffDiagonalCosine` and `patchEffectiveRankNormalized` J−S in each of three layers, and S-v-J matched-index `crossModelCKA` for each layer. There are **9 fixed SCEPT structural predictors**.

## Three model arms, frozen features

Apply exact same 55 patient records and leave-one-patient-out training folds for both targets.

- **B0** (basic): `HGG_indicator`, ten-slice averaged `reference_fraction`, ten-slice averaged original `p_hgg(S)`, ten-slice averaged `p_hgg(J)` (4 source covariates).
- **B1** (amplitude baseline): B0 + 3 locked *patient-average J−S patch RMS* features at Block 0, 5, 11 (7 total).
- **F** (structural SCEPT extension): B1 + **9 fixed** ten-slice-averaged SCEPT structural predictors (16 total).
- All numerical features standardized **on each training fold only**, constants mapped to zero. Ridge penalty `alpha=10.0`, intercept not penalized; no tuning/hyperparameter selection. 55 deterministic leave-one-patient-out (LOPO) fits per outcome × arm, with same single 55-patient feature matrix. `MAE` is the primary utility metric; report held-out `R²` against global patient mean as descriptive and raw residuals.
- Improvements: `ΔMAE B1−F` is the **primary incremental structural information** estimand. Positive indicates lower out-of-patient prediction error from SCEPT structures; negative/zero is a scientifically valid negative result. Also report `B0−B1` and `B0−F`. Patient-level fixed-LOPO-residual paired bootstrap 2000 replicates, seed 240216, 95% interval; explicitly acknowledge it doesn't refit model inside bootstrap.
- **100 fixed within-grade patient permutations** of the **entire 9-column SCEPT feature vector** (seed 240216); preserve B0/B1 and all outcomes, shuffling only additional SCEPT structure within HGG and LGG. Run the same LOPO F arm for both outcomes on each shuffle. Report null mean, 95% interval of `B1−F_{shuffled}`, fraction null ≥ observed (descriptive, no multiplicity-adjusted p-value), no rerolling or selective picking.
- Fixed sensitivity: report HGG (44) and LGG (11) grouped held-out absolute-residual difference B1−F and covariate-only metrics, as descriptive; the small LGG group means weak uncertainty.
- **No significance gating:** a scientifically null or negative improvement must not invalidate a fully executed 03C engineering PASS. Never select features on whole data, leak patient slices across training/test, or change alpha/permutations/metric after reading outcomes.

## Acceptance criteria and scientific limits

1. 55/55 original validation patient records and 1100/1100 RMS probe groups traced to all original 02B/03B SHA-verified raw source ZIPs, matched on true patient, same slice, model, and predeclared layer.
2. All 11 immutable private RMS chunk reports and final RMS manifest exist and are SHA-verified, with no new model forward calls.
3. Deterministic LOPO B0/B1/F on both endpoints with the **same 55 patients**, with fold-local preprocessing, locked alpha, 2000 patient bootstrap replicates, 100 within-grade shuffles, grade-stratified descriptive diagnostics and original per-patient out-of-fold predictions/residuals in immutable private GitHub.
4. Private output + remote pinned commit readback verified. Test set P5 remains sealed, no independent P6, no clinical/causal attribution.
5. The first-phase output is a **conditional association/analysis complement** evaluation, **not** independent clinical test, Grad-CAM replacement, proof of pathologic decision dependence or an independently generalizable XAI method.

The public BrainMirror repository contains only code/protocol and synthetic CI; all patient-derived values and private raw ZIP-derived measures stay in `changxinjiresearch/research-course-brats2020-ijepa-xai`.
