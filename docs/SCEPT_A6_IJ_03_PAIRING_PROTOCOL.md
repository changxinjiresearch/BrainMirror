# SCEPT × Grad-CAM — A6-IJ-03 preregistered paired observation study

**Status: design locked; execute 03A source match before any new MRI capture.** This is NOT a causal evidence-dependence or clinical explanation trial.

## Core principle

P4B already created **1,100 authentic slice-level** Grad-CAM records, with original source CSV chunks, SHA-256 indexed in 11 immutable private manifest records. **Do not rerun Grad-CAM** or alter target definition. The target is each patient's **true-grade logit**, not automatically the predicted class. SCEPT measures internal *forward* activations, not gradient saliency. A relationship between their metrics does not prove one explains the other.

## Frozen source and identities

- The **unchanged** P0 split (SHA256 `1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352`): 55 validation patients, HGG 44 / LGG 11, 10 frozen P1 FLAIR slices per person. Exactly one patient identifier + **1-based P4B slice_rank** + **S/J frozen model** identifies each joined record.
- The exact original P4B `results/p4b/chunks/chunk_{01..11}_metrics.csv`, whose SHA256 equals their `chunk_{01..11}_manifest.json` `csv_sha256`. Each P4B chunk must list the next 5 original lexicographically sorted validation patients. All 1,100 rows must be `VALID`, with nonnull finite metrics. Preserve the P4B `class_idx`, `axial_index`, `true_grade`, `reference_fraction`, `tap`, `iou10`, `p_hgg`.
- The completed private 02B `results/scept_a6_ij/02b_55_patient_representations/analysis.json` (55 `patientRows`, actual frozen P4 S/J model records), and 02C `results/scept_a6_ij/02c_ln_position_controls/analysis.json`. These both use **P1 slice index 0**; this is exactly P4B `slice_rank=1`, not axial plane index 1.
- All sources and per-patient row IDs remain in PRIVATE `research-course-brats2020-ijepa-xai`. Do not copy full derived patient JSON/ZIP or patient source hashes into public BrainMirror.

## 03A — source and one-slice exact-match qualification

Without any fresh model run, construct **110 exact model×patient×slice matched rows** from 02B slice0 and P4B slice_rank1. Reject duplicate/missing join keys, non-validation provenance, grade/target mismatch, 02C to 02B CKA drift (>1e-5), P4B P0 patient report mismatches, or unverified CSV byte checksums. Check 02B observed HGG probability vs P4B recorded FP32 probability; require absolute difference ≤0.03 per pair (this accounts for previously audited GPU/CPU inference precision). **Do not silently fill missing values or rechoose slices.** Keep all 55 patients.

Construct an immutable descriptive patient×model×slice table, and report mean S/J differences in P4B TAP/IoU10, normalized patch effective rank and patch cosine at frozen Block 0/5/11, as well as basic paired Spearman correlations and patient bootstrap (2,000 draws, seed 240216). Descriptive interpretation ONLY.

**03A engineering PASS**: 55 unique patients, 110 unique authentic pair records, every check PASS and private original JSON/CSV archived. No new MRI data processing is needed.

## 03B — match all ten P4B slices

Observe SCEPT frozen P4 S and J using original unchanged pinned P1 image rank `k=0..9` corresponding to P4B rank 1..10, and the same `blocks.0.norm1`, `blocks.5.norm1`, `blocks.11.norm1` forward probes. 550 patient slices × 2 original frozen model runs = 1100 SCEPT single forwards. Exclude CLS from patch rank/cosine, measure SCEPT CKA S vs J on the same patient AND same slice. Preserve full original actual tensors for at least an explicit audit sample; full study must preserve all measured numeric tables and cryptographic provenance. Chunk by 5 patients, write locally per-patient results then archive completed **immutable private shards** with independently redownloaded SHA-256 to protect against Kaggle resets. Do not claim complete 03B if fewer than 55 validated patients, 550 slices and 1100 direct original pairs.

**Scientific operational warning:** 1100 frozen ViT forwards plus token Gram eigendecompositions can be expensive on Kaggle CPU; pilot on one patient and measure actual runtime and archive sizes before undertaking full execution. An adaptive reduction to a different number of patients/slices would require a formally named amendment, not a falsely full PASS.

## 03C — test complementarity, not superiority

Prespecify primary **patient-grouped** evaluation: does adding frozen, strictly forward-only SCEPT patch cosine/rank at Blocks 0/5/11, and cross-S/J centered CKA, help describe/explain held-out P4B `tap`/`iou10` variations **beyond** P4B source covariates (`true_grade`, `reference_fraction`, and `p_hgg`) and simple activation RMS? Do not leak same patient's ten slices between fit/evaluation folds. Use deterministic group split and fixed ridge regularization, report differences in out-of-patient MAE/R² and patient-cluster bootstrap uncertainty. The outcome is **Grad-CAM's descriptive metric**, NOT ground-truth medical decision fidelity.

Negative controls: (i) shuffle SCEPT patient identities within grade and with the slice rank preserved by deterministic RNG, (ii) covariate-only outcome baseline, (iii) grade-stratified descriptive analysis. Preserve all negative or null results; do not optimize significance or reinterpret as causal evidence. Model S and J are analysed separately or paired with model identity as a fixed covariate, always patient-clustered.

**03C acceptance** is correct predetermined execution, publication of all numerical results and falsification checks — **NOT** any particular improvement or significant association.

## Hard boundaries

No P5 sealed TEST, no P6 external cohort, no model retraining, no Grad-CAM recalculation, no choosing alternative Grad-CAM target, no effect-size fishing. Existing validation data were used for checkpoint selection, so this can only establish exploratory research signals. Completeness of 03A–03C does **not** establish that SCEPT replaces Grad-CAM, is causally faithful, or exceeds current attribution baselines. Later A6-IJ-04 must evaluate independent interventions and matched nonlesion controls.
