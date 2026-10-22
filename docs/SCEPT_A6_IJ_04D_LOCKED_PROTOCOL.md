# A6-IJ-04D — preregistered **frozen-model internal causal intervention** study

**Locked scope before executing 04D model intervention outcomes.** This is **not training** and success never depends on improving model prediction. The objective is to test the effect of *directly intervening on the computation graph* after a ViT block and measuring the pre-existing classifier's decision response. 04C's observational 33-patient, 594-archive full-tensor result and its exploratory S/J differences remain untouched.

## Source and cohort

Original P0 55-person VAL first-slice P1 FLAIR cohort; exactly 33 pre-screen eligible (26 HGG/7 LGG), original 22 excluded. Real frozen P4 ViT-B/16 **S and J**, exact original weight SHA256 and P1 NPZ full SHA256 checked. Original 04B all-33 equal input pixel L2 source preflight and complete 7 private original raw source SHA archives required. 04C 33-patient, 594 genuine full activation tensor verifications plus immutable final report and SHA receipt required. **No P5 TEST or P6 independent data.**

Fixed 04B **Gaussian sigma 3, 50% of the common attainable source pixel-L2** intervention only, applied separately to original `lesion` and healthy `control` 16x16 patch, unchanged fixed 04A source region selection. This is the original 04B primary operator, not selected based on 04C exploratory correlation. Repeat original S/J output logits and compare each to exact original 04B patient, model and condition, absolute tolerance <=2e-4.

## Direct computational interventions

**Insertion points are block residual outputs**, not pre-attention `blocks.5.norm1` and `blocks.11.norm1`; PyTorch forward hooks replace **actual post-block outputs** at `blocks.5` and `blocks.11`, before all remaining network computation, without changing model parameters. Using the corresponding original clean and edited MRI forward-derived activations as donor states, compare these three sets at each point:

1. `target`: exactly one prespecified target patch-token index in `1..196`, matching original lesion/control ViT-grid image ROI;
2. `placebo`: the **other original ROI token** in the same patient, as a spatially distinct non-target control; this is not a signal- or anatomical-intensity-matched equivalent and cannot by itself establish selectivity;
3. `all`: all 197 tokens, required positive causal instrumentation control: full patched model output **must** match the authentic donor branch output to 2e-4 (inference-mode CPU, reproducible).

Both patch **directions** are mandatory for each operator arm/layer/token set:
- **rescue**: edited recipient with clean source at the post-block site (may move model output toward clean);
- **insertion**: clean recipient with edited source at the site (may move model output toward edited).

For each frozen model/person: unmodified clean + two unmodified edited Gaussian inputs = three baseline forwards; 2 arms × 2 residual layers × 3 token sets × 2 directions = 24 actual genuine patched forwards. Additionally, 2 unaltered same-source self-patching shams at `blocks.5` and `blocks.11` on **clean→clean**, required to leave predictions unchanged within 2e-4. Total **29 real model forwards per model/patient, 58 per patient, 1,914 across 33 patients**. Distinct full post-block donor activations are persistently recorded as genuine float32 [1,197,768] tensors for clean and each edited arm in an independent privately archived NPZ. Every patched output, effect value, identities, input-provenance hash, source/donor norm, target index, full donor/recipient SHA and numeric instrument controls are persisted to private case JSON (no original patient MRI added).

### A crucial interpretation boundary

The effect of replacing selected hidden computation is a *causal effect of that operation within this frozen model graph*, not proof that a biological tumor itself caused the diagnosis, not unique responsibility of a single neuron, and not causal mediation of the natural data-generating process. Activation patching can create non-natural hybrid model states; the model could route around a patched site, the residual skip could carry information, and token index selection is not equivalent to semantic tumor region importance. A near-zero original output contrast is **inconclusive**, not a SCEPT failure. Even a full donor restore control PASS does not validate the target-token attribution.

## Prespecified outcome and controls

Signed true-grade margin `m_y=z_y-z_(1-y)` for each frozen model. For each patient/model/arm/layer:

- `rescue_fraction = (margin(rescue-target)-margin(edited))/(margin(clean)-margin(edited))` if denominator |margin(clean)-margin(edited)| >= 1e-4, otherwise **INDETERMINATE**, never imputed as zero;
- `insertion_fraction = (margin(insertion-target)-margin(clean))/(margin(edited)-margin(clean))` under same gate;
- same two calculations for placebo and full. The full-token positive control should recover ~=1 for both directions whenever denominators are adequate;
- signed `target_minus_placebo` rescue and insertion effects; distinguish magnitude and direction; **do not** maximize over layers/models/post hoc;
- secondary joint 2-model patient mean per layer and ROI arm with 2,000 **patient-only** seed-240216 bootstrap 95% intervals, plus fraction indeterminate (pre-specified 1e-4 baseline delta). Compare target vs placebo with both directions and clearly label mixed positions, all estimates descriptive. Analysis unit patient n=33; S/J are paired repeated models. If baseline denominators too often near zero, report lower eligible numerator/denominator and **no broad fidelity claim**.

## Controls and fatal engineering gates

- Model weights unchanged before/after all actual interventions (same exact state byte SHA), model always `eval()` and `torch.inference_mode()`, no training/backward/optimizer.
- Required *unpatched* clean and edited output logits compare against 04B original outcome in same patient/model/arm.
- Post-block full replacement `all` exactly reproduces donor branch output for **both directions**, including last block; violation indicates wrong graph location/hook behavior.
- `clean→clean` target self-patch sham must leave output identical for both block positions.
- Validate number of hooks exactly one during patch and zero leaked afterwards, finite full tensors, correct [1,197,768] shape and unchanged inputs; fail closed on mismatch.
- Do not choose new patients, ROIs, layer, operator or thresholds after observing the outputs. A negative effect, contradiction, indeterminate result or failed paired placebo is scientifically reportable, never retuned away.
- Seven SHA256-verified private source archive chunks (5+5+5+5+5+5+3 patients), each containing 5/3 original case metadata JSON and true model-internal donor activation NPZ; immutable private ZIP Release readback before reporting batch complete. Full final report requires 33 cases, 1,914 original independent forward executions, 7 immutable archive receipts and patient-level inferential/statistical analysis.
- Separate independent frozen-model implementation test on a *small synthetic real Torch residual model*, plus direct Hook ordering / full restore / all-sham / outside-token non-leakage; no patient MRI or real model used in public CI.

**Completion definition:** 04D engineering and controlled-computation study is complete only after all 33 original patient experiments and independent original private SHA256 archive checks pass and the 33-patient observational *bounded graph-intervention* report is published in the original private research repository. 04D does **not** certify a new SCEPT paper-level explainability operator or any top-tier generalization; those are future work outside this 04D task.
