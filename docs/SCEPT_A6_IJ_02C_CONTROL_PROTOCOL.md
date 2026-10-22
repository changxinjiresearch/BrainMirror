# A6-IJ-02C — LayerNorm and position-alignment controls (locked protocol)

**State: engineering implementation pending/under construction; no 02C Kaggle outcome exists yet.**

## Question and falsifiable scope

02B measured 55 validation patients: cross-model centered linear CKA is higher at Block 0 than at Blocks 5/11, with additional patch cosine/effective-rank contrasts. **02C** asks whether these descriptive contrasts persist when comparing (i) the layer's residual-stream **pre-LayerNorm** tensor, (ii) the same tensor normalized with LayerNorm **without learned affine gamma/beta**, and (iii) the actual **post-LayerNorm** module output. This decomposes numerical representation geometry only; it cannot fully establish causes or rule out all architecture/training confounds.

## Frozen inputs

- Same P0 SHA256-locked **55 validation patients**, with HGG=44/LGG=11; lexicographic order; **first** P1 224x224 FLAIR slice from each patient's 10-image classification NPZ, repeated as 3 channels. P1 NPZ bytes must match the prior authoritative SHA256 inventory.
- Frozen P4 best S epoch 8 full checkpoint SHA256 `2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf`; J epoch 16 SHA256 `719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640`. No model updating/retraining.
- Use the exact same timm ViT-B/16 module topology, CPU FP32, eval/inference_mode.
- Each model is observed once per patient with the existing independent SCEPT SDK. Real named probe pairs: `norm_pre -> blocks.0.norm1`, `blocks.4 -> blocks.5.norm1`, `blocks.10 -> blocks.11.norm1`. Pairs are **pre-LN residual** and **post-LN normalized** outputs for Blocks 0, 5, 11. All six actual tensors are [1,197,768], plus actual [1,2] logits. Importantly, `norm_pre` is a module output upstream of Block 0, not an invented "norm1 input".
- Explicitly verify for the **first fixed patient** and **both models** using independent *direct PyTorch forward-pre-hook and forward-hook* replay (a separate model forward, not SCEPT), comparing all pre-LN inputs and post-LN outputs and logits against the SCEPT record within max absolute tolerance 1e-6.
- Verify algebraic LayerNorm on **every patient**, every paired layer: `post == F.layer_norm(pre, shape, weight, bias, eps)` within max absolute error 1e-5; store actual max errors. Construct *affine-free* LN vectors by `F.layer_norm(pre, shape, weight=None, bias=None, eps)`. This is an offline mathematical control, not an observed extra forward through a modified model.

## Locked metrics and negative controls

Each patient has 196 patch tokens (CLS index 0 excluded). For each of the three layers:
- Cross-model centered linear CKA at **pre**, **affine-free LN**, **post**, all computed on exact same patch indices.
- Within each model, same-forward pre/post representation CKA, and post/affine-free CKA; post/pre cosine mean and centered normalized effective rank (exploratory descriptive).
- Patch correspondence null: **64 deterministic random permutations** of J's patch **index order** for pre and post stage (fixed seed 240216 with patient/layer/stage deterministic offsets). Report aligned S/J CKA and permutation-null mean / observed difference; random row scrambling is a *correspondence sensitivity control*, **not** a removal of positional embeddings or a valid image intervention.
- Fixed **all-zero 224x224 FLAIR repeated-three** OOD control (one real SCEPT forward per model), same six probes. Use to report pre/post blank-input cross-model CKA and whether nonzero inputs differ from that blank response. **Blank MRI is a computational control, never a clinical negative example and not a clean isolation of position encoding.**
- Compare **post-LN 02C CKA** against individually matched patient/layer 02B post-LN original CKA; require absolute difference <= 1e-5, else fail on possible reproducibility/protocol drift. The original 02B source lives exclusively in the private research repository.

## Statistical reporting

Exactly 55 paired patients; 2000 bootstrap replicates seeded 240216 on the patient unit. Report descriptive means and 95% patient-bootstrap intervals for S/J CKA and the changes across stages, stratified descriptively by original HGG/LGG. No post-hoc variable/threshold selection and no new significance claims; do not infer causality from a change in CKA across stages.

**Engineering pass**: 55/55 patients, 110/110 SHA-verified SCEPT full actual-output archives, 2 extra zero-input references, 110×3 algebraic LN consistency records, first-patient independent direct-hook replay S/J, 02B patient-wise CKA replication, deterministic permutation-null computations, reproducible 2000 patient-bootstrap report, private artifact archive verified with full SHA256. **Scientific finding is OPEN**, including null or contradictory comparisons.

## Privacy and boundaries

All per-patient IDs, actual activations, NPZ fingerprints, patient records, and raw ZIPs remain in the **private** research-course-brats2020-ijepa-xai repo, not public BrainMirror. All release uploads independently re-downloaded and full SHA256 verified; report immutable. Never open P5 sealed test, never use P6 independent external data, never train models, never alter 02B/P4B registered records. This is not a SCEPT scientific XAI validation or M5 full SDK general-purpose release.
