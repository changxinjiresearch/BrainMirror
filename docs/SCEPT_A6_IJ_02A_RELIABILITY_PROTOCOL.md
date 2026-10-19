# A6-IJ-02A: Repeatability and real-input sensitivity

Status: CODE COMMITTED, KAGGLE REAL-RUN PENDING. Scientific results not yet available.

## Prelocked inputs

P0 SHA256: 1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352.

Use first two lexicographically sorted VAL patients: BraTS20_Training_022 (A) and BraTS20_Training_040 (B). Both must be verified against private P1 source inventory; one first 224x224 FLAIR slice each, repeated as three identical channels. Never access sealed 56-patient TEST.

P4 best supervised Model S SHA256: 2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf (epoch 8). I-JEPA-finetuned Model J SHA256: 719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640 (epoch 16).

Fixed CPU FP32 SCEPT PyTorch probes: blocks.0.norm1, blocks.5.norm1, blocks.11.norm1, each 1 x 197 x 768. Real output logits must be 1 x 2.

For EACH unchanged frozen model: perform distinct single-forwards in order A1, A2, B1, A3. Compare all raw layer values plus logits for repeated identical input A. Compare full raw hidden layer tensors between A1 and B1; do NOT rely only on LayerNorm RMS. Different model channels are not coordinate-aligned.

## Preregistered engineering gates

Reliability PASS for a model: for both A2/A3 versus A1, all three captured hidden layers and logits have maximum absolute error <= 1e-6 AND relative L2 error <= 1e-7.

Input sensitivity PASS for a model: B1 versus A1 has relative L2 >= 1e-3 in at least 2 of 3 prespecified hidden layers, AND image tensors are distinct. The thresholds are engineering screening cutoffs, NOT inferential effect sizes or clinical significance.

A gate failure must be preserved as a negative/undetected result, not rerun until it passes. Report both models and every layer including zero differences. No image perturbation, no retraining, no attribution gradients, no output-importance/causal or spatial dependence claims.

Eight complete verified private output-only ZIPs and one JSON report are generated. Each ZIP holds actual captured tensors and recomputed NAF. Offline recomputation does not rerun the model.

## Kaggle steps (existing earlier A6-IJ restore prerequisites)

1. Update local BrainMirror A6 source branch.
2. Run: PYTHONPATH=/kaggle/working/BrainMirror/scept_sdk python /kaggle/working/BrainMirror/scept_sdk/examples/a6_ij_02a_measure.py
3. Run: PYTHONPATH=/kaggle/working/BrainMirror/scept_sdk python /kaggle/working/BrainMirror/scept_sdk/examples/a6_ij_02a_archive_private.py

Requires prior restored best S/J checkpoints, the already SHA-verified private P1 Release ZIP and a Kaggle GITHUB_TOKEN secret with private research repository access.

## Archival

Only the PRIVATE research-course-brats2020-ijepa-xai GitHub prerelease scept-a6-ij-02a-reliability-v1 receives original derived-patient ZIPs. Every upload requires remote full-byte SHA256 re-download verification. Source report and receipt are preserved immutably at results/scept_a6_ij/02a_reliability/. Authenticated readback retry protects against transient GitHub 404s. Do not publish raw derived patient activations, ZIPs, input/model binaries or tokens to public SCEPT BrainMirror.

A PASS means reproducible SCEPT numerical observation and real-input responsiveness in two specific validation slices, NOT scientific faithfulness, an explanation method, biological evidence, generalization, patient-level classification result, superiority to Grad-CAM, or completion of A6 licensing/independent-operator M5 gates.
