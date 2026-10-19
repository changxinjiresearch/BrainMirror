# A6-IJ-02B — 55-patient REAL SCEPT representation cohort

Status: **IMPLEMENTED, REAL KAGGLE 55-PATIENT EXECUTION PENDING**. All patient-level outputs stay in the PRIVATE coursework research repo.

## Scientific question

Can actual frozen P4 Model S (supervised ViT-B/16) and Model J (I-JEPA-initialized and fine-tuned ViT-B/16) internal states be measured across multiple real MRI validation patients with reusable SCEPT, beyond LayerNorm RMS, using patch-token geometry? Can a second, non-SCEPT direct PyTorch Hook run reproduce the captured values?

This is an exploratory *representation* study. The experiment does **not** test spatial evidence dependence or causal importance, does not replace Grad-CAM and cannot make clinical claims.

## Locked cohort and execution

- Research repo: `changxinjiresearch/research-course-brats2020-ijepa-xai` (must remain private).
- P0 split pinned SHA256: `1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352`.
- Exactly **55 validation patients** (44 HGG/11 LGG), lexicographically sorted patient IDs; exactly the **first** P1 classification FLAIR slice per person, 224×224 FP32, repeated to 3 channels, without augmentation.
- First preselected slice comes from the existing P1 cache and is not a random slice; do not interpret single-slice results as a patient's full MRI.
- Frozen P4 best S epoch 8 SHA `2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf` and J epoch 16 SHA `719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640`. No training.
- One real SCEPT single-forward per model/patient = **110 actual forwards** on CPU FP32. Observe fixed named modules `blocks.0.norm1`, `blocks.5.norm1`, `blocks.11.norm1` each [1,197,768] and actual [1,2] logits.
- Export each full actual captured activation ZIP with `include_inputs=False`; independently load and verify every original tensor, NAF recompute, source hashes and model identity. The patient manifest contains patient IDs: keep it private.
- For the first locked validation patient only, reexecute **each** model using direct PyTorch forward hooks without SCEPT and check actual outputs against SCEPT for all three layers plus logits, maximum absolute difference ≤1e-6. It is an independent measurement implementation on the same runtime, **not** an external laboratory replication.
- A run already written locally can be reused only if its original archive, plan SHA, patient label and run identity verify. A partial result is **not** official 55-patient success.

## A priori measurements

For each layer and person, use **196 patch token feature vectors (768 channels)**, excluding CLS. Do not align channel coordinates across independently trained models.

1. Patch mean off-diagonal cosine: unit-normalize each patch feature vector and take the mean off-diagonal cosine similarity among 196 spatial patch embeddings. This describes patch-vector similarity, NOT which patches are important.
2. Patch normalized effective rank: eigenvalues of the centered 196×196 Gram matrix; entropy effective rank exp(-sum p log p) divided by 195. Higher value means a more spectrally distributed centered patch feature Gram, not higher medical intelligence.
3. **Linear centered CKA** for same indexed patch tokens between frozen Model S and J at each matched layer. Channel permutation/orthogonal rotation invariant, but patch-position alignment and positional embeddings can inflate similarity even when pathology is irrelevant.
4. Recorded first-slice model predictions are descriptive only. Recomputed three-layer NAF is a provenance/consistency guard, not the primary representation metric.

Patient-level mean, median and paired J-S difference for the first two metrics across three layers, with exploratory 2,000-resample **patient bootstrap** intervals, fixed seed **240216**; HGG/LGG descriptive paired summaries. Report S/J linear CKA distributions by layer.

Additional **four fixed exploratory Spearman rho comparisons** at Block 11: J-S effective rank and J-S patch cosine vs J-S TAP and J-S IoU10 from **existing P4B** patient metrics. The SCEPT observation is one slice, whereas P4B TAP and IoU10 aggregate **10 slices/patient**: correlation is purely exploratory and has an aggregation mismatch. Do not treat validation reuse as an independent confirmatory test. No p-value mining or post-hoc metric selection.

## Execute with EXISTING Kaggle session

The earlier A6-IJ restoration must already have produced (a) both checkpoint files in `/kaggle/working/a6_ij_models` and (b) the **SHA-verified** `p1_classification_npz.zip` there. Do not rerun any P4 training. If Kaggle reset, rerun the earlier `a6_ij_restore_p4.py` before this phase.

```bash
git -C /kaggle/working/BrainMirror pull --ff-only origin phase/scept-sdk-a6-m5-release-qualification
PYTHONPATH=/kaggle/working/BrainMirror/scept_sdk python /kaggle/working/BrainMirror/scept_sdk/examples/a6_ij_02b_measure.py
PYTHONPATH=/kaggle/working/BrainMirror/scept_sdk python /kaggle/working/BrainMirror/scept_sdk/examples/a6_ij_02b_analyse.py
PYTHONPATH=/kaggle/working/BrainMirror/scept_sdk python /kaggle/working/BrainMirror/scept_sdk/examples/a6_ij_02b_archive_private.py
```

The first step writes `/kaggle/working/a6_ij_02b_private/cases` and `raw` separately for every patient, and can be rerun in the same Kaggle working directory after a transient interruption; it verifies existing results instead of reexecuting those already measured. A provisional operational test may use `--limit 5`, but **cannot** qualify the 55-patient result and will be marked as first-five-only.

The second step only passes after all 55 patients were genuinely processed and re-verifies the source ZIPs and metrics. The third step only passes after full source analysis: 11 shards of five patients × two model ZIPs and five patient JSONs, SHA256-reverified on remote readback, and byte-identical locked plan, analysis and receipt under the private repo path `results/scept_a6_ij/02b_55_patient_representations`. Never upload patient-derived ZIPs to public BrainMirror.

## Acceptance and limitations

- **Engineering qualification** requires 55/55 intact case manifests, 110/110 full raw source archives, frozen checkpoint/source identity checks, real CKA/rank/cosine metrics from each actual output tensor, both independent direct-hook first-patient replay checks, reproducible patient-paired aggregate analysis, and complete private GitHub SHA256 re-download.
- **Scientific conclusion** is not predetermined. Even if all engineering gates PASS, a model difference may arise from LayerNorm/position/architecture or data selection, not I-JEPA biological evidence use.
- P5 sealed TEST set and P6 independent external-validation sources MUST remain unopened. This is not SCEPT SDK A6 full M5 release qualification or medical-XAI scientific M4 release.
