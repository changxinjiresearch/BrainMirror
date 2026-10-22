# A6-IJ P4 real model observation — preregistered engineering smoke

**Status: CODE READY / NOT EXECUTED.** This is an optional research-platform integration, not a change to frozen P4/P5/P6 coursework. The original P4 Grad-CAM comparisons remain unmodified.

- Inputs: exactly one authorized P1 *validation* patient NPZ, first FLAIR slice (or separately declared fixed index); frozen **best** S/J P4 checkpoint SHA256. Never use sealed 56-patient test samples for this qualification.
- Only probe `blocks.0.norm1`, `blocks.5.norm1`, `blocks.11.norm1`: actual per-model hidden tensors 1×197×768. Retain all 197 tokens, document CLS index 0, do not label NAF amplitude as importance.
- Each model is restored with strict state-dict loading, P4 checkpoint meta/epoch/protocol/split and full checkpoint byte SHA256. Same image repeated to three channels, FP32 CPU one-forward per model, no gradient/training/perturbation, actual logits and complete raw captures.
- For each model, produce a verified ZIP with no original model input retained, validate every captured tensor against reimport, recompute NAF offline. Output two local private ZIP archives plus a metadata JSON. ZIP may contain sensitive derived activations: **do not commit ZIP or per-patient identifiers to public repository**. JSON currently records SHA of full P1 patient NPZ (potentially linkable); review and redact before even private archival.
- This smoke does **not** reproduce spatial dependence, U10 evidence gates, rerun the model from archive, test model generalization, demonstrate scientific efficacy, or complete A6 independent operator qualification. It specifically tests observation and ZIP offline fidelity on both real P4 frozen classifiers.
- Kaggle offline prerequisites: P3A-restored P1 validation NPZ; P4 model weights restored and SHA-verified locally; Python dependencies `torch`, `timm` and editable `scept_sdk`. Example:
```bash
python -m pip install -e './BrainMirror/scept_sdk[torch]' timm
python BrainMirror/scept_sdk/examples/a6_ijepa_p4_observe.py \
  --s /kaggle/working/model_s_best.pt \
  --j /kaggle/working/model_j_best.pt \
  --npz /kaggle/working/research_course_p1/classification_npz/val/<ACTUAL_PATIENT>.npz \
  --out /kaggle/working/a6_ij_private
```
Use actual restored paths; placeholders are NOT downloadable data or evidence. Avoid uploading clinical images or patient-level raw tensors to SCEPT's public repository.

**Acceptance:** Before declaring success, run on Kaggle; capture command/versions/exit code, checkpoint SHA, P1 cohort eligibility, real NAF layer geometry, archive integrity and audit failures. Hold M5 blocked until separate A6 CI, independent operator and license gates close.
