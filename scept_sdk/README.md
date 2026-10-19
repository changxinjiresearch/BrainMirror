# SCEPT Research SDK A0–A5 (alpha)

Install from source (this package is **not yet on PyPI**):

```bash
python -m pip install './scept_sdk[torch,onnx]'
# For testing:
python -m pip install './scept_sdk[test]'
python -m pytest scept_sdk/tests -q
```

```python
import torch
from scept import SCEPT
model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
observer = SCEPT(model, layers=['0', '1'], backend='torch', profile='cnn-nchw')
result = observer.observe(torch.randn(1, 1, 32, 32), preprocess_id='synthetic-fixture-v1')
print(result.run_id, result.tensors)
print(result.tensor('0').shape)  # genuine 1,4,32,32 activation
print(result.record['science']) # NOT a causal explanation
```

**A3:** Python NAF is now implemented as run.fingerprint(), run.pattern(tensor, index), and graph-qualified run.representation_update(...). For unproven paths, it explicitly returns not_evaluable. The calculation is numeric state, not output importance.

**A3 reference calibration:** build_reference(naf_samples, label="...", domain="...", kind="within-volume") requires five other real, distinct-input NAFs with identical frozen model SHA and preprocessing fingerprint. calibrate(target_naf, reference) uses the original V6.3 empirical midrank; a target cannot be included in its own reference. Declared cohort independence cannot be established from input hashes alone. Fixed raw RMS scale does not autoscale by individual input.

**A4:** ProtocolRegistry, Mask, U10Plan, run_u10 provide local versioned protocol registration and preregistered controlled A/B/C numeric inference. U10 performs 3 real model forwards per seed/dose, strict ROI checking and seven negative/falsifiable gates; it does not prove clinical or causal attribution. Custom registered Python callbacks are trusted local process code, not an isolated sandbox.

**A5 — complete portable scientific observation archive and reproducibility:**

```python
from scept import load_observation

# Save all actually captured outputs, full NAF and source-provenance hashes.
run.export('my-observation.zip')
# Opt in to save the preprocessed model feed (potentially sensitive):
run.export('with-input.zip', include_inputs=True)

# Either website V6.5.2 or Python zip can be read without changing source schema.
loaded = load_observation('my-observation.zip')
print(loaded.record['schema'], loaded.tensors)
print(loaded.recompute_naf()['science']['model_forward_reexecuted'])  # False

# A separate true model re-execution requires a matching frozen observer:
loaded_with_input = load_observation('with-input.zip')
print(loaded_with_input.verify_model_replay(observer)['status'])
```

The ZIP format is `scept.full-observation.v1` compatible with V6.5.2 browser. It contains every actual measured output tensor (including logits), full NAF, source-record schema (`scept.python-capture.v1` or original `scept.run-record.v1`), raw binary values, canonical per-tensor SHA-256, per-file SHA-256 and ZIP CRC. Inputs are explicit opt-in; original NIfTI, entire hidden uninstrumented graph, frozen model weights and executable preprocessing code are NOT included.

Standalone offline commands: `scept-archive verify archive.zip` or `scept-archive recompute archive.zip`. Offline NAF arithmetic is **not** the same as model execution; real model re-execution is a separate operation requiring an independently supplied exact frozen model and archived input. Different CPU/GPU runtimes may produce genuine small numeric differences.

File SHA and ZIP CRC validate integrity against the manifest, not real-world authorship. A hostile author can rewrite and rehash unsigned archives; only a separately verified model and actual rerun can establish stronger provenance. All downloaded archives, even those without inputs, may contain sensitive information about patient data. No automatic external data transfer is performed.

**Research boundaries:** One *genuine* forward per observe call; all captured tensor arrays are full copies. Tensors must be dense, finite, nonempty and in the V6 numeric dtype set; the default total input+output copy budget is 32 MB. A named PyTorch module must execute exactly once. ONNX layer names must be actual graph producers with verified types; graph outputs are instrumented **in memory** and the original ONNX file is not changed. Training flags and Hook handles are restored even on failure. A0 Python record currently uses `scept.python-capture.v1` rather than falsely using the browser's completed `scept.run-record.v1` NAF semantics.

PyTorch model identity is a *state-dict fingerprint*, **not** proof of unchanged model source code. ONNX identity is SHA-256 of the exact frozen original model bytes. Preprocessing descriptor is user declared (string IDs / integers only in A0 to avoid silently incompatible JS float formatting). Shape/axis profile never implies tumor semantics, model-output importance or causality. For preregistering methods, observer.model_sha256 exposes original ONNX hash or current PyTorch state hash.

**Not yet implemented:** arbitrary nested PyTorch output structures, distributed scheduler, external-user installation qualification, formal license/PyPI publishing, U11 independent scientific validation and clinical explanation. These belong to A6 or separate M4 scientific work. Do not infer that this is a scientifically validated explanation method.
