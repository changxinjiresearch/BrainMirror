# SCEPT Research SDK A0–A4 (alpha)

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

**Research boundaries:** One *genuine* forward per observe call; all captured tensor arrays are full copies. Tensors must be dense, finite, nonempty and in the V6 numeric dtype set; the default total input+output copy budget is 32 MB. A named PyTorch module must execute exactly once. ONNX layer names must be actual graph producers with verified types; graph outputs are instrumented **in memory** and the original ONNX file is not changed. Training flags and Hook handles are restored even on failure. A0 Python record currently uses `scept.python-capture.v1` rather than falsely using the browser's completed `scept.run-record.v1` NAF semantics.

PyTorch model identity is a *state-dict fingerprint*, **not** proof of unchanged model source code. ONNX identity is SHA-256 of the exact frozen original model bytes. Preprocessing descriptor is user declared (string IDs / integers only in A0 to avoid silently incompatible JS float formatting). Shape/axis profile never implies tumor semantics, model-output importance or causality. For preregistering methods, observer.model_sha256 exposes original ONNX hash or current PyTorch state hash.

**Not yet implemented:** portable Python ZIP `.export()`, arbitrary nested PyTorch output structures, distributed scheduler, fully independent scientific U11 replication and validated external clinical explanation. These are reserved for A5–A6 (scientific M4 separately). Do not infer that this is a scientifically validated explanation method.
