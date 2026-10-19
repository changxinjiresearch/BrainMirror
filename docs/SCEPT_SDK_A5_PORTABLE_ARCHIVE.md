# SCEPT Python Research SDK A5 — Portable Scientific Observation Archive

## Status and scope

A5 delivers **file-level compatibility between the existing real SCEPT V6.5.2 website and the independent Python Research SDK**, not a new attribution mechanism. Python package version: **0.2.0a5** (source-install alpha). Neither M5 (A6 qualification) nor U11/M4 (scientific efficacy) is achieved merely by satisfying this stage.

### Authoritative source records are NOT rewritten

- Website: `record.json` **`scept.run-record.v1`** with genuine V6 runtime observations.
- Python: `record.json` **`scept.python-capture.v1`** with genuine A2 source tensor records. A3 NAF is computed **separately**, in `naf.json` (`scept.naf.v1`).

Both live unchanged in the same proven ZIP layout **`scept.full-observation.v1`**. Manifest additionally records `sourceRecordSchema` for Python. Rewriting a Python raw record to claim it was already generated and verified by the JavaScript runtime would be scientifically misleading and is NOT permitted. A5 does not silently merge run identities, hidden graph paths or source model fingerprints.

### ZIP contents and privacy

`record.json`: exact single-forward source record with frozen model/input/preprocess/runtime hashes and actual observed output hashes.
`naf.json`: numeric single-run fingerprint computed from the exact captured real tensors.
`reference.json`: only when source model/preprocessor and real reference identity match.
`tensors/outputs/NNN.bin`: complete original numeric outputs (including logits and all instrumented graph probes), IEEE float/int typed bytes, no subsampling.
`tensors/inputs/NNN.bin`: only after user explicitly chooses `include_inputs=True` **when exporting Python capture**. The website requires the separate pre-inference retain-input opt-in. The exact model input is a *preprocessed feed*, not necessarily the source NIfTI image.
`manifest.json`: ZIP schema, provenance/exclusion status, per-file SHA-256, per-tensor type/shape/SHA and mandatory completeness counts.

No model weights, native MRI NIfTI source, executable preprocessing code, uninstrumented layers, patient identity guarantees, output importance/causal score or full model computation trace is included. **Output activations can themselves expose sensitive patient data** even with input inclusion disabled. Users are responsible for local storage and consent. The SDK never silently uploads this archive to GitHub or the cloud.

### Read, validate, replay (distinct statements)

```python
from scept import SCEPT, load_observation

run = SCEPT(your_model, layers=["your_actual_layer"], backend="torch",
            profile="cnn-nchw").observe(your_preprocessed_input)
run.export("observation.zip")                        # Default: outputs only
run.export("with-model-input.zip", include_inputs=True)

archived = load_observation("observation.zip")
archived.tensor("your_actual_layer")                 # exact numeric copy
replayed_naf = archived.recompute_naf()               # OFFLINE arithmetic only
assert replayed_naf["science"]["model_forward_reexecuted"] is False

# Separately provide the same frozen model instance / ONNX weights
archived_input = load_observation("with-model-input.zip")
report = archived_input.verify_model_replay(
    observer=SCEPT(your_model, layers=["your_actual_layer"],
                   backend="torch", profile="cnn-nchw"),
    atol=1e-6, rtol=1e-5)  # genuine NEW forward, new run ID
```

`recompute_naf()` rederives mean/RMS/sign fractions from ARCHIVED raw tensor bytes; it cannot attest to original model identity or reexecute a model. `verify_model_replay()` requires explicit inputs, separately supplied exact model fingerprint, same observed tensor set and predeclared numeric tolerance; a new forward is performed and each output is compared. Different providers can cause legitimate floating differences. In the website-only case, a real frozen model usually must be downloaded independently because the ZIP never contains the weights.

Use CLI after local pip install: `scept-archive verify observation.zip` or `scept-archive recompute observation.zip`. Both print clear `modelForwardReexecuted: false` and `modelOriginIndependentlyAuthenticated: false`.

### Threat and scientific safety boundaries

1. ZIP format restricted to STORE/uncompressed, 160 MiB / 100 entries, no duplicate or traversal paths, no symlinks, no pickle/unsafe executable load, no decompression bombs.
2. All bytes checked against ZIP CRC and signed-by-manifest SHA (file-level), tensor SHA includes canonical dtype+shape and raw bytes, full actual output inventory must match original record, and stored NAF states are recomputed and compared per unit. Altered binary or NAF, even when a hostile author regenerates ZIP CRC and manifest hashes, will not pass unless the hostile author forges *all* science records coherently.
3. This format has **NO cryptographic author identity signature**. A malicious author can fabricate a consistent fake run and rehash it. File integrity is not authenticated proof that any real claimed model/dataset was executed. An independently checked frozen model with source input must be reexecuted for stronger evidence.
4. Any archived update numerical values are checked against original pre/post float arrays. An actual graph path or valid same-coordinate scientific assertion cannot be inferred solely from ZIP and must be verified against separately supplied frozen model graph.
5. Partial/missing output data, invalid input retention declarations, non-finite observed RMS values, source-NAF mismatches and omitted model/preprocess identity fail closed.

### A5 verification gates

- Python actual PyTorch CNN and ONNX Conv→Relu forward captures, all outputs and optional original inputs exported and imported.
- Independent JS Node browser-oriented verifier parses actual Python ZIP, checks every native ZIP CRC, file SHA, model tensor SHA/shape and Run Record identities.
- Python executes the **existing unchanged** original V6.5.2 JS capture + archive builder; imports real produced ZIP with original browser `scept.run-record.v1`, validates raw typed values and redoes NAF arithmetic.
- Browser Chromium real public pretrained ViT NAF and two downloaded ZIPs (without / with consented preprocessed input) are audited by independent Python loader, including 985+ measured NAF units.
- Explicit real model re-execution verifies every captured output with same independently supplied model and archived input; mismatched model SHA rejected.
- Negative tests for traversal, rehashed NAF forgery, record hash modification, missing input and unsupported archive format.

### Next stage A6

Independent external-operator installation on clean environments, complete public examples and documentation, Python compatibility matrix, version contract/migration policy, third-party reproducibility report, release licensing and (if desired) PyPI publication; scientific/clinical U11 M4 remains a separate, unresolved question. **M5 may only be claimed after A6 engineering acceptance.**
