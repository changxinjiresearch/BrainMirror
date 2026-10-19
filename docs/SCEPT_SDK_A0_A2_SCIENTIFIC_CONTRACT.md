# SCEPT Research SDK A0–A2 — Scientific contract and release gates

## Stage boundary and evidence
A0–A2 = installable Python raw-observation alpha tooling, NOT a validated XAI attribution or completed NAF Python math implementation. Browser V6 is an independent existing JS codebase. U11 found explanatory benchmark weaknesses; do not assert Research M4 scientific success.

The new package is in scept_sdk/ and imports as scept. The legacy brainmirror/ 3D autoencoder is NOT the general-purpose SDK.

### A0 evidence levels (never merge them)
OBSERVE: single true frozen inference, measured named internal/output tensors, independent original input and model SHA.
NAF: browser V6.2 currently implements real RMS/state/pattern; Python A3 is still pending. Python A2 explicitly uses scept.python-capture.v1 (compatibleBase scept.run-record.v1) because calling a raw capture an already computed NAF would be false.
EXPLAIN: U10 is a separate controlled model-output dependence experiment. Internal magnitude or stability does not establish attribution, causal mechanism or tumor-specific evidence.
VALIDATE: software correctness, scientific superiority and clinical validity are distinct.

### Python A2 raw-capture schema
RunID, creation time, model/input/preprocessing/runtime provenance, named input/output declared signatures, actual dtype/shape and output SHA hashes, per-probe measured/unavailable status, adapter/profile and explicit negative scientific claims. JSON records do NOT contain actual patient/activation arrays; private in-memory tensor() and input() access returns full defensive copies.

Tensor digest parity matches existing JavaScript V6:
SHA256(UTF8(sorted JSON object with keys shape/type) + 0x00 + genuine C-contiguous little-endian typed bytes).
Test with actual Node runtime and exact Python bytes. Python A0 preprocessing metadata restricts floating-point properties until A3 can verify cross-language float canonicalization; strings, booleans, null, safe integers allowed.

ONNX model SHA hashes exact unmodified original model file bytes. PyTorch model fingerprint hashes actual state_dict weights and buffers, not its complete forward source code: research provenance must also preserve code/checkpoint separately. User-declared preprocess ID does not independently prove preprocessing code.

### A2 PyTorch
Install from source: python -m pip install './scept_sdk[torch,onnx]'.
Use SCEPT(model, layers=['conv'], backend='torch').observe(tensor). Captures actual module outputs using hooks in exactly one inference_mode call, restores training flags and hook state. Repeatedly used modules without unique call-site IDs fail closed. No fabricated biological coordinate labels. Requires actual Tensor outputs; unsupported dtypes/nonfinite arrays are rejected. Resource budget 32 MB per run by default.

### A2 ONNX
Use SCEPT('frozen.onnx', layers=['/real/Conv_output_0'], backend='onnx').observe(real_input).
Parser verifies named actual graph producer and typed value-info (or real ONNX shape inference). In-memory graph output promotion allows ORT to return actual hidden tensors. Unproven/unknown probes fail; actual original ONNX file is not modified. Default uses CPUExecutionProvider, no hosting GPU fees. Multi-input models accept named input mappings.

## Three scientific engineering issues across A0–A6

1. **JS/Python numeric consistency**: A0 proves canonical float32 tensor SHA via real Node code; A3 formalizes exact cross-language NAF state/update/representation calculations with declared float tolerances; A5–A6 checks archive and full real model outputs on independent platforms. Same-shape output from different model bases is NOT semantically aligned merely due to dimensional equality.
2. **Observation versus explanation**: A0 explicit record statuses; A2 captures only actual tensors; A3 NAF state and verified same-forward module updates; A4 independent U10 experimental protocol. Magnitude of internal activity is not output relevance, a causal neural pathway or an MRI lesion explanation.
3. **Research software engineering versus scientific validity**: A2 unit/real ONNX/torch numeric tests, A3–A5 more benchmarks and replay, A6 external-user reproducibility. M5 engineering may eventually be achievable; U11/M4 comparative XAI scientific validation remains independently necessary.

## Completion and deferred scope

CI builds an installable wheel and imports scept; runs genuine CPU PyTorch CNN (+BatchNorm) and variable-length Transformer module hooks, ONNX Conv/Relu graph instrumented values, exact Python↔JS tensor SHA checks, missing probe/duplicate calls/shape+dtype/resource guards. Real computations use synthetic numeric fixtures, NOT clinical evidence.

A3 Python NAF mathematics + validated JS parity; A4 U10 custom experiments; A5 scientific ZIP/replay and Python/web compatibility; A6 real pretrained public external model acceptance, reproducibility and scientific baseline/reporting.

Passing this stage authorizes the statement **A0–A2 engineering completed**, not completion of A3–A6, M5 or M4.