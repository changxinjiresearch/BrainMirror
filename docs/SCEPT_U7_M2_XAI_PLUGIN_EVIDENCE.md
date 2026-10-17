# SCEPT U7 / M2 — Extensible XAI method and evidence-evaluation protocol

## M2 acceptance criterion

A **new experiment** is described by an external JSON manifest, declaring:
1. **Model**: exact immutable SHA-256 and optional model ID/explicit capabilities.
2. **Target**: an actual ONNX exported numeric tensor plus a flat index, or a mean reducer; never an inferred diagnosis or probability.
3. **Intervention**: U5 registered plugin, parameters, fixed uint32 seed, U6 A/B / sweep / batch plan.
4. **Observation**: explicit runtime ONNX exported outputs and a U4 axis profile; new uninstrumented hidden state names are rejected.
5. **Explanation method**: U7 registered `method.explain` adapter.
6. **Evaluation**: one or more registered `evaluation.score` adapters with preregistered numeric decision criteria.

The host's **main web program is not modified** when a new experiment manifest is imported using the UI file selector or pasted JSON. A new algorithm is an independently registered trusted-JavaScript adapter: `registry.register('explanation', {...})` / `registry.register('evaluation', {...})`, without modifying the U7 core. Loading third-party executable JS in the browser requires a trusted script-loading mechanism; JSON itself never executes JS.

## Interfaces

- `scept-web/core/scept_xai.js` is the M2 protocol engine and adapter registration.
- U1 registry already has `target`, `explanation`, `evaluation` categories. U7 uses those directly.
- U6 supports a reusable optional `analysisAdapter.evaluateTask` hook. For each **real frozen ONNX run**, it receives runtime outputs (A, B, optional C), U4 observations and U5 intervention metadata. **Only after real results are returned** does it compute U7 evidence.
- `SCEPT_XAI.make(...)` supports `registerExperiment(manifest)`, `prepare(id,{feeds,referenceFeeds})`, `run()`, `exportEvidence()`, `snapshot()`, `retryFailed()`, `cancel()`, `resume()`.
- `scept.u7.evidence.v1` includes experiment manifest, model SHA, input hashes, task state/seed/attempts, real output target A/B/C numeric contrasts, per-task explanation and evaluation, sweep-level tests, genuine U4 observations and explicit scientific-limit statements.

Built-in **target** adapters:
- `u7.target.tensor-element`: flattened index of genuine runtime output, no implied class or logit semantics.
- `u7.target.tensor-mean`: mean of actual numeric output tensor; no clinical meaning unless explicitly calibrated externally.

Built-in **method** adapters:
- `u7.method.target-contrast`: target B−A signed change / absolute magnitude with optional C control.
- `u7.method.activation-response`: rank genuine observed tensors by U4 real mean absolute response; this is **descriptive** and not automatic XAI fidelity or causality.

Built-in **evaluation** adapters:
- `u7.eval.directional-shift`: predeclared direction and minimum numeric threshold.
- `u7.eval.control-specificity`: genuine A/B/C comparison, otherwise **not_evaluable**, not "passed".
- `u7.eval.dose-monotonicity`: require >=3 distinct completed intensity tasks; reports finite strength-to-absolute-target-change and violations. Incomplete plans => **not_evaluable**.

All test results use explicit `criterion_met`, `criterion_not_met` or `not_evaluable`, **never** "causality proven", "tumour necessary" or "clinically valid." No numerical test here establishes disease-specific attribution, shortcut reliance or causal dependence.

## Lesion-removal experiment plugin

Template: `scept-web/plugins/lesion_removal.template.json`. **It is intentionally not runnable as distributed.** The caller must supply:
- A genuine frozen **MRI-trained** model hash / compatible inputs / declared target and actual internal observations.
- An independently obtained lesion segmentation mask, converted correctly into the actual model input's flat tensor index convention and dimensional shape.
- The sorted unique `mask.indices` checksum computed with `SCEPT_CORE.sha256(Uint32Array.from(sortedIndices))`, a named source ID and the declaration `coordinateSpace:'model-input-flat-indices'`.
- Explicit fill/replacement policy and appropriate controls chosen for the scientific target. Region masking does not automatically generate counterfactuals free of artifacts.

The built-in `u7.masked-occlusion` plugin runs the index-wise edit **only after** the declared mask matches actual runtime input dimensions and its cryptographic index checksum matches the supplied array. Mask checksum is **integrity only**: it cannot establish correct MRI registration, lesion annotation quality, anatomical validity, causal dependence, or model calibration. An unknown mask, wrong hash, out-of-range ROI, missing actual target tensor or model SHA mismatch fails closed or records task error.

A synthetic **engineering** version (including a mask containing two values) is covered by unit and real Chromium/ONNX tests. Do NOT report its response as a real tumour-removal finding. Any model/dataset-level dependency conclusion requires prospective matched controls, validated MRI processing and external evidence.

## Examples and acceptance

- `scept-web/plugins/m2_engineering_cnn.json`: standalone JSON **new experiment plugin**, pinned to a generated untrained ONNX CNN fixture SHA; input must be supplied, no MRI meaning.
- `scept-web/plugins/m2_external_method_example.js`: separately packaged example of registration of new method/evaluator; never imported by the main program. Trusted JS only.
- `scept-web/tests/u7_xai_protocol.cjs`: strict model/target/adapter/error contracts, independent method registration, genuine numeric protocol across an isolated test fixture, dose evidence, sparse-mask sanity (unit stub).
- `scept-web/tests/u7_m2_browser.cjs`: **real Chrome+ONNX/WASM** new experiment JSON through existing UI; executed U6 multi-strength with genuine hidden tensors, deterministic replay, independent lesion-mask numeric example, and no main program edits.
- Existing U0–U6, pretrained ViT/CNN, MRI and 3D visual regressions stay required in GitHub CI.

**M2 is only verified when the final head's real browser E2E and remaining full CI pass.** U7 is not a universal Grad-CAM replacement; gradient attribution requires a model exposing proper gradients and scientifically validated evaluation. Real clinical lesion dependence is not claimed by the engineering platform.

## Runtime/privacy boundary

All browser-side execution happens in the current user session; no automatic cloud background execution. JSON evidence includes input hashes and real numerical observations, not full raw MRI arrays, but some numerical evidence and mask metadata may still be potentially sensitive. Export is user-controlled. Full replay requires original verified ONNX bytes and inputs; export alone is not a restorable checkpoint.
