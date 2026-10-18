# SCEPT V6.0–V6.1 — Formal NAF Contract and Independent Single-Run Capture

**Status:** V6.0 contract + V6.1 execution core. V6.2 GUI, calibrated NAF rendering, computation-update estimands and channel-pattern UI are **not** part of this milestone. Neither single-run state RMS nor color is decision importance, explanation validity or causal tumor evidence. U11 M4 remains NOT ACHIEVED.

## Purpose and scientific distinction

The **SCEPT Run Record** is an independently useful record of ONE genuine frozen model forward pass. An **NAF** is a representation of one such record; A/B comparison is not a prerequisite for this record or its activity values.

This is distinct from the historical U4 \`observePair\`, U10 controlled intervention (A/B/C), V5 NDF/CRC and ICM. They are retained unchanged and will use the common Run Record in V6.4.

### Formally defined quantities (V6.0)

For actual runtime output tensor \(H_p(X)\) produced at registered probe \(p\), a profile defines a set of actual observation units \(u\) (e.g., Transformer token; CNN channel). Each unit contains the native vector \(h_{p,u}(X)\) of \(d_{p,u}\) scalar channels/features.

- **State:** \(A_{p,u}(X) = \sqrt{d_{p,u}^{-1}\sum_j h_{p,u,j}(X)^2}\) is a *raw RMS state magnitude*. Unit mean, positive and negative fraction, and the complete retained tensor can also be inspected. RMS is invariant to sign reversal and may hide completely different channel patterns; it is NOT a sufficient summary of neural computation.
- **Update:** \(U_{p,u}(X)=\|h^\mathrm{out}_{p,u}(X)-h^\mathrm{in}_{p,u}(X)\|_2/\sqrt{d_{p,u}}\) is **specified but NOT evaluated in V6.1**. It requires true same-coordinate module input and output with a verified correspondence; equality of shapes alone is insufficient. Missing before/after tensors must produce \`not_observed\` / \`unsupported\`.
- **Pattern:** Real raw channel vector and future channel distribution/representation structure. V6.1 privately retains actual typed tensors for future V6.2 visualization. No fixed interpretation or semantic feature names are inferred.
- **Difference:** V5 NDF \(\sqrt{d^{-1}\sum_j(h_{p,u,j}(B)-h_{p,u,j}(A))^2}\) remains a separate optional operation, NOT equivalent to \(A_{p,u}(B)-A_{p,u}(A)\). Comparison requires explicit compatibility.

SCEPT retains distinctions: observable tensors ≠ human neurons; numerical magnitude ≠ compute FLOPs; self-attention weights ≠ causal explanation; model values ≠ medically validated evidence.

## Schema: \`scept.run-record.v1\`

Fields:
- \`runId\` unique cryptographic nonce; \`createdAt\` UTC ISO timestamp (for audit only; not used as scientific identity).
- \`provenance.model.{id,sha256}\`: SHA-256 of the actual instrumented frozen model (not merely source architecture).
- \`provenance.inputs\`: per-feed SHA-256 over a canonical type+shape header AND the exact typed tensor bytes passed to the model. Any unexpected input mutation during runtime aborts capture.
- \`provenance.preprocessing.{id,sha256}\`: named and stable SHA-256 of explicitly declared JSON preprocessing protocol/configuration. This fingerprints the declaration; it is not itself proof the declaration honestly matches code.
- \`provenance.runtime.{engine,version,provider}\`.
- \`signature.inputs/outputs\`: complete declared actual runtime I/O with typed shapes; dynamic output dimensions permitted if the rank is explicitly declared.
- \`outputs[]\`: every declared returned real tensor, name/type/real shape/element count, and SHA-256 over its raw content and descriptor.
- \`observations[]\`: only explicitly requested genuine probes; real recorded tensor per-axis activity summaries, or a \`not_measured\` observation with no numeric activity.
- \`availability[]\`: one status per requested probe: measured, not_observed, unsupported, failed (current V6.1 implementation produces measured/not_observed only; others reserved for future adapters).
- \`limits\`: actual bounded tensor count, total output bytes and per-axis summary cap.
- \`science\`: observation_only; no decision importance, stability, clinical or causal attribution.

Export is JSON metadata and finite numeric summaries ONLY, never raw MRI values or complete hidden tensors. Actual typed output tensor arrays are in a private, bounded in-memory handle \`getTensor(name)\`; accessing it returns a **defensive copy**. \`release()\` invalidates handles. No persistent backend, cloud model inference or paid API is required. The user may opt to save a JSON summary file later.

## Execution API

\`\`\`js
const engine = SCEPT_CAPTURE_ONCE.make({
  observation: SCEPT_OBSERVATION,
  contract: SCEPT_RUN_RECORD
});
const captured = await engine.captureOnce({
  session: verifiedSession, // {sha256, description:{inputs,outputs}, run(feeds)}
  feeds: {pixel_values:{type:'float32',dims:[1,3,224,224],data:actualFloat32Input}},
  modelId: 'verified-instrumented-vit-tiny',
  probes: ['verified_existing_onnx_hidden_output'],
  profileId: 'vit-token-channel',
  preprocessing: {id:'known-transform',config:{...}},
  runtime: {engine:'onnxruntime-web',version:'1.20.1',provider:'wasm'}
});
const record = captured.exportRecord(); // independently usable standalone NAF record
const hidden = captured.getTensor('verified_existing_onnx_hidden_output'); // actual values
captured.release();
\`\`\`

Exactly one \`session.run(feeds)\` is called by \`captureOnce\`, with no baseline, B sample, hidden intervention, or target class. External session adapters must be bound to genuinely verified model bytes. The engine validates the returned metadata, real tensor elements, cryptographic hashes, finite statistics and bounded memory. Partial or unknown runtime outputs fail closed. Names of requested probes not exposed by the actual model are explicitly marked not observed, with **no fabricated color/values**.

## Resource safety and model boundaries

Defaults: max 8,000,000 elements per tensor; max 32 MB of capture budget for output tensors AND separately 32 MB for input feeds; at most 64 declared outputs and 64 requested probes; max 2048 per-axis groups by default. If those limits cannot contain actual model data, the current engine **fails** instead of silently selecting or inventing values. Future V6.2–V6.3 should add streaming/statistical summaries or opt-in external compute for larger models.

\`observeOnce\` is a backward-compatible addition to existing \`scept_observation.js\`; the existing \`observePair\` remains available for V3/V4/V5 and U10. No existing scientific gate or result is changed.

## Explicit acceptance (V6.0–V6.1)

1. One real public pretrained ViT + authentic public MRI example produces 5 genuine hidden-layer activity summaries, 197 true Token axis units each = 985 genuine activity cells, from **one** frozen ONNX inference, with no A/B run.
2. A second independent capture has an independent runId and input/output provenance. Model/preprocessing/input hashes must be present; fake probe names have no activity.
3. A raw tensor is available for exact drill-down in memory, with defensive copies, but is absent from JSON export and cannot be accessed after release.
4. Fail closed on mismatched shape, nonfinite tensor, undeclared outputs, duplicate probes, output resource budget, changed input during inference, invalid preprocessing declaration.
5. Existing U4 pair observation and U10/V5 browser regression remain unchanged.
6. CI success only demonstrates engineering observability, not original XAI algorithm efficacy.

## Subsequent milestones

- **V6.2**: user-facing single-run NAF fingerprint, state/update/pattern views with correct per-unit semantics, show unsupported rather than simulated. Actual module before/after probe identities required for Update.
- **V6.3**: reference calibration, fixed scales, multiple real model profiles (CNN/3D/I-JEPA adapters under task-correct preprocessing).
- **V6.4+**: migrate comparison, controls, explain plugins to versioned Run Records, independent scientific validation and generalization.

Do not replace the stable public V5 homepage merely because V6.1 core passes. The V6 public interface belongs to V6.2 / later after real user-facing acceptance.
