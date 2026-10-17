# SCEPT U5 — Universal Intervention Engine

**Engineering implementation:** U5 introduces first-class `intervention` plugins on top of the U1 registry, U3 typed-feed contract and U4 genuine tensor observation. No MRI-specific hardcoded spatial conventions, model retraining or browser-only simulated activations.

## Core architecture

Files:
- `scept-web/core/scept_intervention.js`: registered plugins + fail-closed `interventionEngine.execute({pluginId,description,feeds,config,referenceFeeds})`.
- `scept-web/core/onnx_adapter.js`: verifies actual `Mul` graph nodes with named SCEPT instrumentation, live mask input, positive static float32 dimensions, genuine producing and consuming nodes. Only verified instrumented graphs advertise `hidden.intervention`.
- `scept-web/universal.html` / `universal.js`: plugin selection, explicit JSON parameters, user reference inputs, genuine two-pass (A/B) or three-pass (A/B/C) ONNX WASM execution, U4 observations and downloadable audit.
- `scripts/expose_onnx_probes.py`: existing **offline** graph transformer that inserts a real `scept_mask` Multiply, not model retraining. The original model and the modified graph are different ONNX artifacts; always persist/check hashes.

### Plugin contracts and scientific limitations

| Plugin ID | Operation | Required prerequisites | Main output |
| --- | --- | --- | --- |
| `u5.noise` | Seeded Box–Muller Gaussian additive noise | Float32/64 input; sigma>=0, integer uint32 seed; optional explicit ROI | Original and edited feeds; seed/distribution/changed count |
| `u5.occlusion` | Fill selected tensor index hyperrectangle | Explicit `region.bounds` (one half-open [start,end) pair **per tensor dimension**) | Genuine edited input, ROI and count |
| `u5.replace` | Replace selected ROI with same coordinates from an **explicit reference feed** | Exact reference tensor names, types and dimensions | Provenance marked reference-based numeric replacement |
| `u5.matched-control` | Pair one ROI occlusion with a spatially disjoint, equal-sized comparison ROI, run A/B/C | Explicit `region`, `controlRegion`, `fill`, `meanTolerance`. Check baseline region mean difference <= tolerance | Main and control observations, distinct ROI, measured means and sizes |
| `u5.internal-mask` | Modify actual hidden activations through an added float32 ONNX `Mul` input | Verified instrumented graph, declared `scept_mask` or `__scept_mask__...` and `config.indices`, multiplier 0..1 | Baseline ones mask, edited mask; truly downstream computation changes |

A numeric region is **not** a medical ROI. The match test checks equal support count, disjointness and **mean only**, not tissue type, scanner provenance, sex, age, SNR, anatomical validity or causal identification. The label `matched-control` is explicitly scoped to this numerical definition. Region replacement is coordinate-aligned input substitution, **not** geometric co-registration.

`u5.noise` uses a deterministic seeded pseudo-random generator, **not a secure random source**. A nonzero sigma does not guarantee an output change. Failed edits, nonfinite values, unsupported dtype, invalid indices, missing reference feeds and false internal-mask claims are rejected. Neither input perturbation nor internal-state masking automatically constitutes a scientific explanation.

### Example configurations

JSON examples (change region rank/bounds to match actual model input shape):

```json
{"sigma":0.1,"seed":42,"region":{"bounds":[[0,1],[0,1],[0,4],[0,4]]}}
```

```json
{"region":{"bounds":[[0,1],[0,1],[0,4],[0,4]]},"fill":0}
```

```json
{"region":{"bounds":[[0,1],[0,1],[0,1],[0,2]]},"controlRegion":{"bounds":[[0,1],[0,1],[1,2],[0,2]]},"meanTolerance":1,"fill":0}
```

```json
{"maskInput":"scept_mask","indices":[0,4],"value":0}
```

For true internal gating without retraining:

```bash
python scripts/expose_onnx_probes.py original.onnx masked.onnx --ablate conv_hidden
```

Import **masked.onnx**, provide input values matching every declared model input (including the added float32 mask shape), select `u5.internal-mask`, and use exact indices. U5 forces all-one mask for baseline A; intervention B alters only the mask input. No claim is made for arbitrary uninstrumented networks.

## Acceptance

- Unit: `node scept-web/tests/u5_interventions.cjs`: reproducibility, typed/axis/ROI guards, reference and match validation, graph-mask capability denial, extension adapter registration, immutability.
- Genuine CPU browser: `node scept-web/tests/u5_browser.cjs`: original CNN Conv+Relu fixture receives numeric noise, occlusion, replacement, numeric matched A/B/C, rejects noninstrumented hidden mask, and runs actual masked Conv ONNX with changed downstream final output. No fake hidden tensor / training / clinical data.
- Existing U0–U4 plus MRI/ViT visual regression tests stay mandatory.
- PASS requires actual green CI and the latest branch/commit. The CNN fixture is **not pretrained**, and the pretrained ViT from M1 remains a separate frozen-model regression.

## Deferred to U6+

Scientifically meaningful anatomy-aware ROI mapping, data-target coupling, automatically generated matched controls, more sophisticated observational confound matching, intervention schedules/experiment orchestration, cross-model semantics, causal attribution and fidelity evaluation. Only *compatible pre-instrumented internal states* are now supported.
