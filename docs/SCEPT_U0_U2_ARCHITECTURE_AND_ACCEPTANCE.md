# SCEPT Universal Core — U0–U2 implementation and acceptance contract

Status: **Engineering integration branch**; not a scientific explanation algorithm and not a validated diagnostic device.

## U0 — consolidate without sacrificing tested demos

Canonical source branch `phase/scept-u0-u2-universal-core` is based on the latest validated v3.1 visual clarity branch `fix/scept-v31-token-visual-clarity`. This ancestry retains the v2 single-input demo, v3 authentic MNI MRI frozen inference, multiple ViT intermediate probes, token activity, CNN exploratory inference, and all existing browser acceptance tests. The new standalone `scept-web/universal.html` is linked from both `index.html` and `v3.html`; no legacy or MNI functionality has been deleted.

No changes to the original production `main` until the complete PR acceptance gates pass. Old preview deployments remain separate. Existing `main` is not assumed to be running v3.1 yet.

## U1 — contracts, not pretend plugin implementations

`core/scept_core.js` exports `SCEPT_CORE.VERSION`, a registry, input/output tensor contracts, SHA-256 verified model loading and a strict fail-closed lifecycle.

Categories in the **registration protocol**: `model`, `data`, `target`, `observation`, `intervention`, `explanation`, `evaluation`. Declaring an extension point is **not** the same as implementing an adapter. At U1/U2, only the `model` implementation is available. Unknown capability calls fail with a typed `SceptError`.

Model adapter implementation contract:
- `{id, kind:'model', capabilities, inspect(bytes, context), load(bytes, context)}`.
- `inspect` returns exact named inputs and outputs with types, dimensions, and actual graph provenance.
- `load` returns a session with `run(namedFeeds,{outputs})`, `dispose()`.
- `openModel(registry, manifest, bytes)` checks declared source SHA-256 (optional for a local file, but *always computes and records* one), names, supported types, declared shapes and output existence. No unknown node or fake tensor can be requested successfully.
- Native GPU training, network storage and model weights modification are outside the scope.

## U2 — working model-agnostic ONNX adapter

`core/onnx_adapter.js` and `core/onnx_probes.js` provide real ONNX Runtime Web/WASM inference for compatible ONNX models. The parser reads actual graph metadata and supports optional intermediate tensor output promotion **only for existing annotated GraphProto value_info records** (up to 16). If no intermediate metadata exists, only declared graph outputs are available. A request for unobserved/untyped data is refused.

Input arrays are explicitly named/typed per model. For dynamic input axes, concrete user-provided dimensions and typed values are required; the core will not guess. Some uncommon ONNX dtypes/operators or large models will be rejected, not simulated. Output values are direct ORT tensor data.

`universal.html` is a separate Chinese workbench. It can download pinned and SHA-256 verified sample ViT/CNN, or import a compatible local ONNX file; select actual annotated intermediate probes; supply typed input JSON or use a **clearly marked synthetic numeric engineering fixture**; perform two frozen forward passes with a trivial numeric perturbation; and export model/inputs/outputs/provenance/audit. This does NOT imply correct biomedical preprocessing, self-supervised target semantics, decision explanations or causal claims.

## Acceptance, boundary and dependency notes

- `tests/u1_core_contract.cjs`: duplicate adapter rejection, unknown capability rejection, source SHA mismatch rejection, invalid/mismatched tensor shape rejection, invalid outputs and disposed session.
- `tests/generate_u2_fixture.py`: create two fresh ONNX graphs with real weights/ops, an MLP `MatMul → Add → Relu` and a CNN `Conv → Relu`, each with declared intermediate value_info.
- `tests/u2_generic_adapter.cjs`: browser-import both generated ONNX files through the **same new adapter** without changing the engine; select genuine intermediate outputs, independently verify numeric outputs against MatMul/Conv calculations, run input A/B, and reject a fabricated probe name.
- All original v3.1 MRI, ViT, 3D, CNN and Chinese UI tests remain regression gates in `.github/workflows/scept-browser.yml`.

U3–U11 **not complete**: MRI-specific normalization, generic data adapters, classification/segmentation/self-supervised target adapters, gradients and activation patching, intervention protocol and experiment scheduling, scientific explanation plugins, faithfulness assessment, external I-JEPA checkpoint validation.

## Code ownership

| Current stable capability | Source |
| --- | --- |
| Latest MRI + ViT + CNN visual study | `v3.html`, `v3_engine.js`, `v3_mri.js`, `v3_network3d.js`, `scept_activity3d.js` |
| Legacy v2 synthetic input demo | `index.html`, `studio.js`, `lab.html` |
| New U1 interface and registry | `core/scept_core.js` |
| U2 generic real ONNX intermediate annotation | `core/onnx_probes.js` |
| U2 frozen ONNX engine adapter | `core/onnx_adapter.js` |
| Model-independent user workbench | `universal.html`, `universal.js`, `universal.css` |

**Architecture rule:** no new experiment is allowed to add a special case to `scept_core.js`. Each new model is registered via a `model` adapter and must pass capability tests. Missing support is a first-class scientific result, not a rendering prompt.
