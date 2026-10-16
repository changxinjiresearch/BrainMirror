# SCEPT v1.1 Browser Studio — engineering audit / release candidate

## Core principle
SCEPT observes **frozen pretrained networks**. Model training remains an independent scientific workflow. SCEPT is neither an MRI model trainer nor a synthetic animation generator.

## Source of truth
- Canonical implementation: \`scept-studio/index.html\`, \`studio.css\`, \`studio.js\`
- Offline ONNX graph instrumenter: \`scripts/scept_instrument_onnx.py\`
- Tests: \`tests/test_scept_static.py\`, \`tests/test_scept_instrument.py\`
- CI: \`.github/workflows/scept-studio.yml\`
- Historical BrainMirror web demo: \`web/\` (not used as live inference code)

## Modules
1. Model registry: local ONNX/.ort import, pretrained ONNX Model Zoo examples, ViT-base transformer demo, identity SHA-256.
2. Inputs: drawing/image loader, NIfTI-1 and tensor-JSON, normalization manifest, local orthogonal raw-storage slice preview.
3. Inference: browser WASM/WebGPU session, actual ONNX forward execution. No cloud/paid model service.
4. Observation: actual exported intermediate values; graph metadata from ONNX exporter; compute node/tensor dependence is not inferred causal influence.
5. Experiment: paired same-checkpoint baseline and perturbation; a separately instrumented mask graph can gate selected activation units without retraining.
6. Record: local IndexedDB, trace schema, output SHA hashes, input hashes, CSV/JSON, explicit large-output truncation.
7. Website: responsive research layout, experiment logging and scientific caveats.

## Nontrivial limitations / release gates
- A deployed static site returning READY **does not prove** a pretrained model loads and runs in each target browser.
- CDN and ONNX Model Zoo demo fetch may be blocked by CORS/network policy or browser memory.
- A generic ONNX checkpoint has no guaranteed internal state visibility until graph outputs are exported.
- Dynamic multi-input, tokenizer, audio/video, operator sets, external data, huge 3D networks and custom formats require separate adapters.
- Current volume editing is a numeric (primarily intensity) perturbation in the model's input grid. It is **not** a clinically credible 3D anatomical warp with affine/Jacobian quality control.
- The model and data remain local by default. The interface is not a multi-user authenticated research server.
- Verified numerical intervention on a constructed ONNX graph validates the software semantics only, NOT the generality or causal-scientific hypothesis of SCEPT.
- Formal scientific evaluation requires preregistered metrics, controls, cross-model tests, participant- and seed-level samples.

## Acceptance status as of current code
| Gate | Status |
|---|---|
| Source in GitHub branch and independent Vercel production site | Built |
| Static JS/Python/DOM CI | Passed |
| CPU ONNX mask reference fixture | Passed in CI |
| Model Zoo CNN examples in actual browser | Not independently observed |
| Real pretrained Vision Transformer in browser | Not independently observed |
| Actual user ONNX intermediate extraction | Converter implemented; E2E not observed |
| Accurate MRI anatomical warping/ROI validity | Not implemented |
| All-purpose pretrained model compatibility | Not claimed |
| Cross-browser/mobile accessibility | Not independently tested |
| I-JEPA research | Explicitly excluded |

The first three passed engineering gates are **not** a substitute for the outstanding scientific/field tests.
