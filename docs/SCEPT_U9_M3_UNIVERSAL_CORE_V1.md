# SCEPT Universal Core V1 — U9 / M3 Engineering Acceptance

## Completion conditions

M3 requires the complete U0–U8 regression suite to remain green, genuine Chromium/WASM ONNX plugin/experiment E2E acceptance, reproducible integrity-checked audit/replay, explicit plugin security and performance limits, and a verified no-new-fee static publication path. A successful code merge and browser CI do not themselves establish a public deployment.

### Plugin SDK

- scept-web/core/scept_sdk.js provides validateAdapter, install, compatibility, template and exampleManifest. It uses the existing U1 kind/capability registry and U7 experiment schema.
- scept-web/plugins/u9_method_evaluation_sdk_template.js is a standalone trusted script with an explanation adapter and evaluation adapter. Installing it does not require modifying SCEPT's core program. This is a numerical engineering extension, not an established XAI technique.
- SDK compatibility checks declared plugin capabilities, adapter existence, session model SHA-256, actual exposed runtime outputs, requested target/observations, and optional U4 axis profiles. A static pass is not proof the model runs on the given device, or of clinical validity.
- JSON manifests are data. JavaScript method plugins are **trusted host code**, not isolated or safe-to-run untrusted scripts. Review extensions before installing.

### Audit and replay

- scept-web/core/scept_audit.js exports scept.u9.audit.v1 with canonical sorted-key SHA-256 content fingerprint, original experiment definition hash, model SHA-256, input/ref tensor SHA-256 and U6/U7 genuine task evidence.
- Redaction removes raw numeric tensor data fields, mask index arrays and known identifier fields. Derived observations and custom plugin metadata can still be sensitive. There is no guarantee of anonymization of custom metadata.
- Fully completed U6/U7 experiments only are eligible for replay-grade records. The browser deliberately **does not** store weights, raw medical input arrays or secrets inside the audit.
- Verification requires the unchanged bundle checksum, original model, original typed feeds/reference inputs and the original *unredacted* U6 plan or U7 manifest. An exported audit alone is not a restorable checkpoint.
- Replay actually re-runs U6/U7 ONNX inference, comparing identical task IDs/seeds, output names/shapes, actual A/B/delta and evaluation verdicts. Default tolerance is atol=1e-5 and rtol=1e-4; model-provider floating point portability is not guaranteed. Tampered checksum, wrong source inputs/model/definition, uncompleted tasks and numeric divergence are explicit failures.
- SHA256 is an **integrity checksum, not an author-authenticating digital signature**.

### Performance

- scept-web/core/scept_performance.js measures real baseline session.run calls with user-selectable warmup, 1–15 forward passes, individual timings, median, mean, p95 approximation and known input typed-buffer MiB.
- Memory is an *input tensor size estimate*, **not** browser peak heap, WebAssembly allocations, model-weight footprint or GPU usage. Timing depends on actual CPU/OS/runtime; no invented numeric latency or arbitrary performance promise.
- Input budget and optional max mean latency budget can be configured and verified. The main site provides three measured runs after one warm-up.

### Safety and zero-fee hosting

- scept-web/universal.html CSP disallows inline JS/general eval, frames/objects, and restricts remote scripts to a pinned-version ONNX Runtime Web CDN. WASM is permitted explicitly.
- An external version-pinned CDN remains a supply-chain and network dependency: M3 does not claim offline-only or fully self-hosted dependencies.
- scept-web/tests/u9_security_check.cjs enforces static safety and no-unknown-fee publish gates.
- .github/workflows/scept-pages.yml publishes the existing PUBLIC repository's scept-web static pages using GitHub Pages only **after successful main-branch SCEPT Browser Engine Acceptance**. No new paid GPU, hosting account, backend or inference API is created. GitHub Pages must first be enabled by a repository owner at GitHub repository Settings -> Pages -> Build and deployment -> Source: GitHub Actions. The default GitHub Actions token **cannot automatically enable Pages** using actions/configure-pages. This first-time settings action does not add a paid account or fee. The release workflow cannot bypass main-branch CI. Provider quotas still apply. Current Vercel connector access to the old project is forbidden and must not be used to claim a deployed URL.
- Explicit expected Pages path, only if the Pages deployment finishes READY: https://changxinjiresearch.github.io/BrainMirror/universal.html . Do not claim it exists or works merely from configuration.

### Tests and source

- scept-web/tests/u9_m3_contract.cjs: trusted plugin template and SDK validation; U6/U7 full audit and actual reruns; tamper, wrong model/input/definition, numeric drift, privacy, runtime performance bounds.
- scept-web/tests/u9_m3_browser.cjs: Chrome + genuine ONNX WASM, U9 SDK capability report, measured inference performance, actual downloaded U6 and U7 audit and real replays after re-import, rejects altered file and altered typed source.
- scept-web/tests/u9_security_check.cjs: static CSP/no-eval, audit policy and Pages workflow safeguards.
- Existing M1/M2, U8 3D/Token/volume, pretrained ViT/CNN, real MRI, previous v3.1 tests retained.

## Supported use cases

A researcher can write a trusted explanation/evaluator adapter in an independent JS module, declare a compatible frozen ONNX model, explicit target/output observations, interventions and evaluations in a JSON experiment plugin, run it using existing U6/U7, observe the outcomes through U8, export a content-checked U9 audit, and reproduce the experiment after supplying the original sources. This is reusable computational research infrastructure, not an MRI+Transformer-specific website.

## Known limitations

- No guarantee of arbitrary model architecture, preprocessing, model ops or off-browser model formats.
- No sandbox for unknown JavaScript plugins. No clinical validation, causal claims, verified atlas mapping, lesion segmentation, genuine counterfactuals or tested cross-hardware bitwise reproducibility.
- No background persistent scheduler or secure online patient-data repository.
- Derived records are not necessarily privacy-preserving; researchers remain responsible for data governance.
- Public deployment is a separate gate from engineering CI, and cannot be silently assumed.

## Release verdict

**M3 Engineering Complete** requires exact final GitHub PR head green on all new and old CI, merged main SHA verified, and the above deliverables present. **M3 Public Deployment Complete** additionally requires an actual successful Pages deployment plus URL live check. If not available, explicitly state the deployment gap and the single owner Settings -> Pages activation step. Neither engineering criterion is evidence of scientific or medical validity.
