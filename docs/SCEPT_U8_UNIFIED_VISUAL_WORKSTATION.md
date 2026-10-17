# SCEPT U8 — Unified Data-Driven Visual Research Workstation

## Delivery and scientific scope

U8 replaces the separate fixed-shape assumptions **in the Universal Core web workbench** with a shared read-only visualization-data contract. The legacy v3.1 MRI/ViT 197-token × 192-channel viewer is **not deleted or rewritten**; it remains available for comparison and compatibility at `v3.html`. The universal `universal.html` page now hosts the new U8 visual panels and is updated automatically when U5, U6 or U7 produce genuine ONNX results.

- **3D operator canvas:** exact model graph operators and *actual producer→consumer tensor dependency edges* from the inspected ONNX graph; a deterministic **capped subset** for huge models, with observed producing nodes retained. Observed A/B/Δ colors are derived from actual U4 runtime tensor results; unobserved graph nodes remain gray. Camera rotation/zoom only affects the illustration layout. No invented neuron locations or edges.
- **3D token/channel/feature canvas:** axis sizes from the true U4 runtime shape and explicit architecture profile; no fixed 197/192 assumption outside the retained v3.1 dedicated adapter. For each axis unit, U4 calculates **mean absolute actual A / mean absolute actual B / mean absolute A-to-B change / RMS difference** directly over real typed ONNX outputs. Note A/B mean absolute magnitude differs from raw signed activation; do not read visual colors as tissue saliency.
- **Model input / MRI-volume section:** paired actual A/B typed model input arrays, captured in **ephemeral browser memory only** after ONNX inference, with an explicit required NCHW/NHWC/NCDHW/NDHWC layout. For 3D tensors, three orthogonal **native tensor-axis** slices can be examined. Optional deterministic nearest-neighbor sampling to <=72 elements per spatial dimension limits memory; the sampling is disclosed with full/sampled dimensions. Identical grayscale A/B scale, separate signed Δ scale with a zero-change neutral color. **No MRI identity, anatomical orientation, segmentation, registration, clinical diagnosis or causal inference is inferred** from tensor dimension rank or user-specified axes.
- **Source/experiment selector:** switch among U5 A/B, U6 A/B/C runs and U7 evidence tasks. Only **completed and genuine** inference tasks can drive measurable scene values. Each scene carries model SHA-256, originating task, observation provenance and real exposed ONNX output names.

## Interface contract and integration

Files:
- `scept-web/core/scept_visual_data.js`: browser/CommonJS U8 schema `scept.u8.scene.v1`; strict real-runtime evidence checks, graph sampling with actual ONNX edges, dynamic token/channel axis interpretation, `captureInput()` and three `plane()` orientations for typed model inputs.
- `scept-web/scept_visual_studio.js`: one cohesive UI renderer: genuine 3D perspective operator dependency graph, dynamic 3D tensor unit layouts and real paired model input slices A/B/Δ. Drag and zoom are presentation only.
- `scept-web/core/scept_observation.js`: adds actual baseline and edited axis/total mean absolute magnitudes and new **explicit** 5D CNN volume profiles. Previous U4 fields retain their semantics.
- `scept-web/core/scept_scheduler.js`: optional `onTaskVisual({taskId,baseline,edited,...})` callback **only after success** and genuine ORT computations; visualization failure never changes scientific result status or exports image tensors in U6 JSON/CSV.
- `scept-web/core/scept_xai.js`: forwards this optional ephemeral callback for U7 task execution.
- `scept-web/universal.html` / `universal.js` / `universal.css`: top-level U8 station, automatic synchronization from the real U5/U6/U7 flows. Existing older controls retained.

The graph's sparse rendering and node coordinate positions are for screen navigation, not a claim of all model units or anatomical localization. Dynamic Token shapes are taken from runtime results, **not inferred patch-to-lesion alignment**. MRI-plane claim is restricted to *actual tensor samples* and input-layout metadata; for real anatomical MRI interpretation, voxel affine, preprocessing and verified alignment must be added by an upstream medical data adapter.

## Reproducibility and privacy

The U8 volume buffers are private **temporary page-memory samples** (at most a small number of bounded captures) and are not written to experimental JSON/CSV. Closing the browser loses the input view; existing source inputs and model are needed for replay. Model SHA/Task IDs help identify the associated inference. For large volumetric tensors, downsampled previews are explicitly indicated and must not be used as full-resolution quantitative lesion segmentations.

## Acceptance

1. U8 unit contract checks genuine runtime fields, explicit axes and 5×7 ViT-style tokens, dynamic CNN channels, honest graph edges, provenance failure closure, true A/B/Δ from 2D inputs, native 3D volume 3 orthogonal views and explicit shape/layout protections.
2. Actual browser Chrome + Web/WASM ONNX: untrained 2D CNN hidden node and input A/B/Δ, automatic U5→U6→U7 scene changes, no fabricated unmeasured node colors, plus **different 3D convolutional model** with real 5D input and native 3 orthogonal tensor-space planes.
3. Previous U0–U7, pretrained ViT, MRI 3D v3.1 and full regression suite remain required.

Fixtures are **deterministic untrained engineering models**, not biomedical validation. Do not claim MRI-trained 3D model acceptance from them; the dedicated v3.x real MRI-trained model tests are separate.

**Production deployment remains a separate operation.** Engineering acceptance does not establish that an existing public Vercel URL serves U8.
