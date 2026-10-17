# SCEPT v3.1 — fine-grained real MRI / ViT observation acceptance

**Engineering research preview, not a validated medical explanation or clinical diagnostic system.** I-JEPA was neither trained nor integrated. No GPU training or paid inference is required.

## Five implemented upgrades

1. **Strict tensor-to-Token/channel adapter** (`scept_tensor_adapter.js`): validate actual ONNX output shape `[1,197,192]`; parse 197 observed Tokens (including non-spatial CLS) and 192 actual channels with per-Token A, B, mean absolute difference, RMS difference, signed mean and changed-element count. No invented activations. Independent arithmetic from full captured before/after tensors is the gold-standard verification.
2. **Actual 3D Token observatory** (`scept_activity3d.js`): five observed ViT layers times 197 Tokens = 985 data-backed glyphs. Orbit, zoom, all-layer / per-layer scope, click to inspect all 192 actual channel differences. Three-dimensional positions are *display geometry*, not a learned brain or synapse map. Crucially, there are **no unmeasured inter-Token edges**. The original independent real ONNX operator/edge viewer remains accessible.
3. **Synchronized A/B/Δ**: same camera, same pairwise values and shared quantitative reference calibration; default Token-view Δ uses RMS of *channelwise* differences rather than the change of the mean absolute activation. Zero input perturbation must produce precisely zero Token deltas; subsequent experiments do not silently rescale the reference colors. Positive/negative per-channel differences appear separately.
4. **Genuine MRI-to-Patch mapping** (`scept_patch_mapper.js`): every baseline/edited input is derived from `SCEPT_MRI.makeSlice` (actual 3-channel 224×224 tensor). Exactly 196 model input patches, each 16×16. Each patched pixel and its actual changed value is checked before a Patch is highlighted. Click a Patch to inspect Token index `patch+1` (Token 0 is CLS). Raw image coordinate conventions are documented; MRI display slice and 3D ROI coordinates are now separately maintained so moving the displayed slice does not silently reposition the 3D ROI.
5. **Additional *actual* ONNX probes** (`v3_engine.js`): expose further observation candidates from verified graph `ValueInfoProto`: Patch projection Conv, Query and Key MatMul pre-bias projections at ViT layer 6, actual Attention Softmax and residual Add. The engine only records these if the public model graph explicitly exposes the named tensor; actual ONNX runtime output arrays/shapes, per-element paired differences and independent numerical summaries are included in JSON export and optional scientific details. Distinct tensor types are never all relabeled 'neurons' or pooled into one incomparable activity score.

## Scientific and privacy boundaries

- Public MNI152 atlas MRI plus frozen natural-image pretrained ViT is **out-of-domain**; the per-Token and Attention outputs are bona fide computations but **not medical interpretation**. Model training/validation for clinical MRI, graded intervention controls, output-dependence ablations and Grad-CAM comparisons remain future scientific stages.
- Current 5 Value-path tensors represent 5 actual network outputs, not all 626 operators; additional probes increase sparse direct observation coverage but do not authorize filling the rest of the network with invented values.
- A patch is an *input* grid cell. A downstream Token may change from the full attention/context pathway even if its source patch was untouched. Patch-to-Token positional alignment must not be misinterpreted as proof that information flowed only along that spatial link.
- The raw public model and public MRI template are downloaded to the browser via their publishers/CDN. A locally selected MRI file remains in browser memory; the UI itself requires no paid backend.
- Baseline and edited arrays in the audit are real runtime outputs, with model hash and MRI hash. Large JSON exports may consume browser memory.
- A/B comparisons and signed channel deltas are distinct quantities: A/B are mean *absolute* activation, Token Δ is RMS of elementwise differences, and the per-channel chart shows signed `B-A`.

## Browser acceptance

`scept-web/tests/v31_token_mri.cjs` checks real public 3D MRI NIfTI, 196 Patch mapping, decoupled MRI ROI vs slice, shape `[1,197,192]`, 985 rendered Tokens, independent raw-number parity, orbit and zoom, per-layer 197 Token drilldown, exact Patch→Token coupling, 192-channel selection, changed ROI re-run, fixed reference scales, zero-delta falsification and responsive viewport widths.

`scept-web/tests/v31_extra_outputs.cjs` checks independently recomputed statistics and actual ONNX shapes/values of additional intermediate outputs.

The original v3 real MRI, ViT and public MRI-trained EfficientNet smoke tests, real ONNX operator network navigation and A/B 3D regression checks remain part of CI.

## Limitations to address after v3.1

- This upgrade does not make the MRI-specific CNN's internal activations accessible. Its displayed classifier output remains unsuitable as clinical evidence.
- Token grid nodes are real tensor-coordinate values displayed in 3D, not individually executed operators.
- The current raw Q/K MatMul probes are pre-bias *linear projections*; the actual attention-score pathway should be analyzed separately from the attention Softmax output.
- Larger models may require tensor streaming and GPU/WebGPU acceleration to maintain Chromebook responsiveness; zero-cost CPU feasibility must be measured for each newly adapted model.
