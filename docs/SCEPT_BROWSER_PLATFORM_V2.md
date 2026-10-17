# SCEPT v2 — Training-Independent Browser Observatory

Status: **ENGINEERING PREVIEW / SCIENTIFIC VALIDATION NOT COMPLETE**. I-JEPA not started.

## Product contract

SCEPT is a **model-observation and controlled-perturbation layer**, not a training framework. A pre-trained checkpoint can be made observable without retraining when exported as a supported ONNX graph with required intermediate tensors.

### Modules delivered as code

| Capability | Implementation | Important bound |
|---|---|---|
| Local model import | `scept-web/app.js` | ONNX, CPU float32 inputs, browser memory limit |
| Genuine forward execution | ONNX Runtime Web / WASM, WebGPU optional | Requires compatible model and runtime assets |
| Real ONNX operator topology | `scept-web/onnx_inspector.js` | Structural edges only; not quantified weights |
| Observable intermediate tensors | ONNX graph output probes | Requires instrumentation/export |
| Paired input perturbation | `app.js` | One or multiple fixed-shape float32 feeds |
| True internal activation masking | `scept_mask` instrumentation | Requires adapted graph, same unmodified checkpoint weights |
| NIfTI MRI input and display | `mri.js` | NIfTI-1 only, 3D / singleton 4D |
| MRI 3-axis slices and 3D point projection | `mri.js` | Voxel-native axes, not full anatomical reorientation |
| ROI intensity and geometric editing | `mri.js` | **EXPLORATORY ONLY**; affine retained, geometric QC approximate |
| Numeric trace/export | JSON, CSV, hashes | No external scientific replication yet |
| Two actual ONNX graph reference fixtures | `fixtures.js` | Hand-selected fixed weights; **NOT pretrained** |
| ONNX probe/ablation exporter | `scripts/expose_onnx_probes.py` | Requires onnx Python package |
| Checkpoint-to-ONNX converter | `scripts/export_pytorch_to_onnx.py` | Requires trusted architecture code & compatible export |
| Automated QA | `scept-web/tests` and GitHub Actions workflow | Await CI and real browser confirmation |

## Safety and evidence rules

1. A Vercel READY status **does not certify functional acceptance**.
2. Handcrafted ONNX engineering fixtures cannot be presented as medical, pretrained or scientific validation.
3. Real model output only: no animated or fabricated activation values. All graph edges from the inspector represent computational tensor dependence, not anatomical or causal connections.
4. User MRI and model files are accessed by local browser JavaScript. The runtime is currently delivered via a public CDN, which is an additional third-party script/privacy consideration. Use only de-identified/licensed images; a strict offline/no-third-party security build requires self-hosting approved runtime assets and CSP hardening.
5. Geometry-only ROI edit uses an exploratory voxel-space displacement; its Jacobian checks are limited. **Do not** infer tissue plausibility, clinical validity, or biomarker changes from it.
6. Model-specific preprocessing (spacing, affine reorientation, channel order, intensity scaling) must be verified against each pretrained model card. The UI does not silently guess these.
7. Internal ablation uses true graph activation masking only when a static-shape float32 probe has been instrumented. It is a local causal effect on the specified model computation; it does not prove a higher-level mechanism.
8. Avoid uploading model checkpoints, token secrets, or raw subject data to public GitHub.
9. Observability is portable; **arbitrary PyTorch/TF weights cannot simply be interpreted by the browser**. Export to supported ONNX format first, with custom adapters for unsupported operators.
10. I-JEPA training and adapter development are deferred.

## Preparation for true pretrained model test

Export or obtain a trusted pretrained ONNX model and model-specific preprocessor; open the SCEPT website, choose model, run a baseline, perturb supported input, verify per-layer exported probes, export audit.

To expose intermediate ONNX values without training:

```shell
pip install onnx
python scripts/expose_onnx_probes.py model.onnx checked.onnx --list
python scripts/expose_onnx_probes.py model.onnx observed.onnx --probe tensor_name
python scripts/expose_onnx_probes.py model.onnx ablatable.onnx --ablate tensor_name
```

To convert a pretrained PyTorch state_dict:

```shell
pip install torch onnx
python scripts/export_pytorch_to_onnx.py --factory mymodule:build_model \
  --checkpoint model.pt --shape 1,3,224,224 --output exported.onnx
```

No new training, but model conversion and graph export use local CPU computation.

## Final acceptance gates

| Gate | Must verify | Current evidence |
|---|---|---|
| A | GitHub branch matches Vercel deployment | Explicit preview file deployment, branch source |
| B | Real ONNX numeric reference fixture | Committed tests; not yet verified in browser |
| C | Chrome real WASM inference / JSON export | E2E test authored; independent run pending |
| D | Internal mask changes downstream output without changing model parameters | Actual ONNX graph fixture and Python test authored; independent run pending |
| E | NIfTI load, slice view, edit, export and import bridge | Pure parser smoke executed; browser end-to-end pending |
| F | Two **pretrained** models, distinct architectures, correct preprocessing / output parity | NOT VERIFIED |
| G | High-resolution 3D MRI model and medically sound geometry QC | NOT COMPLETE |
| H | General internal graph operator numerical edge validation | NOT COMPLETE (structural graph only) |
| I | Privacy / security audit on third-party runtime, CSP and large real clinical data | NOT COMPLETE |
| J | Independent scientific evaluation and reproducibility | NOT COMPLETE |

Only gate-independent PASS may be reported as passed. Production promotion must follow integration testing; keep legacy production untouched until then.
