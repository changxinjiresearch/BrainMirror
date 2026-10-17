# SCEPT ViT zero-upload browser acceptance — 2026-10-17

**Scope:** Engineering demonstration of training-independent pretrained Transformer import, inference and true hidden-state observation. Not an MRI or causal-explanation scientific validation.

## Reproducible inputs and model
- Public pretrained model: `onnx-community/vit-tiny-patch16-224-ONNX`, FP32 `onnx/model.onnx`, pinned Hugging Face revision `ebffd7b9c0c53b51bada44116abd8f01aed41be6`.
- Original model SHA-256: `32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3`.
- The browser validates source SHA-256 before adding output metadata for `/vit/encoder/layer.6/attention/attention/value/MatMul_output_0`; original weights are unchanged.
- Deterministic synthetic input: `Float32Array(1*3*224*224)` where `data[i]=((i*73)%256)/255`. This is not a clinically meaningful MRI or natural-image preprocessor.
- Perturbation: add `0.15` to tensor element index 0; run frozen model twice using ONNX Runtime Web CPU/WASM.
- Observed real hidden state: 37,824 float elements; 37,818 changed at comparison threshold 1e-7.

## Verification
- Automated live Chromium run: https://github.com/changxinjiresearch/BrainMirror/actions/runs/39001694347
- Three passing full-pipeline cases: local button click; actual Vercel preview URL button click; actual Vercel preview URL with auto-start query parameter **no user uploads and no click**.
- Latest deployed preview: https://brainmirror-observatory-5s7sb6ng5-jayson-s-projects6.vercel.app/?run=transformer
- Verified GitHub Actions step output: `PASS AUTO IMPORT/REAL VI-T: verified source weights, real WASM inference, 37824 hidden-layer activations, audit exported; changed 37818/37824`.
- User's browser downloads model weights only after intentionally opening the auto-test link or clicking demo button. All model inference and data are local in that browser; test records generated in CI are not injected into the user's unrelated browser profile.

## Scientific boundaries
- Model interface validated for a real pretrained ViT, not arbitrary model formats without export.
- Changed activations alone do not establish decision dependence or causality.
- Full biomedical MRI model support, preprocessing equivalence, interpretability comparison with Grad-CAM, and I-JEPA training are not covered.
- Public model licensing and endpoint availability remain external dependencies. A zero-paid-cloud-use setup is the target; model download and local CPU still use real resources.
