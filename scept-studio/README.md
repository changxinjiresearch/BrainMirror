# SCEPT Browser Studio — model-independent inference and observation

Status: research engineering candidate, **not independently E2E verified**. I-JEPA is outside this release.

## What changed

- Self-contained browser UI in \`scept-studio/\`, independent of any model training pipeline.
- Local ONNX Runtime Web WASM execution; experimental optional WebGPU. No paid inference endpoint.
- Two official pretrained ONNX model zoo demos (MNIST-8 and SqueezeNet 1.1) plus a pretrained ViT-base Transformer (88 MB, ImageNet), each verified against its published SHA-256 when downloaded.
- Local .onnx file import, a \`scept.model.v1\` manifest, .nii/.nii.gz NIfTI-1 basic decoding, JPG/PNG image import and tensor JSON import.
- Baseline and intervention are two genuine ONNX Runtime forward passes on the same frozen weights.
- Visualizes **only tensors actually exported as model outputs**. Model-hidden activations not exposed by the model are *not* magically observable.
- Records hashes, input/edited signatures, output hashes, numeric deltas, in-browser timings and provenance. IndexedDB and JSON/CSV export.
- \`scripts/scept_instrument_onnx.py\` can expose intermediate tensors and add a mask input to the computation graph without training or altering pretrained parameter values.

## Run locally

A static HTTP server is sufficient:

\`\`\`bash
cd scept-studio
python -m http.server 8000
\`\`\`

Navigate to \`http://localhost:8000\` in Chrome. The first model load requires network access to the ONNX Runtime CDN and, for demos, Hugging Face. Locally uploaded files stay on-device. No browser app server processes MRI or weights.

## Model adaptation

To inspect arbitrary intermediate tensors in a pretrained ONNX graph, explicitly mark them as graph outputs:

\`\`\`bash
pip install onnx
python scripts/scept_instrument_onnx.py example.onnx -o example.scept.onnx --list
python scripts/scept_instrument_onnx.py example.onnx -o example.scept.onnx \
  --expose "tensor_name_a,tensor_name_b" --mask-tensor "tensor_name_a"
\`\`\`

This creates \`example.scept.onnx\` and a \`example.scept.scept.json\` manifest.

Upload both files to SCEPT, and, if the graph contains a mask input, select it in the paired-experiment controls. The baseline uses an all-ones mask, the intervention sets one mask index to zero; the **source input is held fixed** for this type of intervention.

The mask target must be a float32 internal tensor with static dimensions and downstream consumers. The converter rejects unsupported cases instead of claiming intervention success. Attention scores are not automatically causal edges.

### Runtime constraints

- Multiple arbitrary inputs, dynamic dimensions, custom ONNX operators, float16 pre/post-processing, arbitrary MRI orientation/affine transforms and nonstandard tokenizers require a dedicated adapter. Do not claim universal plug-and-play for unsupported checkpoints.
- Browser memory budget caps input length at 35 million tensor elements. Per-output exports cap raw values at 2 million elements; the full-output SHA remains recorded and the truncation flag is explicit.
- NIfTI viewer previews raw storage orientation and does **not** constitute clinically validated anatomical editing. Spatial non-rigid edits require affine-aware QC, Jacobian tests and medical review.
- Browser inference requires model size and ops compatible with local CPU/WASM or supported WebGPU.
- Downloaded pretrained model data is fetched by the browser from external hosts; it is **not uploaded** to a proprietary SCEPT inference service.
- No user accounts or medical-data server persistence. This is a single-device research workstation; don't market it as a clinical product.

## Engineering validation gates

1. Static JS syntax and DOM contract must pass.
2. Load two pretrained models on a tested browser and compare their outputs to a reference ORT run.
3. Run repeat baseline and zero-perturbation input: output should match within numeric tolerance.
4. Run edited input: output hashes and numerical differences must be computed from real session outputs.
5. Instrument a simple ONNX graph, compare masked/unmasked output with trusted reference calculations.
6. Verify JSON and CSV audit contents, hashes, export truncation flags and local-only data processing.
7. Verify ChromeOS/Chrome mobile responsive behavior and screen-reader keyboard actions.
8. Cross-check data protection and deployment origin; never claim pass before recorded evidence.

## Release and source of truth

The previous web v1.0 trace replay and older P0 PyTorch model remain in the repository as legacy. New work is maintained in \`scept-studio/\` and may be deployed as a static site. **Update and protect this repository as the canonical code source**, then use preview deployment/CI and deliberate production promotion after E2E proof.

See \`docs/02_详细开发实施计划.md\` for the historical training-first plan. That plan is *not* the implementation roadmap for SCEPT Browser Studio.

## Scientific claims

All experiments here start as exploratory \`L1_NUMERIC\` evidence (actual numeric forward traces). Model-internal mask effects are computational interventions **only within the stated scope** and do not automatically prove semantic explanations. Trained model ≠ brain simulator; 3D visual layout ≠ anatomical mapping.
