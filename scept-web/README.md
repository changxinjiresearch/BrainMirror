# SCEPT Observatory — Browser-First Research Preview

The static website lives in `scept-web/` and is **training-independent**. It runs compatible ONNX models on-device through ONNX Runtime Web (CPU WASM by default), compares baseline/perturbed input forward passes, displays raw tensor differences, and exports experiment audits.

## Scope and evidence boundaries

- A real ONNX file must be imported before any numeric experiment can run. No fabricated model inference or MRI is displayed.
- Internal activations and attention tensors must be exposed as ONNX outputs. The generic browser adapter cannot introspect every hidden tensor automatically.
- The current input adapter accepts a **single float32 input tensor** as JSON `{"shape":[1,4],"data":[0.1,0.2,0.3,0.4]}`. Multiple-input models and typed image/NIfTI preprocessing require model-specific adapters.
- Two forward passes share unchanged model weights. The first 64 output tensor elements are plotted and the first 200 displayed in a table; full outputs are included in the JSON export within safety limits.
- The numeric perturbation edits flattened tensor elements, NOT an anatomically QC-validated MRI intervention. No clinical claim or causal mechanism claim is made.
- `L1_REAL_NUMERIC` means an instrumented model's actual forward values were measured. Internal causal patching and independent scientific validation are NOT implemented.
- CPU inference is free but subject to browser memory, model compatibility, CDN availability, and latency.

## Running locally

From the repository root:

```bash
python -m http.server 8765 -d scept-web
```

Visit http://127.0.0.1:8765. No GPU, paid API, or server-side model processing is used.

## Making internal tensors observable without retraining

```bash
python -m pip install onnx
python scripts/expose_onnx_probes.py original.onnx ignored.onnx --list
python scripts/expose_onnx_probes.py original.onnx observed.onnx --probe selected_internal_tensor
```

This adds existing ONNX graph values as graph outputs, when shape information is available; it does **not** train or alter weights. For unavailable values, adjust the original framework's export graph.

## Acceptance gates

- [x] Static UI and versioned implementation source present
- [x] Import/metadata/paired-inference/trace-export code implemented
- [x] JavaScript syntax checked
- [ ] Browser automation / real model integration smoke PASS
- [ ] Two independent pretrained models with verified reference output PASS
- [ ] Model-specific MRI NIfTI adapter, affine-aware visual editing, anatomical QC PASS
- [ ] True internal node patching or ablation PASS
- [ ] I-JEPA integration — intentionally excluded

Do not label the system scientifically validated until these checks have independent evidence. The GitHub branch is authoritative for this preview; do not overwrite the existing production site until acceptance gates pass.
