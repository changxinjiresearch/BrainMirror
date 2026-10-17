# SCEPT Observatory v2 — engineering acceptance handoff (2026-10-17)

## Preview and immutable evidence

- Candidate preview URL: https://brainmirror-observatory-g085g6ecm-jayson-s-projects6.vercel.app
- Vercel deployment: `dpl_ME98sBK14Cq2MvkbiFdjgZty6iiE`, READY. This does **not** prove public accessibility or user browser functionality.
- Git branch: `phase/scept-browser-observatory-v2`
- Draft pull request: https://github.com/changxinjiresearch/BrainMirror/pull/18
- Verified GitHub Actions acceptance run: https://github.com/changxinjiresearch/BrainMirror/actions/runs/38948169637
- Additional browser studio run: https://github.com/changxinjiresearch/BrainMirror/actions/runs/38948169682
- Real pretrained architecture tests used:
  - Vision Transformer, FP32 23.1 MB, SHA256 `32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3`
  - SqueezeNet 1.1 CNN, SHA256 `1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088`
- Both models executed **real Chrome/WASM forward passes** on synthetic numeric tensors in the acceptance CI. These are engineering compatibility tests, not medically meaningful inferences or validation of activation-based explanations.

## Manual acceptance — operator instructions

1. Open preview on Chromebook Chrome. Ensure the page presents the SCEPT workspace, a model import card, the two engineering ONNX controls, and NIfTI editor.
2. Choose **Try real ONNX computation**, click **Run baseline + intervention**. Confirm measured numeric output differences in the right panel, a populated operator graph and 3D graph, no red JavaScript error.
3. Choose **Try internal ablation**, click Run. Ensure output says **INTERNAL ABLATION**; verify a downstream output changes. This uses a fixed-weight engineering model, not a pretrained biomedical model.
4. Click **Export audit** and **Export CSV**, check downloaded files and hashes/values.
5. Open **Experiment Records** and confirm your runs are recorded locally. Reload; history summaries should persist. (Do not store raw patient data on shared devices.)
6. Download a **pretrained FP32 ONNX** model from the provided publisher link, import, then select **Generate reproducible numeric input** and run. Numerical test tensor is synthetic and not an MRI or meaningful picture.
7. Import authorized/de-identified `.nii` or `.nii.gz`; inspect three planes and 3D voxel projection, edit ROI with intensity change, undo, and export edited `.nii`. Confirm the warning describing voxel-native axes, not fully normalized anatomy.
8. Try a browser with incompatible ONNX model: an explicit informative error is expected (no silent fabricated output).
9. Repeat on mobile to check responsive layout. Monitor memory usage on large 3D MRI; unsupported/full 3D clinical models may hit browser limits.

## Honesty and engineering vs scientific limitations

- Zero incremental paid cloud inference is an architectural aim; browser CPU requires the device's local compute and the static deployment/CDN have quota & usage limits.
- Current UI supports compatible ONNX graphs and tested two dissimilar **pretrained** image models; does not prove all model classes/ops work.
- Dynamic middle-layer inspection requires offline ONNX graph export with extra probe outputs, or model-specific adapters. For trained ViT/CNN, only published forward output is confirmed in browser; **internal attention probing on those pretrained models has not been separately validated**.
- Current 3D network graph maps **ONNX operators and directed tensor dependencies**. It is a visual graph layout and does not represent actual biological neurons or every single weight coefficient.
- NIfTI import works for constrained NIfTI-1 3D images; spatial axes are shown in voxel coordinates. Geometric edits only carry approximate local Jacobian QA, **not** robust anatomy-valid deformation claims.
- Clinical privacy, third-party CDN exposure, strict offline assets / CSP hardening and independent patient-level replicability do **not** yet have production-grade approvals.
- True clinical/high-resolution MRI ONNX inference with matching original model preprocessing has not been verified.
- I-JEPA training and its specific adapter were explicitly not started.
- **Do not merge PR to main / promote preview until user acceptance plus the scientific/medical accuracy requirements appropriate to release scope.**
