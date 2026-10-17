# SCEPT v3 — authentic brain MRI to frozen network observation (engineering preview)

Status: **Preview; biomedical model validation NOT complete**. This does not imply causal interpretation or Grad-CAM replacement. I-JEPA is not used and no model training is performed.

## Implemented

- **True MRI data**: MNI152 atlas-derived NIfTI-1 file from `niivue/niivue-demo-images` (public demo dataset, not a patient-specific scan). Triplanar voxel-coordinate views, rotatable volumetric point cloud and mouse-addressable three-dimensional spherical ROI; native affine available for provenance but anatomical axes are NOT automatically reoriented.
- **Perturbations**: additive intensity changes to only voxels within an actual 3D sphere; original `Float32Array` preserved. This does **not** model a physiologically plausible glioma or causal anatomical intervention.
- **Actual model**: pinned SHA256 `32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3`, public pretrained `onnx-community/vit-tiny-patch16-224-ONNX`, frozen weights.
- **Actual internal state**: existing layer 2/4/6/8/10 attention-value intermediate tensors added as observable graph outputs (minimum three required). All measured delta charts and network node colors derive strictly from those outputs. Unobserved blocks are grey.
- **Model input boundary**: a T1-like axial MRI atlas slice, nearest-neighbor resampled to 224×224, percentile clipped, repeated to 3 channels. This is out-of-distribution for the natural-image trained ViT and **does not match any MRI-specialist preprocessor**. Not a medical imaging inference.
- **Audit**: model and MRI SHA256, voxel grid, ROI center/radius/amplitude, sampling and intensity normalization, exact per-layer before/after arrays, statistics and CSV/JSON interface. Browsers keep data locally (public runtime JS + public model/MRI template fetched online).
- **Testing**: Chromium CI `scept-web/tests/v3_brain_mri.cjs` runs real public MRI, frozen model, paired inference, independent audit readback and responsive checks.

## MRI-specific research model upgrade (not an acceptance claim)

Candidate: `MONAI/brats_mri_segmentation` with published pre-trained BraTS 2018 four-modality glioma segmentation weights. Its original inference contract requires four registered MRI modalities, spatial pre-processing, verified checkpoint conversion, 3D model export and actual ChromeOS memory/latency qualification. **Not a current one-click browser module**; do not substitute a natural-image ViT prediction for segmentation evidence.

An alternative 2D model candidate `Udit013/brain-tumor-efficientnetb3` publishes ONNX weights with 256×256×3 T1-style MRI preprocessing but its authors disclose dataset train/test leakage. It may be investigated for **browser-engineering compatibility only**, not relied upon for claims of scientific generalization or clinical accuracy.

## Interpretation controls

- Network drawing nodes are groups of real hidden-layer tensor elements. Each observed layer is divided into eight deterministic contiguous partitions; each node color derives from the mean absolute measured difference in that partition.
- Block-to-block arrows represent a simplified sequence of frozen ViT architecture blocks, not a precisely weighted synaptic network or inferred causal attribution.
- Genuine model computation is asynchronous two-pass inference; presentation changes afterward, not fabricated during a forward pass.
- A user's file stays in the current browser, but the runtime CDN and open model template/model sources are fetched from public endpoints. Do not load identifiable patient data without institutional privacy approval.

## Acceptance

- [ ] Real MNI152 HTTP import and NIfTI decode validated in Chromium
- [ ] Actual frozen ViT middle-layer probes verified with real 3D ROI volume edit
- [ ] Second edit rerun generates a new audit with changed numeric results
- [ ] Responsive v3 triplanar and layer visualization on Chromebook
- [ ] Audit export using user browser click checked (if it fails, retain only in-memory audit)
- [ ] MRI-trained specialist ONNX model accepted with verified preprocessor and output parity
- [ ] Scientific claims supported by controls and comparison baselines

This is a deliberate stage-gated system: passing the first four items makes an authentic MRI **engineering observatory**, not a scientifically validated medical explainability system.
