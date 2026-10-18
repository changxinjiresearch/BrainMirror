# SCEPT V5 — Genuine Neural Counterfactual Observatory

## Engineering status, not clinical validation
V5 is a research/engineering preview, NOT a clinically validated lesion explanation. The existing U10 method and U11 M4_NOT_ACHIEVED remain unchanged. Passing CI proves genuine runtime data and protocol checks only.

The built-in ViT is natural-image pretrained, NOT an MRI tumor diagnostic model. The default MNI152 volume is a healthy atlas template with a numeric ROI intensity editor, NOT a tumor case. The B NIfTI and tumor mask are user supplied. Matching grid, sform and spacing cannot guarantee biologically plausible tumor removal.

## Implemented
- Real frozen-ONNX MRI A/B NDF: RMS delta of actual [1,197,192] middle-layer Token+channel arrays, with MAE and signed mean; 5 real layers x 197 Tokens. Clicking a Token in the NDF grid locates it in the existing actual 3D neural Token viewer. Visual logarithmic contrast is per-run only, not a calibrated inter-layer significance scale.
- Two mutually distinguishable inputs: numerical ROI intensity modification (explicitly NOT tumor removal); or research pair uploaded actual MRI A, counterfactual MRI B (up to 12 versions) plus binary lesion mask. Strict NIfTI dimension, voxel spacing, spatial sform, mask-slice overlap, nonzero in-mask edited pixels and excessive out-of-mask change checks.
- CRC from at least 3 actual distinct B input hashes, same A/model/probe shape: median pairwise Top-10% Jaccard, Spearman rank correlation and within-unit response MAD per layer. No independent-patient generalization inferred; no automatic pass verdict.
- ICM genuine activation patching: Python ONNX graph rewriter scripts/scept_build_icm_patch_model.py creates Where(mask,override,original) at a real static float32 intermediate producer. Browser imports it and performs no-op A and B forwards, then actual A-unit into B and B-unit into A patch forwards. No-op mismatch, missing probe, wrong source SHA or near-zero original output effect => not evaluable. Results are unbounded signed score-recovery ratios; not probabilities, independent node contributions or tumor proof.
- Output is traceable scientific summary JSON (raw patient voxels are not embedded) plus original v3 operator/Token observatory and v4 CEDS advanced panel.

## Formal numerical estimands
NDF(l,u) = sqrt(mean_j((H(l,u,j,B)-H(l,u,j,A))^2)).
CRC(l): K=ceil(10% of actual units), pairwise top-K Jaccard and midrank Spearman on genuine repeated-B response magnitude vectors; median over all B pairs. Undefined or all-zero response is reported not evaluable.
ICM(l,u) uses a=s(A), b=s(B), rAB=s(B with internal A unit), rBA=s(A with internal B unit). If |a-b| is above predeclared tolerance and patched no-op returns a and b within tolerance, then recovery=(rAB-b)/(a-b) and reverse=(a-rBA)/(a-b). Otherwise not evaluable.

## Zero-new-fee user workflow
1. Open v5.html; select the frozen ViT and original MRI A. In numerical mode, change the current ROI intensity and click Run NDF. Do not call this tumor removal.
2. In paired mode, replace the default MNI152 template with your true A NIfTI. Upload already validated B NIfTI and same-space tumor segmentation mask, select a slice intersecting the mask, click Compare actual A/B. Three distinct B versions unlock CRC.
3. Optional ICM research: click Download exact instrumented ViT ONNX (exports the SHA-256 checked model). Locally run on free CPU:
   pip install onnx
   python scripts/scept_build_icm_patch_model.py scept-v5-exact-instrumented-vit.onnx /vit/encoder/layer.6/attention/attention/value/MatMul_output_0 scept-v5-patched.onnx
   Upload the generated patched ONNX, select a NDF Token from the same layer, confirm the original model hash and logit index, then run true ICM. Current browser ICM profile is strict ViT [1,tokens,channels] with named logits and pixel_values.
4. Never interpret results as medically validated until appropriate MRI-task model, clinically sound inpainting, matched negative controls and held-out statistics have been performed.

## Outstanding scientific milestones
Actual registered tumor segmentation or diagnosis pretrained task model; proven faithful 3D lesion replacement and generated artifact controls; independent held-out patient comparisons, preregistration, uncertainty/CI; Grad-CAM/IG/Occlusion baselines; ICM random controls in trained medical models and independent generalization. These are not completed by this release. M4 remains NOT ACHIEVED.

## Tests
- node scept-web/tests/v5_science_contract.cjs — mathematical negative tests, not medical findings.
- node scept-web/tests/v5_fingerprint_browser.cjs — real public MNI template + genuinely pretrained ONNX five-layer internal tensors and 985-Token 3D linkage.
- node scept-web/tests/v5_paired_mri_browser.cjs — genuine public NIfTI, but a synthetic mask/B modification strictly for software integration testing, not a tumor experiment.
- python -m unittest discover -s scept-web/tests -p test_v5_icm_patch.py -v — actual ONNX graph patch runtime and exact no-op parity.
- Original v3, U10 and v4 CI regression must remain green.

Privacy: patient MRI arrays remain in local browser memory; exported provenance may expose filenames/ROI metadata. Never commit patient MRI, actual scans or private checkpoints to public GitHub.
