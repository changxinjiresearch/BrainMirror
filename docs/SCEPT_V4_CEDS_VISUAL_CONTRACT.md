# SCEPT Observatory v4 — CEDS Visual Contract

## Actual new functionality
- v4 preserves authentic NIfTI MRI editing, the real ONNX operator graph, five genuine ViT probe layers and the interactive 985-Token 3D view.
- After real MRI A/B observation the v4 default becomes real measured RMS Delta rather than paired similar-looking absolute activations. Logarithmic contrast affects ONLY visibility, not values.
- Principal new scientific visual: CEDS (Controlled Evidence Dependence Signature), with U10 paired P vs Q response curves, real nuisance-seed ranges and unchanged seven-gate verdict.
- Single-action U10/U7 trusted-plugin run: 3 doses x 3 seeded nuisances x 3 genuine A/B/C frozen ONNX forwards, plus verified baseline inference.
- Actual captured U10 internal output tensor summaries P/Q are presented in a separately labeled drilldown; these are not the original MRI A/B 3D experiment.

## Frozen U10 visual quantities
Let a be the original frozen model target output; b(q,r) and c(q,r) are target P and matched-control Q results at dose q and shared nuisance seed r; S0=1 in the reported raw target logit units.

Target line = median over r of |b(q,r)-a| / S0.
Control line = median over r of |c(q,r)-a| / S0.
Per-seed specificity = (|b-a|-|c-a|)/S0.
Per-seed controlled magnitude = |b-c|/S0 (not in general the distance of aggregate lines).

The displayed doses are ONLY the three actually measured q values 0, .5, 1. Whiskers are experimental min/max seed ranges; NOT confidence intervals. B and C share a nuisance, and the unchanged A does not. Therefore at q=0 the two tracks can be nonzero relative to A even though their B-C controlled difference must be zero.

All seven U10 criteria (specificity, controlled effect, replicate consistency, nuisance spread, direction, graded response and zero-dose null) are calculated by the existing unchanged U10 suite. The frontend cannot override the scientific verdict and must display U11 M4_NOT_ACHIEVED transparently.

## ViT auto ROI profile and limits
This initial profile handles the true 3x224x224 frozen natural-image ViT input derived from current MRI slice. Choose a 4x4x3 model-input target P within actually changed pixels of the user-edited MRI ROI, with maximal affected-pixel count. Choose disjoint unmodified 4x4x3 numeric control Q by closest mean/std, with tolerance .2 and noise margin. Verify flat indices, source identities, SHA-256, ROI lengths, no overlap and exact typed input. Fail closed when the MRI ROI is out of plane or no matched control exists.

Important: these 4x4x3 blocks are a measured subset of the model input, NOT a full 3D lesion mask. Mean/std matching is NOT anatomy-aware, biological, causal or spatial exchangeability. The natural-image ViT is out of distribution on MRI; selected maximum logit index is NOT a diagnostic class. This version is a real numerical ONNX study but not clinical explanation.

The MRI CNN is deliberately not automatically connected to U10 in v4; it currently lacks a verified full hidden observation and anatomy-aware control profile. Universal Core remains available for explicit researcher-specified model inputs/outputs/protocols. No silent universal compatibility claim is made.

## Acceptance
1. Old MRI, 3D network navigation, 985 real Tokens, MRI edits and audit exports remain operational.
2. CEDS starts empty; 9 completed authentic U10 triplet jobs and valid evidence are required before any plotted science.
3. Full U10 seven-gate verdict is unchanged; unsuccessful and not evaluable results display explicitly.
4. Real full-dose U10 internal P/Q data are separately labeled from v3 ordinary A/B.
5. U11 negative comparison and M4 not achieved are preserved.
6. Zero new paid model inference, GPUs, or web hosting.

Automated acceptance:
- node scept-web/tests/v4_ceds_contract.cjs — synthetic UI contract, not empirical result.
- node scept-web/tests/v4_ceds_browser.cjs — genuine public MRI and frozen ONNX ViT with actual P/Q A/B/C.
- Existing v3 and U10 browser and quantitative regression suite.
