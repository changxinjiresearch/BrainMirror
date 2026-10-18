# SCEPT V6.2–V6.3: genuine single-run NAF observatory and guarded calibration

## Scope
V6.2 upgrades V6.1 independently measured run-records into actual state/RMS, measured raw channel pattern and verified-same-coordinate **same-forward** computation update when both model tensors exist. V6.3 adds fixed raw display scales, provenance-bound empirical reference distributions, and a separate real CNN model adapter. A standalone v6.html preserves V5 and the U10 research console.

Not a new causal explanation algorithm. No assertion of medical diagnosis or U11 M4 success.

## Scientific observables
State: per registered tensor unit u, RMS of all genuine unit feature scalars. Pattern: real full feature channel vector from the retained bounded runtime tensor. The UI may subsample already measured values only for plotting (state exact for all units), not fill in unobserved values.
Update: \`sqrt(mean_j((h_out - h_in)^2))\` from the **same model forward**, with explicit same-residual-stream coordinate declaration, equal runtime shape/dtype and demonstrated actual graph dependency path. An input or output tensor alone, or merely same shape, is NOT sufficient: no valid pair -> not_evaluable. Never represents FLOPs, attention importance or input A/B effect.
Three-dimensional view plots **genuinely measured units**, not biological neurons. 3D positions are a layout, not geometric location, topology or a causal network. No fabricated connecting edges.

## Model profiles
ViT pinned source SHA is verified upstream by original v3 engine, with true exposed [1,197,192] attention-value projections at layer 2/4/6/8/10. For V6 same-forward update, instrumented residual output tensors from block 5 and 6 are exposed when ONNX provides their genuine signatures, graph path is checked and a pinned-model same-residual-stream declaration is required. Natural-image ViT is outside biomedical domain.
Second separate **actual pretrained** EfficientNetB3 CNN pinned to the source SHA; strict ONNX inspector and bounded real annotated 4D intermediate tensors are promoted only when genuinely present, otherwise solely genuine output features are shown with no claimed internal convolutional activation. Native channel axis follows actual known input/output tensor layout, never uses fake Transformer Token naming. Model author discloses train/test leakage; labels/scores are NOT clinically valid.

## Calibration
- Default scientific rendering: RAW RMS with a declared, editable, **fixed** upper bound (default 2) that does not dynamically change per single-run image; values beyond it saturate with the source numeric values still available.
- Reference view requires at least 5 actual distinct model input fingerprints from a single exact model SHA, preprocessing SHA, identical named probe shapes and profile. The candidate's own run is excluded; duplicate reference input hashes rejected. Score is the empirical midrank relative to source reference values. It is not a p-value, anomaly probability or medical reference. An example reference set from multiple slices of the same MNI brain has only *within-volume descriptive* interpretation; samples are not independent people.
- Zero invented clinical normal cohort or normed reference. No reference -> grey/not evaluable, the UI defaults to raw fixed scale.

## Runtime + limitations
The V6 page uses the genuine V6.1 captureOnce and actual ORT WASM; one true session.run per NAF. Multiple user-triggered independent runs create independent Run Records. Last detailed raw tensor is held in bounded memory, older runs retain only hashed summaries; old detailed values require real rerun. User MRI voxels are not serialized to exported default JSON.
The temporary V5/old controls remain in the static page's DOM for stable MRI and model infrastructure but are hidden in V6; v5.html remains unchanged. The public root homepage is not automatically redirected until full review of this isolated V6 observatory.

## Verifications
- \`node scept-web/tests/v6_23_naf_contract.cjs\`: real numeric state/update, negative missing-pair cases, signed native CNN features, rank-calibration exactness and incompatible inputs/model gate.
- \`node scept-web/tests/v6_23_naf_browser.cjs\`: real public MRI + frozen ViT one-forward NAF; five layer states with 985 true units; new independent run after changing selected slice; details and no-reference gate; actual frozen EfficientNetB3 ONNX with genuine CNN profile; no clinical claim.
- \`node scept-web/tests/v5_fingerprint_browser.cjs\`: confirm original V5 A/B retains normal behavior.
- Existing V6.0–6.1, U4 and U10 tests must pass.

## Future
V6.4 will migrate NDF/CRC/ICM/U10 to unified Run Record comparisons. Additional genuine architectures, calibrated independent healthy reference datasets, semantic feature validation and causal importance tests are future work. V6.2–6.3 is **software and numeric observations**, not original XAI validation.
