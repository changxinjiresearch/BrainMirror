# SCEPT v3 — measured 3D activation response & synchronized A/B (2026-10-17)

## What is now represented
- Existing graph nodes remain real ONNX operator instances and edges remain exact named tensor producer-to-consumer dependencies. 3D layout is an artificial *display* geometry and remains fixed when MRI data change.
- Only the output-producing operators for five **actually exposed** pretrained ViT hidden tensor probes (layers 2, 4, 6, 8 and 10) can show measured activation-dependent changes. Non-observed operators remain neutral. No values are imputed to those nodes and lines are never animated as unverified signals.
- For each actual hidden tensor and each paired model inference, the visualizer computes:
  - (A_l = mean_i |h_l(x)_i|)
  - (B_l = mean_i |h_l(x')_i|)
  - (Δ_l = mean_i |h_l(x')_i - h_l(x)_i|)
  These numbers are calculated from the **actual before and after arrays**, not layer proxies, attention weights, static mocks, UI input sliders or made-up animation tracks.
- Measured producer node size and color are mapped continuously to the selected quantity. Quantitative reference scales for A/B (shared) and Δ are calibrated on the first actual run and **not silently re-calibrated** when the user changes the ROI. Further differences are distinguishable using `u/(u+reference_scale)` instead of clipping all larger values at one color. The user can explicitly press the recalibration control.
- If there is no MRI input perturbation then A and B have the exact same model input, Δ is zero, and the Δ-mode observed-node glyphs must become small and neutral. A and B can still exhibit nonzero activation: baseline activations must not be confused with activation *differences*.

## Controls and interaction
- **A/B 并排对比**: original and edited conditions side-by-side, with the same 3D camera and shared activation scale.
- **差异 Δ**: a single 3D viewport with color and glyph radius controlled by actual mean absolute hidden-tensor differences.
- **原始 A** / **修改后 B**: isolated state views for inspection.
- Orbit, zoom, node selection, per-layer drilldown continue to work. Clicking a measured probe producer now displays its exact tensor name, original mean absolute activation, edited mean absolute activation, observed mean |B−A|, and element count.
- Most nodes show no activation overlay because their output tensors are not currently exposed. The original static 3D network geometry and truthful edges are preserved.

## Checks in GitHub Actions
`scept-web/tests/v3_activation_compare.cjs` asserts browser-local real MRI and frozen pretrained ViT inference; A/B mode selection, five real observed tensor producers, exact independent numeric parity with raw saved outputs, unchanged shared scales on rerun, changed response under a different real ROI intensity, exactly zero numerical delta for no intervention, no fabricated glow at zero delta, and responsive layouts. All earlier tests for real MRI, genuine 3D edges, layer drilldown and MRI-specialist exploratory model also remain gates.

## Scientific limitations
- These are measurable intermediate **Value-path MatMul output activations**, not direct measured Attention probability matrices or verified decision-relevant causal effects.
- Visual glyph radius and color scale are display encodings; no claim of physical neuron size, learned connection strength or causal propagation.
- The default ViT was pretrained on natural images. MNI152 anatomical slices are out of distribution. This is an observability test, **not a medically validated model, tumor detection, Grad-CAM replacement or published explanation result**.
- Activations are sampled from five nodes, not all 626 operators. More nodes must be instrumented and independently validated before claiming full-network activity coverage.
- I-JEPA untouched. No paid inference infrastructure.
