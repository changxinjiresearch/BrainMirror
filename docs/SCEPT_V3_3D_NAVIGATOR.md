# SCEPT v3 — 3D ONNX operator and real dependency navigator

## Scope of this stage

Only the right-side **actual computational architecture viewer** is upgraded. The original authentic MRI reader, frozen ViT/CNN inference, ROI input intervention and scientific audit logic are unchanged.

### Verified structure versus presentation

- Nodes are real ONNX `NodeProto` computational operators taken from `SCEPT_INSPECT_ONNX`. A node may be `MatMul`, `Add`, `Reshape` etc.; **not an individual biological neuron**.
- Drawn edges are **exact producer→consumer intermediate tensor dependencies** parsed from ONNX graph input/output names. No inferred synaptic links, synthetic layer connectors, or parameter-value strengths are added.
- Three-dimensional positions are deterministic spatial **layout coordinates** for inspection, not real anatomical locations, embedding coordinates or physiological neuron positions.
- For normal ViT overview, an evenly sampled subset of actual operators is drawn (with exact exposed probe producers always preserved); the UI states displayed vs total counts. A drilled-down ViT layer shows all actual operators in that layer and direct edges where both endpoints belong to that layer.
- In a model without ViT layer-name metadata, 12 neutral order bands are used for display, explicitly labeled as structural **operator-order sections**, not claimed biological or training layers.
- Operator color is categorical by **operator type**, except when a truly exposed probe's own producing node has a measured delta. Other nodes are not colored as if activation data were available.

### Interaction

- Drag the 3D canvas to orbit (yaw / pitch)
- Wheel to zoom; **＋/－** buttons also work; view distance is bounded
- Click a visible operator: inspect exact `op_type`, `name`, original named `inputs`, `outputs`, and actual incoming/outgoing dependency counts
- Select **第 6 层** in the existing layer selector (or click an operator in a layer), then click **进入选中层**
- **结构总览** or Escape to return; **重置视角** restores the camera without rerunning inference
- When WebGL2 is unavailable (Chrome CI software runners, blocked GPUs), a 2D canvas rasterizer still displays and interacts with exactly the **same rotating 3D perspective graph**. Its software rendering mode is stated explicitly.

### Regression evidence

`scept-web/tests/v3_network3d.cjs` verifies: actual node/edge counts, that every visible edge is present in ONNX source, orbit camera changes, wheel zoom, entering layer 6, selecting a real operator and displaying tensor names, return to overview, and conservation of actual frozen MRI→ViT paired inference. All visual graph operations are read-only and do not change model weights or input tensor values.

### Scientific boundaries

A 3D topology visualization is **not** SCEPT's complete explanation validation. Showing connected operators and changes in selected probes does not establish causal dependence of a medical prediction. Full per-node activations, causal mediation, counterfactual studies and evaluation against Grad-CAM remain separate scientific phases.
