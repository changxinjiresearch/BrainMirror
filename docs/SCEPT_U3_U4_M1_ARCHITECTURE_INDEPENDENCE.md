# SCEPT U3–U4: M1 architecture-independence engineering contract

## Scope

M1 is about **one engine API** accepting more than one model architecture and observing genuine exported tensors. It does not establish a universal XAI algorithm, pretrained MRI interpretation, causal semantic explanations, MRI preprocessing, or clinical validity.

U3 supplies a real **typed data adapter** (\`core/scept_data_adapter.js\`), which validates precisely named/typed tensors against the inspected model signature, copies baselines and builds an explicitly *numeric-only* intervention. It does **not** guess MRI affine, channel normalization or target labels.

U4 supplies a **separate observation adapter** (\`core/scept_observation.js\`). This calculates A/B statistics from actual ONNX Runtime outputs with an explicit observation profile. It supports dynamic tensor dimensions and axis-wise real response summaries. Profiles currently offered: \`vit-token-channel\`, \`cnn-nchw\`, \`cnn-nhwc\`, \`generic\`, and strictly bounded \`legacy-vit-197x192\`.

The original \`scept_tensor_adapter.js\` / \`scept_activity3d.js\` and \`v3_engine.js\` retain a deliberately **legacy ViT-specific** [1,197,192] visual path. They are NOT imported by, nor do they define contracts for, the U3/U4 universal workbench. A later visual migration is needed before every existing MRI visual widget is fully architecture-neutral.

## M1 acceptance tests

| Gate | Concrete proof | Scope |
| --- | --- | --- |
| M1.1 | \`tests/u3_u4_contract.cjs\` | Named typed feed validation, immutable A/B copies, synthetic 5×7 ViT and 3×2×4 CNN axis stats, rejected unknown probes and mismatched legacy shapes |
| M1.2 | \`tests/u4_m1_cross_architecture.cjs\` | Actual frozen **pretrained** public Tiny ViT and separately generated **untrained** ONNX CNN traverse SAME registry, WASM loader, typed-feed adapter and observation API |
| M1.3 | CNN known Conv response | Real \`conv_hidden\` value after perturbation has independently calculable first-element delta 0.06 |
| M1.4 | ViT exported intermediate | Real layer-6 attention-value output is promoted from existing graph metadata and measured A/B |
| M1.5 | Regression | Existing U0–U2 and v3.1 tests remain unchanged; new tests run on the same CI job |

All sample inputs in these gates are deterministic **non-medical** synthetic numeric tensors. Do not call this a pretrained-CNN internal observability pass: the CNN proving arbitrary rank/shape here is a generated test model, whereas the separately exposed public MRI-CNN demo currently promises only end outputs. Cross-architecture ENGINEERING verification does not qualify the medical science or robust generality across pretrained checkpoints.

**Pass/fail rule:** code committed ≠ CI pass. M1 counts as verified only when the U3/U4 unit gate and headless Chromium cross-architecture inference gate both succeed on the same branch/commit. If the ViT download, CORS, WASM compatibility or declared probe cannot run, report a blocking failure without substituting animations.

## Extending

For each new ONNX architecture, retain \`SCEPT_CORE.openModel\` and \`SCEPT_DATA_ADAPTER.prepare/offset\`, expose genuine graph outputs for internal inspection, select a declared layout or \`generic\`, and call \`SCEPT_OBSERVATION.observePair\`. The model's actual runtime shape controls the axis computations. Attention connections are not inferred causal explanations.
