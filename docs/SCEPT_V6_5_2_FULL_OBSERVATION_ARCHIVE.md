# SCEPT V6.5.2 — complete captured-observation scientific ZIP

## Deliverable and limits
Each genuine single-forward model run can export all outputs ACTUALLY CAPTURED, including complete raw float32 (or declared dtype) hidden-layer tensors, genuine logits, NAF statistical arrays, matching calibration reference if created, full Run Record and SHA-256/CRC provenance. No subsampling of stored tensor values. This is NOT every intermediate operation in the model: uninstrumented layers, model weights, native MRI NIfTI and unrecorded memory are not fabricated or included. U11 M4 remains NOT ACHIEVED.

The original MRI may be sensitive and model activations themselves may leak private data. The ZIP is generated and downloaded only in the local browser, with no automatic remote upload. Include the exact preprocessed MODEL INPUT tensor only if the user selects retention before inference. This is not the original source NIfTI. This is off by default. The current existing summary-only JSON journal continues to exclude all raw bytes.

## User actions
- Run one genuine NAF with actual frozen ViT or CNN.
- Choose Download complete ZIP to save all captured hidden/output data.
- Optional: opt in to retain the exact preprocessed input before running.
- Optional: select auto-download complete ZIP after EVERY new real inference (browser may request repeated download permission).
- Raw tensors from earlier runs are released on later runs, so save ZIP immediately or enable automatic download before experiments.

## ZIP schema
Format: ZIP/STORE, UTF-8, max 160 MiB and 100 entries, locally generated:
- record.json — source hashes, actual output signatures, provenance, measured vs unavailable.
- naf.json — complete single-run numeric fingerprint.
- reference.json — full matching empirical reference distribution, if available.
- tensors/outputs/000.bin etc — every real measured ONNX output, including logits and all exposed probes.
- tensors/inputs/000.bin etc — only explicitly consented actual model feed.
- manifest.json — archive scope flags, exact dtype/shape, per-file SHA-256, tensor name and canonical SHA.

The exporter cross-checks EVERY declared runtime output against its actual original capture SHA, fails closed on absent/mutated tensors, refuses invalid calibration references and never labels unobserved probes as stored.

Independent verifier: python scept-web/tests/verify_full_archive.py archive.zip
This checks standard ZIP CRC, file lengths, per-file SHA-256, original tensor type/shape/byte hash, and agreement with Run Record. It proves archive integrity and record consistency, not real-world identity of third-party model weights or clinical/causal validity.

## Validation
V6.5.2 CI: Node contract ZIP and Python stdlib recheck; browser genuine pretrained ViT sample Token 7/channel 0 direct raw float32 equality; manual full ZIP download; second actual independent run with consented model-input retention and automatic ZIP; independent Python verification of both downloaded real ViT archives; existing V6.3, V5 and U10 regression.

Compatibility: no change to original scientific experimental definitions; generic U10 controlled experiments still await V6.4 universal Run Record integration.
