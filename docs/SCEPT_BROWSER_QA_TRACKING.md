# SCEPT browser-runtime release gate

This change intentionally triggers the full PR CI suite with the pretrained ONNX Chromium workflow. The final pass/fail must be checked from actual CI logs; successful static deployment alone does not validate browser inference.
