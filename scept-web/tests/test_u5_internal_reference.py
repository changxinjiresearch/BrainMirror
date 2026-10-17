"""U5 true ONNX numerical parity: unmodified original weights vs all-ones-mask instrumented CNN.
Source models are deterministic engineering fixtures, not pretrained medical networks.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent / "generated"

def test_mask_true_internal_downstream_effect_and_original_parity():
    original = (ROOT / "u2_cnn.onnx").read_bytes()
    instrumented = (ROOT / "u5_cnn_mask.onnx").read_bytes()
    raw = onnx.load_from_string(instrumented)
    onnx.checker.check_model(raw)
    mask_ops = [n for n in raw.graph.node if n.name == "SCEPT_ActivationMask"]
    assert len(mask_ops) == 1
    assert mask_ops[0].op_type == "Mul"
    assert "scept_mask" in mask_ops[0].input
    output_of_mask = mask_ops[0].output[0]
    assert any(output_of_mask in n.input for n in raw.graph.node if n.name != "SCEPT_ActivationMask")
    session_a = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
    session_b = ort.InferenceSession(instrumented, providers=["CPUExecutionProvider"])
    image = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    ones = np.ones((1, 1, 3, 3), dtype=np.float32)
    masked = ones.copy()
    masked[0, 0, 0, 0] = 0.0
    a = session_a.run(None, {"image_data": image})[0]
    observations = [x.name for x in session_b.get_outputs()]
    base = dict(zip(observations, session_b.run(None, {"image_data": image, "scept_mask": ones})))
    perturbed = dict(zip(observations, session_b.run(None, {"image_data": image, "scept_mask": masked})))
    np.testing.assert_allclose(base["conv_output"], a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(base["conv_output"][0,0,0,0], 0.0, atol=1e-6) if a[0,0,0,0] == 0 else None
    assert not np.array_equal(base["conv_hidden"], perturbed["conv_hidden"])
    assert not np.array_equal(base["conv_output"], perturbed["conv_output"])
    assert np.count_nonzero(base["conv_output"] != perturbed["conv_output"]) == 1
    np.testing.assert_allclose(perturbed["conv_output"][0,0,0,0], 0.0, atol=1e-6)
    np.testing.assert_allclose(perturbed["conv_output"][0,0,0,1:], base["conv_output"][0,0,0,1:], atol=1e-6)
    manifest_path = ROOT / "u5_cnn_mask.onnx.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_training_performed"] is False
    assert manifest["internal_ablation"]["tensor"] == "conv_hidden"
    assert manifest["internal_ablation"]["input"] == "scept_mask"
    assert manifest["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["instrumented_sha256"] == hashlib.sha256(instrumented).hexdigest()
