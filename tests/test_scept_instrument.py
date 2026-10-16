"""Independent ONNX Runtime numerical checks for SCEPT instrumentation.

This constructs a deterministic untrained toy network solely to validate code
math. It is NOT evidence of medical-model or Transformer scientific validity.
"""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper
from scripts.scept_instrument_onnx import inject


def make_graph():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    w1 = numpy_helper.from_array(np.array([[1, 0], [0, 1]], np.float32), "W1")
    w2 = numpy_helper.from_array(np.array([[1], [2]], np.float32), "W2")
    graph = helper.make_graph([
        helper.make_node("MatMul", ["x", "W1"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["activation"]),
        helper.make_node("MatMul", ["activation", "W2"], ["y"]),
    ], "known-numerical-fixture", [x], [y], [w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def test_observed_tensor_and_one_unit_ablation_exact_math():
    graph, mask_name = inject(make_graph(), ["activation"], "activation")
    session = ort.InferenceSession(graph.SerializeToString(),
                                   providers=["CPUExecutionProvider"])
    x = np.array([[1, 3]], np.float32)
    ones = np.ones((1, 2), np.float32)
    masked = np.array([[1, 0]], np.float32)
    normal = session.run(None, {"x": x, mask_name: ones})
    treated = session.run(None, {"x": x, mask_name: masked})
    names = session.get_outputs()
    mapping_a = {n.name: v for n, v in zip(names, normal)}
    mapping_b = {n.name: v for n, v in zip(names, treated)}
    assert np.array_equal(mapping_a["activation"], np.array([[1, 3]], np.float32))
    assert np.array_equal(mapping_b["activation"], np.array([[1, 3]], np.float32))
    assert np.allclose(mapping_a["y"], [[7.0]], atol=1e-6)
    assert np.allclose(mapping_b["y"], [[1.0]], atol=1e-6)
    assert not np.array_equal(mapping_a["y"], mapping_b["y"])


def test_identity_mask_matches_reference_model():
    original = ort.InferenceSession(make_graph().SerializeToString(),
                                    providers=["CPUExecutionProvider"])
    graph, mask_name = inject(make_graph(), ["activation"], "activation")
    instrumented = ort.InferenceSession(graph.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    x = np.array([[2, 4]], np.float32)
    baseline = original.run(["y"], {"x": x})[0]
    with_mask = instrumented.run(["y"], {
        "x": x, mask_name: np.ones((1, 2), np.float32)
    })[0]
    np.testing.assert_allclose(baseline, with_mask, rtol=0, atol=1e-6)


def test_refuses_out_of_graph_target():
    with pytest.raises(ValueError, match="not found"):
        inject(make_graph(), ["unreal_tensor"], None)
