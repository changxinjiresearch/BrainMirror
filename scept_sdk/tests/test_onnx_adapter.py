"""A2 real ONNX Runtime numeric graph output instrumentation without rewriting user file."""
import hashlib
import numpy as np
import pytest
import onnx
from onnx import TensorProto, helper, numpy_helper
from scept import SCEPT, SCEPTError, tensor_array, tensor_sha

def model_bytes():
    x=helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,1,3,3])
    y=helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,2,3,3])
    weights=np.array([1.,-1.],dtype=np.float32).reshape(2,1,1,1)
    initializer=numpy_helper.from_array(weights,"kernel")
    nodes=[
        helper.make_node("Conv",["input","kernel"],["conv_out"],name="actual_conv"),
        helper.make_node("Relu",["conv_out"],["relu_out"],name="actual_relu"),
        helper.make_node("Identity",["relu_out"],["output"],name="actual_output"),
    ]
    graph=helper.make_graph(nodes,"actual-cnn", [x],[y],initializer=[initializer])
    m=helper.make_model(graph,opset_imports=[helper.make_operatorsetid("",13)])
    m.ir_version=9
    onnx.checker.check_model(m)
    return m.SerializeToString()

def test_actual_single_forward_onnx_conv_probes():
    original=model_bytes()
    x=np.arange(1,10,dtype=np.float32).reshape(1,1,3,3)
    copy=x.copy()
    observer=SCEPT(original,["conv_out","relu_out"],backend="onnx",profile="cnn-nchw")
    assert observer._adapter.model_sha256==hashlib.sha256(original).hexdigest()
    assert "conv_out" in observer._adapter.outputs
    r=observer.observe(x,preprocess_id="actual-test-v1",preprocess_config={"seed":17})
    assert r.tensors==("output","conv_out","relu_out")
    np.testing.assert_array_equal(r.tensor("conv_out")[0,0],x[0,0])
    np.testing.assert_array_equal(r.tensor("conv_out")[0,1],-x[0,0])
    np.testing.assert_array_equal(r.tensor("relu_out")[0,1],np.zeros((3,3),dtype=np.float32))
    np.testing.assert_array_equal(x,copy)
    assert r.record["provenance"]["model"]["sha256"]==hashlib.sha256(original).hexdigest()
    assert r.record["provenance"]["inputs"]["input"]==tensor_sha(*tensor_array(x))
    assert r.record["science"]["status"]=="raw_tensor_capture_only"
    assert r.record["science"]["causal"]=="NOT_ESTABLISHED"
    # Actual named graph outputs from a single actual ORT run, not fake activations.
    for o in r.record["outputs"]:
        assert o["sha256"]==tensor_sha(*tensor_array(r.tensor(o["name"])))

def test_unknown_onnx_tensor_and_memory_guard():
    model=model_bytes()
    with pytest.raises(SCEPTError,match="PROBE_NOT_FOUND"):
        SCEPT(model,["not_a_node"],backend="onnx")
    sdk=SCEPT(model,["relu_out"],backend="onnx",max_capture_bytes=32)
    with pytest.raises(SCEPTError,match="RESOURCE_LIMIT"):
        sdk.observe(np.ones((1,1,3,3),dtype=np.float32))

def test_onnx_input_dtype_and_shape_rejected():
    model=model_bytes()
    sdk=SCEPT(model,["conv_out"],backend="onnx")
    # ONNX Runtime itself rejects mismatched dtype and shape; we must not
    # silently re-cast the tensor and reinterpret values.
    with pytest.raises(Exception):
        sdk.observe(np.ones((1,1,3,3),dtype=np.int32))
    with pytest.raises(Exception):
        sdk.observe({"other":np.ones((1,1,3,3),dtype=np.float32)})
