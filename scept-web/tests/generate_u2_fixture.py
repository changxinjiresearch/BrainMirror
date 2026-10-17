"""U2 compatibility fixtures: two deliberately different real ONNX graphs.
These are tiny deterministic TEST models; they are not medical or pretrained.
Their op outputs are genuine runtime-computed tensors, not fabricated UI data.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

out = Path("scept-web/tests/generated")
out.mkdir(parents=True, exist_ok=True)

def save(model, name):
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out / name)
    print("GENERATED", out / name, (out / name).stat().st_size)

w = np.array([[0.7,-0.3,0.5],[0.2,0.8,0.4],[-0.4,0.1,0.9],[0.3,0.7,-0.6]], dtype=np.float32)
bias = np.array([0.1,-0.2,0.05],dtype=np.float32)
g = helper.make_graph(
    [helper.make_node("MatMul",["features","W"],["mm_hidden"],name="hidden_projection"),
     helper.make_node("Add",["mm_hidden","bias"],["biased_hidden"],name="bias_addition"),
     helper.make_node("Relu",["biased_hidden"],["scores"],name="relu_output")],
    "tiny_mlp_u2",
    [helper.make_tensor_value_info("features",TensorProto.FLOAT,[1,4])],
    [helper.make_tensor_value_info("scores",TensorProto.FLOAT,[1,3])],
    [numpy_helper.from_array(w,name="W"),numpy_helper.from_array(bias,name="bias")],
    value_info=[
        helper.make_tensor_value_info("mm_hidden",TensorProto.FLOAT,[1,3]),
        helper.make_tensor_value_info("biased_hidden",TensorProto.FLOAT,[1,3])])
save(helper.make_model(g,producer_name="scept-u2-fixture",opset_imports=[helper.make_operatorsetid("",13)]),"u2_mlp.onnx")

kernel=np.array([[[[0.6,-0.1],[0.25,0.5]]]],dtype=np.float32)
g2=helper.make_graph(
    [helper.make_node("Conv",["image_data","K"],["conv_hidden"],name="real_convolution"),
     helper.make_node("Relu",["conv_hidden"],["conv_output"],name="conv_nonlinearity")],
    "tiny_cnn_u2",
    [helper.make_tensor_value_info("image_data",TensorProto.FLOAT,[1,1,4,4])],
    [helper.make_tensor_value_info("conv_output",TensorProto.FLOAT,[1,1,3,3])],
    [numpy_helper.from_array(kernel,name="K")],
    value_info=[helper.make_tensor_value_info("conv_hidden",TensorProto.FLOAT,[1,1,3,3])])
save(helper.make_model(g2,producer_name="scept-u2-fixture",opset_imports=[helper.make_operatorsetid("",13)]),"u2_cnn.onnx")
