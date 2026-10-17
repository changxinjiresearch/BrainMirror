"""U8: independently generated tiny 3D ONNX Conv fixture (NOT a trained MRI model)."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
out=Path("scept-web/tests/generated")
out.mkdir(parents=True,exist_ok=True)
kernel=np.array([[[[[0.2,-0.1],[0.3,0.5]],[[0.7,0.0],[-0.2,0.4]]]]],dtype=np.float32)
graph=helper.make_graph([
  helper.make_node("Conv",["vol_input","K"],["vol_hidden"],name="actual_3d_convolution"),
  helper.make_node("Relu",["vol_hidden"],["vol_output"],name="actual_3d_relu")
 ],"u8_tiny_3d_conv",
 [helper.make_tensor_value_info("vol_input",TensorProto.FLOAT,[1,1,3,4,4])],
 [helper.make_tensor_value_info("vol_output",TensorProto.FLOAT,[1,1,2,3,3])],
 [numpy_helper.from_array(kernel,name="K")],
 value_info=[helper.make_tensor_value_info("vol_hidden",TensorProto.FLOAT,[1,1,2,3,3])])
m=helper.make_model(graph,producer_name="SCEPT-U8-engineering-fixture",opset_imports=[helper.make_operatorsetid("",13)])
m.ir_version=8
onnx.checker.check_model(m)
onnx.save(m,out/"u8_3d_cnn.onnx")
print("GENERATED",out/"u8_3d_cnn.onnx")
