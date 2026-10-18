"""Test REAL ONNX internal activation patching; synthetic fixture is engineering only."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("patch_builder",ROOT/"scripts"/"scept_build_icm_patch_model.py")
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

class TestActualRuntimePatch(unittest.TestCase):
    def test_exact_noop_and_nonzero_real_mask(self):
        import onnx,onnxruntime as ort
        from onnx import TensorProto,helper,numpy_helper
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            x=helper.make_tensor_value_info("pixel_values",TensorProto.FLOAT,[1,3,2])
            out=helper.make_tensor_value_info("logits",TensorProto.FLOAT,[1,3,2])
            hidden=helper.make_tensor_value_info("true_hidden",TensorProto.FLOAT,[1,3,2])
            w=numpy_helper.from_array(np.array([[2.,0.],[0.,3.]],dtype=np.float32),name="W")
            nodes=[helper.make_node("MatMul",["pixel_values","W"],["true_hidden"]),
                   helper.make_node("Relu",["true_hidden"],["logits"])]
            model=helper.make_model(helper.make_graph(nodes,"real-rewrite-fixture",[x],[out,hidden],[w]),
                opset_imports=[helper.make_opsetid("",17)])
            model.ir_version=min(model.ir_version,10)
            onnx.checker.check_model(model)
            original=tmp/"model.onnx";onnx.save(model,original)
            target=tmp/"icm_patch.onnx"
            manifest=mod.build(original,"true_hidden",target)
            self.assertEqual(manifest["probe"],"true_hidden")
            a=np.ones((1,3,2),dtype=np.float32)
            b=a.copy();b[0,1,:]=[.2,.4]
            original_session=ort.InferenceSession(str(original),providers=["CPUExecutionProvider"])
            patched_session=ort.InferenceSession(str(target),providers=["CPUExecutionProvider"])
            off=np.zeros_like(a,dtype=bool);mask=off.copy();mask[:,1,:]=True
            HA=original_session.run(["true_hidden"],{"pixel_values":a})[0]
            HB=original_session.run(["true_hidden"],{"pixel_values":b})[0]
            def real_run(v,rep,m):
                return patched_session.run(["logits"],{
                 "pixel_values":v,"scept_patch_override":rep,"scept_patch_mask":m})[0]
            origA=original_session.run(["logits"],{"pixel_values":a})[0]
            origB=original_session.run(["logits"],{"pixel_values":b})[0]
            np.testing.assert_array_equal(real_run(a,HA,off),origA)
            np.testing.assert_array_equal(real_run(b,HB,off),origB)
            newB=real_run(b,HA,mask);newA=real_run(a,HB,mask)
            self.assertGreater(abs(float(newB[0,1,0])-float(origB[0,1,0])),.1)
            self.assertGreater(abs(float(newA[0,1,0])-float(origA[0,1,0])),.1)
            np.testing.assert_array_equal(newB[0,0,:],origB[0,0,:])
            np.testing.assert_array_equal(newA[0,2,:],origA[0,2,:])

if __name__=="__main__":
    unittest.main()
