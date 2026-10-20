"""A6-IJ-02D: zero-patient-data independent-runner contract.

Two genuinely different PyTorch module structures + PyTorch-vs-ONNXRuntime
engine parity with same frozen weights. No weights download, patient MRI, secrets
or clinical/scientific attribution. Run on fresh Python 3.10/3.11/3.12 runners.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile

import numpy as np
import pytest

torch=pytest.importorskip("torch")
onnx=pytest.importorskip("onnx")
ort=pytest.importorskip("onnxruntime")
from scept import SCEPT,load_observation

MAX_ABS=1e-3
MAX_REL_L2=1e-3

class TinyResidualCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=torch.nn.Conv2d(1,5,3,padding=1,bias=True)
        self.residual=torch.nn.Conv2d(5,5,3,padding=1,bias=True)
        self.fc=torch.nn.Linear(5,2)
    def forward(self,x):
        raw=self.stem(x)
        skip=self.residual(torch.relu(raw))
        logits=self.fc(torch.relu(raw+skip).mean(dim=(-2,-1)))
        return raw,skip,logits

class TokenBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=torch.nn.MultiheadAttention(12,3,dropout=0.0,batch_first=True)
        self.norm1=torch.nn.LayerNorm(12)
        self.linear=torch.nn.Linear(12,12)
    def forward(self,x):
        result,_=self.attn(x,x,x,need_weights=False)
        y=self.norm1(x+result)
        return torch.tanh(self.linear(y))

class TinyTokenTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=torch.nn.Linear(8,12)
        self.block0=TokenBlock()
        self.block1=TokenBlock()
        self.head=torch.nn.Linear(12,2)
    def forward(self,x):
        return self.head(self.block1(self.block0(self.proj(x))).mean(dim=1))

def compare(a,b):
    av=np.asarray(a,dtype=np.float64)
    bv=np.asarray(b,dtype=np.float64)
    assert av.shape==bv.shape
    max_abs=float(np.max(np.abs(av-bv)))
    relative=float(np.linalg.norm((av-bv).ravel())/max(np.linalg.norm(av.ravel()),1e-12))
    return {"maxAbs":max_abs,"relativeL2":relative}

def direct_hooks(model,x,names):
    captured={}
    refs=dict(model.named_modules())
    handles=[]
    for name in names:
        def capture(_m,_inp,y,key=name):
            assert isinstance(y,torch.Tensor)
            captured[key]=y.detach().cpu().numpy().copy()
        handles.append(refs[name].register_forward_hook(capture))
    try:
        model.eval()
        with torch.inference_mode():
            y=model(x)
        if isinstance(y,torch.Tensor):out={"model.output":y.detach().cpu().numpy().copy()}
        else:out={"model.output."+str(i):v.detach().cpu().numpy().copy() for i,v in enumerate(y)}
    finally:
        for h in handles:h.remove()
    return {**captured,**out}

def test_two_architectures_real_torch_zip_and_independent_reexecution():
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(240216)
    cnn=TinyResidualCNN().eval()
    transformer=TinyTokenTransformer().eval()
    cases=[
        (cnn,torch.linspace(-1,1,256,dtype=torch.float32).reshape(1,1,16,16),
         ("stem","residual"),"cnn-nchw"),
        (transformer,torch.linspace(-1,1,48,dtype=torch.float32).reshape(1,6,8),
         ("block0.norm1","block1.norm1"),"vit-token-channel"),
    ]
    report={"schema":"scept.a6-ij-02d.ci.v1",
            "host":{"python":platform.python_version(),"system":platform.system(),
                    "torch":torch.__version__,"onnx":onnx.__version__,
                    "onnxruntime":ort.__version__},
            "scientificInterpretation":"ENGINEERING_ONLY_NO_PATIENT_OR_CLINICAL_EVIDENCE",
            "modelTests":[]}
    with tempfile.TemporaryDirectory() as td:
        for model,x,names,profile in cases:
            obs=SCEPT(model,names,backend="torch",profile=profile,max_capture_bytes=16_000_000)
            run=obs.observe(x,model_id=model.__class__.__name__,preprocess_id="synthetic-02d")
            assert {g["tensor"] for g in run.fingerprint()["groups"] if g["status"]=="measured"}==set(names)
            independent=direct_hooks(model,x,names)
            assert set(independent)==set(run.tensors)
            for name in run.tensors:
                np.testing.assert_array_equal(run.tensor(name),independent[name])
            again=obs.observe(x,model_id=model.__class__.__name__,preprocess_id="synthetic-02d")
            for name in run.tensors:np.testing.assert_array_equal(run.tensor(name),again.tensor(name))
            path=Path(td)/(model.__class__.__name__+".zip")
            run.export(str(path),include_inputs=False)
            loaded=load_observation(str(path))
            assert loaded.run_id==run.run_id
            assert loaded.manifest["captureScope"]["inputsIncluded"] is False
            for name in run.tensors:np.testing.assert_array_equal(loaded.tensor(name),independent[name])
            recalc=loaded.recompute_naf()
            assert recalc["science"]["model_forward_reexecuted"] is False
            report["modelTests"].append({"architecture":model.__class__.__name__,
                "torchHookEquality":"EXACT","repeatedTorchEquality":"EXACT",
                "originalZipSha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "captureTensors":list(run.tensors)})
        x=cases[0][1]
        cnn=cases[0][0]
        model_path=Path(td)/"frozen-cnn.onnx"
        torch.onnx.export(cnn,x,str(model_path),opset_version=17,
                          input_names=["image"],output_names=["stem_raw","residual_raw","logits"],
                          do_constant_folding=True)
        onnx.checker.check_model(onnx.load(str(model_path)))
        from_torch=SCEPT(cnn,("stem","residual"),backend="torch",profile="cnn-nchw").observe(
            x,preprocess_id="synthetic-02d")
        from_onnx=SCEPT(model_path,("stem_raw","residual_raw"),backend="onnx",
                       providers=("CPUExecutionProvider",),profile="cnn-nchw").observe(
                           {"image":x.numpy()},preprocess_id="synthetic-02d")
        pairs=(("stem","stem_raw"),("residual","residual_raw"),("model.output.2","logits"))
        errors={}
        for torch_name,onnx_name in pairs:
            errors[torch_name]=compare(from_torch.tensor(torch_name),from_onnx.tensor(onnx_name))
            assert errors[torch_name]["maxAbs"]<=MAX_ABS
            assert errors[torch_name]["relativeL2"]<=MAX_REL_L2
        dst=Path(td)/"frozen-cnn-onnx.zip"
        from_onnx.export(str(dst),include_inputs=False)
        loaded=load_observation(str(dst))
        assert loaded.manifest["captureScope"]["inputsIncluded"] is False
        for _,onnx_name in pairs:
            np.testing.assert_array_equal(loaded.tensor(onnx_name),from_onnx.tensor(onnx_name))
        assert loaded.recompute_naf()["science"]["model_forward_reexecuted"] is False
        report["crossEngine"]={"status":"PASS","torchVsONNXError":errors,
                               "maxAbsoluteTolerance":MAX_ABS,"relativeL2Tolerance":MAX_REL_L2,
                               "originalOnnxSHA256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
                               "originalOnnxZipSHA256":hashlib.sha256(dst.read_bytes()).hexdigest()}
    outfile=os.environ.get("SCEPT_02D_REPORT")
    if outfile:
        p=Path(outfile);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("A6-IJ-02D CLEAN RUNNER CROSS-ARCH+ONNX PASS "+json.dumps(report,sort_keys=True))
