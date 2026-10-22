"""Independent zero-patient timm ResNet18 PyTorch/ONNX SCEPT true-CNN test.

Runs ONLY with full timm+onnxscript dependencies (dedicated 02D runner).
Weights are deterministic RANDOM (not MRI trained); no patient/private data.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

torch=pytest.importorskip("torch")
onnx=pytest.importorskip("onnx")
ort=pytest.importorskip("onnxruntime")
timm=pytest.importorskip("timm")

from scept import SCEPT,load_observation

PROBES=("net.layer1","net.layer3")
PAIRS=(("net.layer1","layer1"),
       ("net.layer3","layer3"),
       ("model.output.2","logits"))

class ActualResNet18Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=timm.create_model("resnet18",pretrained=False,num_classes=2,in_chans=3)
    def forward(self,x):
        n=self.net
        x=n.conv1(x)
        x=n.bn1(x)
        x=n.act1(x)
        x=n.maxpool(x)
        p1=n.layer1(x)
        x=n.layer2(p1)
        p3=n.layer3(x)
        x=n.layer4(p3)
        logits=n.fc(n.global_pool(x))
        return p1,p3,logits

def err(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape
    return {"maxAbs":float(np.max(np.abs(a-b))),
            "relativeL2":float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(a.ravel()),1e-12))}

def test_actual_timm_resnet18_named_probes_onnx_graph_parity():
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(240216)
    model=ActualResNet18Probe().eval()
    x=torch.linspace(0,1,3*224*224,dtype=torch.float32).reshape(1,3,224,224)
    hooked={}
    handles=[]
    for key in PROBES:
        def capture(_m,_inputs,t,key=key):
            hooked[key]=t.detach().cpu().numpy().copy()
        handles.append(dict(model.named_modules())[key].register_forward_hook(capture))
    try:
        with torch.inference_mode():baseline=model(x)
    finally:
        for h in handles:h.remove()
    hooked["model.output.2"]=baseline[2].detach().cpu().numpy().copy()
    torchobserver=SCEPT(model,PROBES,backend="torch",profile="cnn-nchw",
                        max_capture_bytes=32_000_000)
    torchrun=torchobserver.observe(x,model_id="real-timm-resnet18-random-no-clinical-value",
        preprocess_id="synthetic-02d-real-resnet")
    assert [g["tensor"] for g in torchrun.fingerprint()["groups"]]==list(PROBES)
    for key in hooked:
        np.testing.assert_array_equal(torchrun.tensor(key),hooked[key])
    assert torchrun.tensor("net.layer1").shape==(1,64,56,56)
    assert torchrun.tensor("net.layer3").shape==(1,256,14,14)
    with tempfile.TemporaryDirectory() as td:
        local=Path(td)
        path=local/"torch.zip"
        torchrun.export(str(path),include_inputs=False)
        tzip=load_observation(str(path))
        assert tzip.manifest["captureScope"]["inputsIncluded"] is False
        for key in hooked:
            np.testing.assert_array_equal(tzip.tensor(key),hooked[key])
        assert tzip.recompute_naf()["science"]["model_forward_reexecuted"] is False
        onnx_path=local/"resnet18-real.onnx"
        kwargs={"opset_version":17,"do_constant_folding":True,
                "input_names":["flair_rgb"],"output_names":["layer1","layer3","logits"]}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs["dynamo"]=False
        torch.onnx.export(model,x,str(onnx_path),**kwargs)
        checker=onnx.load(str(onnx_path))
        onnx.checker.check_model(checker)
        assert {t.name for t in checker.graph.output}=={"layer1","layer3","logits"}
        assert onnx_path.stat().st_size<150_000_000
        original=SCEPT(onnx_path,("layer1","layer3"),backend="onnx",
                       profile="cnn-nchw",providers=("CPUExecutionProvider",),
                       max_capture_bytes=32_000_000)
        onnxrun=original.observe({"flair_rgb":x.numpy()},model_id="same-resnet18-export",
                                 preprocess_id="synthetic-02d-real-resnet")
        errors={}
        for t,o in PAIRS:
            errors[t]=err(torchrun.tensor(t),onnxrun.tensor(o))
            assert errors[t]["maxAbs"]<=1e-3
            assert errors[t]["relativeL2"]<=1e-3
        target=local/"onnx.zip"
        onnxrun.export(str(target),include_inputs=False)
        oz=load_observation(str(target))
        assert oz.manifest["captureScope"]["inputsIncluded"] is False
        for t,o in PAIRS:
            np.testing.assert_array_equal(oz.tensor(o),onnxrun.tensor(o))
        report={"schema":"scept.a6-ij-02d.real-resnet.synthetic.v1",
            "status":"PASS_SYNTHETIC_REAL_RESNET18_TORCH_ONNX",
            "architecture":"actual timm ResNet18, untrained deterministic RANDOM weights",
            "scope":"NO MEDICAL OR PATIENT DATA",
            "weightStateSha256":torchobserver.model_sha256,
            "torchZipSHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "onnxModelSHA256":hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "onnxZipSHA256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "perTensor":errors,
            "versions":{"torch":torch.__version__,"onnx":onnx.__version__,
                        "onnxruntime":ort.__version__,"timm":timm.__version__}}
        dest=os.environ.get("SCEPT_02D_REPORT")
        if dest:
            q=Path(dest)
            q=q.with_name(q.stem+"_resnet18"+q.suffix)
            q.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("A6-IJ-02D REAL ResNet18 architecture synthetic ONNX parity PASS "+json.dumps(report,sort_keys=True))
