import numpy as np
import pytest
import torch
from torch import nn
from scept import SCEPT, SCEPTError, tensor_array, tensor_sha

def test_torch_actual_one_forward_cnn_and_record():
    torch.manual_seed(29)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Conv2d(1,4,3,padding=1)
            self.bn=nn.BatchNorm2d(4)
            self.head=nn.Conv2d(4,2,1)
            self.calls=0
        def forward(self,x):
            self.calls+=1
            return self.head(torch.relu(self.bn(self.conv(x))))
    model=CNN().train()
    x=torch.randn(1,1,8,8)
    before=x.clone()
    sdk=SCEPT(model,["conv","head"],backend="torch",profile="cnn-nchw")
    result=sdk.observe(x,preprocess_id="synthetic-test",preprocess_config={"seed":29})
    assert model.calls==1
    assert model.training and model.bn.training
    assert torch.equal(x,before)
    assert result.record["schema"]=="scept.python-capture.v1"
    assert result.record["science"]["causal"]=="NOT_ESTABLISHED"
    assert result.record["science"]["NAF"]=="NOT_YET_COMPUTED_A3"
    assert result.tensors==("model.output","conv","head")
    assert result.tensor("conv").shape==(1,4,8,8)
    assert result.tensor("model.output").shape==(1,2,8,8)
    assert result.record["outputs"][0]["sha256"]==tensor_sha(*tensor_array(result.tensor("model.output")))
    altered=result.tensor("conv")
    altered[0,0,0,0]=999.
    assert result.tensor("conv")[0,0,0,0]!=999.
    assert not any(m._forward_hooks for m in model.modules())
    with pytest.raises(NotImplementedError,match="A3"):
        result.fingerprint()
    with pytest.raises(NotImplementedError,match="A5"):
        result.export("archive.zip")

def test_true_transformer_layer_not_fixed_197_tokens():
    torch.manual_seed(31)
    model=nn.TransformerEncoderLayer(d_model=8,nhead=2,dim_feedforward=16,batch_first=True,dropout=0.)
    observer=SCEPT(model,["linear1","linear2"],backend="torch",profile="vit-token-channel")
    previous_fastpath=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        r=observer.observe(torch.randn(1,5,8))
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    assert r.tensor("linear1").shape==(1,5,16)
    assert r.tensor("linear2").shape==(1,5,8)
    assert r.record["science"]["stability"]=="NOT_EVALUATED"

def test_reused_module_rejected_and_hooks_removed():
    class R(nn.Module):
        def __init__(self):super().__init__();self.conv=nn.Linear(2,2)
        def forward(self,x):return self.conv(self.conv(x))
    m=R()
    observer=SCEPT(m,["conv"])
    with pytest.raises(SCEPTError,match="REPEATED_MODULE"):
        observer.observe(torch.ones(1,2))
    assert len(m.conv._forward_hooks)==0

def test_unknown_layer_budget_and_invalid_probe():
    model=nn.Sequential(nn.Conv2d(1,2,1))
    with pytest.raises(SCEPTError,match="PROBE_NOT_FOUND"):
        SCEPT(model,["not_real"])
    with pytest.raises(SCEPTError,match="INVALID_PROBES"):
        SCEPT(model,[""])
    limited=SCEPT(model,["0"],max_capture_bytes=100)
    with pytest.raises(SCEPTError,match="RESOURCE_LIMIT"):
        limited.observe(torch.ones(1,1,16,16))
