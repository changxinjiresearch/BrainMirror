"""Synthetic 04D direct post-residual causal computation test: no MRI or patient info."""
import ast
import sys
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from torch import nn

EX=Path(__file__).resolve().parents[1]/"examples"
sys.path.insert(0,str(EX))
from a6_ij_04d_hook import (
 LAYERS,PLAN,plan_sha,indices_for_roi,capture_original,patched_forward,
 assert_logits,truth_margin,bounded_fraction
)

class ToyBlock(nn.Module):
 def __init__(self,index):
  super().__init__()
  self.register_buffer("scale",torch.tensor((index+1)*.001))
 def forward(self,x):
  return x+x.mean(dim=1,keepdim=True)*self.scale

class TinyVit(nn.Module):
 def __init__(self):
  super().__init__()
  self.blocks=nn.ModuleList([ToyBlock(i) for i in range(12)])
 def forward(self,x):
  for block in self.blocks:x=block(x)
  p=x[:,1:,:]
  return torch.stack([p.mean(dim=(1,2)),p[:,0,:].mean(dim=1)],dim=1)

def fixture():
 m=TinyVit().eval()
 clean=torch.zeros(1,197,768)
 clean[:,1,:]=.5
 edit=clean.clone()
 edit[:,2,:]=2.0
 return m,clean,edit

def test_python_310_protocol_and_graph_source_contract():
 src=(EX/"a6_ij_04d_hook.py").read_text()
 ast.parse(src,feature_version=(3,10))
 assert tuple(PLAN["layers"])==("blocks.5","blocks.11")
 assert PLAN["forwardPassesPerModelPatient"]==29
 assert PLAN["totalActualModelForwardPasses"]==1914
 assert PLAN["minimumInputOutputMarginDeltaForRatios"]==1e-4
 assert len(plan_sha())==64
 assert "torch.inference_mode()" in src
 assert "optimizer" not in src[src.index("def patched_forward"):]

@pytest.mark.parametrize("layer",LAYERS)
def test_full_clean_recovery_and_reverse_full_insertion(layer):
 model,clean,edit=fixture()
 a,sources=capture_original(model,clean)
 b,edited=capture_original(model,edit)
 r=patched_forward(model,edit,layer,sources[layer],range(197))
 i=patched_forward(model,clean,layer,edited[layer],range(197))
 assert assert_logits(r,a,"full-target-back")<1e-7
 assert assert_logits(i,b,"full-target-in")<1e-7
 assert assert_logits(patched_forward(model,clean,layer,sources[layer],[2]),a,"clean-sham")<1e-7
 assert len(dict(model.named_modules())[layer]._forward_hooks)==0
 assert torch.allclose(model(clean),torch.tensor(a).view(1,2))

def test_single_token_intervention_and_placebo_from_frozen_graph():
 model,clean,edit=fixture()
 a,ca=capture_original(model,clean)
 b,ed=capture_original(model,edit)
 new=patched_forward(model,edit,"blocks.5",ca["blocks.5"],[2])
 placebo=patched_forward(model,edit,"blocks.5",ca["blocks.5"],[3])
 assert abs(new[0]-b[0])>1e-6
 assert np.isfinite(placebo).all() # placebo may or may not change output
 assert assert_logits(patched_forward(model,edit,"blocks.5",ed["blocks.5"],[2]),b,"edit-self")==0

def test_prespecified_selection_and_indeterminate_margin_rule():
 selection={"lesion":{"row":2,"col":3},"control":{"row":9,"col":10}}
 a=indices_for_roi(selection,"lesion")
 b=indices_for_roi(selection,"control")
 assert a["target"]==b["placebo"]
 assert a["placebo"]==b["target"]
 assert set(a["all"])==set(range(197))
 assert bounded_fraction(.1,.1+.00001,.1+.00001,"rescue")["signedFraction"] is None
 assert bounded_fraction(1,0,1,"rescue")["signedFraction"]==1
 assert bounded_fraction(1,0,0,"insertion")["signedFraction"]==1
 assert truth_margin([0,1],"HGG")==1
 assert truth_margin([0,1],"LGG")==-1

def test_hook_exception_cleanup_guarantee():
 model,clean,edit=fixture()
 _,donor=capture_original(model,clean)
 with pytest.raises(RuntimeError):
  patched_forward(model,edit,"blocks.5",donor["blocks.5"][:,:,:50],[2])
 assert len(model.blocks[5]._forward_hooks)==0
