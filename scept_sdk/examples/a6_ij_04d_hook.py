"""A6-IJ-04D fixed genuine PyTorch residual-block causal intervention plugin.

Unlike capture-only SCEPT, this adapter intervenes on actual forward
computation of a frozen network. No backward, optimizer, weights modification
or retraining. The extra module hook must be temporary and will only change
specified token positions of the post-block tensor for that single forward.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import numpy as np
import torch

LAYERS=("blocks.5","blocks.11")
ARMS=("lesion","control")
SELECTORS=("target","placebo","all")
DIRECTIONS=("rescue","insertion")
OPERATOR="gaussian_local_sigma3"
DOSE=0.50
TOL=2e-4
MIN_MARGIN_DELTA=1e-4
PLAN={
 "schema":"scept.a6-ij-04d.locked-direct-post-residual-patching.v1",
 "original04BCondition":"gaussian_local_sigma3_dose50",
 "nPatients":33,"originalVALExcluded":22,"grades":{"HGG":26,"LGG":7},
 "models":["S","J"],"layers":list(LAYERS),
 "arms":list(ARMS),"selectors":list(SELECTORS),
 "directions":list(DIRECTIONS),
 "sourceState":"same patient's authentic frozen-model clean branch",
 "recipientState":"same patient's authentic frozen-model equal-L2 gaussian edited branch",
 "postBlockHookTensorShape":[1,197,768],"tokenGrid":"14x14, CLS token 0",
 "placebo":"other predeclared original same-patient ROI token (not anatomically matched)",
 "fullAllTokenPositiveControl":"patched branch logits within 0.0002 of authentic donor logits",
 "selfSham":"clean source into clean recipient at target lesion token; unchanged logits",
 "decision":"signed original true-grade logit margin; intervention output != training",
 "minimumInputOutputMarginDeltaForRatios":MIN_MARGIN_DELTA,
 "forwardPassesPerModelPatient":29,
 "totalActualModelForwardPasses":1914,
 "privateChunkSizes":[5,5,5,5,5,5,3],
 "nPatientBootstrap":2000,"seed":240216,
 "sealedP5Test":"NOT_ACCESSED","externalP6":"NOT_ACCESSED",
 "scientificInterpretation":"bounded causal effect of a specified post-block computational intervention, not medical causality"
}
def plan_sha():
 return hashlib.sha256(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def indices_for_roi(selection,arm):
 if arm not in ARMS:raise RuntimeError("Unregistered 04D ROI arm")
 entry=selection[arm]
 row,col=int(entry["row"]),int(entry["col"])
 if not(0<=row<14 and 0<=col<14):raise RuntimeError("04D true original 14x14 ROI invalid")
 idx=1+14*row+col
 other="control" if arm=="lesion" else "lesion"
 q=selection[other]
 otheridx=1+14*int(q["row"])+int(q["col"])
 if idx==otheridx or otheridx not in range(1,197):
  raise RuntimeError("04D original target-placebo token positions indistinguishable")
 return {"target":(idx,),"placebo":(otheridx,),"all":tuple(range(197))}

def truth_margin(logits,grade):
 values=np.asarray(logits,dtype=np.float64).reshape(-1)
 if values.shape!=(2,) or not np.isfinite(values).all() or grade not in ("LGG","HGG"):
  raise RuntimeError("04D finite original binary logits and grade required")
 label=1 if grade=="HGG" else 0
 return float(values[label]-values[1-label])

def bounded_fraction(clean,edited,patched,direction):
 """Signed fraction of authentic decision-margin displacement rescued/inserted."""
 if direction not in DIRECTIONS:raise RuntimeError("04D unknown patch direction")
 delta=float(clean-edited)
 if abs(delta)<MIN_MARGIN_DELTA:
  return {"status":"INDETERMINATE_ORIGINAL_OUTPUT_CONTRAST_BELOW_FIXED_GATE",
          "signedFraction":None,"originalMarginContrast":delta}
 if direction=="rescue":
  frac=(float(patched)-float(edited))/delta
 else:
  frac=(float(patched)-float(clean))/(-delta)
 return {"status":"MEASURABLE_GRAPH_INTERVENTION",
         "signedFraction":float(frac),"originalMarginContrast":delta}

def check_model(model):
 if model.training or any(p.requires_grad and p.grad is not None for p in model.parameters()):
  raise RuntimeError("04D model must be eval, without accumulated gradients")
 return True

def layer_module(model,layer):
 if layer not in LAYERS:raise RuntimeError("Unregistered 04D residual output layer")
 modules=dict(model.named_modules())
 if layer not in modules:raise RuntimeError("Original ViT residual layer unavailable")
 return modules[layer]

def capture_original(model,x,layers=LAYERS):
 """Single genuine eval/no-grad forward; snapshots actual postblock residual outputs."""
 check_model(model)
 modules={key:layer_module(model,key) for key in layers}
 got={}
 handles=[]
 for key,mod in modules.items():
  def capture(_mod,_args,out,at=key):
   if not torch.is_tensor(out) or tuple(out.shape)!=(1,197,768):
    raise RuntimeError("04D actual ViT postblock tensor expected [1,197,768]")
   got[at]=out.detach().to("cpu",dtype=torch.float32).clone()
  handles.append(mod.register_forward_hook(capture))
 try:
  with torch.inference_mode():
   output=model(x)
 finally:
  for handle in handles:handle.remove()
 if set(got)!=set(layers) or tuple(output.shape)!=(1,2) or \
    not torch.isfinite(output).all() or any(not torch.isfinite(t).all() for t in got.values()):
  raise RuntimeError("04D genuine frozen model baseline capture incomplete")
 return output.detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),got

def patched_forward(model,x,layer,donor,indices,verify=True):
 """Direct intervention at block output in a SINGLE actual downstream forward."""
 check_model(model)
 mod=layer_module(model,layer)
 token_ids=tuple(int(i) for i in indices)
 if len(token_ids) not in (1,197) or any(i<0 or i>196 for i in token_ids) or len(set(token_ids))!=len(token_ids):
  raise RuntimeError("04D token replacement selector invalid")
 if len(token_ids)==1 and token_ids[0]==0:
  raise RuntimeError("04D target/placebo must be an actual image patch token")
 if not torch.is_tensor(donor) or tuple(donor.shape)!=(1,197,768) or \
    donor.dtype!=torch.float32 or not bool(torch.isfinite(donor).all()):
  raise RuntimeError("04D source must be original finite full residual [1,197,768] float32")
 before=len(mod._forward_hooks)
 counter=[0]
 idx=torch.tensor(token_ids,dtype=torch.long,device=x.device)
 def replace(_module,_args,actual):
  counter[0]+=1
  if not torch.is_tensor(actual) or tuple(actual.shape)!=(1,197,768):
   raise RuntimeError("04D true postblock physical tensor wrong shape")
  cloned=actual.clone()
  src=donor.to(device=cloned.device,dtype=cloned.dtype)
  cloned[:,idx,:]=src[:,idx,:]
  if verify:
   if not torch.equal(cloned[:,idx,:],src[:,idx,:]):
    raise RuntimeError("04D donor full-token replacement failed")
   if len(token_ids)<197:
    untouched=torch.ones(197,device=cloned.device,dtype=torch.bool)
    untouched[idx]=False
    if not torch.equal(cloned[:,untouched,:],actual[:,untouched,:]):
     raise RuntimeError("04D edited non-target model state unexpectedly")
  return cloned
 handle=mod.register_forward_hook(replace)
 try:
  with torch.inference_mode():
   logits=model(x)
 finally:
  handle.remove()
 if len(mod._forward_hooks)!=before or counter[0]!=1 or \
    tuple(logits.shape)!=(1,2) or not bool(torch.isfinite(logits).all()):
  raise RuntimeError("04D real post-block hook count / cleanup / model output check FAILED")
 return logits.detach().cpu().numpy().reshape(-1).astype(np.float64).tolist()

def assert_logits(actual,expected,context,tol=TOL):
 u=np.asarray(actual,dtype=float);v=np.asarray(expected,dtype=float)
 if u.shape!=(2,) or v.shape!=(2,) or not np.isfinite(u).all() or \
    not np.isfinite(v).all() or float(np.max(np.abs(u-v)))>tol:
  raise RuntimeError("04D "+context+" ORIGINAL FROZEN GRAPH INVARIANT FAILED; "+
                     str(float(np.max(np.abs(u-v))) if u.shape==v.shape else "SHAPE"))
 return float(np.max(np.abs(u-v)))
