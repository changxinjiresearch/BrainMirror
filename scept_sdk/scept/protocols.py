"""A4 local protocol SDK and preregistered numerical U10 contrast.

Scientific scope: frozen-model numerical dependence ONLY, NOT clinical or
causally identified model explanation. Local Python plugins are trusted code,
not sandboxed browser scripts.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
import math
import statistics
from typing import Callable, Any
import numpy as np
from .contracts import fail, ensure_hash, tensor_array

class ProtocolRegistry:
    def __init__(self): self._registered = {}
    def register(self, name: str, version: str, fn: Callable):
        if not isinstance(name,str) or not name or not isinstance(version,str) or not version or not callable(fn):
            fail("INVALID_PLUGIN","Versioned callable protocol required")
        if name in self._registered: fail("DUPLICATE_PLUGIN",name)
        self._registered[name]=(version,fn)
    def run(self,name:str,**kwargs):
        if name not in self._registered:fail("PLUGIN_NOT_FOUND",name)
        return self._registered[name][1](**kwargs)
    def list_protocols(self):
        return tuple({"name":n,"version":v} for n,(v,_) in self._registered.items())

@dataclass(frozen=True)
class Mask:
    input_name: str
    shape: tuple[int,...]
    indices: tuple[int,...]
    source_id: str
    indices_sha256: str

    @classmethod
    def from_indices(cls,input_name: str,shape: tuple[int,...],
                     indices: tuple[int,...],source_id: str):
        if not isinstance(indices,tuple) or not indices or any(
            not isinstance(x,int) or isinstance(x,bool) or x<0 or x>0xffffffff for x in indices):
            fail("MASK_CONTRACT","Sorted nonempty uint32 tuple required")
        raw=np.asarray(indices,dtype="<u4").tobytes()
        return cls(input_name,tuple(shape),indices,source_id,sha256(raw).hexdigest())

    def validate(self, name,shape):
        if self.input_name!=name or self.shape!=tuple(shape) or not self.source_id or len(self.source_id)>160:
            fail("MASK_CONTRACT","Real input/shape/source mismatch")
        if len(self.indices)>100000 or tuple(sorted(set(self.indices)))!=self.indices or any(
            i<0 or i>=math.prod(shape) for i in self.indices):
            fail("MASK_CONTRACT","Mask indices must be sorted unique and in bounds")
        actual=sha256(np.asarray(self.indices,dtype="<u4").tobytes()).hexdigest()
        if actual!=ensure_hash(self.indices_sha256,"mask"):fail("MASK_HASH_MISMATCH","Index hash mismatch")

@dataclass(frozen=True)
class U10Plan:
    input_name: str
    target_tensor: str
    target_index: int
    target_mask: Mask
    control_mask: Mask
    model_sha256: str
    strengths: tuple[float,...]=(0.,.5,1.)
    nuisance_seeds: tuple[int,...]=(109,211,307)
    nuisance_sigma: float=.001
    fill: float=0.
    mean_tolerance: float=.02
    std_tolerance: float=.02
    output_scale: float=1.
    min_target_effect: float=.01
    min_specificity: float=.005
    min_positive_fraction: float=1.
    max_spread: float=.02
    minimum_sign_agreement: float=1.
    dose_tolerance: float=.005
    null_tolerance: float=.00001
    expected_sign: int | None=None

    def validate(self,shape):
        ensure_hash(self.model_sha256,"model")
        if not self.input_name or not self.target_tensor or not isinstance(self.target_index,int) or self.target_index<0:
            fail("INVALID_PLAN","Named input and true numeric target required")
        self.target_mask.validate(self.input_name,shape)
        self.control_mask.validate(self.input_name,shape)
        if len(self.target_mask.indices)!=len(self.control_mask.indices) or set(self.target_mask.indices)&set(self.control_mask.indices):
            fail("CONTROL_UNMATCHED","Masks must be disjoint and equal-count")
        if not 3<=len(self.strengths)<=10 or self.strengths[0]!=0 or self.strengths[-1]!=1 or any(
            not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 or x>1 or
            (i and x<=self.strengths[i-1]) for i,x in enumerate(self.strengths)):
            fail("INVALID_PLAN","Locked distinct ordered 0..1 dose levels (3..10) required")
        if not 3<=len(self.nuisance_seeds)<=12 or len(set(self.nuisance_seeds))!=len(self.nuisance_seeds) or any(
            not isinstance(x,int) or isinstance(x,bool) or not 0<=x<=0xffffffff for x in self.nuisance_seeds):
            fail("INVALID_PLAN","3..12 distinct uint32 nuisance seeds required")
        if len(self.strengths)*len(self.nuisance_seeds)>128:fail("RESOURCE_LIMIT","Study exceeds 128 cells")
        for name,lo,hi in [
            ("nuisance_sigma",0,.1),("mean_tolerance",0,100),("std_tolerance",0,100),
            ("output_scale",1e-12,1e12),("min_target_effect",0,1e5),
            ("min_specificity",0,1e5),("min_positive_fraction",0,1),
            ("max_spread",0,1e5),("minimum_sign_agreement",0,1),
            ("dose_tolerance",0,1e5),("null_tolerance",0,1e5)]:
            x=getattr(self,name)
            if not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi:
                fail("INVALID_PLAN",name+" outside preregistered bounds")
        if self.expected_sign not in (None,1,-1) or not math.isfinite(self.fill):
            fail("INVALID_PLAN","Invalid sign or fill")
 
def _u32(x): return x&0xffffffff
def _imul(a,b):return (a*b)&0xffffffff

def js_uniform(seed:int):
    """Mulberry32 precisely matching JS U10 core unsigned 32-bit steps."""
    state=_u32(seed)
    while True:
        state=_u32(state+0x6d2b79f5)
        t=state
        t=_imul(t^(t>>15),t|1)
        t=_u32(t^_u32(t+_imul(t^(t>>7),t|61)))
        yield _u32(t^(t>>14))/4294967296.

def js_nuisance(seed:int,n:int):
    rng=js_uniform(seed)
    values=np.empty(n,dtype=np.float64)
    for i in range(n):
        u=max(next(rng),float.fromhex("0x0.0000000000001p-1022"))
        v=next(rng)
        values[i]=math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)
    return values

def paired_inputs(source,plan:U10Plan,*,seed:int,strength:float):
    """Unchanged A, identical seeded nuisance to B/C, separate ROI operations."""
    a,dtype=tensor_array(source)
    if dtype not in ("float32","float64"):fail("INVALID_INPUT","Float input required")
    plan.validate(a.shape)
    if seed not in plan.nuisance_seeds or strength not in plan.strengths:
        fail("UNREGISTERED_CELL","Dose/seed was not preregistered")
    b=a.copy();c=a.copy()
    z=js_nuisance(seed,a.size)
    src=a.reshape(-1);p=b.reshape(-1);q=c.reshape(-1)
    for i in range(a.size):
        shifted=float(src[i])+plan.nuisance_sigma*float(z[i])
        p[i]=shifted;q[i]=shifted
    ti=np.asarray(plan.target_mask.indices,dtype=np.int64)
    ci=np.asarray(plan.control_mask.indices,dtype=np.int64)
    def stats(array,idx):
        vals=array[idx].astype(np.float64)
        mean=float(np.sum(vals,dtype=np.float64))/len(vals)
        var=max(0.,float(np.sum(vals*vals,dtype=np.float64))/len(vals)-mean*mean)
        return mean,math.sqrt(var)
    tm,ts=stats(p,ti);cm,cs=stats(q,ci)
    md,sd=abs(tm-cm),abs(ts-cs)
    if md>plan.mean_tolerance or sd>plan.std_tolerance:
        fail("CONTROL_UNMATCHED","True nuisance-adjusted ROI/control mean/std mismatch")
    for idx in ti:
        v=float(p[idx]);p[idx]=v+strength*(plan.fill-v)
    for idx in ci:
        v=float(q[idx]);q[idx]=v+strength*(plan.fill-v)
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        fail("NONFINITE_INPUT","Actual B/C overflow")
    return a,b,c,{"meanDiff":md,"stdDiff":sd,"targetMaskSha256":plan.target_mask.indices_sha256,
                  "controlMaskSha256":plan.control_mask.indices_sha256,"nuisanceSeed":seed,
                  "strength":strength,"nuisanceSigma":plan.nuisance_sigma}

def _sign(v):return (v>0)-(v<0)
def _median(v):return float(statistics.median(v))
def u10_gates(plan:U10Plan,rows:list[dict[str,Any]]):
    """Seven frozen JS U10 gate estimands, empty/duplicated grid NOT_EVALUABLE."""
    base={"schema":"scept.u10.gates.v1",
          "scientificStatus":"NUMERIC_PREDECLARED_COMPUTATIONAL_CRITERIA_ONLY",
          "causal":"NOT_ESTABLISHED","clinical":"NOT_ESTABLISHED"}
    def no(msg):return {**base,"verdict":"not_evaluable","reason":msg,
                        "gates":[],"nTasks":len(rows)}
    expected=len(plan.nuisance_seeds)*len(plan.strengths)
    if len(rows)!=expected:return no("Incomplete preregistered seed×dose matrix")
    mat={}
    for row in rows:
        k=(row.get("seed"),row.get("dose"))
        if k in mat or k[0] not in plan.nuisance_seeds or k[1] not in plan.strengths:
            return no("Duplicate or unexpected cell")
        if not row.get("genuineInference") or not row.get("modelIdentityVerified"):
            return no("Unverified A/B/C true runtime evidence")
        info=row.get("matching")
        if not isinstance(info,dict) or info.get("targetMaskSha256")!=plan.target_mask.indices_sha256 or info.get("controlMaskSha256")!=plan.control_mask.indices_sha256 or info.get("nuisanceSigma")!=plan.nuisance_sigma or info.get("meanDiff",float("inf"))>plan.mean_tolerance or info.get("stdDiff",float("inf"))>plan.std_tolerance:
            return no("Mismatched masks, nuisance or numeric matching across cells")
        if not all(isinstance(row.get(v),(int,float)) and math.isfinite(row[v]) for v in
                   ("score","targetedEffect","matchedControlEffect","controlledMagnitude",
                    "controlledSignedDelta","targetSignedDelta")):
            return no("Nonfinite/missing actual target statistic")
        if not isinstance(row.get("runIds"),list) or len(row["runIds"])!=3 or len(set(row["runIds"]))!=3:
            return no("No three distinct real A/B/C run identities")
        mat[k]=row
    if any((s,d) not in mat for s in plan.nuisance_seeds for d in plan.strengths):
        return no("Missing dose for nuisance seed")
    full=[mat[(s,1.)] for s in plan.nuisance_seeds]
    controlled=[x["controlledSignedDelta"] for x in full]
    magnitude=[x["controlledMagnitude"] for x in full]
    scores=[x["score"] for x in full]
    mcontrol=_median(magnitude);mscore=_median(scores)
    mtarget=_median([x["targetedEffect"] for x in full])
    pos=sum(x>0 and x>=plan.min_specificity for x in scores)/len(scores)
    signed=_median(controlled);spread=max(abs(x-signed) for x in controlled)
    sign=plan.expected_sign if plan.expected_sign is not None else _sign(signed)
    agreement=sum(_sign(x)==sign for x in controlled)/len(controlled)
    violations=sum(mat[(s,plan.strengths[i])]["controlledMagnitude"]+plan.dose_tolerance<
                   mat[(s,plan.strengths[i-1])]["controlledMagnitude"]
                   for s in plan.nuisance_seeds for i in range(1,len(plan.strengths)))
    null=max(mat[(s,0.)]["controlledMagnitude"] for s in plan.nuisance_seeds)
    checks=[("paired-control-specificity",mscore>=plan.min_specificity and mscore>0 and mtarget>0),
        ("controlled-intervention-effect",mcontrol>=plan.min_target_effect and mcontrol>0),
        ("replicate-specificity",pos>=plan.min_positive_fraction),
        ("nuisance-effect-stability",spread<=plan.max_spread),
        ("nuisance-sign-agreement",sign!=0 and signed!=0 and agreement>=plan.minimum_sign_agreement),
        ("graded-sensitivity",violations==0),("zero-dose-null",null<=plan.null_tolerance)]
    return {**base,"verdict":"criterion_met" if all(ok for _,ok in checks) else "criterion_not_met",
        "gates":[{"name":n,"verdict":"criterion_met" if ok else "criterion_not_met"} for n,ok in checks],
        "nTasks":len(rows),"nReplicates":len(plan.nuisance_seeds),
        "strengths":list(plan.strengths),"seeds":list(plan.nuisance_seeds),
        "metrics":{"medianTarget":mtarget,"medianControlled":mcontrol,
          "medianSpecificity":mscore,"positiveFraction":pos,"effectSpread":spread,
          "medianSigned":signed,"signAgreement":agreement,"doseViolations":violations,
          "maxZeroDose":null,"medianControlAtFullDose":_median(
              [x["matchedControlEffect"] for x in full]),"outputScale":plan.output_scale},
        "notes":["Numerical preregistered threshold gates, not statistical p-values",
                 "A unchanged; same nuisance in B/C only; no clinical causal claim"]}
 
def run_u10(*,observer,input_tensor,plan:U10Plan):
    """Genuine three forward A/B/C per seed-dose cell, no fake results."""
    a,dtype=tensor_array(input_tensor)
    if dtype not in ("float32","float64"):fail("INVALID_INPUT","Float input required")
    plan.validate(a.shape)
    if observer.backend=="onnx":
        if tuple(observer._adapter.inputs)!=(plan.input_name,):
            fail("INPUT_MISMATCH","Protocol must target actual ONNX input name")
        model_sha=observer._adapter.model_sha256
    elif observer.backend=="torch":
        if plan.input_name!="input":fail("INPUT_MISMATCH","PyTorch positional input is called input")
        from .adapters import _weights_sha
        model_sha=_weights_sha(observer._adapter.model)
    else: fail("INVALID_BACKEND","Unsupported real model adapter")
    if model_sha!=ensure_hash(plan.model_sha256,"model"):
        fail("MODEL_MISMATCH","Preregistered model fingerprint differs from frozen model")
    locked=sha256(json.dumps(asdict(plan),sort_keys=True).encode()).hexdigest()
    if plan.target_tensor not in getattr(observer._adapter,"probes",()):
        # Public outputs are captured even if they are not included in named probes.
        if observer.backend=="onnx" and plan.target_tensor not in observer._adapter.outputs:
            fail("TARGET_NOT_OBSERVED","Target is not an actual runtime output")
        if observer.backend=="torch" and not plan.target_tensor.startswith("model."):
            fail("TARGET_NOT_OBSERVED","Only actual configured modules/model output accepted")
    rows=[]
    for seed in plan.nuisance_seeds:
        for dose in plan.strengths:
            A,B,C,diagnostics=paired_inputs(a,plan,seed=seed,strength=dose)
            def feed(value):
                if observer.backend=="onnx": return {plan.input_name:value}
                import torch
                params=list(observer._adapter.model.parameters())
                device=params[0].device if params else "cpu"
                return torch.from_numpy(value.copy()).to(device=device)
            runs=[observer.observe(feed(v),
                       preprocess_id="u10-locked-numeric-v1",
                       preprocess_config={"nuisance_seed":seed,"dose_basis_points":round(dose*10000)},
                       model_id="preregistered-u10") for v in (A,B,C)]
            if any(r.record["provenance"]["model"]["sha256"]!=model_sha for r in runs):
                fail("MODEL_MUTATION","Model state did not remain frozen across triple")
            if any(plan.target_tensor not in r.tensors for r in runs):
                fail("TARGET_NOT_OBSERVED","User-selected actual model output missing")
            actual=[r.tensor(plan.target_tensor).reshape(-1) for r in runs]
            if any(plan.target_index>=len(x) for x in actual):
                fail("TARGET_NOT_OBSERVED","Model target index outside measured output")
            x,y,z=(float(v[plan.target_index]) for v in actual)
            dp=(y-x)/plan.output_scale;dq=(z-x)/plan.output_scale
            contrast=dp-dq
            score=(abs(y-x)-abs(z-x))/plan.output_scale
            if not all(map(math.isfinite,(dp,dq,contrast,score))):
                fail("NONFINITE","Actual model target overflow")
            layer_responses=[]
            for probe in observer.layers:
                if any(probe not in r.tensors for r in runs):
                    fail("OBSERVATIONS_REQUIRED","Internal selected layer missing in A/B/C")
                original,case,control=[r.tensor(probe).astype(np.float64) for r in runs]
                diff_b=float(np.mean(np.abs(case-original)))
                diff_c=float(np.mean(np.abs(control-original)))
                layer_responses.append({"tensor":probe,"shape":list(original.shape),
                    "targetedMeanAbsoluteChange":diff_b,
                    "matchedControlMeanAbsoluteChange":diff_c,
                    "matchedDifference":diff_b-diff_c,
                    "status":"actual-numeric-internal-response-only"})
            rows.append({"seed":seed,"dose":dose,"baseline":x,"target":y,"control":z,
             "layerResponses":layer_responses,
             "targetedEffect":abs(dp),"matchedControlEffect":abs(dq),
             "targetSignedDelta":dp,"controlSignedDelta":dq,
             "controlledSignedDelta":contrast,"controlledMagnitude":abs(contrast),
             "score":score,"genuineInference":True,"modelIdentityVerified":True,
             "runIds":[r.run_id for r in runs],
             "inputHashes":[r.record["provenance"]["inputs"] for r in runs],
             "matching":diagnostics})
    return {"schema":"scept.python.u10.study.v1",
        "protocol":"SCEPT-Evidential-Dependence-v1","lockedPlanSha256":locked,
        "plan":asdict(plan),"modelSha256":model_sha,
        "nGenuineForwards":len(rows)*3,"cells":rows,"gates":u10_gates(plan,rows),
        "science":{"status":"NUMERIC_CONTROLLED_DEPENDENCE_ONLY",
            "causality":"NOT_ESTABLISHED","clinical":"NOT_ESTABLISHED",
            "outputImportance":"NOT_ESTABLISHED","performanceSuperiority":"NOT_ESTABLISHED"}}
