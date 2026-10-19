"""Real, once-per-request PyTorch/ONNX adapters.

No invented probes or outputs. An A2 capture is NOT a validated explanation.
All arrays are actual copies from exactly one frozen inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

from .contracts import SCEPTError, fail, tensor_array, tensor_sha, MAX_CAPTURE_BYTES, MAX_TENSOR_ELEMENTS

def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("Install scept-research[torch] to observe PyTorch models") from e
    return torch

def _onnx():
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("Install scept-research[onnx] for ONNX graph observation") from e
    return onnx, ort

def _freeze(result: Any, *, label: str) -> tuple[np.ndarray, str]:
    arr, dtype = tensor_array(result)
    return arr, dtype

def _input_sources(x: Any, prefix: str = "input") -> dict[str, Any]:
    if isinstance(x, dict):
        if not x or any(not isinstance(k,str) or not k for k in x):
            fail("INPUT_MISMATCH", "Named nonempty model inputs required")
        return dict(x)
    return {prefix: x}

def _model_output_torch(result: Any) -> dict[str, Any]:
    torch = _torch()
    if isinstance(result, torch.Tensor):
        return {"output": result}
    if isinstance(result, (tuple, list)) and result and all(isinstance(x, torch.Tensor) for x in result):
        return {f"output.{i}": v for i,v in enumerate(result)}
    if isinstance(result, dict) and result and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k,v in result.items()):
        return {"output."+k: v for k,v in sorted(result.items())}
    fail("OUTPUT_UNSUPPORTED", "Model outputs must be Tensor / flat Tensor tuple / dict[str, Tensor]")

def _weights_sha(model: Any) -> str:
    """Fingerprint model's actual state, not entire Python model code or graph identity."""
    torch = _torch()
    d=hashlib.sha256()
    states=model.state_dict()
    if not states:
        fail("MODEL_PROVENANCE", "No frozen parameter/buffer identity; supply a stateful model in A2")
    for name,t in sorted(states.items()):
        # State may contain scalar BatchNorm counters and half/bfloat16 weights:
        # fingerprint every original state byte without pretending it is an
        # observable V6 activation dtype.
        raw=t.detach().cpu().contiguous()
        binary=raw.reshape(-1).view(torch.uint8).numpy().tobytes()
        descriptor=json.dumps({"name":name,"shape":list(t.shape),"dtype":str(t.dtype)},
                              sort_keys=True,separators=(",",":")).encode()
        d.update(descriptor+b"\0"+binary+b"\0")
    return d.hexdigest()

class PyTorchAdapter:
    def __init__(self, model: Any, probes: tuple[str,...], *, max_capture_bytes: int = MAX_CAPTURE_BYTES):
        torch = _torch()
        if not isinstance(model,torch.nn.Module):
            fail("INVALID_MODEL", "Provide a real torch.nn.Module")
        if not probes or len(probes)>64 or len(set(probes))!=len(probes):
            fail("INVALID_PROBES", "1..64 distinct module names")
        layers=dict(model.named_modules())
        if any(not p or p not in layers for p in probes):
            fail("PROBE_NOT_FOUND", f"Unknown actual module: {[p for p in probes if p not in layers]}")
        if not isinstance(max_capture_bytes,int) or max_capture_bytes<=0:
            fail("RESOURCE_LIMIT","Positive byte limit required")
        self.model=model
        self.layers=layers
        self.probes=tuple(probes)
        self.max_capture_bytes=max_capture_bytes

    def observe_once(self, inputs: Any) -> tuple[dict[str,tuple[np.ndarray,str]],
                                                       dict[str,tuple[np.ndarray,str]],str,dict[str,str]]:
        torch=_torch()
        original=_input_sources(inputs)
        if len(original)>16:
            fail("INPUT_MISMATCH","Max 16 named input tensors")
        if any(not isinstance(t,torch.Tensor) for t in original.values()):
            fail("INPUT_MISMATCH","PyTorch inputs must be actual torch.Tensor values")
        named={name:_freeze(t,label=name) for name,t in original.items()}
        if sum(v[0].nbytes for v in named.values())>self.max_capture_bytes:
            fail("RESOURCE_LIMIT","Actual input snapshots exceed permitted capture byte budget")
        original_hash={n:tensor_sha(*named[n]) for n in named}
        model_hash=_weights_sha(self.model)
        modules=list(self.model.modules())
        states=[m.training for m in modules]
        captured:dict[str,tuple[np.ndarray,str]]={}
        counts={name:0 for name in self.probes}
        handles=[]
        def hook(name):
            def callback(_module,_inp,output):
                counts[name]+=1
                if counts[name]>1:
                    fail("REPEATED_MODULE","Module "+name+" executed >1 time; select a unique call site or use ONNX")
                if not isinstance(output,torch.Tensor):
                    fail("PROBE_NOT_TENSOR","Probe "+name+" returns non-Tensor; select an actual tensor-producing module")
                captured[name]=_freeze(output,label=name)
            return callback
        try:
            for name in self.probes:
                handles.append(self.layers[name].register_forward_hook(hook(name)))
            self.model.eval()
            with torch.inference_mode():
                result=(self.model(**original) if isinstance(inputs,dict)
                        else self.model(inputs))
                result=_model_output_torch(result)
                outputs={"model."+k:_freeze(v,label=k) for k,v in result.items()}
        finally:
            for h in handles: h.remove()
            for module,was_training in zip(modules,states): module.training=was_training
        if any(counts[p]!=1 for p in self.probes):
            fail("PROBE_NOT_EXECUTED", "Some requested real module did not execute exactly once")
        if any(tensor_sha(*_freeze(original[n],label=n))!=original_hash[n] for n in original):
            fail("INPUT_MUTATION","Model modified its original tensor input")
        if _weights_sha(self.model)!=model_hash:
            fail("MODEL_MUTATION","Model state changed during a supposedly frozen forward")
        all_outputs={**outputs,**captured}
        if set(outputs).intersection(captured):
            fail("PROBE_COLLISION","Model output/module namespace collided")
        size=sum(a.nbytes for a,_ in [*named.values(),*all_outputs.values()])
        if size>self.max_capture_bytes:
            fail("RESOURCE_LIMIT",f"Actual input/output budget >{self.max_capture_bytes} bytes")
        return named,all_outputs,model_hash,{
            "engine":"PyTorch","version":torch.__version__,"provider":str(next(self.model.parameters()).device)
                if any(True for _ in self.model.parameters()) else "cpu",
        }

class ONNXAdapter:
    def __init__(self, source: str | bytes | Path, probes: tuple[str,...],
                 *, providers: tuple[str,...] = ("CPUExecutionProvider",),
                 max_capture_bytes: int = MAX_CAPTURE_BYTES):
        onnx,ort=_onnx()
        if not probes or len(probes)>64 or len(set(probes))!=len(probes):
            fail("INVALID_PROBES","1..64 distinct ONNX graph output/probe names")
        if not providers or any(not isinstance(p,str) for p in providers):
            fail("INVALID_PROVIDER","Explicit ONNX Runtime providers required")
        content=Path(source).read_bytes() if isinstance(source,(str,Path)) else bytes(source)
        if not content or len(content)>160*1024*1024:
            fail("RESOURCE_LIMIT","ONNX source missing or >160MiB")
        self.model_sha256=hashlib.sha256(content).hexdigest()
        graph=onnx.load_model_from_string(content)
        onnx.checker.check_model(graph)
        self.graph_edges=tuple((tuple(node.input),tuple(node.output)) for node in graph.graph.node)
        producers={out for node in graph.graph.node for out in node.output if out}
        graph_out={v.name for v in graph.graph.output}
        missing=[p for p in probes if p not in graph_out and p not in producers]
        if missing: fail("PROBE_NOT_FOUND","No real graph producer/output: "+repr(missing))
        # Work on an IN-MEMORY clone, without changing frozen source hash or disk model.
        try:
            inferred=onnx.shape_inference.infer_shapes(graph, strict_mode=False)
        except Exception:
            inferred=graph
        metadata={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.output)+list(inferred.graph.input)}
        graph=onnx.ModelProto.FromString(inferred.SerializeToString())
        for p in probes:
            if p in graph_out: continue
            value_info=metadata.get(p)
            if value_info is None or not value_info.type.HasField("tensor_type") or value_info.type.tensor_type.elem_type==0:
                fail("PROBE_TYPE_UNVERIFIED",
                     "ONNX inferred dtype missing for "+p+"; explicitly expose typed graph output upstream")
            graph.graph.output.add().CopyFrom(value_info)
        onnx.checker.check_model(graph)
        self.session=ort.InferenceSession(graph.SerializeToString(),providers=list(providers))
        self.probes=tuple(probes)
        self.max_capture_bytes=max_capture_bytes
        self.runtime={"engine":"onnxruntime","version":ort.__version__,
                      "provider":";".join(self.session.get_providers())}
        self.inputs=tuple(meta.name for meta in self.session.get_inputs())
        self.outputs=tuple(meta.name for meta in self.session.get_outputs())
        if any(p not in self.outputs for p in probes):
            fail("PROBE_NOT_FOUND","Real ONNX Runtime did not expose declared graph probe")
    def observe_once(self, inputs: dict[str,Any] | Any) -> tuple[dict[str,tuple[np.ndarray,str]],
                                                           dict[str,tuple[np.ndarray,str]],str,dict[str,str]]:
        if not isinstance(inputs,dict):
            if len(self.inputs)!=1:
                fail("INPUT_MISMATCH","Multi-input ONNX model requires dict[name,tensor]")
            inputs={self.inputs[0]:inputs}
        if set(inputs)!=set(self.inputs):
            fail("INPUT_MISMATCH","Must supply exactly real named ONNX inputs")
        source={name:_freeze(inputs[name],label=name) for name in self.inputs}
        if sum(v[0].nbytes for v in source.values())>self.max_capture_bytes:
            fail("RESOURCE_LIMIT","Actual ONNX inputs exceed capture budget before execution")
        original_hash={name:tensor_sha(*x) for name,x in source.items()}
        # ONNX Runtime CPU accepts numpy; no A/B or hidden reruns.
        feeds={name:a for name,(a,_dtype) in source.items()}
        results=self.session.run(list(self.outputs),feeds)
        if len(results)!=len(self.outputs):
            fail("OUTPUT_MISMATCH","ONNX Runtime returned fewer actual tensors")
        outputs={name:_freeze(value,label=name) for name,value in zip(self.outputs,results)}
        if any(tensor_sha(*_freeze(inputs[name],label=name))!=original_hash[name] for name in self.inputs):
            fail("INPUT_MUTATION","Caller input modified during ONNX run")
        size=sum(a.nbytes for a,_ in [*source.values(),*outputs.values()])
        if size>self.max_capture_bytes:
            fail("RESOURCE_LIMIT","Actual observed input+output bytes exceeds limit")
        return source,outputs,self.model_sha256,self.runtime
