"""A3: model/preprocessing-bound empirical NAF reference calibration.

Implements existing JS V6.3 midrank (not probability, p value, or diagnosis).
Python adds explicit target-input exclusion and cohort attestations.
"""
from __future__ import annotations
import math
from typing import Any
from .contracts import fail

SCHEMA="scept.naf.reference.v1"

def _key(group: str,index:int)->str:
    return group+"\x01"+str(index)

def _input_signature(naf:dict[str,Any]):
    return tuple(sorted(naf["provenance"]["inputs"].items()))

def _valid(naf:dict[str,Any]):
    if not isinstance(naf,dict) or naf.get("schema")!="scept.naf.v1" or not naf.get("groups"):
        fail("INVALID_NAF","True measured NAF is required")
    for group in naf["groups"]:
        if group.get("status")!="measured" or not group.get("units"):
            fail("UNOBSERVED","Unmeasured layer cannot be calibrated")
    return naf

def build_reference(samples: list[dict[str,Any]], *, label: str,
                    domain: str,kind: str="within-volume",
                    patient_ids: tuple[str,...] | None=None) -> dict[str,Any]:
    """Require >=5 real distinct INPUT hashes and identical measured tensor shapes.

    independent-cohort is only a user-supplied sample independence assertion;
    nothing about images verifies biological independence automatically.
    """
    if not isinstance(samples,(list,tuple)) or len(samples)<5 or len(samples)>512 or \
       not isinstance(label,str) or not label.strip() or not isinstance(domain,str) or not domain.strip():
        fail("INVALID_REFERENCE","At least five actual NAFs and nonempty provenance label/domain required")
    if kind not in ("within-volume","independent-cohort"):
        fail("INVALID_REFERENCE","Explicit reference domain kind required")
    if kind=="independent-cohort" and (not isinstance(patient_ids,(tuple,list)) or
          len(patient_ids)!=len(samples) or len(set(patient_ids))!=len(samples) or
          any(not isinstance(v,str) or not v for v in patient_ids)):
        fail("COHORT_NOT_ATTESTED","Distinct declared sample identities needed; not proof of clinical independence")
    first=_valid(samples[0])
    model=first["provenance"]["model"]["sha256"]
    preprocessing=first["provenance"]["preprocessing"]["sha256"]
    profile=first["profile"]
    def axes(s):
        return [(g["tensor"],tuple(g["shape"]),g["role"]) for g in s["groups"]]
    signature=axes(first)
    seen_inputs=set()
    dist:dict[str,list[float]]={}
    for s in samples:
        _valid(s)
        if s["profile"]!=profile or s["provenance"]["model"]["sha256"]!=model or \
           s["provenance"]["preprocessing"]["sha256"]!=preprocessing or axes(s)!=signature:
            fail("INCOMPATIBLE_REFERENCE","Model, preprocessing, actual tensor shape or axis profile differ")
        sig=_input_signature(s)
        if sig in seen_inputs:fail("DUPLICATE_INPUT","Repeated exact input cannot count as new reference sample")
        seen_inputs.add(sig)
        for g in s["groups"]:
            for u in g["units"]:
                x=u["rms"]
                if not isinstance(x,(float,int)) or not math.isfinite(x) or x<0:
                    fail("INVALID_NAF","Finite positive RMS required")
                dist.setdefault(_key(g["tensor"],u["index"]),[]).append(float(x))
    if any(len(v)!=len(samples) for v in dist.values()):
        fail("INCOMPATIBLE_REFERENCE","Missing or mismatched unit indices in reference samples")
    return {"schema":SCHEMA,"version":"6.3.0-python-a3",
            "modelSha256":model,"preprocessSha256":preprocessing,
            "profile":profile,"label":label,"domain":domain,"kind":kind,
            "nSamples":len(samples),"distributions":{k:sorted(v) for k,v in dist.items()},
            "inputSignatures":[[list(pair) for pair in signature] for signature in sorted(seen_inputs)],
            "provenanceAttestation":"USER_DECLARED_NOT_EXTERNALLY_VERIFIED" if kind=="independent-cohort"
                                      else "DISTINCT_HASHES_WITHIN_VOLUME_ONLY",
            "scientificMeaning":(
                "descriptive within-volume local slice rank; NOT population reference and NOT clinical abnormality"
                if kind=="within-volume" else
                "empirical rank versus declared independent samples; NOT significance or pathology probability")}

def empirical(x:float,sorted_values:list[float] | None):
    if not isinstance(x,(int,float)) or not math.isfinite(x) or \
       not isinstance(sorted_values,list) or len(sorted_values)<5:
        return None
    smaller=sum(v<x for v in sorted_values)
    equal=sum(v==x for v in sorted_values)
    return (smaller+.5*equal)/len(sorted_values)

def calibrate(naf:dict[str,Any],reference:dict[str,Any]):
    _valid(naf)
    if not isinstance(reference,dict) or reference.get("schema")!=SCHEMA or \
       reference["profile"]!=naf["profile"] or \
       reference["modelSha256"]!=naf["provenance"]["model"]["sha256"] or \
       reference["preprocessSha256"]!=naf["provenance"]["preprocessing"]["sha256"]:
        fail("INCOMPATIBLE_REFERENCE","Exact model/preprocessing/profile match required")
    signatures=reference.get("inputSignatures")
    if not isinstance(signatures,list) or len(signatures)<5 or len(signatures)!=reference.get("nSamples"):
        fail("REFERENCE_PROVENANCE","Reference must contain five other actual source input signatures")
    if [list(pair) for pair in _input_signature(naf)] in signatures:
        fail("REFERENCE_LEAKAGE","Target input must be excluded from other-sample reference")
    groups=[]
    for g in naf["groups"]:
        if g["status"]!="measured":
            fail("UNOBSERVED","Unmeasured layer cannot be calibrated")
        new=dict(g)
        new["units"]=[{**u,"empiricalRank":empirical(u["rms"],
          reference["distributions"].get(_key(g["tensor"],u["index"])))}
          for u in g["units"]]
        groups.append(new)
    return groups

def scale(value:float,*,mode:str="raw",raw_max:float=2.,reference_rank:float | None=None):
    if not math.isfinite(value) or value<0:fail("INVALID_RMS","RMS must be finite and >=0")
    if mode=="raw":
        if not math.isfinite(raw_max) or raw_max<=0:fail("INVALID_SCALE","Fixed rawMax > 0 required")
        return min(1.,value/raw_max)
    if mode=="reference":
        if reference_rank is None or not math.isfinite(reference_rank) or not 0<=reference_rank<=1:
            fail("INVALID_REFERENCE","True empirical reference rank required")
        return reference_rank
    fail("UNSUPPORTED_CALIBRATION","Within-run autoscaling is not valid scientific reference")
