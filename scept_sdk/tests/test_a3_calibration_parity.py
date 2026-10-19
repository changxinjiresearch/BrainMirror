"""A3 V6.3 reference calibration parity with production JS, no faux clinical cohort."""
import copy
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from scept import SCEPT, SCEPTError
from scept.calibration import build_reference, calibrate, empirical, scale

ROOT=Path(__file__).resolve().parents[2]

def test_true_independent_input_hashes_reference_and_js_empirical_midrank():
    torch.manual_seed(119)
    model=nn.Sequential(nn.Linear(4,3,bias=False), nn.ReLU())
    observer=SCEPT(model,["0"],backend="torch",profile="vit-token-channel")
    naf_samples=[]
    for i in range(6):
        inp=(torch.arange(20,dtype=torch.float32).reshape(1,5,4)+i*.0625)
        assert inp.shape==(1,5,4)
        naf_samples.append(observer.observe(inp).fingerprint())
    ref=build_reference(naf_samples[:5],label="sample-views",
                        domain="one synthetic numeric tensor family",kind="within-volume")
    assert ref["nSamples"]==5
    assert ref["provenanceAttestation"]=="DISTINCT_HASHES_WITHIN_VOLUME_ONLY"
    result=calibrate(naf_samples[5],ref)
    assert len(result[0]["units"])==5
    assert all(0<=x["empiricalRank"]<=1 for x in result[0]["units"])
    payload={"samples":naf_samples[:5],"target":naf_samples[5]}
    js=r"""const fs=require('fs'),C=require('./scept-web/core/scept_naf_calibration.js');
const inp=JSON.parse(fs.readFileSync(0,'utf8'));
const r=C.buildReference(inp.samples,{label:'sample-views',
 domain:'one synthetic numeric tensor family',kind:'within-volume'});
const units=C.calibrated(inp.target,r);
console.log(JSON.stringify({reference:r,units:units}));
"""
    proc=subprocess.run(['node','-e',js],cwd=ROOT,input=json.dumps(payload),
                        text=True,capture_output=True,check=True)
    actual=json.loads(proc.stdout)
    assert ref["distributions"]==actual["reference"]["distributions"]
    for row,expected in zip(result[0]["units"],actual["units"][0]["units"]):
        assert row["empiricalRank"]==pytest.approx(expected["empiricalRank"],abs=0)
    with pytest.raises(SCEPTError,match="REFERENCE_LEAKAGE"):
        calibrate(naf_samples[0],ref)
    ref_dup=naf_samples[:4]+[naf_samples[0]]
    with pytest.raises(SCEPTError,match="DUPLICATE_INPUT"):
        build_reference(ref_dup,label="dup",domain="same-numeric")
    bad=copy.deepcopy(naf_samples[1])
    bad["provenance"]["model"]["sha256"]="f"*64
    with pytest.raises(SCEPTError,match="INCOMPATIBLE_REFERENCE"):
        build_reference([naf_samples[0],bad,*naf_samples[2:5]],
                        label="wrong",domain="model")
    with pytest.raises(SCEPTError,match="COHORT_NOT_ATTESTED"):
        build_reference(naf_samples[:5],label="cohort",domain="unknown",
                        kind="independent-cohort")

def test_empirical_exact_ties_and_fixed_no_auto_scale():
    assert empirical(2.,[1.,2.,2.,2.,3.])==pytest.approx(.5)
    assert scale(3,raw_max=2)==1.
    assert scale(.25,raw_max=2)==.125
    assert scale(20,mode="reference",reference_rank=.75)==.75
    with pytest.raises(SCEPTError,match="UNSUPPORTED_CALIBRATION"):
        scale(1,mode="within-run-autoscale")
