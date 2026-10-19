"""A3: real PyTorch/ONNX NAF and cross-language JS numerical parity."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from scept import SCEPT
from scept.naf import fingerprint, pattern
from scept.contracts import SCEPTError

ROOT=Path(__file__).resolve().parents[2]
JS=r"""
const fs=require('fs'),O=require('./scept-web/core/scept_observation.js');
const N=require('./scept-web/core/scept_naf.js');
const x=JSON.parse(fs.readFileSync(0,'utf8'));
const t={type:'float32',dims:x.shape,data:Float32Array.from(x.values)};
const desc={outputs:[{name:'real-probe',type:'float32',shape:x.shape}]};
const a=O.observeOnce({description:desc,outputs:{'real-probe':t},
  probes:['real-probe'],profileId:x.profile,maxAxisUnits:2048});
const record={observations:a};
const group=N.units(record,x.profile,'real-probe');
const cap={record:{schema:'scept.run-record.v1'},
  getTensor:()=>t};
const p=x.unit!==null?N.getPattern(cap,'real-probe',x.unit,x.profile):null;
console.log(JSON.stringify({group,pattern:p}));
"""
def browser_numeric(array,profile,unit=0):
    args={'shape':list(array.shape),'values':array.reshape(-1).tolist(),
          'profile':profile,'unit':unit}
    p=subprocess.run(['node','-e',JS],input=json.dumps(args),cwd=ROOT,
                     text=True,capture_output=True,check=True)
    return json.loads(p.stdout)

class ViT(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(8,8,bias=False)
        self.calls=0
    def forward(self,x):
        self.calls+=1
        return self.linear(x)

def test_real_vit_naf_matches_original_js_units_and_patterns():
    torch.manual_seed(721)
    model=ViT()
    r=SCEPT(model,['linear'],backend='torch',profile='vit-token-channel').observe(
        torch.randn(1,5,8))
    assert model.calls==1
    naf=r.fingerprint()
    g=naf['groups'][0]
    expected=browser_numeric(r.tensor('linear'),'vit-token-channel',3)
    assert g['status']=='measured' and g['role']=='token'
    assert len(g['units'])==5
    for a,b in zip(g['units'],expected['group']['units']):
        assert a['index']==b['index'] and a['count']==b['count']
        for k in ['rms','mean','positiveFraction','negativeFraction']:
            assert a[k]==pytest.approx(b[k],abs=1e-10,rel=1e-10)
    p=r.pattern('linear',3)
    assert p['values']==pytest.approx(expected['pattern']['values'],abs=0)
    assert len(p['values'])==8
    assert p['provenance']=='verified_same_forward_private_runtime_tensor'
    assert naf['science']['causality']=='NOT_ESTABLISHED'
    assert r.representation_update('linear','model.output',
        coordinate_semantics='same-residual-stream',
        verified_identity=True,declared_by='unverified-user-guess')['status']=='not_evaluable'
    with pytest.raises(SCEPTError,match='UNOBSERVED'):
        r.pattern('nonexistent',0)

def test_actual_cnn_nchw_channel_pattern_and_js_axis_parity():
    model=nn.Sequential(nn.Conv2d(1,3,3,padding=1),nn.ReLU())
    r=SCEPT(model,['0'],backend='torch',profile='cnn-nchw').observe(
        torch.randn(1,1,4,4))
    a=r.tensor('0');g=r.fingerprint()['groups'][0]
    js=browser_numeric(a,'cnn-nchw',1)
    assert g['role']=='channel'
    assert len(g['units'])==3
    for x,y in zip(g['units'],js['group']['units']):
        assert x['rms']==pytest.approx(y['rms'],abs=1e-10)
        assert x['mean']==pytest.approx(y['mean'],abs=1e-10)
    assert r.pattern('0',1)['values']==pytest.approx(js['pattern']['values'],abs=0)

def test_real_onnx_graph_qualified_update():
    import onnx
    from onnx import helper,TensorProto
    x=helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,5,4])
    y=helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,5,4])
    # Producer/consumer path from measured residual_a to residual_b.
    nodes=[helper.make_node('Identity',['input'],['residual_a']),
           helper.make_node('Identity',['residual_a'],['residual_b']),
           helper.make_node('Identity',['residual_b'],['output'])]
    model=helper.make_model(helper.make_graph(nodes,'real-residual',[x],[y]),
        opset_imports=[helper.make_operatorsetid('',13)])
    model.ir_version=9
    r=SCEPT(model.SerializeToString(),['residual_a','residual_b'],
            backend='onnx',profile='vit-token-channel').observe(
        np.arange(20,dtype=np.float32).reshape(1,5,4))
    invalid=r.representation_update('residual_a','residual_b')
    assert invalid['status']=='not_evaluable'
    measured=r.representation_update('residual_a','residual_b',
        coordinate_semantics='same-residual-stream',
        verified_identity=True,declared_by='scientifically-declared-residual-adapter')
    assert measured['status']=='measured'
    assert len(measured['units'])==5
    assert all(u['rmsUpdate']==0 for u in measured['units'])
    reverse=r.representation_update('residual_b','residual_a',
        coordinate_semantics='same-residual-stream',
        verified_identity=True,declared_by='scientifically-declared-residual-adapter')
    assert reverse['status']=='not_evaluable'
    assert reverse['reason']=='no_real_computation_path'

def test_generic_profile_explicit_unavailable_not_fabricated():
    model=nn.Linear(3,2)
    # Observe named module's actual output from Sequential, generic no role.
    r=SCEPT(nn.Sequential(model),['0'],backend='torch',profile='generic').observe(
        torch.ones(1,3))
    group=r.fingerprint()['groups'][0]
    assert group['status']=='unsupported' and not group['units']
