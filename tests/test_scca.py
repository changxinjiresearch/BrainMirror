"""SCCA engineering falsification tests: synthetic-only, deterministic CPU."""
import math
import pytest
import torch
from torch import nn

from brainmirror.scept import TransformerVIB3D, Region
from brainmirror.scept.scca import (SCCAConfig, explain_scca, gradient_screen,
    valid_declared_edges, reference_volume)
from brainmirror.scept.trace import captured_forward, allowed_modules


torch.set_num_threads(2)


class KnownDependency(nn.Module):
    """Known functional edge; a decoy exists but has zero effect by design."""
    def __init__(self):
        super().__init__()
        self.signal_to_residual = nn.Identity()
        self.decoy_to_residual = nn.Identity()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def scca_edge_routes(self):
        return {
            'signal_to_residual': {'source':'signal', 'consumer':'prediction_add',
                                   'intervention_unit':'whole_branch_tensor'},
            'decoy_to_residual': {'source':'decoy', 'consumer':'prediction_add',
                                  'intervention_unit':'whole_branch_tensor'},
        }

    def forward(self, x, sample=False, return_details=False):
        primary = self.signal_to_residual(x[:, 0:1] * self.scale)
        decoy = self.decoy_to_residual(x[:, 1:2])
        score = primary + 0.0 * decoy
        output = torch.cat([torch.zeros_like(score), torch.zeros_like(score), score], dim=1)
        return (output, {}) if return_details else output


@pytest.fixture
def known_case():
    x = torch.zeros(1, 4, 8, 8, 8)
    x[:, 0, 2:6, 2:6, 2:6] = 2.0
    roi = torch.zeros(8,8,8,dtype=torch.bool)
    roi[2:6,2:6,2:6] = True
    return KnownDependency().eval(), x, roi, Region(2,6,2,6,2,6)


def test_ground_truth_signal_not_decoy_and_full_recovery(known_case):
    model, x, roi, region = known_case
    result = explain_scca(model, x, region, roi, SCCAConfig(channel=2))
    assert result['status'] == 'candidate_selected_not_independently_validated'
    assert result['input_effect'] > 1.0
    assert result['selected']['edges'] == ['signal_to_residual']
    assert abs(result['selected']['disruption']-result['input_effect']) < 1e-6
    assert abs(result['selected']['restoration']-result['input_effect']) < 1e-6
    assert result['selected']['relative_restoration_error'] < 1e-6
    assert result['gradient_screen']['decoy_to_residual'] == 0
    assert result['validation']['status'] == 'evaluated_on_reserved_reference'
    assert result['validation']['direction_aligned'] is True
    assert result['independent_patient_validation'] is False
    assert result['selected']['evidence_status'] == 'intervened_selection_only'


def test_zero_effect_fail_closed(known_case):
    model, x, roi, region = known_case
    x.zero_()
    result = explain_scca(model, x, region, roi)
    assert result['status'] == 'unresolved_low_input_effect'
    assert result['selected'] is None
    assert result['tested_edge_sets'] == []


def test_reference_uses_only_selected_region(known_case):
    _, x, _, r = known_case
    for strategy in ('channel_mean','local_blur'):
        alt = reference_volume(x,r,strategy)
        assert alt.shape == x.shape
        outside = torch.ones((8,8,8),dtype=torch.bool)
        outside[2:6,2:6,2:6] = False
        assert torch.equal(alt[...,outside],x[...,outside])
    with pytest.raises(ValueError): reference_volume(x,r,'clinical_fake')


def test_config_input_and_invalid_reference_fail_closed(known_case):
    m,x,roi,r = known_case
    with pytest.raises(ValueError): explain_scca(m,x,r,roi,validation_strategy='channel_mean')
    with pytest.raises(ValueError): explain_scca(m,x,r,roi,SCCAConfig(max_candidates=0))
    with pytest.raises(ValueError): explain_scca(m,x,r,torch.ones_like(roi))
    with pytest.raises(ValueError): explain_scca(m,x,r,roi,SCCAConfig(channel=4))


def test_no_artificial_causal_paths_and_no_hook_leakage():
    torch.manual_seed(101)
    model=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    roi=torch.zeros((8,8,8),dtype=torch.bool);roi[2:6,2:6,2:6]=True
    edges=valid_declared_edges(model)
    assert len(edges)==4
    assert all('residual_add' in v['consumer'] for v in edges.values())
    before,_trace,_=captured_forward(model,x)
    result=explain_scca(model,x,Region(2,6,2,6,2,6),roi,
                        SCCAConfig(max_circuit_size=2,beam_width=2))
    assert result['interpretation'].startswith('Single-consumer residual')
    if result['selected']:
        assert set(result['selected']['edges']) <= set(edges)
        assert len(result['selected']['edges']) <= 2
        assert result['selected']['evidence_status'] == 'intervened_selection_only'
        assert math.isfinite(result['selected']['objective'])
    after,_,_=captured_forward(model,x)
    assert torch.equal(before,after)
    assert all(len(mod._forward_hooks) == 0 for mod in model.modules())
    assert all(p.grad is None for p in model.parameters())


def test_exact_residual_edge_patch_and_noop():
    torch.manual_seed(7)
    model=TransformerVIB3D(dim=16,heads=4,depth=1,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    base,trace,_=captured_forward(model,x)
    edge='blocks.0.mlp_to_residual'
    noop,_,_=captured_forward(model,x,patch=(edge,trace[edge]))
    assert torch.equal(base,noop)
    replaced,_,_=captured_forward(model,x,patch=(edge,torch.zeros_like(trace[edge])))
    assert not torch.equal(base,replaced)
    attention_route='blocks.0.attn_to_residual'
    # Two independent residual branches can be patched in one forward pass.
    together,_,_=captured_forward(model,x,patch={edge:torch.zeros_like(trace[edge]),
        attention_route:torch.zeros_like(trace[attention_route])})
    assert not torch.equal(base,together)
    assert all(len(mod._forward_hooks)==0 for mod in model.modules())


def test_token_selection_patch_and_tensor_strictness():
    model=TransformerVIB3D(dim=16,heads=4,depth=1,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    base,trace,_=captured_forward(model,x)
    site='blocks.0.attn_to_residual'
    mask=torch.zeros(1,8,dtype=torch.bool)
    nothing,_,_=captured_forward(model,x,patch=(site,torch.zeros_like(trace[site]),mask))
    assert torch.equal(nothing,base)
    mask[0,0]=True
    changed,_,_=captured_forward(model,x,patch=(site,torch.zeros_like(trace[site]),mask))
    assert not torch.equal(changed,base)
    with pytest.raises(ValueError):
        captured_forward(model,x,patch=(site,trace[site].to(torch.float64)))
    with pytest.raises(ValueError):
        captured_forward(model,x,patch=(site,trace[site],torch.ones(1,8)))
    assert all(len(mod._forward_hooks)==0 for mod in model.modules())


def test_declared_edges_reject_uninstrumented_models():
    with pytest.raises(TypeError):valid_declared_edges(nn.Linear(3,3))


def test_gradient_screen_has_no_parameter_update(known_case):
    m,x,roi,r=known_case
    _,trace,_=captured_forward(m,x)
    xalt=reference_volume(x,r)
    old=m.scale.detach().clone()
    grads=gradient_screen(m,xalt,trace,list(m.scca_edge_routes()),roi,2)
    assert grads['signal_to_residual']>0
    assert grads['decoy_to_residual']==0
    assert torch.equal(m.scale,old)
    assert m.scale.grad is None


class TwoNecessaryRoutes(nn.Module):
    """Both explicitly wired branches contribute half of the target score."""
    def __init__(self):
        super().__init__()
        self.left_to_residual = nn.Identity()
        self.right_to_residual = nn.Identity()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def scca_edge_routes(self):
        return {name:{'source':name.split('_to_')[0], 'consumer':'joint_add',
                      'intervention_unit':'whole_branch_tensor'}
                for name in ('left_to_residual','right_to_residual')}

    def forward(self,x,sample=False,return_details=False):
        left=self.left_to_residual(x[:,0:1]*self.anchor*0.5)
        right=self.right_to_residual(x[:,1:2]*self.anchor*0.5)
        score=left+right
        output=torch.cat([torch.zeros_like(score),torch.zeros_like(score),score],1)
        return (output,{}) if return_details else output


def test_joint_search_recovers_both_contributing_edges():
    x=torch.zeros(1,4,8,8,8)
    x[:,0:2,2:6,2:6,2:6] = 2.0
    roi=torch.zeros(8,8,8,dtype=torch.bool);roi[2:6,2:6,2:6]=True
    result=explain_scca(TwoNecessaryRoutes().eval(),x,Region(2,6,2,6,2,6),roi,
                        SCCAConfig(max_circuit_size=2,beam_width=2))
    assert result['selected']['edges']==['left_to_residual','right_to_residual']
    assert abs(result['selected']['restoration']-result['input_effect'])<1e-6
    assert result['selected']['relative_disruption_error']<1e-6
    assert len(result['tested_edge_sets'])==3
    single=[a for a in result['tested_edge_sets'] if len(a['edges'])==1]
    assert all(abs(s['fractional_recovery']-.5)<1e-5 for s in single)


def test_negative_signed_effect_is_preserved(known_case):
    model,x,roi,region=known_case
    with torch.no_grad(): model.scale.fill_(-1.0)
    result=explain_scca(model,x,region,roi)
    assert result['input_effect']<0
    assert result['selected']['edges']==['signal_to_residual']
    assert result['selected']['restoration']<0
    assert result['selected']['direction_aligned'] is True
