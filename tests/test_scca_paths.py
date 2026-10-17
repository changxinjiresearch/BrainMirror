"""Intervention semantics, controlled oracle and negative controls; synthetic only."""
import math

import pytest
import torch

from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.explain import Region
from brainmirror.scept.paths import (
    PathHop, PathSearchConfig, capture_path_baseline,
    evaluate_path, explain_scca_paths, intervene_path, route_delta, validate_path)
from brainmirror.scept.trace import captured_forward

torch.set_num_threads(2)


@pytest.fixture
def case():
    torch.manual_seed(310)
    model=TransformerVIB3D(dim=16,heads=4,depth=3,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    ref=x.clone();ref[...,2:6,2:6,2:6]=0
    roi=torch.zeros(8,8,8,dtype=torch.bool);roi[2:6,2:6,2:6]=True
    return model,x,ref,roi


def no_hooks(model):
    assert all(len(m._forward_hooks)==0 and len(m._forward_pre_hooks)==0
               for m in model.modules())


def test_path_requires_real_forward_order_and_declared_edges(case):
    model,*_=case
    good=(PathHop('blocks.0.attn_to_residual'),PathHop('blocks.1.attn_to_residual','k'),
          PathHop('blocks.2.mlp_to_residual'))
    assert validate_path(model,good)==good
    for bad in [
        (PathHop('blocks.1.attn_to_residual'),),
        (PathHop('blocks.2.mlp_to_residual'),PathHop('blocks.1.attn_to_residual')),
        (PathHop('blocks.1.mlp_to_residual'),PathHop('blocks.1.attn_to_residual')),
        (PathHop('blocks.0.attn_to_residual'),PathHop('blocks.0.attn_to_residual')),
        (PathHop('blocks.0.attn_to_residual'),PathHop('blocks.1.mlp_to_residual','k')),
        (PathHop('blocks.0.attn_to_residual'),PathHop('blocks.1.attn_to_residual','bad')),
        (PathHop('blocks.0.attn_to_residual'),PathHop('blocks.9.mlp_to_residual')),
    ]:
        with pytest.raises(ValueError):validate_path(model,bad)


def test_cross_layer_exact_frozen_offpath_oracle(case):
    """An independent hook-driven hybrid run must match the path engine.

    Source gate = original clean activation on reference execution.
    All offpath branches between source and destination are frozen as reference;
    pass identity residual difference to target MLP. Inject only target output
    into reference forward, deliberately disallowing source's direct bypass.
    """
    model,x,ref,_=case
    hops=[PathHop('blocks.0.attn_to_residual'),PathHop('blocks.2.mlp_to_residual')]
    _,cg,_=capture_path_baseline(model,x)
    _,rg,rr=capture_path_baseline(model,ref)
    source=hops[0].edge;dest=hops[-1].edge
    _,delta=intervene_path(model,ref,hops,rg,rr,cg[source]-rg[source])
    # Independent, algebraic oracle of the strictly frozen offpath residual DAG:
    # x_target_pre_mlp^hybrid = x_target_pre_mlp^ref + source_delta.
    block=model.blocks[2]
    hybrid_input=rr['blocks.2.mlp']+(cg[source]-rg[source])
    exact_target=block.mlp(block.norm2(hybrid_input))
    expected_delta=exact_target-rg[dest]
    assert torch.allclose(delta,expected_delta,atol=1e-7,rtol=1e-5)
    patched,_,_=captured_forward(model,ref,patch=(dest,exact_target))
    real, _=intervene_path(model,ref,hops,rg,rr,cg[source]-rg[source])
    assert torch.allclose(real,patched,atol=1e-7,rtol=1e-6)
    no_hooks(model)


def test_attention_q_k_v_are_separate_real_mha_inputs(case):
    model,x,ref,_=case
    _,cg,_=capture_path_baseline(model,x)
    _,rg,rr=capture_path_baseline(model,ref)
    source='blocks.0.mlp_to_residual'
    target='blocks.1.attn_to_residual'
    diff=cg[source]-rg[source]
    block=model.blocks[1]
    baseline=block.norm1(rr['blocks.1.attn'])
    changed=block.norm1(rr['blocks.1.attn']+diff)
    signals={}
    for role in ('q','k','v','qkv'):
        hops=[PathHop(source),PathHop(target,role)]
        output=route_delta(model,hops,diff,rg,rr)
        q=changed if role in ('q','qkv') else baseline
        k=changed if role in ('k','qkv') else baseline
        v=changed if role in ('v','qkv') else baseline
        hand,_=block.attn(q,k,v,need_weights=False)
        assert torch.allclose(output,hand-rg[target],rtol=1e-5,atol=1e-7)
        signals[role]=output
    # Nonzero Q/K/V route variation means these are not invented node-edge lines.
    assert signals['q'].abs().max()>0
    assert signals['k'].abs().max()>0
    assert signals['v'].abs().max()>0
    assert not torch.allclose(signals['q'],signals['v'],atol=1e-7,rtol=1e-5)
    no_hooks(model)


def test_three_hop_path_does_not_leak_direct_bypass(case):
    model,x,ref,_=case
    _,cg,_=capture_path_baseline(model,x)
    _,rg,rr=capture_path_baseline(model,ref)
    s='blocks.0.attn_to_residual'
    middle='blocks.1.attn_to_residual'
    end='blocks.2.mlp_to_residual'
    diff=cg[s]-rg[s]
    mid_delta=route_delta(model,[PathHop(s),PathHop(middle,'q')],diff,rg,rr)
    end_delta=route_delta(model,[PathHop(middle),PathHop(end)],mid_delta,rg,rr)
    joint_delta=route_delta(model,[PathHop(s),PathHop(middle,'q'),PathHop(end)],diff,rg,rr)
    assert torch.allclose(joint_delta,end_delta,atol=1e-7,rtol=1e-5)
    # A direct source->end bypass need NOT equal the constrained 3-hop route.
    shortcut=route_delta(model,[PathHop(s),PathHop(end)],diff,rg,rr)
    assert not torch.allclose(joint_delta,shortcut,atol=1e-7,rtol=1e-5)
    no_hooks(model)


def test_zero_delta_zero_mask_and_identical_baselines_are_noops(case):
    model,x,ref,roi=case
    base,rg,rr=capture_path_baseline(model,ref)
    hops=[PathHop('blocks.0.mlp_to_residual'),PathHop('blocks.2.attn_to_residual','v')]
    zero=torch.zeros_like(rg[hops[0].edge])
    result,delta=intervene_path(model,ref,hops,rg,rr,zero)
    assert torch.equal(result,base)
    assert torch.equal(delta,torch.zeros_like(delta)) or delta.abs().max()<1e-7
    _,cg,_=capture_path_baseline(model,x)
    mask=torch.zeros(zero.shape[:2],dtype=torch.bool)
    result,delta=intervene_path(model,ref,hops,rg,rr,cg[hops[0].edge]-rg[hops[0].edge],mask)
    assert torch.equal(result,base)
    match=evaluate_path(model,x,x,hops,roi)
    assert match['input_effect']==0 and match['restoration']==0
    assert match['disruption']==0
    no_hooks(model)


def test_token_mask_restricts_source_not_global_attention(case):
    model,x,ref,_=case
    _,cg,_=capture_path_baseline(model,x)
    _,rg,rr=capture_path_baseline(model,ref)
    hops=[PathHop('blocks.0.attn_to_residual'),PathHop('blocks.2.attn_to_residual','v')]
    diff=cg[hops[0].edge]-rg[hops[0].edge]
    mask=torch.zeros(diff.shape[:2],dtype=torch.bool);mask[:,0]=True
    delta=route_delta(model,hops,diff,rg,rr,mask)
    assert torch.isfinite(delta).all()
    assert delta[:,0].abs().sum()>0
    # Global MHA means changing one SOURCE token may change destination other tokens.
    assert delta[:,1:].abs().sum()>0
    with pytest.raises(ValueError):
        route_delta(model,hops,diff,rg,rr,torch.zeros(diff.shape[:2],dtype=torch.float))
    no_hooks(model)


def test_no_weights_changed_no_hidden_grad_or_hooks_leak(case):
    model,x,ref,roi=case
    old={name:p.detach().clone() for name,p in model.named_parameters()}
    orig,_,_=captured_forward(model,x)
    path=[PathHop('blocks.0.attn_to_residual'),PathHop('blocks.2.mlp_to_residual')]
    res=evaluate_path(model,x,ref,path,roi)
    assert all(math.isfinite(res[k]) for k in ('restoration','disruption','objective'))
    assert isinstance(res['direction_aligned'],bool)
    new,_,_=captured_forward(model,x)
    assert torch.equal(orig,new)
    for name,p in model.named_parameters():
        assert torch.equal(old[name],p) and p.grad is None
    no_hooks(model)


def test_invalid_path_input_fails_cleanly(case):
    model,x,ref,roi=case
    hops=[PathHop('blocks.0.attn_to_residual'),PathHop('blocks.2.mlp_to_residual')]
    with pytest.raises(ValueError):evaluate_path(model,x,ref[:,:,:4],hops,roi)
    # Wrong size/shape source (with identical device and dtype) rejected.
    _,cg,_=capture_path_baseline(model,x)
    _,rg,rr=capture_path_baseline(model,ref)
    with pytest.raises(ValueError):route_delta(model,hops,torch.zeros(1,1,1),rg,rr)
    with pytest.raises(ValueError):evaluate_path(model,x,ref,hops,torch.ones(8,8,8,dtype=torch.bool))
    no_hooks(model)


def test_bounded_search_reserved_reference_and_determinism(case):
    model,x,_,roi=case
    # Distinct references are used for selection vs the frozen holdout replay.
    cfg=PathSearchConfig(channel=2,max_candidates=5,max_hops=3,
                         max_tested_paths=5,attention_roles=('q','v'),
                         min_input_effect=1e-12)
    r1=explain_scca_paths(model,x,Region(2,6,2,6,2,6),roi,cfg)
    r2=explain_scca_paths(model,x,Region(2,6,2,6,2,6),roi,cfg)
    assert r1['version']=='cross_layer_path_alpha_v0.3'
    assert 0<len(r1['tested_paths'])<=cfg.max_tested_paths
    assert r1['selected']==r2['selected']
    assert r1['validation']['status']=='evaluated_same_case_reserved_reference'
    assert r1['validation']['fixed_path']['path']==r1['selected']['path']
    assert all(len(z['path'])>=2 for z in r1['tested_paths'])
    assert r1['independent_patient_validation'] is False
    no_hooks(model)


def test_low_input_effect_search_does_not_fabricate_path(case):
    model,x,ref,roi=case
    cfg=PathSearchConfig(min_input_effect=999.0)
    result=explain_scca_paths(model,x,Region(2,6,2,6,2,6),roi,cfg)
    assert result['status']=='unresolved_low_input_effect'
    assert result['selected'] is None
    assert result['tested_paths']==[]
    no_hooks(model)


def test_same_block_attention_to_mlp_consumer_oracle(case):
    model,x,ref,_=case
    _,clean,_=capture_path_baseline(model,x)
    _,base,res=capture_path_baseline(model,ref)
    source='blocks.1.attn_to_residual';target='blocks.1.mlp_to_residual'
    hop=[PathHop(source),PathHop(target)]
    delta=clean[source]-base[source]
    output=route_delta(model,hop,delta,base,res)
    # Same block MLP input is AFTER attention residual sum.
    manual=model.blocks[1].mlp(model.blocks[1].norm2(res['blocks.1.mlp']+delta))-base[target]
    assert torch.allclose(output,manual,atol=1e-7,rtol=1e-5)
    no_hooks(model)


def test_path_real_vib_and_decoder_output_changes(case):
    model,x,ref,_=case
    base,clean,_=capture_path_baseline(model,x)
    reference,ref_gate,res=capture_path_baseline(model,ref)
    hops=[PathHop('blocks.0.mlp_to_residual'),PathHop('blocks.1.attn_to_residual','v'),
          PathHop('blocks.2.mlp_to_residual')]
    modified,delta=intervene_path(model,ref,hops,ref_gate,res,
                                   clean[hops[0].edge]-ref_gate[hops[0].edge])
    assert delta.abs().max()>1e-8
    assert (modified-reference).abs().max()>1e-9
    assert modified.shape==(1,3,8,8,8)
    assert torch.isfinite(modified).all()
    no_hooks(model)


def test_source_attention_role_is_not_mislabelled(case):
    model,*_=case
    with pytest.raises(ValueError,match='Source gate'):
        validate_path(model,[PathHop('blocks.0.attn_to_residual','q'),
                             PathHop('blocks.1.mlp_to_residual')])
    no_hooks(model)


def test_known_q_projection_knockout_distinguishes_qk_from_v(case):
    """Known mechanistic control: zero Q projection makes attention independent
    of query/key inputs, but leaves the value path with its real effect."""
    model,x,ref,_=case
    layer=model.blocks[1].attn
    d=layer.embed_dim
    with torch.no_grad():
        layer.in_proj_weight[:d].zero_()
        layer.in_proj_bias[:d].zero_()
    _,cg,_=capture_path_baseline(model,x)
    _,rg,rr=capture_path_baseline(model,ref)
    source='blocks.0.attn_to_residual';target='blocks.1.attn_to_residual'
    delta=cg[source]-rg[source]
    q=route_delta(model,[PathHop(source),PathHop(target,'q')],delta,rg,rr)
    k=route_delta(model,[PathHop(source),PathHop(target,'k')],delta,rg,rr)
    v=route_delta(model,[PathHop(source),PathHop(target,'v')],delta,rg,rr)
    assert q.abs().max() < 1e-6
    assert k.abs().max() < 1e-6
    assert v.abs().max() > 1e-6
    no_hooks(model)
