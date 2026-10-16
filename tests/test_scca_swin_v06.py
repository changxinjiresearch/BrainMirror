"""Independent numerical controls for Swin v0.6, including negative cases."""
import dataclasses
import math
import pytest
import torch
from torch.nn import functional as F
from brainmirror.scept.swin3d import (WindowAttention3D,MultiLayerSwinVIB3D,
                                      execution_topology,unflatten,grid_index)
from brainmirror.scept.swin_discovery import (
    WindowAtom,SwinDiscoveryConfig,_atom_delta,_trace,_bridge,
    _inject_one_source,execute_route,discover_scca_swin,validate_fixed_route,
    _candidate_vectors)


@pytest.fixture
def setting():
    torch.manual_seed(165)
    model=MultiLayerSwinVIB3D(dim=12,heads=3,patch=2,window=2,
                              fine_depth=2,coarse_depth=2,max_tokens=512).eval()
    x=torch.randn(1,4,16,16,16)
    roi=torch.zeros(16,16,16,dtype=torch.bool)
    roi[2:10,2:10,2:10]=True
    return model,x,roi


def test_window_partition_restoration_and_sdpa_numerical_equivalence():
    torch.manual_seed(3)
    for grid in ((4,4,4),(2,2,2)):
        for shift in (0,1):
            attn=WindowAttention3D(12,3,2,shift).double().eval()
            x=torch.randn(1,math.prod(grid),12,dtype=torch.float64)
            out,trace=attn(x,grid,record=True)
            reference=F.scaled_dot_product_attention(trace['q'],trace['k'],trace['v'],
                attn_mask=trace['allowed'][None,:,None],dropout_p=0.0)
            assert torch.allclose(reference,trace['head_output'],rtol=1e-12,atol=1e-12)
            assert torch.allclose(attn(x,grid),out,rtol=0,atol=0)
            assert int((~trace['allowed']).sum())>0 if trace['effective_shift'] else int((~trace['allowed']).sum())==0


def test_shifted_windows_allow_real_cross_boundary_and_mask_fake_wrap():
    attn=WindowAttention3D(12,3,2,1)
    loc,allowed,shift=attn.geometry((4,4,4),torch.device('cpu'))
    assert shift==1
    def connectivity(a,b):
        for wi in range(len(loc)):
            q=(loc[wi]==a).nonzero().flatten()
            k=(loc[wi]==b).nonzero().flatten()
            if len(q)==1 and len(k)==1:
                return bool(allowed[wi,int(q[0]),int(k[0])])
        return False
    assert connectivity(grid_index(1,1,1,(4,4,4)),grid_index(2,1,1,(4,4,4)))
    assert not connectivity(grid_index(0,1,1,(4,4,4)),grid_index(3,1,1,(4,4,4)))
    assert not connectivity(grid_index(0,0,0,(4,4,4)),grid_index(3,3,3,(4,4,4)))


def test_model_trace_equivalence_and_gradient_flow(setting):
    model,x,_=setting
    o,rec=model(x,record=True)
    assert torch.equal(o,model(x))
    assert o.shape==(1,3,16,16,16)
    assert len(rec['fine_steps'])==2 and len(rec['coarse_steps'])==2
    assert [r['attention']['effective_shift'] for r in rec['fine_steps']]==[0,1]
    assert [r['attention']['effective_shift'] for r in rec['coarse_steps']]==[0,1]
    o.square().mean().backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.fine_blocks[1].attn.qkv.weight.grad).all()
    assert torch.isfinite(model.vib.mu.weight.grad).all()


def test_atom_delta_equals_explicit_message_substitution(setting):
    model,x,_=setting
    _,rec=model(x,record=True)
    block=model.fine_blocks[1]
    base=rec['fine_steps'][1]['attention']
    hybrid_input=rec['fine_steps'][1]['input'].detach().clone()
    hybrid_input[:,grid_index(1,1,1,rec['fine_grid'])]+=0.25
    _,changed=block.attn(block.norm1(hybrid_input),rec['fine_grid'],record=True)
    wi=next(i for i,ids in enumerate(base['token_ids']) if int(ids[0])==grid_index(1,1,1,rec['fine_grid']))
    q=int(base['token_ids'][wi,0])
    valid_idx=[j for j in range(base['token_ids'].shape[1]) if bool(base['allowed'][wi,0,j])]
    k=int(base['token_ids'][wi,valid_idx[-1]])
    for mode in ('weighted_value','logit'):
        site=WindowAtom('fine',1,wi,0,q,k,mode)
        direct=_atom_delta(base,changed,site,block.attn.proj.weight)
        qi=0;ki=valid_idx[-1];dh=base['head_width']
        if mode=='weighted_value':
            raw=(changed['weights'][:,wi,0,qi,ki,None]*changed['v'][:,wi,0,ki]-
                 base['weights'][:,wi,0,qi,ki,None]*base['v'][:,wi,0,ki])
        else:
            logits=base['logits'][:,wi,0,qi].clone()
            logits[:,ki]=changed['logits'][:,wi,0,qi,ki]
            raw=((logits.softmax(-1)-base['weights'][:,wi,0,qi]).unsqueeze(-1)*
                  base['v'][:,wi,0]).sum(1)
        control=raw@block.attn.proj.weight[:,:dh].T
        assert torch.allclose(direct,control,rtol=1e-5,atol=1e-7)


def _simple_route(rec):
    atom_lists=[]
    for stage in ('fine','coarse'):
        atoms=[]
        for j,step in enumerate(rec[stage+'_steps']):
            a=step['attention']; global_idx=0
            wi=next(i for i,ids in enumerate(a['token_ids']) if (ids==global_idx).any())
            atoms.append(WindowAtom(stage,j,wi,0,global_idx,global_idx,'weighted_value'))
        atom_lists.append(atoms)
    return atom_lists


def test_route_noop_and_deterministic_reproducibility(setting):
    model,x,roi=setting
    out,base,_=_trace(model,x)
    fine,coarse=_simple_route(base)
    same=execute_route(model,base,base,0,fine,coarse)
    assert torch.allclose(same,out,rtol=1e-5,atol=1e-6)
    donor=x.clone();donor[:,:,0:2,0:2,0:2]+=3
    _,clean,_=_trace(model,donor)
    first=execute_route(model,base,clean,0,fine,coarse)
    second=execute_route(model,base,clean,0,fine,coarse)
    assert torch.equal(first,second)
    assert torch.isfinite(first).all()
    measured=validate_fixed_route(model,donor,x,0,fine,coarse,roi,2)
    assert all(math.isfinite(v) for v in measured.values())


def test_offpath_skip_is_frozen(setting):
    model,x,_=setting
    _,base,_=_trace(model,x)
    donor=x.clone();donor[:,:,0:2,0:2,0:2]+=2
    _,clean,_=_trace(model,donor)
    fine,coarse=_simple_route(base)
    seen=[]
    def hook(_m,args):
        seen.append(args[0][:,model.up.out_channels:])
    handle=model.fuse.register_forward_pre_hook(hook)
    try:
        execute_route(model,base,clean,0,fine,coarse)
    finally:handle.remove()
    assert len(seen)==1
    assert torch.equal(seen[0],base['fine_skip'])
    assert not torch.equal(clean['fine_skip'],base['fine_skip'])


def test_invalid_atoms_are_rejected_no_silent_path_fabrication(setting):
    model,x,_=setting
    _,base,_=_trace(model,x)
    fine,coarse=_simple_route(base)
    with pytest.raises(ValueError):
        execute_route(model,base,base,0,[dataclasses.replace(fine[0],layer=1)]+fine[1:],coarse)
    with pytest.raises(ValueError):
        execute_route(model,base,base,0,fine[:-1],coarse)
    with pytest.raises(ValueError):
        execute_route(model,base,base,10_000,fine,coarse)
    with pytest.raises(ValueError):
        bad=dataclasses.replace(fine[1],query=10_000)
        execute_route(model,base,base,0,[fine[0],bad],coarse)


def test_patch_merge_exact_cell_and_decoder_link(setting):
    model,x,_=setting
    _,base,_=_trace(model,x)
    delta=torch.zeros_like(base['fine_pre_norm']); delta[:,grid_index(1,1,1,base['fine_grid']),0]=0.15
    changed=_bridge(model,base,delta)
    nonzero=(changed.abs().sum(dim=-1)>1e-10)
    assert int(nonzero.sum())==1
    target=grid_index(0,0,0,base['coarse_grid'])
    assert bool(nonzero[0,target])
    assert tuple(model.decode(base['coarse_normed'],base['fine_skip'],base['coarse_grid']).shape)==tuple(x.shape[:1]+(3,)+x.shape[2:])


def test_execution_topology_matches_actual_executed_mask_and_shift(setting):
    model,x,_=setting
    manifest=execution_topology(model,x)
    o,t=model(x,record=True)
    for stage in ('fine','coarse'):
        for j,block in enumerate(manifest['stages'][stage]['blocks']):
            real=t[stage+'_steps'][j]['attention']
            assert block['window_token_ids']==real['token_ids'].tolist()
            assert block['effective_shift']==real['effective_shift']
            assert len(block['per_head_query_key_edges'])==int(real['allowed'].sum())
    assert len(manifest['patch_merge_receptive_links'])==math.prod(t['fine_grid'])
    with pytest.raises(ValueError):execution_topology(model,x,max_edges=10)


def test_source_negative_control_and_no_answer_for_zero_effect(setting):
    model,x,roi=setting
    result=discover_scca_swin(model,x,roi,config=SwinDiscoveryConfig())
    assert result['exact_tested_count']>0
    assert all(len(p['fine_atoms'])==2 and len(p['coarse_atoms'])==2 for p in result['candidates'])
    assert 0<result['screened_atomic_count']
    assert all(math.isfinite(p['objective']) for p in result['candidates'])
    assert result['independent_patient_validation'] is False
    # Zero tensor has no information to remove using blur/mean references.
    zeros=torch.zeros_like(x)
    zero=discover_scca_swin(model,zeros,roi)
    assert zero['status']=='unresolved_low_primary_effect'
    assert zero['exact_tested_count']==0


def test_ranking_does_not_bypass_window_mask(setting):
    model,x,roi=setting
    o,t=model(x,record=True)
    step=t['fine_steps'][1]
    attn=step['attention'];block=model.fine_blocks[1]
    grad=torch.randn_like(step['gate'])
    fake=torch.randn_like(step['input'])*0.1
    _,hybrid=block.attn(block.norm1(step['input']+fake),t['fine_grid'],record=True)
    found=_candidate_vectors(attn,hybrid,block.attn.proj.weight,grad,
             ('weighted_value','logit'),0,256,'fine',1)
    for _,__,atom,_ in found:
        ids=attn['token_ids'][atom.window]
        qi=int((ids==atom.query).nonzero().item())
        ki=int((ids==atom.key).nonzero().item())
        assert bool(attn['allowed'][atom.window,qi,ki])


def test_invalid_reference_and_shape_rejected(setting):
    model,x,roi=setting
    with pytest.raises(ValueError):discover_scca_swin(model,x,roi,config=SwinDiscoveryConfig(primary_reference='blur',reserved_reference='blur'))
    with pytest.raises(ValueError):discover_scca_swin(model,x,roi,channel=5)
    with pytest.raises(ValueError):discover_scca_swin(model,x,torch.ones_like(roi))
    with pytest.raises(ValueError):model(torch.ones(1,4,12,12,12))
    model.train()
    with pytest.raises(ValueError):discover_scca_swin(model,x,roi)


def test_qk_logit_zero_projection_has_zero_atomic_effect(setting):
    model,x,roi=setting
    block=model.fine_blocks[0]
    with torch.no_grad():
        dim=model.dim
        block.attn.qkv.weight[:dim].zero_()
        block.attn.qkv.bias[:dim].zero_()
    _,rec=model(x,record=True)
    base=rec['fine_steps'][0]['attention']
    inp=rec['fine_steps'][0]['input'].clone()
    inp[:,0,:]+=3
    _,hybrid=block.attn(block.norm1(inp),rec['fine_grid'],record=True)
    atom=WindowAtom('fine',0,0,0,int(base['token_ids'][0,0]),int(base['token_ids'][0,0]),'logit')
    delta=_atom_delta(base,hybrid,atom,block.attn.proj.weight)
    assert torch.allclose(delta,torch.zeros_like(delta),atol=1e-12,rtol=0)


def test_zeroed_first_attention_projection_cannot_fabricate_a_path(setting):
    model,x,roi=setting
    with torch.no_grad():
        model.fine_blocks[0].attn.proj.weight.zero_()
        model.fine_blocks[0].attn.proj.bias.zero_()
    r=discover_scca_swin(model,x,roi,config=SwinDiscoveryConfig(min_input_effect=1e-12))
    # Source MRI may still change predictions through OTHER routes, especially
    # the fine skip. That does not authorise a fabricated selected atom route.
    if r['status']!='unresolved_low_primary_effect':
        assert r['status']=='unresolved_no_eligible_path'
    assert r['exact_tested_count']==0


def test_full_swin_route_json_roundtrip_and_reference_reservation(setting):
    import json
    model,x,roi=setting
    cfg=SwinDiscoveryConfig(beam_width=2,max_source_tokens=2,
                            per_state_proposals=4,max_exact_paths=2,validate_top=1)
    a=discover_scca_swin(model,x,roi,config=cfg)
    b=discover_scca_swin(model,x,roi,config=cfg)
    assert a==b
    s=json.dumps(a,ensure_ascii=False,allow_nan=False)
    assert json.loads(s)==a
    assert a['exact_tested_count']<=cfg.max_exact_paths
    assert len(a['reserved_validation'])<=cfg.validate_top
    assert a['status'] in ('same_case_cross_reference_supported','unresolved_or_conflicting_references',
                            'unresolved_low_primary_effect','unresolved_low_reserved_effect',
                            'unresolved_no_eligible_path','unresolved_no_changed_embedding')


def test_known_uniform_attention_single_value_message_has_exact_effect():
    """Controlled-positive ground truth: 1/8 of the donor's token value."""
    torch.manual_seed(17)
    mod=WindowAttention3D(dim=4,heads=1,window=2,shift=0).double().eval()
    with torch.no_grad():
        mod.qkv.weight.zero_();mod.qkv.bias.zero_()
        mod.qkv.weight[8:12].copy_(torch.eye(4,dtype=torch.float64))
        mod.proj.weight.copy_(torch.eye(4,dtype=torch.float64));mod.proj.bias.zero_()
    inp=torch.zeros((1,8,4),dtype=torch.float64)
    donor=inp.clone();donor[:,0,1]=8.0
    _,b=mod(inp,(2,2,2),record=True)
    _,h=mod(donor,(2,2,2),record=True)
    site=WindowAtom('fine',0,0,0,1,0,'weighted_value')
    delta=_atom_delta(b,h,site,mod.proj.weight)
    target=torch.tensor([[0.,1.,0.,0.]],dtype=torch.float64)
    assert torch.allclose(delta,target,rtol=0,atol=1e-12)
    # Mutating one QK *logit* is powerless here: Q and K are all zero.
    logit_delta=_atom_delta(b,h,dataclasses.replace(site,mode='logit'),mod.proj.weight)
    assert torch.count_nonzero(logit_delta)==0


def test_fine_to_coarse_merge_links_exact_spatial_grouping(setting):
    model,x,_=setting
    manifest=execution_topology(model,x)
    fine,coarse=model.grids(x)
    for source,target in manifest['patch_merge_receptive_links']:
        f=unflatten(source,fine)
        assert target==grid_index(*(v//2 for v in f),coarse)
    for coarse_index in range(math.prod(coarse)):
        assert sum(dst==coarse_index for _,dst in manifest['patch_merge_receptive_links'])==8


def test_full_four_block_path_has_nonzero_isolated_effect_in_float64():
    """End-to-end positive control, not merely a one-atom arithmetic check.

    A single donor patch is injected, then each of 4 selected attention sites
    is recomputed on the hybrid residual. The frozen reference skip must remain
    out of this route. Double precision retains a very small 4-hop effect.
    """
    torch.manual_seed(165)
    model=MultiLayerSwinVIB3D(dim=12,heads=3,patch=2,window=2,
                              fine_depth=2,coarse_depth=2,max_tokens=512).double().eval()
    baseline=torch.zeros(1,4,16,16,16,dtype=torch.float64)
    donor=baseline.clone()
    donor[:,1,0:2,0:2,0:2]+=10
    base_logits,ref,_=_trace(model,baseline)
    _,clean,_=_trace(model,donor)
    fine,coarse=_simple_route(ref)
    patched=execute_route(model,ref,clean,0,fine,coarse)
    effect=(patched-base_logits).abs().max().item()
    assert 1e-13 < effect < 0.2
    with torch.no_grad():
        full_effect=(model(donor)-base_logits).abs().max().item()
    assert effect < full_effect  # other routes and skips remain frozen
