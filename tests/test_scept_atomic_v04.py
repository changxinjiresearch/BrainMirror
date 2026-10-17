"""Exact attention and VIB invariants: synthetic numerical verification only."""
import pytest
import torch

from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.attention_atomic import (
    AtomicAttentionSite, attention_parts, intervene_atomic_attention, validate_site)
from brainmirror.scept.latent_bridge import LatentSite, intervene_spatial_vib
from brainmirror.scept.paths import capture_path_baseline
from brainmirror.scept.trace import captured_forward

torch.set_num_threads(2)

@pytest.fixture
def case():
    torch.manual_seed(491)
    model=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8).eval()
    ref=torch.randn(1,4,8,8,8)
    clean=ref.clone();clean[:,:,:4,:4,:4]+=0.8
    return model,clean,ref


def test_attention_decomposes_identical_pytorch_head_weights_and_output(case):
    model,clean,_=case
    _,gates,res=capture_path_baseline(model,clean)
    for i,block in enumerate(model.blocks):
        t=res[f'blocks.{i}.attn'];parts=attention_parts(block,t)
        h=block.norm1(t)
        gold,w=block.attn(h,h,h,need_weights=True,average_attn_weights=False)
        assert torch.allclose(parts['output'],gold,rtol=1e-5,atol=2e-6)
        assert torch.allclose(parts['weights'],w,rtol=1e-5,atol=2e-6)
        assert torch.allclose(parts['output'],gates[f'blocks.{i}.attn_to_residual'],rtol=1e-5,atol=2e-6)


def test_single_message_only_one_head_query_and_exact_decoder_rerun(case):
    model,clean,ref=case
    site=AtomicAttentionSite(layer=0,head=1,query=0,key=1)
    r=intervene_atomic_attention(model,clean,ref,site)
    diff=r['patched_branch']-r['reference_branch']
    assert diff[:,1:].abs().max()==0
    assert diff[:,0].abs().max()>0
    restored,_,_=captured_forward(model,ref,patch=('blocks.0.attn_to_residual',r['patched_branch']))
    assert torch.allclose(restored,r['intervened_logits'],atol=1e-7,rtol=1e-6)
    assert r['semantics']=='post_softmax_weighted_value_message_swap'


def test_single_qk_logit_renormalisation_and_independent_oracle(case):
    model,clean,ref=case
    site=AtomicAttentionSite(layer=0,head=2,query=3,key=1,mode='logit')
    r=intervene_atomic_attention(model,clean,ref,site)
    _,_,cr=capture_path_baseline(model,clean)
    _,_,rr=capture_path_baseline(model,ref)
    block=model.blocks[0]
    cp=attention_parts(block,cr['blocks.0.attn']);bp=attention_parts(block,rr['blocks.0.attn'])
    row=bp['logits'][:,2,3].clone();row[:,1]=cp['logits'][:,2,3,1]
    weights=row.softmax(-1)
    expected=((weights-bp['weights'][:,2,3]).unsqueeze(-1)*bp['v'][:,2]).sum(1)
    expected=expected @ block.attn.out_proj.weight[:,2*4:3*4].T
    assert torch.allclose(r['branch_delta'],expected,atol=1e-7,rtol=1e-5)
    assert (r['patched_branch']-r['reference_branch'])[:,torch.arange(8)!=3].abs().max()==0


def test_atomic_identical_inputs_are_noop_and_bad_sites_rejected(case):
    model,_,ref=case
    r=intervene_atomic_attention(model,ref,ref,AtomicAttentionSite(0,0,0,0))
    assert torch.allclose(r['intervened_logits'],r['reference_logits'],atol=2e-6)
    for site in [AtomicAttentionSite(-1,0,0,0),AtomicAttentionSite(0,8,0,0),
                 AtomicAttentionSite(0,0,8,0),AtomicAttentionSite(0,0,0,8),
                 AtomicAttentionSite(0,0,0,0,'other')]:
        with pytest.raises(ValueError):validate_site(model,site,8)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_spatial_vib_token_decoder_matches_independent_linear_oracle(case):
    model,clean,ref=case
    r=intervene_spatial_vib(model,clean,ref,LatentSite(token=0,channel=2))
    assert (r['latent_delta'][:,1:] == 0).all()
    assert (r['latent_delta'][:,:,torch.arange(8)!=2] == 0).all()
    assert torch.allclose(r['intervened_logits']-r['reference_logits'],r['expected_decoder_delta'],atol=2e-6)
    same=intervene_spatial_vib(model,ref,ref,LatentSite(token=2))
    assert torch.allclose(same['intervened_logits'],same['reference_logits'],atol=1e-6)


def test_vib_invalid_sites_rejected(case):
    model,clean,ref=case
    for site in [LatentSite(8),LatentSite(-1),LatentSite(0,99)]:
        with pytest.raises(ValueError):intervene_spatial_vib(model,clean,ref,site)
