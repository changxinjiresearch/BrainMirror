"""Single fine token -> merge -> exact one coarse head message -> VIB/decoder."""
import pytest
import torch
from brainmirror.scept.multiscale import (
    MultiScaleTransformerVIB3D,CrossScaleSite,intervene_cross_scale_atomic)
from brainmirror.scept.attention_atomic import AtomicAttentionSite

torch.set_num_threads(2)

@pytest.fixture
def case():
    torch.manual_seed(1011)
    model=MultiScaleTransformerVIB3D(dim=16,heads=4,patch=2,max_tokens=512).eval()
    ref=torch.randn(1,4,8,8,8)
    clean=ref.clone();clean[...,:2,:2,:2]+=2.0
    return model,clean,ref


def test_multi_scale_atomic_chain_nonzero_and_vib_changes(case):
    m,x,r=case
    site=AtomicAttentionSite(0,1,0,0)
    res=intervene_cross_scale_atomic(m,x,r,CrossScaleSite(0),site)
    assert res['atomic_branch_delta'].shape==(1,16)
    assert res['atomic_branch_delta'].abs().max()>1e-9
    assert res['latent_mu_delta'].abs().max()>1e-9
    assert res['isolated_logits'].shape==(1,3,8,8,8)
    assert not any(b._forward_hooks or b._forward_pre_hooks for b in m.modules())


def test_multi_scale_atomic_negative_control_reference_equals_clean(case):
    m,_,ref=case
    r=intervene_cross_scale_atomic(m,ref,ref,CrossScaleSite(0),AtomicAttentionSite(0,0,0,0))
    assert r['atomic_branch_delta'].abs().max()<1e-7
    assert torch.allclose(r['isolated_logits'],r['reference_logits'],atol=2e-6)


def test_multi_scale_atomic_rejects_unavailable_layers_and_sites(case):
    m,x,ref=case
    with pytest.raises(ValueError):
        intervene_cross_scale_atomic(m,x,ref,CrossScaleSite(0),AtomicAttentionSite(3,0,0,0))
    with pytest.raises(ValueError):
        intervene_cross_scale_atomic(m,x,ref,CrossScaleSite(0),AtomicAttentionSite(0,0,50,0))
    with pytest.raises(ValueError):
        intervene_cross_scale_atomic(m,x,ref,CrossScaleSite(99),AtomicAttentionSite(0,0,0,0))
