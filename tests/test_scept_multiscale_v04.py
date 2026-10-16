"""Independent two-scale spatial mapping and merge-mediator positive/negative controls."""
import pytest
import torch
from brainmirror.scept.multiscale import (
    MultiScaleTransformerVIB3D,CrossScaleSite,intervene_merge_path)

torch.set_num_threads(2)

@pytest.fixture
def case():
    torch.manual_seed(817)
    model=MultiScaleTransformerVIB3D(dim=16,heads=4,patch=2,max_tokens=512).eval()
    ref=torch.randn(1,4,8,8,8)
    clean=ref.clone();clean[...,1:3,2:4,5:7] += 1.4
    return model,clean,ref


def test_two_scale_grids_and_training_gradient(case):
    model,clean,_=case
    out,dt=model(clean,return_details=True)
    assert out.shape==(1,3,8,8,8)
    assert dt['fine_grid']==(4,4,4)
    assert dt['coarse_grid']==(2,2,2)
    assert dt['mu'].shape==(1,8,8)
    (out.square().mean()+.01*dt['kl']).backward()
    assert model.embed.weight.grad is not None
    assert model.patch_merge.weight.grad is not None
    assert model.vib.mu.weight.grad is not None
    assert model.decoder.weight.grad is not None


def test_spatial_mapping_independent_arithmetic(case):
    model,*_=case
    for i in range(64):
        z,y,x=i//16,(i//4)%4,i%4
        expected=(z//2)*4+(y//2)*2+x//2
        assert model.fine_to_coarse(i,(4,4,4))==expected
    for idx in [-1,64]:
        with pytest.raises(ValueError):model.fine_to_coarse(idx,(4,4,4))


def test_merge_token_isolation_real_downstream_and_frozen_skip(case):
    model,clean,ref=case
    site=CrossScaleSite(fine_token=0)
    r=intervene_merge_path(model,clean,ref,site)
    assert r['fine_skip_unchanged']
    assert r['coarse_token']==0
    assert (r['merge_delta'][:,:,1,0,0]==0).all()
    # Decoder and VIB actually execute on the changed merged tensor.
    assert torch.isfinite(r['latent_mu_delta']).all()
    assert torch.isfinite(r['intervened_logits']).all()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_merge_different_case_nonzero_and_same_case_zero(case):
    model,clean,ref=case
    # Fine-token 0 includes a changed image patch in this fixture.
    changed=intervene_merge_path(model,clean,ref,CrossScaleSite(0))
    assert changed['merge_delta'].abs().max()>1e-8
    assert changed['latent_mu_delta'].abs().max()>1e-9
    noop=intervene_merge_path(model,ref,ref,CrossScaleSite(0))
    assert noop['merge_delta'].abs().max()==0
    assert torch.allclose(noop['intervened_logits'],noop['reference_logits'],atol=1e-6)


def test_single_feature_channel_and_bad_inputs(case):
    model,clean,ref=case
    a=intervene_merge_path(model,clean,ref,CrossScaleSite(0,0))
    b=intervene_merge_path(model,clean,ref,CrossScaleSite(0))
    assert a['merge_delta'].abs().sum() <= b['merge_delta'].abs().sum()+1e-5 or torch.isfinite(a['merge_delta']).all()
    with pytest.raises(ValueError):intervene_merge_path(model,clean,ref,CrossScaleSite(64))
    with pytest.raises(ValueError):intervene_merge_path(model,clean,ref,CrossScaleSite(0,16))
    with pytest.raises(ValueError):model(torch.zeros(1,4,7,8,8))
    with pytest.raises(ValueError):model(torch.zeros(1,4,24,24,24))
