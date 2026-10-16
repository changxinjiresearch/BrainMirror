"""Cross-layer single-message plus isolated VIB/decoder mediation controls."""
import pytest
import torch
from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.atomic_chain import AtomicPathHop,intervene_atomic_chain
from brainmirror.scept.paths import PathHop

torch.set_num_threads(2)

@pytest.fixture
def setup():
    torch.manual_seed(419)
    model=TransformerVIB3D(dim=16,heads=4,depth=3,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    ref=x.clone();ref[:,:,:4,:4,:4]+=1.2
    path=[PathHop('blocks.0.mlp_to_residual'),AtomicPathHop(1,2,0,0),
          PathHop('blocks.2.mlp_to_residual')]
    return model,x,ref,path


def test_atomic_chain_into_actual_vib_decoder_and_its_single_latent(setup):
    model,x,ref,path=setup
    r=intervene_atomic_chain(model,x,ref,path,latent_token=0,latent_channel=3)
    assert r['route_delta'].shape==(1,8,16)
    assert r['latent_delta'].shape==(1,8,8)
    assert len(r['steps'])==2
    assert torch.isfinite(r['isolated_chain_logits']).all()
    assert torch.isfinite(r['latent_mediated_logits']).all()
    assert r['latent_delta'].abs().max()>0
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_atomic_chain_source_token_mask_zero_means_no_change(setup):
    model,x,ref,path=setup
    mask=torch.zeros(1,8,dtype=torch.bool)
    r=intervene_atomic_chain(model,x,ref,path,source_mask=mask,latent_token=0)
    assert r['route_delta'].abs().max()<1e-7
    assert torch.allclose(r['isolated_chain_logits'],r['reference_logits'],atol=2e-6)
    assert torch.allclose(r['latent_mediated_logits'],r['reference_logits'],atol=2e-6)


def test_chain_rejects_paths_not_closed_at_final_mlp(setup):
    model,x,ref,path=setup
    with pytest.raises(ValueError):
        intervene_atomic_chain(model,x,ref,path[:-1])
    with pytest.raises(ValueError):
        intervene_atomic_chain(model,x,ref,path[:-1]+[PathHop('blocks.2.attn_to_residual')])
    with pytest.raises(ValueError):
        intervene_atomic_chain(model,x,ref,[path[0],AtomicPathHop(1,99,0,0),path[-1]])
    with pytest.raises(ValueError):
        intervene_atomic_chain(model,x,ref,path,latent_token=500)


def test_message_and_logit_chain_operationally_distinct(setup):
    model,x,ref,path=setup
    weighted=intervene_atomic_chain(model,x,ref,path)
    logit=intervene_atomic_chain(model,x,ref,[path[0],AtomicPathHop(1,2,0,0,'logit'),path[-1]])
    assert torch.isfinite(weighted['route_delta']).all()
    assert torch.isfinite(logit['route_delta']).all()
    assert not torch.allclose(weighted['route_delta'],logit['route_delta'],atol=1e-9,rtol=1e-5)
