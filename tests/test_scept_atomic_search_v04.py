"""Search budget, signed metrics, no false validation claims."""
import torch
from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.atomic_chain import search_atomic_chains

torch.set_num_threads(2)


def test_atomic_search_bounded_and_signed_effects():
    torch.manual_seed(2204)
    model=TransformerVIB3D(dim=16,heads=4,depth=3,patch=4,latent_dim=8).eval()
    ref=torch.randn(1,4,8,8,8)
    clean=ref.clone();clean[...,0:4,0:4,0:4]+=1.
    roi=torch.zeros(8,8,8,dtype=torch.bool);roi[:4,:4,:4]=1
    r=search_atomic_chains(model,clean,ref,'blocks.0.mlp_to_residual',1,
        queries=[0,1],keys=[0,1],heads=[0,2],roi=roi,max_tested=4)
    assert len(r['candidates'])==4
    assert r['best'] is None or r['best'] in r['candidates']
    assert not r['independent_patient_validation']
    assert all('chain_restoration' in x and 'input_effect' in x for x in r['candidates'])


def test_atomic_search_zero_effect_is_unresolved():
    torch.manual_seed(2205)
    model=TransformerVIB3D(dim=16,heads=4,depth=3,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    roi=torch.zeros(8,8,8,dtype=torch.bool);roi[:4,:4,:4]=1
    r=search_atomic_chains(model,x,x,'blocks.0.attn_to_residual',1,
        queries=[0],keys=[0],heads=[0],roi=roi)
    assert r['status']=='unresolved_low_input_effect'
    assert r['best'] is None
