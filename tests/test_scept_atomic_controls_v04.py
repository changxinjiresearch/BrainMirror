"""Parametric negative/positive controls on real MHA projection weights."""
import torch
from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.attention_atomic import AtomicAttentionSite,intervene_atomic_attention

torch.set_num_threads(2)


def test_zero_query_key_projection_removes_single_logit_effect_but_not_value_message():
    torch.manual_seed(1401)
    model=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8).eval()
    clean=torch.randn(1,4,8,8,8)
    ref=clean.clone();ref[:,:,:4,:4,:4]+=2.
    with torch.no_grad():
        attn=model.blocks[0].attn
        attn.in_proj_weight[:32].zero_()
        attn.in_proj_bias[:32].zero_()
    by_logit=intervene_atomic_attention(model,clean,ref,
                                      AtomicAttentionSite(0,1,0,0,'logit'))
    by_value=intervene_atomic_attention(model,clean,ref,
                                      AtomicAttentionSite(0,1,0,0,'weighted_value'))
    assert by_logit['branch_delta'].abs().max()<1e-8
    assert by_value['branch_delta'].abs().max()>1e-9


def test_zero_one_head_output_projection_blocks_its_atomic_message():
    torch.manual_seed(1402)
    model=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8).eval()
    clean=torch.randn(1,4,8,8,8);ref=clean.clone()
    ref[...,0:4,0:4,0:4]+=1
    with torch.no_grad():
        model.blocks[0].attn.out_proj.weight[:,4:8].zero_()
    result=intervene_atomic_attention(model,clean,ref,
                                    AtomicAttentionSite(0,1,0,0))
    assert result['branch_delta'].abs().max()==0
    assert torch.allclose(result['intervened_logits'],result['reference_logits'],atol=2e-6)
