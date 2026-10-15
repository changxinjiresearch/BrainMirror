import numpy as np
import torch
from brainmirror.model import BrainAE, incoming_edges, LAYERS
from brainmirror.mri import phantom, local_expand, normalize_mask, normalize_volume, to_mesh


def test_network_tensor_shape_and_exact_kernel_contributions():
    torch.manual_seed(17)
    m = BrainAE().eval()
    x = torch.rand((1,1,32,32,32))
    with torch.no_grad():
        result, activations = m(x, capture=True)
    assert result.shape == (1,1,32,32,32)
    for layer in ('conv1','conv2','conv3'):
        for c,z,y,xx in ((0,0,0,0),(1,1,1,1)):
            g=incoming_edges(m,activations,layer,c,z,y,xx)
            assert abs(g['verified_sum']-g['preactivation'])<2e-5
            assert 1 <= len(g['edges']) <= (1 if layer=='conv1' else (4 if layer=='conv2' else 8))*27
            assert g['activation']>=0


def test_hidden_unit_ablation_changes_the_downstream_computation():
    m=BrainAE().eval()
    x=torch.rand((1,1,32,32,32))
    with torch.no_grad():
        clean,a=m(x,capture=True)
        altered,_=m(x,capture=True,ablate={'layer':'conv3','channel':0,'z':1,'y':1,'x':1})
    assert a['conv3_pre'].shape == (1,8,4,4,4)
    assert torch.isfinite(altered).all()
    assert altered.shape==clean.shape


def test_local_warp_identity_and_perturbed_input():
    v,m=phantom(1)
    i,mi,q0=local_expand(v,m,0)
    assert np.array_equal(i,v)
    assert np.array_equal(mi,m)
    assert q0['changed_voxel_fraction']==0
    after,new_mask,q=local_expand(v,m,1.25)
    assert after.shape==v.shape
    assert new_mask.shape==m.shape
    assert np.isfinite(after).all()
    assert np.mean(np.abs(after-v))>0
    assert q['target_volume_after'] > q['target_volume_before']
    assert q['non_target_mean_abs_change'] < .03


def test_3d_mesh_and_resample():
    v,m=phantom(2)
    mesh=to_mesh(m)
    assert len(mesh['i'])>0
    assert len(mesh['x'])>0
    assert len(mesh['i'])==len(mesh['j'])==len(mesh['k'])
    big=np.ones((64,48,40),dtype=np.float32)*.3
    big[10:38,5:33,6:28]=.9
    out=normalize_volume(big)
    assert out.shape==(32,32,32)
    mask=np.zeros_like(big);mask[10:18,10:25,10:25]=1
    assert normalize_mask(mask).shape==(32,32,32)
