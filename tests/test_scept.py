"""CPU-only engineering tests. No patient data, no medical validity claims."""
import pytest
import torch
from brainmirror.scept import (TransformerVIB3D, IJEPAPretrainer,
    transfer_pretrained_encoder, Region, captured_forward, numeric_trace_equivalence,
    explain_region, brats_dice_bce, local_reference, fixed_target_score,
    allowed_modules)

torch.set_num_threads(2)

@pytest.fixture
def fixture():
    torch.manual_seed(23)
    model = TransformerVIB3D(dim=16, heads=4, depth=2, patch=4, latent_dim=8).eval()
    x = torch.rand(1, 4, 16, 16, 16)
    roi = torch.zeros(16, 16, 16, dtype=torch.bool)
    roi[4:12, 4:12, 4:12] = True
    return model, x, roi


def test_3d_transformer_vib_shapes_gradients_and_kl(fixture):
    model, x, _ = fixture
    out, d = model(x, sample=True, return_details=True)
    assert out.shape == (1,3,16,16,16)
    assert d['mu'].shape == (1,64,8)
    assert d['latent'].shape == (1,64,8)
    assert d['grid'] == (4,4,4)
    assert d['kl'].item() >= 0
    target = torch.zeros_like(out)
    loss = brats_dice_bce(out, target) + 0.01 * d['kl']
    loss.backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_input_validation_and_token_limit(fixture):
    model, x, _ = fixture
    with pytest.raises(ValueError): model(x[:,:,:15])
    with pytest.raises(ValueError): model(torch.rand(1,1,16,16,16))
    with pytest.raises(ValueError): model(torch.rand(1,4,36,36,36))


def test_trace_equivalence_shapes_and_real_attention(fixture):
    model,x,_=fixture
    assert numeric_trace_equivalence(model,x)
    output,records,details=captured_forward(model,x)
    assert output.shape==(1,3,16,16,16)
    assert len(allowed_modules(model)) == 5
    assert records['blocks.0.attn.weights'].shape==(1,4,64,64)
    assert records['blocks.1.mlp'].shape==(1,64,16)
    assert records['vib'].shape==(1,64,8)
    assert torch.allclose(records['blocks.0.attn.weights'].sum(-1),torch.ones(1,4,64),atol=1e-5)
    assert torch.isfinite(details['kl'])


def test_clean_patch_is_noop_and_hooks_are_removed(fixture):
    model,x,_=fixture
    clean,acts,_=captured_forward(model,x)
    for name in ('blocks.0.attn','blocks.1.mlp','vib'):
        same,_,_=captured_forward(model,x,patch=(name,acts[name]))
        assert torch.allclose(same,clean,atol=1e-6)
    with pytest.raises(KeyError): captured_forward(model,x,patch=('invalid',torch.tensor(0)))
    assert numeric_trace_equivalence(model,x)
    assert all(len(mod._forward_hooks)==0 for mod in model.modules())


def test_reference_and_explanation_are_measured(fixture):
    model,x,roi=fixture
    region=Region(3,12,2,11,4,12)
    alt=local_reference(x,region)
    assert torch.equal(alt[..., :3,:,:],x[..., :3,:,:])
    assert not torch.allclose(alt,x)
    r=explain_region(model,x,region,roi,channel=2)
    assert r['reference'].endswith('SYNTHETIC_ONLY')
    assert r['mechanistic_claim'].startswith('node-level')
    assert r['independent_validation'] is False
    assert len(r['modules'])==5
    assert abs(r['input_effect']-(r['clean_score']-r['altered_score']))<1e-6
    assert all(torch.isfinite(torch.tensor(m['disruption'])) for m in r['modules'])
    assert abs(r['input_effect']) > 1e-9


def test_invalid_regions_and_roi(fixture):
    model,x,roi=fixture
    for region in (Region(-1,5,1,4,1,4),Region(7,7,1,4,1,4),Region(1,20,1,4,1,4)):
        with pytest.raises(ValueError): local_reference(x,region)
    with pytest.raises(ValueError): fixed_target_score(model(x),0,torch.zeros_like(roi))
    with pytest.raises(ValueError): explain_region(model,x,Region(1,3,1,3,1,3),roi,max_sites=-1)


def test_ijepa_pretraining_teacher_and_transfer(fixture):
    model,x,_=fixture
    pre=IJEPAPretrainer(model,mask_ratio=.5,momentum=.9)
    opt=torch.optim.Adam(list(model.encoder_parameters())+list(pre.predictor.parameters())+[pre.mask_token],lr=1e-3)
    before=pre.student.embed.weight.detach().clone()
    loss=pre.pretrain_step(x,opt)
    assert loss>=0 and torch.isfinite(torch.tensor(loss))
    assert not torch.equal(pre.student.embed.weight,before)
    assert not any(p.requires_grad for p in pre.teacher.parameters())
    scratch=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8)
    previous_vib=scratch.vib.mu.weight.detach().clone()
    transfer_pretrained_encoder(pre.student,scratch)
    assert torch.equal(scratch.embed.weight,pre.student.embed.weight)
    assert torch.equal(scratch.vib.mu.weight,previous_vib)
    assert not torch.equal(scratch.embed.weight,before)


def test_train_and_infer_after_encoder_transfer(fixture):
    model,x,_=fixture
    pre=IJEPAPretrainer(model)
    scratch=TransformerVIB3D(dim=16,heads=4,depth=2,patch=4,latent_dim=8)
    transfer_pretrained_encoder(pre.student,scratch)
    out, d=scratch(x, sample=True,return_details=True)
    loss=brats_dice_bce(out,torch.zeros_like(out))+.01*d['kl']
    optim=torch.optim.Adam(scratch.parameters(),lr=1e-3)
    optim.zero_grad();loss.backward();optim.step()
    assert torch.isfinite(scratch(x)).all()
