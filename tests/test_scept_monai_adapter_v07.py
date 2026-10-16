import pytest
import torch
from torch import nn
from brainmirror.scept.monai_adapter import (
    AttentionAtom, decompose_monai_attention, patched_attention_output,
    MonaiSwinObserver, MonaiDecoderTrace, substitute_decoder_skip, target_mean_logit,
    require_monai,
)


class FakeMonaiWindowAttention(nn.Module):
    """Independent MONAI-shaped mathematical oracle for local CPU tests.

    This mock is NOT evidence that MONAI 1.5.1 integration succeeded.
    Real integration tests live in tests/test_monai_integration_v07.py.
    """
    def __init__(self):
        super().__init__();self.num_heads=2;self.dim=8;self.window_size=(2,2,2)
        self.scale=(8//2)**-.5
        self.qkv=nn.Linear(8,24)
        self.proj=nn.Linear(8,8)
        self.attn_drop=nn.Dropout(0)
        self.proj_drop=nn.Dropout(0)
        self.relative_position_bias_table=nn.Parameter(torch.randn(27,2)*0.05)
        self.register_buffer('relative_position_index',torch.randint(0,27,(8,8)))
    def forward(self,x,mask=None):
        b,n,c=x.shape
        z=self.qkv(x).reshape(b,n,3,2,4).permute(2,0,3,1,4)
        q,k,v=z[0],z[1],z[2]
        scores=(q*self.scale) @ k.transpose(-1,-2)
        bias=self.relative_position_bias_table[self.relative_position_index[:n,:n].reshape(-1)].reshape(n,n,2).permute(2,0,1)
        scores=scores+bias.unsqueeze(0)
        if mask is not None:
            nw=mask.shape[0]
            scores=(scores.reshape(b//nw,nw,2,n,n)+mask.unsqueeze(0).unsqueeze(2)).reshape(b,2,n,n)
        weights=torch.softmax(scores,dim=-1)
        out=(weights@v).transpose(1,2).reshape(b,n,c)
        return self.proj_drop(self.proj(out))


class TinySwin(nn.Module):
    def __init__(self):
        super().__init__();self.attn=FakeMonaiWindowAttention()
        self.out=nn.Linear(8,1)
    def forward(self,x):
        return self.out(self.attn(x,mask=None)).mean()


def test_exact_maths_unshifted():
    torch.manual_seed(3)
    m=FakeMonaiWindowAttention().eval();x=torch.randn(3,8,8)
    assert torch.allclose(m(x),decompose_monai_attention(m,x)['output'],atol=1e-7)


def test_exact_maths_shifted_additive_mask():
    torch.manual_seed(4)
    m=FakeMonaiWindowAttention().eval();x=torch.randn(4,8,8)
    mask=torch.zeros(2,8,8)
    mask[0,:4,4:]=-100;mask[1,4:,:4]=-100
    assert torch.allclose(m(x,mask),decompose_monai_attention(m,x,mask)['output'],atol=1e-7)


@pytest.mark.parametrize('kind',['value_message','qk_logit'])
def test_single_atomic_message_exact_and_nonzero(kind):
    torch.manual_seed(9)
    m=FakeMonaiWindowAttention().eval()
    x=torch.randn(3,8,8);reference=x.clone();reference[0,2,:]+=4
    atom=AttentionAtom('attn',0,0,0,2,kind)
    y=patched_attention_output(m,x,reference,None,atom)
    assert y.shape==m(x).shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(y[0,0],m(x)[0,0])
    assert torch.allclose(y[1:],m(x)[1:],atol=1e-7)


def test_identity_patch_same_inputs():
    m=FakeMonaiWindowAttention().eval();x=torch.randn(2,8,8)
    for kind in ('value_message','qk_logit'):
        atom=AttentionAtom('attn',0,0,0,0,kind)
        assert torch.allclose(m(x),patched_attention_output(m,x,x,None,atom),atol=1e-7)


def test_forbidden_shift_and_bounds_fail():
    m=FakeMonaiWindowAttention().eval();x=torch.randn(2,8,8)
    mask=torch.zeros(2,8,8);mask[0,0,5]=-100
    with pytest.raises(ValueError,match='forbidden'):
        patched_attention_output(m,x,x,mask,AttentionAtom('attn',0,0,0,5))
    with pytest.raises(IndexError):
        patched_attention_output(m,x,x,None,AttentionAtom('attn',0,50,0,5))


def test_hooks_and_cleanup():
    m=TinySwin().eval();x=torch.randn(1,8,8);ref=x.clone();ref[:,2,:]*=3
    obs=MonaiSwinObserver(m)
    baseline=m(x)
    with obs.recording(selected=['attn']):
        assert torch.equal(m(x),baseline)
    assert obs.windows['attn']['input'].shape==x.shape
    with obs.atom_patch(AttentionAtom('attn',0,0,0,2),ref) as called:
        changed=m(x)
    assert called==[1]
    assert not torch.allclose(changed,baseline)
    assert torch.equal(m(x),baseline)
    assert obs.inventory()[0]['type']=='window_attention'


def test_missing_site_and_exception_cleanup():
    m=TinySwin().eval();obs=MonaiSwinObserver(m)
    with pytest.raises(KeyError):
        with obs.recording(selected=['does_not_exist']):pass
    with pytest.raises(RuntimeError):
        with obs.atom_patch(AttentionAtom('attn',0,0,0,0),torch.randn(1,8,8)):
            raise RuntimeError('deliberate')
    assert torch.isfinite(m(torch.randn(1,8,8)))


def test_target_output_validation():
    x=torch.arange(64,dtype=torch.float).reshape(1,1,4,4,4)
    mask=torch.zeros(4,4,4,dtype=torch.bool);mask[0,0,0]=True
    assert float(target_mean_logit(x,0,mask))==0
    with pytest.raises(ValueError):target_mean_logit(x,0,torch.ones(2,2,2))


def test_wrong_module_and_training_fail():
    m=FakeMonaiWindowAttention().train();x=torch.randn(1,8,8)
    with pytest.raises(ValueError,match='eval'):
        decompose_monai_attention(m,x)
    with pytest.raises(TypeError):decompose_monai_attention(nn.Linear(8,8).eval(),x)


def test_missing_library_graceful():
    try:
        import monai
    except ImportError:
        with pytest.raises(RuntimeError,match='MONAI 1.5.1'):
            require_monai()


def test_window_geometry_shift_padding_and_invalid_positions():
    from brainmirror.scept.monai_adapter import window_geometry
    class Block:
        window_size=(2,2,2)
        shift_size=(1,1,1)
    g=window_geometry(Block(),(5,5,5))
    assert g.window_counts==(3,3,3)
    assert g.total_windows==27
    assert g.position(0,0)==(1,1,1)
    assert g.position(26,7)==(0,0,0)  # wraparound from rolled padded grid
    assert g.position(26,0) is None   # padded position
    with pytest.raises(IndexError):g.position(27,0)


def test_window_geometry_shift_disabled_on_small_axis():
    from brainmirror.scept.monai_adapter import window_geometry
    class Block:
        window_size=(3,3,3)
        shift_size=(1,1,1)
    g=window_geometry(Block(),(2,4,5))
    assert g.window_size==(2,3,3)
    assert g.shift_size==(0,1,1)


def test_rank_roi_atoms_respects_geometry_and_mask():
    from brainmirror.scept.monai_adapter import WindowGeometry,rank_monai_roi_atoms
    torch.manual_seed(54)
    model=TinySwin().eval();obs=MonaiSwinObserver(model)
    x=torch.randn(1,8,8)
    with obs.recording(selected=['attn']):
        model(x)
    obs.geometry['attn']=WindowGeometry((2,2,2),(2,2,2),(0,0,0))
    roi=torch.zeros(4,4,4,dtype=torch.bool)
    roi[:2,:2,:2]=True
    results=rank_monai_roi_atoms(obs,roi,max_atoms=3)
    assert len(results)==3
    assert all(r['evidence_status']=='screened_only' for r in results)
    assert all(r['atom']['module']=='attn' for r in results)
    with pytest.raises(ValueError):
        rank_monai_roi_atoms(obs,torch.zeros_like(roi),max_atoms=3)
