"""Dependency-free graph-control tests; real MONAI is tested in separate CI."""
import pytest
import torch
from torch import nn
from brainmirror.scept.monai_adapter import AttentionAtom
from brainmirror.scept.monai_crosslayer import (
    capture_monai_reference,run_isolated_monai_chain,discover_monai_crosslayer,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim=2;self.num_heads=1;self.scale=2**-0.5;self.window_size=(1,1,2)
        self.qkv=nn.Linear(2,6,bias=False)
        self.proj=nn.Linear(2,2,bias=False)
        self.attn_drop=nn.Dropout(0);self.proj_drop=nn.Dropout(0)
        self.relative_position_bias_table=nn.Parameter(torch.zeros(3,1))
        self.register_buffer('relative_position_index',torch.tensor([[1,0],[2,1]]))
        with torch.no_grad():
            self.qkv.weight.copy_(torch.eye(2).repeat(3,1))
            self.proj.weight.copy_(torch.eye(2))

    def forward(self,x,mask=None):
        b,n,c=x.shape
        qkv=self.qkv(x).reshape(b,n,3,1,2).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        scores=(q*self.scale)@k.transpose(-2,-1)
        scores=scores+self.relative_position_bias_table[self.relative_position_index].permute(2,0,1)
        weights=scores.softmax(dim=-1)
        return self.proj((weights@v).transpose(1,2).reshape(b,n,c))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__();self.attn=TinyAttention();self.mlp=nn.Linear(2,2,bias=False)
        self.window_size=(1,1,2);self.shift_size=(0,0,0)
        with torch.no_grad():self.mlp.weight.copy_(torch.eye(2)*0.17)

    def forward(self,x):
        x=x+self.attn(x.reshape(1,2,2)).reshape_as(x)
        return x+self.mlp(x)


class TinyStage(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([TinyBlock()])

    def forward(self,x):
        for b in self.blocks:x=b(x)
        return x


class TinyDecoder(nn.Module):
    def forward(self,x,skip):return x+skip


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinViT=nn.Module();self.swinViT.layers1=nn.ModuleList([TinyStage()]);self.swinViT.layers2=nn.ModuleList([TinyStage()]);
        self.swinViT.layers3=nn.ModuleList([TinyStage()]);self.swinViT.layers4=nn.ModuleList([TinyStage()])
        self.decoder5=TinyDecoder()

    def forward(self,x):
        tok=x[:, :2].permute(0,2,3,4,1).contiguous()
        for stage in (self.swinViT.layers1,self.swinViT.layers2,self.swinViT.layers3,self.swinViT.layers4):
            tok=stage[0](tok)
        tok=self.decoder5(tok,torch.zeros_like(tok))
        return tok.permute(0,4,1,2,3)


def fixtures():
    torch.manual_seed(128)
    model=TinyModel().eval()
    base=torch.tensor([0.4,1.3],dtype=torch.float32).reshape(1,1,1,1,2)
    reference=base.repeat(1,2,1,1,1)
    clean=reference.clone();clean[0,0,0,0,0]+=2.0
    roi=torch.ones((1,1,2),dtype=torch.bool)
    atom=lambda layer:AttentionAtom(f'swinViT.layers{layer}.0.blocks.0.attn',0,0,0,0,'value_message')
    return model,clean,reference,roi,atom


def test_real_numeric_baseline_and_explicit_chain_bypass_cut():
    m,clean,ref,roi,atom=fixtures()
    with torch.no_grad():
        snapshot=capture_monai_reference(m,clean,ref,roi=roi)
        path=[atom(1),atom(2)]
        full=run_isolated_monai_chain(m,ref,snapshot,path,roi=roi)
        cut=run_isolated_monai_chain(m,ref,snapshot,path,roi=roi,cut_at=1)
        source=run_isolated_monai_chain(m,ref,snapshot,path[:1],roi=roi)
        baseline=m(ref)
    assert len(full['hops'])==2
    assert full['hops'][0]['attention_delta_l2']>0
    assert abs(full['effect_from_reference'])>1e-8
    assert torch.allclose(cut['logits'],baseline,atol=1e-5)
    assert abs(cut['effect_from_reference'])<1e-5
    assert torch.equal(baseline,m(ref))
    assert not torch.allclose(source['logits'],baseline)


def test_order_reversal_and_duplicate_rejected_and_hooks_clean():
    m,clean,ref,roi,a=fixtures()
    with torch.no_grad():
        snap=capture_monai_reference(m,clean,ref,roi=roi)
        original=m(ref).clone()
        for path in ([a(2),a(1)],[a(1),a(1)]):
            with pytest.raises(ValueError):
                run_isolated_monai_chain(m,ref,snap,path,roi=roi)
        with pytest.raises(ValueError):
            run_isolated_monai_chain(m,ref,snap,[a(1),a(2)],roi=roi,cut_at=5)
        after=m(ref)
    assert torch.equal(original,after)
    assert all(not v._forward_hooks for v in m.modules())


def test_full_auto_discovery_and_second_reference_abstention_contract():
    m,clean,ref,roi,_=fixtures()
    held=ref.clone();held[0,0,0,0,0]*=-0.7
    with torch.no_grad():
        result=discover_monai_crosslayer(m,clean,ref,roi,max_hops=3,max_sites=3,
            per_step=2,beam_width=2,max_exact=6,kinds=('value_message',),heldout_reference=held,effect_tol=1e-8)
    assert result['forward_budget_used']<=6
    assert result['screened']>0
    assert result['scope']=='controlled_residual_atom_cascade_not_unique_circuit'
    assert result['status'] in ('cross_reference_supported','conflicting','unresolved')
    if result['selected_path']:
        assert len(result['selected_path'])>=2
        assert result['negative_control']['max_logit_delta']<3e-5
        assert all('hops' in x for x in result['candidate_paths'])
    assert all(len(x['path'])<=3 for x in result['candidate_paths'])


def test_source_reference_equal_refuses_fabricated_circuit():
    m,clean,ref,roi,_=fixtures()
    with torch.no_grad():
        report=discover_monai_crosslayer(m,ref,ref,roi,max_hops=2,kinds=('value_message',),max_exact=3)
    assert report['status']=='unresolved'
    assert report['reason']=='no_changed_roi_atoms'
    assert report.get('selected_path') is None


def test_fail_closed_on_invalid_shapes_and_training():
    m,clean,ref,roi,a=fixtures()
    with torch.no_grad():
        with pytest.raises(ValueError):
            capture_monai_reference(m,clean,ref.repeat(2,1,1,1,1))
    m.train()
    with torch.no_grad():
        with pytest.raises(ValueError):capture_monai_reference(m,clean,ref)
    m.eval()
    with torch.no_grad():
        with pytest.raises(ValueError):discover_monai_crosslayer(m,clean,ref,torch.zeros_like(roi))
