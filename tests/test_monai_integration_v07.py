"""Integration tests against the INSTALLED official MONAI package, run in dedicated CI."""
import pytest
import torch

monai=pytest.importorskip('monai')
from brainmirror.scept.monai_adapter import (
    AttentionAtom,MonaiSwinObserver, MonaiDecoderTrace,
    atomic_bidirectional_effect, decompose_monai_attention, build_monai_model,
    substitute_decoder_skip
)
from brainmirror.scept.monai_vib import MonaiSwinVIBBridge


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return build_monai_model().eval()


@pytest.fixture(scope='module')
def input_volume():
    torch.manual_seed(100)
    return torch.randn(1,4,32,32,32)


def test_actual_monai_attention_eq_default_and_shifted(model,input_volume):
    observer=MonaiSwinObserver(model)
    with torch.no_grad(),observer.recording():
        result=model(input_volume)
    assert result.shape==(1,3,32,32,32)
    assert len(observer.windows)>=4
    shifted=0
    for name,info in observer.windows.items():
        module=observer.sites[name]
        x=info['input'];mask=info['mask']
        with torch.no_grad():
            expected=module(x,mask=mask)
            actual=decompose_monai_attention(module,x,mask)['output']
        assert torch.allclose(expected,actual,atol=2e-5,rtol=2e-5), (name,(expected-actual).abs().max().item())
        shifted+=int(mask is not None)
    assert shifted >= 0  # depth 1 means no shifted-window block


def test_actual_monai_bidirectional_and_hook_cleanup(model,input_volume):
    observer=MonaiSwinObserver(model)
    site=next(iter(observer.sites))
    reference=input_volume.clone()
    reference[:,:,:6,:6,:6]*=-0.8
    with torch.no_grad():
        before=model(input_volume).clone()
        result=atomic_bidirectional_effect(model,input_volume,reference,
            AttentionAtom(site,0,0,0,0,'value_message'))
        after=model(input_volume)
    assert torch.equal(before,after)
    assert result['evidence_status']=='intervened_not_crossvalidated'
    assert abs(result['spatial_effect'])>0


def test_actual_monai_original_decoder_trace_and_vib_equivalence(model,input_volume):
    tracer=MonaiDecoderTrace(model)
    with torch.no_grad(),tracer.recording(capture_tensors=True):
        original=model(input_volume)
    assert 'decoder5' in tracer.input_args
    assert 'encoder10' in tracer.activations
    bridge=MonaiSwinVIBBridge(model,feature_size=12,enabled=False).eval()
    with torch.no_grad():
        equivalent=bridge(input_volume)
    assert torch.allclose(original,equivalent,atol=1e-6,rtol=1e-6)
    bridge.enabled=True
    bridge.vib.mu_delta.bias.data.fill_(0.08)
    with torch.no_grad():
        altered=bridge(input_volume)
    assert torch.isfinite(altered).all()
    assert bridge.last_evidence['mu'].shape[1] == 192
    assert not torch.allclose(altered,original)


def test_actual_monai_decoder_skip_intervention(model,input_volume):
    trace=MonaiDecoderTrace(model)
    with torch.no_grad(),trace.recording(capture_tensors=True):
        original=model(input_volume)
    args=trace.input_args['decoder5']
    assert len(args)==2
    with torch.no_grad(),substitute_decoder_skip(model,'decoder5',torch.zeros_like(args[1])) as called:
        patched=model(input_volume)
    assert called==[True]
    assert torch.isfinite(patched).all()
    with torch.no_grad():
        same=model(input_volume)
    assert torch.equal(original,same)
    assert not torch.allclose(patched,original)


def test_actual_monai_shifted_and_window_geometry(input_volume):
    model_shift=build_monai_model(depths=(2,1,1,1)).eval()
    obs=MonaiSwinObserver(model_shift)
    with torch.no_grad(),obs.recording():
        model_shift(input_volume)
    shifted=[(name,record) for name,record in obs.windows.items() if record['mask'] is not None]
    assert shifted, 'No shifted window was executed; test configuration invalid'
    for name,record in shifted:
        m=obs.sites[name]
        actual=m(record['input'],mask=record['mask'])
        recomputed=decompose_monai_attention(m,record['input'],record['mask'])['output']
        assert torch.allclose(actual,recomputed,atol=2e-5,rtol=2e-5)
        geom=obs.geometry[name]
        assert geom.total_windows>=1
        assert geom.position(0,0) is not None
        break


def test_actual_monai_roi_to_atom_screening(model,input_volume):
    from brainmirror.scept.monai_adapter import rank_monai_roi_atoms,verify_ranked_monai_atoms
    obs=MonaiSwinObserver(model)
    first=next(iter(obs.sites))
    roi=torch.zeros(32,32,32,dtype=torch.bool)
    roi[:8,:8,:8]=True
    with torch.no_grad(),obs.recording(selected=[first]):
        model(input_volume)
    ranked=rank_monai_roi_atoms(obs,roi,max_atoms=2,sites=[first])
    assert len(ranked)==2
    reference=input_volume.clone();reference[:,:,:8,:8,:8]*=-0.8
    checked=verify_ranked_monai_atoms(model,input_volume,reference,ranked,class_index=0)
    assert len(checked)==2
    assert all(r['evidence_status']=='intervened_not_crossvalidated' for r in checked)
