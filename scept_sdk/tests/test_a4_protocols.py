"""A4: exact U10 JS nuisance parity, actual ONNX three-forward cells and falsification."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from scept import SCEPT, SCEPTError
from scept.protocols import ProtocolRegistry, Mask, U10Plan, js_nuisance, paired_inputs, run_u10, u10_gates

ROOT=Path(__file__).resolve().parents[2]

def make_plan(model_hash='a'*64, **changes):
    target=Mask.from_indices('x',(1,8),(0,1),'target-numeric-not-anatomy')
    control=Mask.from_indices('x',(1,8),(6,7),'matched-control-not-anatomy')
    args=dict(input_name='x',target_tensor='out',target_index=0,
              target_mask=target,control_mask=control,model_sha256=model_hash,
              nuisance_sigma=0.,mean_tolerance=.05,std_tolerance=.05,
              output_scale=1.,min_target_effect=.1,min_specificity=.1,
              min_positive_fraction=1.,max_spread=.1,
              minimum_sign_agreement=1.,dose_tolerance=.05,
              expected_sign=-1)
    args.update(changes)
    return U10Plan(**args)

def make_real_onnx(weights):
    import onnx
    from onnx import helper,TensorProto,numpy_helper
    src=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,8])
    out=helper.make_tensor_value_info('out',TensorProto.FLOAT,[1,1])
    w=numpy_helper.from_array(np.asarray(weights,dtype=np.float32).reshape(8,1),'weights')
    graph=helper.make_graph([helper.make_node('MatMul',['x','weights'],['out'])],
                            'real-numeric-matmul',[src],[out],initializer=[w])
    model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid('',13)])
    model.ir_version=9
    return model.SerializeToString()

def test_python_js_exact_u10_seeded_nuisance_and_shared_masks():
    js=subprocess.run(['node','tests/u11_js_python_parity.cjs'],cwd=ROOT,
                      text=True,capture_output=True,check=True)
    reference=json.loads(js.stdout)
    plan=make_plan('a'*64,nuisance_sigma=.001,
                   mean_tolerance=.2,std_tolerance=.2)
    source=np.ones((1,8),dtype=np.float32)
    for cell in reference['results']:
        a,b,c,diagnostics=paired_inputs(source,plan,seed=109,
                                         strength=cell['dose'])
        np.testing.assert_array_equal(a,source)
        np.testing.assert_allclose(b.reshape(-1),cell['B'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(c.reshape(-1),cell['C'],atol=1e-7,rtol=0)
        assert diagnostics['targetMaskSha256']==cell['targetMaskSha256']
        assert diagnostics['controlMaskSha256']==cell['controlMaskSha256']
        assert diagnostics['meanDiff']==pytest.approx(cell['meanDiff'],abs=1e-8)
        assert diagnostics['stdDiff']==pytest.approx(cell['stdDiff'],abs=1e-8)
        if cell['dose']==0: np.testing.assert_array_equal(b,c)
    # Sanity check state PRNG deterministic, not NumPy RandomState.
    np.testing.assert_array_equal(js_nuisance(109,9),js_nuisance(109,9))

def test_real_onnx_u10_full_preregistered_study_passes_without_claiming_causality():
    model=make_real_onnx([1,1,0,0,0,0,0,0])
    obs=SCEPT(model,['out'],backend='onnx',profile='generic')
    plan=make_plan(obs._adapter.model_sha256)
    registered=ProtocolRegistry()
    registered.register('scept.u10.v1','1.0',run_u10)
    assert registered.list_protocols()[0]['name']=='scept.u10.v1'
    result=registered.run('scept.u10.v1',observer=obs,
                          input_tensor=np.ones((1,8),dtype=np.float32),plan=plan)
    assert result['nGenuineForwards']==27
    assert len(result['cells'])==9
    assert len({rid for r in result['cells'] for rid in r['runIds']})==27
    assert result['gates']['verdict']=='criterion_met'
    assert len(result['gates']['gates'])==7
    assert result['science']['causality']=='NOT_ESTABLISHED'
    assert all(r['matching']['targetMaskSha256']==plan.target_mask.indices_sha256
               for r in result['cells'])
    assert all(len(r['layerResponses'])==1 for r in result['cells'])
    assert all(r['controlledMagnitude']==0 for r in result['cells'] if r['dose']==0)
    assert result['gates']['metrics']['medianControlled']==pytest.approx(2.)
    assert u10_gates(plan,result['cells'][:-1])['verdict']=='not_evaluable'
    duplicated=list(result['cells'])
    duplicated[-1]=duplicated[0]
    assert u10_gates(plan,duplicated)['verdict']=='not_evaluable'

def test_zero_model_true_inference_must_fail_seven_gates():
    model=make_real_onnx([0]*8)
    obs=SCEPT(model,['out'],backend='onnx')
    plan=make_plan(obs._adapter.model_sha256)
    result=run_u10(observer=obs,input_tensor=np.ones((1,8),dtype=np.float32),plan=plan)
    assert result['nGenuineForwards']==27
    assert result['gates']['verdict']=='criterion_not_met'
    assert result['gates']['metrics']['medianControlled']==0.
    assert result['science']['performanceSuperiority']=='NOT_ESTABLISHED'

def test_mask_bad_index_source_hash_and_model_identity_rejected_before_inference():
    t=Mask.from_indices('x',(1,8),(0,1),'target')
    bad=Mask('x',(1,8),(0,1),'target','f'*64)
    with pytest.raises(SCEPTError,match='MASK_HASH_MISMATCH'):
        bad.validate('x',(1,8))
    with pytest.raises(SCEPTError,match='CONTROL_UNMATCHED'):
        make_plan(control_mask=t).validate((1,8))
    with pytest.raises(SCEPTError,match='INVALID_PLAN'):
        make_plan(strengths=(0.,.8,.4,1.)).validate((1,8))
    obs=SCEPT(make_real_onnx([1]*8),['out'],backend='onnx')
    with pytest.raises(SCEPTError,match='MODEL_MISMATCH'):
        run_u10(observer=obs,input_tensor=np.ones((1,8),dtype=np.float32),
                plan=make_plan('a'*64))

def test_plugin_registration_is_versioned_local_and_duplicate_rejected():
    registry=ProtocolRegistry()
    registry.register('custom.real','0.1',lambda x: x+4)
    assert registry.run('custom.real',x=3)==7
    with pytest.raises(SCEPTError,match='DUPLICATE_PLUGIN'):
        registry.register('custom.real','0.2',lambda x:x)
    with pytest.raises(SCEPTError,match='PLUGIN_NOT_FOUND'):
        registry.run('unknown')
