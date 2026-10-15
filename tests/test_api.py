from fastapi.testclient import TestClient
from brainmirror.server import app


def test_api_observation_has_real_edges():
    c=TestClient(app)
    s=c.get('/api/state')
    assert s.status_code==200
    assert s.json()['layers']['conv3']['side']==4
    r=c.post('/api/analyze',json={'strength':1.0,'layer':'conv3','channel':1,'z':2,'y':2,'x':2,'ablate':True})
    assert r.status_code==200, r.text
    body=r.json()
    g=body['graph']
    assert len(g['edges'])>0
    assert abs(g['preactivation']-g['verified_sum']) < 2e-5
    assert body['summary']['ablation_mean_abs_output_change'] is not None
    assert c.get('/api/runs/'+body['run_id']+'.json').status_code==200


def test_invalid_neuron_is_rejected():
    c=TestClient(app)
    r=c.post('/api/analyze',json={'layer':'conv3','channel':99})
    assert r.status_code==400
