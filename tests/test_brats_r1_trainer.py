"""R1 regression: durable dual-slot recovery, exact optimizer/RNG resume."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from brainmirror.scept.brats_r1_trainer import (
    MirroredCheckpointStore, R1Config, run_resumable_training,
)


def stores(tmp_path):
    a=tmp_path/'local';b=tmp_path/'backup'
    return (MirroredCheckpointStore(a,b,prefix='training'),
            MirroredCheckpointStore(a,b,prefix='best'))


def inputs(salt):
    def f(case_id,step,py_rng,np_rng):
        a=py_rng.random() + float(np_rng.random())+float(torch.rand(1))
        x=torch.ones(1,4,4,4,4)*a
        y=torch.ones(1,3,4,4,4)*(.3 if case_id=='A' else .7)
        return x,y
    return f


def metrics(model):
    # Artificial deterministic validation instrument, not clinical performance.
    return {'WT':.42,'TC':.40,'ET':.39}


def run(tmp_path, *, resume=False, stop=None, seed=12, split=None, epochs=2):
    torch.manual_seed(40)
    model=nn.Conv3d(4,3,1,bias=True)
    ts,bs=stores(tmp_path)
    split=split or {'train':['A','B'],'validation':['C'],'test':['D']}
    return run_resumable_training(model=model,splits=split,
            inventory_sha256='a'*64,architecture={'mock':'conv'},
            config=R1Config(epochs=epochs,steps_per_epoch=4,checkpoint_every=2,
                            patience_evaluations=10,seed=seed,amp=False),
            training_store=ts,best_store=bs,sample_fn=inputs(0),
            validation_fn=metrics,device='cpu',resume=resume,
            stop_after_steps=stop),model


def state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


def test_interrupted_resume_matches_uninterrupted_bitwise(tmp_path):
    direct,md=run(tmp_path/'control')
    first,m1=run(tmp_path/'recover',stop=3)
    assert first['interrupted'] and first['steps']==3 and not first['completed']
    resumed,m2=run(tmp_path/'recover',resume=True)
    assert resumed['completed'] and resumed['steps']==8
    assert resumed['validation_history']==direct['validation_history']
    assert all(torch.equal(state(md)[k],state(m2)[k]) for k in state(md))
    assert resumed['best_epoch']==1 and resumed['best_available']


def test_backup_restores_corrupted_local_and_rejects_corrupted_slot(tmp_path):
    ts,bs=stores(tmp_path)
    ts.save({'x':torch.tensor([10])})
    ts.save({'x':torch.tensor([20])})
    latest=ts.verified()[0]
    assert latest[0]==1
    lp=ts.local/f'training_slot{latest[1]}.pt'
    lp.write_bytes(b'corrupt-local')
    item,_=ts.load()
    assert item['x'].item()==20
    assert lp.read_bytes()==latest[2].read_bytes()
    # Corrupt latest backup: fall back to previous intact generation.
    latest[2].write_bytes(b'bad-backup')
    item,manifest=ts.load()
    assert item['x'].item()==10 and manifest['generation']==0


def test_config_and_split_mismatch_rejected(tmp_path):
    run(tmp_path,stop=2)
    with pytest.raises(ValueError,match='fingerprint'):
        run(tmp_path,resume=True,seed=13)
    with pytest.raises(ValueError,match='fingerprint'):
        run(tmp_path,resume=True,split={'train':['A','C'],'validation':['B'],'test':['D']})


def test_new_run_cannot_clobber_existing_checkpoint(tmp_path):
    run(tmp_path,stop=2)
    with pytest.raises(ValueError,match='Existing checkpoints'):
        run(tmp_path,resume=False)


def test_invalid_storage_and_validation_fail_closed(tmp_path):
    with pytest.raises(ValueError,match='separate'):
        MirroredCheckpointStore(tmp_path/'a',tmp_path/'a'/'sub',prefix='unsafe')
    ts,bs=stores(tmp_path)
    m=nn.Conv3d(4,3,1)
    with pytest.raises(ValueError,match='Patient leakage'):
        run_resumable_training(model=m,splits={'train':['A'],'validation':['C'],'test':['A']},
              inventory_sha256='a'*64,architecture={},config=R1Config(epochs=1,steps_per_epoch=1),
              training_store=ts,best_store=bs,sample_fn=inputs(0),validation_fn=metrics)
    def invalid(m):return {'WT':.3,'TC':float('nan'),'ET':.4}
    with pytest.raises(ValueError,match='validation Dice'):
        run_resumable_training(model=m,splits={'train':['A'],'validation':['C'],'test':['D']},
              inventory_sha256='a'*64,architecture={},config=R1Config(epochs=1,steps_per_epoch=1),
              training_store=ts,best_store=bs,sample_fn=inputs(0),validation_fn=invalid)


def test_best_model_metadata_and_epoch_boundary_preemption(tmp_path):
    report,_=run(tmp_path,stop=4)
    assert not report['completed'] and report['best_epoch']==1
    ts,bs=stores(tmp_path)
    best,bm=bs.load()
    assert best['provenance']['optimization_steps']==4
    assert best['provenance']['train_ids']==['A','B']
    assert best['provenance']['validation_ids']==['C']
    assert 'D' not in best['provenance']['train_ids']
    assert bm['metadata']['score']==pytest.approx((.42+.40+.39)/3)
    resumed,_=run(tmp_path,resume=True)
    assert resumed['steps']==8 and len(resumed['validation_history'])==2
