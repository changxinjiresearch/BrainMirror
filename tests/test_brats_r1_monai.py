"""Runs on official MONAI 1.5.1 in CI (not a substitute for BraTS patients)."""
from __future__ import annotations

import pytest
import torch

monai=pytest.importorskip('monai')

from brainmirror.scept.monai_adapter import build_monai_model
from brainmirror.scept.brats_r1_trainer import R1Config, MirroredCheckpointStore, run_resumable_training
from brainmirror.scept.brats_v09_pipeline import load_attested_model


def test_official_monai_one_step_train_and_best_checkpoint_round_trip(tmp_path):
    torch.set_num_threads(2)
    model=build_monai_model(in_channels=4,out_channels=3,feature_size=12,
        depths=(1,1,1,1),num_heads=(3,3,6,12),window_size=2,
        norm_name='batch',use_checkpoint=True)
    arch=dict(in_channels=4,out_channels=3,feature_size=12,
        depths=[1,1,1,1],num_heads=[3,3,6,12],window_size=2,
        norm_name='batch',use_checkpoint=True)
    splits={'train':['p1','p2'],'validation':['v1'],'test':['holdout1']}
    inventory='e'*64
    ts=MirroredCheckpointStore(tmp_path/'work',tmp_path/'backup',prefix='training')
    bs=MirroredCheckpointStore(tmp_path/'work',tmp_path/'backup',prefix='best')
    def example(name,step,py,np_rng):
        x=torch.ones(2,4,32,32,32)*(0.2 if name=='p1' else 0.7)
        y=torch.zeros(2,3,32,32,32)
        y[:,:,10:16,10:16,10:16]=1.
        return x,y
    res=run_resumable_training(model=model,splits=splits,
        inventory_sha256=inventory,architecture=arch,
        config=R1Config(epochs=1,steps_per_epoch=1,checkpoint_every=1,
                         amp=False,patience_evaluations=2),
        training_store=ts,best_store=bs,sample_fn=example,
        validation_fn=lambda m:{'WT':.35,'TC':.25,'ET':.15},device='cpu')
    assert res['completed'] and res['steps']==1 and res['best_available']
    best,meta=bs.load()
    assert best['provenance']['train_ids']==['p1','p2']
    assert best['provenance']['validation_ids']==['v1']
    best_path=bs.backup/f"best_slot{meta['slot']}.pt"
    reloaded,stamp=load_attested_model(best_path,splits,sha256=meta['sha256'],
                                       inventory_digest=inventory,device='cpu')
    with torch.inference_mode():
        out=reloaded(torch.ones(1,4,32,32,32))
    assert out.shape==(1,3,32,32,32)
