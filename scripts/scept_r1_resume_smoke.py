"""Auditable SYNTHETIC fault recovery drill; not a BraTS patient model."""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from brainmirror.scept.brats_r1_trainer import MirroredCheckpointStore, R1Config, run_resumable_training


def experiment(dest, resume=False, stop=None):
    torch.manual_seed(40)
    model=nn.Conv3d(4,3,1)
    ts=MirroredCheckpointStore(dest/'local',dest/'backup',prefix='train')
    bs=MirroredCheckpointStore(dest/'local',dest/'backup',prefix='best')
    splits={'train':['train_a','train_b'],'validation':['valid_a'],'test':['heldout_a']}
    def sample(who,step,py,ng):
        v=py.random()+ng.random()+torch.rand(1).item()
        return torch.full((1,4,4,4,4),float(v)),torch.full((1,3,4,4,4),.5)
    report=run_resumable_training(model=model,splits=splits,
         inventory_sha256='0'*64,architecture={'synthetic':'conv3d'},
         config=R1Config(epochs=2,steps_per_epoch=4,checkpoint_every=2,amp=False),
         training_store=ts,best_store=bs,sample_fn=sample,
         validation_fn=lambda m:{'WT':.4,'TC':.3,'ET':.2},
         resume=resume,stop_after_steps=stop)
    return report,{k:v.detach().clone() for k,v in model.state_dict().items()},ts,bs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a=p.parse_args()
    with tempfile.TemporaryDirectory() as root:
        base=Path(root)
        direct,weights,_,_=experiment(base/'full')
        partial,_,_,_=experiment(base/'interrupted',stop=3)
        resumed,restored,ts,bs=experiment(base/'interrupted',resume=True)
        equality=all(torch.equal(weights[k],restored[k]) for k in weights)
        last=ts.verified()[0]
        local=ts.local/f'train_slot{last[1]}.pt'
        local.write_bytes(b'fault-injection-corrupt-local')
        from_backup,_=ts.load()
        recovered_local_checksum=(local.read_bytes()==last[2].read_bytes())
        audit={'schema':'scept/r1-synthetic-fault-drill/v1',
               'data':'synthetic_only_NO_BraTS_patient_NO_pretrained_model',
               'uninterrupted_steps':direct['steps'],
               'stopped_steps':partial['steps'],'resumed_steps':resumed['steps'],
               'restored_weight_tensors_bit_identical':bool(equality),
               'validation_history_identical':direct['validation_history']==resumed['validation_history'],
               'local_corruption_restored_from_sha256_verified_backup':recovered_local_checksum,
               'backup_generations':len(ts.verified()),
               'best_model_verified':bool(bs.verified()),
               'actual_patient_training':'NOT_PERFORMED',
               'independent_heldout_dice':'NOT_MEASURED'}
        if not all((audit['restored_weight_tensors_bit_identical'],audit['validation_history_identical'],
                   audit['local_corruption_restored_from_sha256_verified_backup'],audit['best_model_verified'])):
            raise RuntimeError('Synthetic resilience smoke failed')
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(audit,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
        print(json.dumps(audit))

if __name__=='__main__':main()
