"""SCEPT v0.9-R1 — durable, resumable MONAI BraTS2020 training.

Requires writable, ACTUALLY persistent backup storage mounted independently of
Kaggle's ephemeral working directory. Checkpoint folders are never uploaded to
public GitHub. Separate v0.9 `evaluate` performs frozen test-set evaluation.

Run `python -m scripts.scept_brats_v09_r1 --help` from repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from brainmirror.scept.brats2020 import discover_training_cases, audited_inventory, load_case, dice_regions
from brainmirror.scept.brats_v09_pipeline import (ARCH_DEFAULT, patch_for_train, predict_full,
    load_attested_model, evaluate_heldout)
from brainmirror.scept.monai_adapter import build_monai_model
from brainmirror.scept.brats_r1_trainer import R1Config, MirroredCheckpointStore, run_resumable_training
from scripts.scept_brats_v09 import read_split, save_json


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Mounted licensed BraTS2020 training images')
    p.add_argument('--split', required=True, help='Frozen prepare split JSON')
    p.add_argument('--work-dir', required=True, help='Local /kaggle/working checkpoint directory')
    p.add_argument('--backup-dir', required=True, help='Independently persistent mounted storage, writable')
    p.add_argument('--confirm-persistent-backup', action='store_true', help='Acknowledge backup-dir is on persistent storage, not /kaggle/working')
    p.add_argument('--report', required=True, help='Write no-PHI training status report JSON')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps-per-epoch', type=int, default=120)
    p.add_argument('--checkpoint-every', type=int, default=20)
    p.add_argument('--validate-every', type=int, default=1)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--patch-size', type=int, default=96)
    p.add_argument('--validation-patch', type=int, default=96)
    p.add_argument('--validation-overlap', type=float, default=.25)
    p.add_argument('--seed', type=int, default=240216)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-full-preflight', action='store_true', help='Engineering only; not a formal scientific run')
    p.add_argument('--stop-after-steps', type=int, default=None, help='Engineering preemption drill ONLY')
    p.add_argument('--skip-heldout-evaluation', action='store_true', help='Only for engineering tests; full real run should evaluate once on 56 untouched test cases')
    return p.parse_args()


def main():
    args=parse_args()
    if not args.confirm_persistent_backup:
        raise SystemExit('Refusing to train: --confirm-persistent-backup required. Local Kaggle Save Version is not a durable backup.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but no GPU is available; refusing to silently use CPU for real training')
    if args.patch_size%32 or args.patch_size<64 or args.validation_patch%32 or args.validation_patch<64:
        raise SystemExit('Formal InstanceNorm SwinUNETR requires patch >=64 and divisible by 32')
    cases=discover_training_cases(Path(args.root),strict=True)
    inv=audited_inventory(cases,root=Path(args.root),expected_count=369)
    split=read_split(args.split,[x.case_id for x in cases],inv['case_manifest_sha256'])
    case_map={x.case_id:x for x in cases}
    if not args.skip_full_preflight:
        for case in cases:
            load_case(case)  # Read each of the four modalities + labels; reject geometry/QC errors.
        print(json.dumps({'full_data_preflight':'PASS','n_checked':len(cases)}),flush=True)
    if set(split['train']) & set(split['validation']) or set(split['train']) & set(split['test']):
        raise SystemExit('Data leakage found: refusing to train')
    arch=dict(ARCH_DEFAULT)
    arch['use_checkpoint']=True
    # Seed BEFORE weight initialisation. Resume restores captured RNG thereafter.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    model=build_monai_model(**arch)
    working=Path(args.work_dir)
    backup=Path(args.backup_dir)
    training_store=MirroredCheckpointStore(working,backup,prefix='scept_r1_training')
    best_store=MirroredCheckpointStore(working,backup,prefix='scept_r1_best')
    # Preserve the frozen split outside the ephemeral Kaggle work directory.
    canonical_split=backup/'scept_r1_split.json'
    split_bytes=Path(args.split).read_bytes()
    if canonical_split.exists():
        if canonical_split.read_bytes()!=split_bytes:
            raise ValueError('Persistent frozen split differs from the requested split')
    else:
        from brainmirror.scept.brats_r1_trainer import _atomic_bytes
        _atomic_bytes(canonical_split,split_bytes)
    config=R1Config(epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,
         checkpoint_every=args.checkpoint_every,validate_every_epochs=args.validate_every,
         patience_evaluations=args.patience,seed=args.seed,lr=args.lr)

    # Limit resident normalized patient data; real BraTS 240x240x155 x 4 is large.
    cached={'case':None,'value':None}
    def fetch(name):
        if cached['case'] != name:
            cached['value']=load_case(case_map[name])
            cached['case']=name
        return cached['value']

    def sample_fn(case_id,step,py_rng,np_rng):
        x,y,_=fetch(case_id)
        return patch_for_train(x,y,size=args.patch_size,seed=args.seed+step,
                               positive=step%3!=0)

    def validate_fn(m):
        # The full, immutable 55-subject validation set, not the held-out 56.
        m.eval()
        records={k:[] for k in ('WT','TC','ET')}
        for case_id in sorted(split['validation']):
            x,y,_=fetch(case_id)
            pred=predict_full(m,x,patch_size=args.validation_patch,
                              overlap=args.validation_overlap,device=args.device)
            scores=dice_regions(pred,y)
            for region in records:
                if scores[region]['dice'] is not None:
                    records[region].append(scores[region]['dice'])
        if not all(records.values()):
            raise ValueError('Full validation cohort has undefined region score')
        return {k:float(np.mean(v)) for k,v in records.items()}

    def on_progress(event):
        # No PHI; latest snapshot is also stored in the external backup mount.
        save_json(working/'scept_r1_progress.json',event)
        save_json(backup/'scept_r1_progress.json',event)
        print(json.dumps(event,allow_nan=False),flush=True)

    report=run_resumable_training(model=model,splits=split,
          inventory_sha256=inv['case_manifest_sha256'],architecture=arch,
          config=config,training_store=training_store,best_store=best_store,
          sample_fn=sample_fn,validation_fn=validate_fn,device=args.device,
          resume=args.resume,stop_after_steps=args.stop_after_steps,
          on_progress=on_progress)
    report.update({'dataset':'BraTS2020_369','subjects':{k:len(v) for k,v in split.items()},
         'backup_is_user_attested_not_independently_verified':True,
         'model_selection':'full_validation_macro_WT_TC_ET_Dice',
         'test_set_used':False,
         'best_checkpoint_directory':str(backup),
         'remaining_task':'Evaluate BEST checkpoint independently using v0.9 evaluate on 56 held-out cases'})
    if report['completed'] and not report['interrupted'] and not args.skip_heldout_evaluation:
        selected,stamp=best_store.load()
        path=best_store.backup/f"scept_r1_best_slot{stamp['slot']}.pt"
        report['best_checkpoint_sha256']=stamp['sha256']
        report['best_checkpoint_path']=str(path)
        target=backup/'scept_r1_heldout_evaluation.json'
        if target.exists():
            earlier=json.loads(target.read_text(encoding='utf-8'))
            if (earlier.get('checkpoint_sha256')!=stamp['sha256'] or
                earlier.get('inventory_sha256')!=inv['case_manifest_sha256'] or
                earlier.get('n_evaluated')!=56 or earlier.get('partial_evaluation')):
                raise ValueError('Frozen test evaluation already exists for DIFFERENT model; refuse repeated test-set reuse')
            report['heldout_evaluation_status']='reused_existing_pinned_evaluation'
        else:
            trained,meta=load_attested_model(path,split,sha256=stamp['sha256'],
                       inventory_digest=inv['case_manifest_sha256'],device=args.device)
            eval_report=evaluate_heldout(trained,cases,split,
                       checkpoint_sha256=stamp['sha256'],
                       inventory_digest=inv['case_manifest_sha256'],
                       patch_size=args.validation_patch,overlap=args.validation_overlap,
                       device=args.device)
            if eval_report['n_evaluated']!=56 or eval_report.get('partial_evaluation'):
                raise RuntimeError('Independent evaluation is not complete; refusing success')
            eval_report.update({'selection':'best_full_55_validation_macro_dice',
                 'test_untouched_during_training':True,
                 'evaluation_protocol_pinned':True,
                 'user_supplied_real_brats_required':True,
                 'patient_count_expected':56})
            save_json(target,eval_report)
            report['heldout_evaluation_status']='completed_56_independent_cases'
            report['heldout_mean_dice']=eval_report['mean_dice']
            report['scientific_evaluation']='REAL_HELDOUT_SEGMENTATION_DICE_ONLY_NOT_SCEPT_MECHANISM_VALIDATION'
    else:
        report['heldout_evaluation_status']='NOT_RUN'
    save_json(args.report,report)
    save_json(backup/'scept_r1_training_report.json',report)
    print(json.dumps({'completed':report['completed'],'steps':report['steps'],
                      'best_score':report['best_score'],'best_epoch':report['best_epoch'],
                      'heldout_status':report['heldout_evaluation_status'],
                      'scientific_evaluation':report['scientific_evaluation']}))

if __name__=='__main__':main()
