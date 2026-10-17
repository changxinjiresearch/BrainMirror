"""SCEPT v0.9: controlled BraTS2020 inventory/train/test/SCCA CLI.

Examples (with real authorised data and trained weights only):
 PYTHONPATH=. python scripts/scept_brats_v09.py prepare --root /kaggle/input/.../BraTS2020_TrainingData --output runs/brats_v09/split.json
 PYTHONPATH=. python scripts/scept_brats_v09.py train --root ... --split runs/brats_v09/split.json --checkpoint /kaggle/working/scept_v09.pt --output /kaggle/working/train_provenance.json
 PYTHONPATH=. python scripts/scept_brats_v09.py evaluate --root ... --split ... --checkpoint /kaggle/working/scept_v09.pt --checkpoint-sha256 64_HEX ...
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from brainmirror.scept.brats2020 import audited_inventory, discover_training_cases, patient_split
from brainmirror.scept.brats_v09_pipeline import (
    attest_metadata, evaluate_heldout, explain_heldout_case, load_attested_model,
    train_brats, sha256_file,
)
from brainmirror.scept.brats2020 import load_case


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def get_inventory(root,*,expected_count):
    cases=discover_training_cases(Path(root),strict=True)
    inv=audited_inventory(cases,root=Path(root),expected_count=expected_count)
    return cases,inv


def read_split(path,ids,inventory_digest):
    item=json.loads(Path(path).read_text())
    if item.get('schema')!='scept/brats2020-split/v1':raise ValueError('Split schema mismatch')
    if item.get('inventory_sha256')!=inventory_digest:raise ValueError('Split inventory digest mismatch')
    splits=item['splits']
    from brainmirror.scept.brats_v09_pipeline import require_clean_splits
    require_clean_splits(splits)
    if set().union(*map(set,splits.values()))!=set(ids):raise ValueError('Split does not cover exactly the discovered cohort')
    return splits


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','train','evaluate','explain'):
        x=sub.add_parser(name)
        x.add_argument('--root',required=True,help='Local authorised BraTS2020 training NIfTI directory')
        x.add_argument('--expected-cases',type=int,default=369,
                       help='Use 369 for full official training cohort; can lower only for engineering smoke tests')
        x.add_argument('--output',required=True)
        if name!='prepare':x.add_argument('--split',required=True)
        if name in ('evaluate','explain'):
            x.add_argument('--checkpoint',required=True)
            x.add_argument('--checkpoint-sha256',required=True,help='Pinned SHA256 from training record')
            x.add_argument('--device',default='cuda')
        if name=='prepare':x.add_argument('--seed',type=int,default=240216)
        if name=='train':
            x.add_argument('--checkpoint',required=True)
            x.add_argument('--device',default='cuda')
            x.add_argument('--seed',type=int,default=240216)
            x.add_argument('--epochs',type=int,default=30)
            x.add_argument('--steps-per-epoch',type=int,default=120)
            x.add_argument('--patch-size',type=int,default=96)
            x.add_argument('--lr',type=float,default=1e-4)
        if name=='evaluate':
            x.add_argument('--patch-size',type=int,default=96)
            x.add_argument('--overlap',type=float,default=.25)
            x.add_argument('--max-cases',type=int)
        if name=='explain':
            x.add_argument('--case-id',required=True)
            x.add_argument('--region',choices=['WT','TC','ET'],default='WT')
            x.add_argument('--patch-size',type=int,default=64)
            x.add_argument('--max-exact',type=int,default=8)
            x.add_argument('--evaluation-report',required=True,
                           help='Frozen whole-cohort independent evaluation report')
    return p


def main():
    args=parser().parse_args()
    cases,inv=get_inventory(args.root,expected_count=args.expected_cases)
    ids=[x.case_id for x in cases]
    if args.command!='prepare' and args.expected_cases!=369:
        raise ValueError('Real-data training/evaluation require the full 369-case BraTS2020 cohort')
    if args.command=='prepare':
        splits=patient_split(ids,seed=args.seed)
        save_json(args.output,{'schema':'scept/brats2020-split/v1',
                  'inventory_sha256':inv['case_manifest_sha256'],'seed':args.seed,
                  'splits':splits,'data':'external_not_embedded',
                  'split_sha256':hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
                  'grade_counts':inv['grade_counts']})
        print(json.dumps({'stage':'prepared','subjects':len(cases),
                          'split_counts':{k:len(v) for k,v in splits.items()},
                          'inventory_sha256':inv['case_manifest_sha256']}))
        return
    splits=read_split(args.split,ids,inv['case_manifest_sha256'])
    if args.command=='train':
        audit=train_brats(cases,splits,inventory_digest=inv['case_manifest_sha256'],
              output=args.checkpoint,patch_size=args.patch_size,epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,lr=args.lr,seed=args.seed,device=args.device)
        save_json(args.output,audit)
        print(json.dumps({'stage':'trained_not_qualified','steps':len(audit['loss_trace']),
                          'checkpoint_sha256':audit['sha256']}));return
    model,stamp=load_attested_model(args.checkpoint,splits,sha256=args.checkpoint_sha256,
                                   inventory_digest=inv['case_manifest_sha256'],device=args.device)
    if args.command=='evaluate':
        report=evaluate_heldout(model,cases,splits,checkpoint_sha256=stamp['sha256'],
                inventory_digest=inv['case_manifest_sha256'],patch_size=args.patch_size,
                overlap=args.overlap,max_cases=args.max_cases,device=args.device)
        report['evaluation_protocol_pinned']=True
        save_json(args.output,report)
        print(json.dumps({'stage':'heldout_evaluation','cases':report['n_evaluated'],
                          'mean_dice':report['mean_dice'],'partial':report['partial_evaluation']}));return
    if args.command=='explain':
        prior=json.loads(Path(args.evaluation_report).read_text())
        if prior.get('schema')!='scept/brats2020-evaluation/v1' or prior.get('checkpoint_sha256')!=stamp['sha256'] \
            or prior.get('inventory_sha256')!=inv['case_manifest_sha256']:
            raise ValueError('No matching frozen independent checkpoint evaluation')
        if prior.get('partial_evaluation') or prior.get('n_evaluated',0)<10:
            raise ValueError('At least 10 independent test cases and complete evaluation required')
        # Explicit, predeclared suitability thresholds; not claims of clinical quality.
        standards={'WT':.65,'TC':.45,'ET':.35}
        for reg,limit in standards.items():
            dice=prior.get('mean_dice',{}).get(reg)
            if dice is None or dice<limit:
                raise ValueError('Insufficient segmentation quality for mechanism study: '+reg)
        if args.case_id not in splits['test']:
            raise ValueError('SCCA interpretation restricted to precommitted independent test IDs')
        selected=next(c for c in cases if c.case_id==args.case_id)
        x,y,qc=load_case(selected)
        report=explain_heldout_case(model,x,y,roi_region=args.region,
                 patch_size=args.patch_size,max_exact=args.max_exact,device=args.device)
        report.update({'checkpoint_sha256':stamp['sha256'],
                'inventory_sha256':inv['case_manifest_sha256'],
                'test_case_id':args.case_id,'real_case_qc':qc,
                'not_a_patient_independent_reference':'Second reference is another perturbation of same subject',
                'baseline_quality_gate_passed':True})
        save_json(args.output,report)
        print(json.dumps({'stage':'scca_local_patch','status':report['status'],
                          'case_id':args.case_id,'region':args.region}));return

if __name__=='__main__':main()
