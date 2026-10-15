#!/usr/bin/env python3
"""BrainMirror IXI T1 private train-only population MRI candidate builder.

Pre-registration: An affine-to-one-reference *candidate*, not an unbiased
population atlas, segmented brain atlas, or dynamic whole-brain simulation.
Sharded to protect Kaggle CPU sessions. Individual images NEVER leave Kaggle.

Commands:
  shard --manifest PRIVATE_ixi_subject_split_DRAFT.csv --pilot-root ...
        --synthstrip-dir ... --shard-index 0 [--shard-size 12] --out ...
  merge --manifest ... --pilot-root ... --shards-root ... --out ...
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SEED = 240215
ATLAS_SCHEMA = 'brainmirror.ixi.affine-candidate.v1'
ID_PATTERN = re.compile(r'^IXI\d{3}$', flags=re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for piece in iter(lambda: inp.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def manifest_train(manifest: Path, *, approved: bool = False) -> list[dict[str, str]]:
    with manifest.open(newline='', encoding='utf-8') as inp:
        reader = csv.DictReader(inp)
        required = {'subject_id', 'site', 'image_path', 'split', 'qc_approved', 'used_in_pilot5'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Split manifest missing required columns')
        rows = list(reader)
    all_ids = [r['subject_id'] for r in rows]
    if len(all_ids) != len(set(all_ids)) or not all(ID_PATTERN.fullmatch(s) for s in all_ids):
        raise ValueError('Invalid or duplicated subject IDs')
    if len(rows) != 581 or set(r['split'] for r in rows) != {'train', 'dev', 'test'}:
        raise ValueError('Cohort split missing subjects or held-out groups')
    if any(r['split'] != 'train' for r in rows if r['used_in_pilot5'] == 'true'):
        raise ValueError('Pilot 5 subject leaked into held-out dataset')
    train = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: r['subject_id'])
    if not train:
        raise ValueError('Empty training cohort')
    if approved and any(r['qc_approved'].lower() != 'true' for r in train):
        raise ValueError('Unapproved subjects in training cohort')
    # Input paths must be existing Kaggle input files; public Git never stores them.
    for r in train:
        p = Path(r['image_path'])
        if not (p.is_file() and p.name.lower().endswith(('.nii', '.nii.gz'))):
            raise FileNotFoundError('Training image missing from mounted input')
        if not str(p.resolve()).startswith('/kaggle/input/'):
            raise ValueError('Expected immutable Kaggle-mounted source image')
    return train


def shard_rows(train: list[dict[str, str]], index: int, size: int):
    if size < 1 or size > 60 or index < 0:
        raise ValueError('Invalid shard index/size')
    start = index * size
    if start >= len(train):
        raise ValueError('Shard index past training cohort')
    return train[start:start+size]


def metadata(manifest: Path, pilot_root: Path, shard_size: int):
    reference = pilot_root / 'case01' / 'brain.nii.gz'
    refmask = pilot_root / 'case01' / 'mask.nii.gz'
    if not reference.is_file() or not refmask.is_file():
        raise FileNotFoundError('Missing 5-case reference brain or mask')
    return {'schema': ATLAS_SCHEMA, 'manifest_sha256': sha256_file(manifest),
            'reference_sha256': sha256_file(reference),
            'reference_mask_sha256': sha256_file(refmask),
            'shard_size': shard_size, 'reference_voxel_mm': 2.0,
            'seed': SEED, 'alignment': 'affine_to_one_subject',
            'qc_status': 'REQUIRES_INDEPENDENT_REVIEW'}


def ref_geometry(reference_path, sitk):
    orig = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    sizes = [max(8, round((n-1)*s/2.0)+1) for n,s in zip(orig.GetSize(),orig.GetSpacing())]
    return sitk.Resample(orig, sizes, sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkLinear, orig.GetOrigin(), [2.0]*3,
                         orig.GetDirection(), 0., sitk.sitkFloat32)


def masked_normalize(data, mask, np):
    a = np.asarray(data, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if a.shape != m.shape or not np.isfinite(a).all() or m.sum() < 10000:
        raise ValueError('Invalid nonfinite image or empty brain mask')
    p1, p99 = np.percentile(a[m], [1, 99])
    if not np.isfinite((p1,p99)).all() or p99 <= p1:
        raise ValueError('Bad foreground intensity distribution')
    return np.where(m, np.clip((a-p1)/(p99-p1),0,1), 0).astype(np.float32)


def sitk_register(fixed, moving, sitk):
    start = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.025,seed=SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.5, minStep=.0005, numberOfIterations=120, relaxationFactor=.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4,2,1])
    reg.SetSmoothingSigmasPerLevel([2,1,0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(start, inPlace=False)
    transform = reg.Execute(fixed,moving)
    return transform, reg.GetMetricValue()


def save_aggregates(path: Path, sums, squares, count, np):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as out:
        np.savez_compressed(out, sums=sums, squares=squares, count=count)
    os.replace(tmp,path)


def run_shard(args):
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    manifest = Path(args.manifest)
    train = manifest_train(manifest, approved=args.approved)
    rows = shard_rows(train,args.shard_index,args.shard_size)
    pilot = Path(args.pilot_root)
    info = metadata(manifest,pilot,args.shard_size)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refuse overwrite of shard {out}')
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    script = Path(args.synthstrip_dir)/'mri_synthstrip'
    weights = Path(args.synthstrip_dir)/'synthstrip.1.pt'
    if not script.is_file() or not weights.is_file():
        raise FileNotFoundError('Existing SynthStrip script+weights required')
    fixed = ref_geometry(pilot/'case01'/'brain.nii.gz', sitk)
    ref_mask = sitk.Resample(
        sitk.ReadImage(str(pilot/'case01'/'mask.nii.gz'),sitk.sitkUInt8),
        fixed, sitk.Transform(3,sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ref_fore = sitk.GetArrayFromImage(ref_mask) > 0
    shape = ref_fore.shape
    sums=np.zeros(shape, dtype=np.float64)
    squares=np.zeros(shape,dtype=np.float64)
    count=np.zeros(shape,dtype=np.int16)
    errors=[]; qc=[]
    scratch = out/'scratch'
    scratch.mkdir()
    for i,row in enumerate(rows,1):
        brain=scratch/'brain.nii.gz'
        mask=scratch/'mask.nii.gz'
        for p in (brain,mask):
            p.unlink(missing_ok=True)
        source=row['image_path']
        proc=subprocess.run([sys.executable,str(script),'-i',source,'-o',str(brain),
                             '-m',str(mask),'--model',str(weights),'-t','2'],
                            capture_output=True,text=True)
        if proc.returncode:
            errors.append({'subject_id':row['subject_id'],'reason':'synthstrip_failed',
                           'return_code':proc.returncode})
            print(f'[{i}/{len(rows)}] extraction ERROR',flush=True)
            continue
        try:
            native=nib.load(source)
            mri_mask=nib.load(str(mask))
            if native.shape != mri_mask.shape or not np.allclose(native.affine,mri_mask.affine,atol=.001,rtol=0):
                raise ValueError('mask_affine_mismatch')
            moving=sitk.ReadImage(str(brain),sitk.sitkFloat32)
            tx,metric = sitk_register(fixed,moving,sitk)
            aligned=sitk.Resample(moving,fixed,tx,sitk.sitkLinear,0.,sitk.sitkFloat32)
            aligned_mask=sitk.Resample(sitk.ReadImage(str(mask),sitk.sitkUInt8),
                                       fixed,tx,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
            m=sitk.GetArrayFromImage(aligned_mask)>0
            dice=2*(m & ref_fore).sum()/max(1,int(m.sum()+ref_fore.sum()))
            if dice < args.min_dice:
                raise ValueError(f'low_overlap_dice_{dice:.4f}')
            a=masked_normalize(sitk.GetArrayFromImage(aligned),m,np)
            sums[m] += a[m]
            squares[m] += np.square(a[m])
            count[m] += 1
            qc.append({'subject_id':row['subject_id'],'site':row['site'],
                       'reference_mask_dice':float(dice),'registration_metric':float(metric),
                       'source_sha256':sha256_file(Path(source))})
            print(f'[{i}/{len(rows)}] PASS {row["site"]} maskDice={dice:.4f}',flush=True)
        except Exception as exc:
            errors.append({'subject_id':row['subject_id'],'reason':str(exc)[:180]})
            print(f'[{i}/{len(rows)}] review required: {type(exc).__name__}',flush=True)
    for p in (brain,mask): p.unlink(missing_ok=True)
    scratch.rmdir()
    save_aggregates(out/'PRIVATE_aggregates.npz',sums,squares,count,np)
    summary={**info,'shard_index':args.shard_index,
             'planned_subjects':len(rows),'included_subjects':len(qc),
             'flagged_subjects':len(errors),'source_tier': 'DRAFT_UNREVIEWED' if not args.approved else 'APPROVED_MANIFEST',
             'template_approved':False,'P1_PASS':False,'P2_TRAINING_AUTHORIZED':False,
             'private_qc':qc,'private_failures':errors}
    (out/'PRIVATE_shard_receipt.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('SHARD_COMPLETE',args.shard_index,'included',len(qc),'flagged',len(errors),flush=True)


def checked_shard_receipts(root:Path, meta:dict, shards:int):
    out=[]
    for k in range(shards):
        folder=root/f'shard_{k:03d}'
        receipt_path=folder/'PRIVATE_shard_receipt.json'
        aggregate=folder/'PRIVATE_aggregates.npz'
        if not receipt_path.exists() or not aggregate.exists():
            raise FileNotFoundError(f'Missing completed shard {k}: {folder}')
        rec=json.loads(receipt_path.read_text())
        if any(rec.get(field) != meta.get(field) for field in
               ('schema','manifest_sha256','reference_sha256','reference_mask_sha256','shard_size')):
            raise ValueError(f'Shard {k} has mismatched provenance')
        if rec['shard_index']!=k or rec['planned_subjects']!=rec['included_subjects']+rec['flagged_subjects']:
            raise ValueError(f'Shard {k} incomplete receipt')
        out.append((aggregate,rec))
    return out


def run_merge(args):
    import numpy as np
    import SimpleITK as sitk
    manifest=Path(args.manifest)
    train=manifest_train(manifest,approved=args.approved)
    meta=metadata(manifest,Path(args.pilot_root),args.shard_size)
    shards=(len(train)+args.shard_size-1)//args.shard_size
    records=checked_shard_receipts(Path(args.shards_root),meta,shards)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory not empty')
    fixed=ref_geometry(Path(args.pilot_root)/'case01'/'brain.nii.gz',sitk)
    shape=tuple(reversed(fixed.GetSize()))
    sums=np.zeros(shape,np.float64)
    squares=np.zeros(shape,np.float64)
    counts=np.zeros(shape,np.int32)
    complete=[]
    for path,rec in records:
        with np.load(path) as data:
            if data['sums'].shape != shape or data['count'].shape!=shape:
                raise ValueError('Shard aggregate geometry mismatch')
            sums+=data['sums']
            squares+=data['squares']
            counts+=data['count']
        complete.extend(rec['private_qc'])
    if len(complete)!=len({x['subject_id'] for x in complete}):
        raise ValueError('Duplicate subjects between shards')
    flagged=[x for _,rec in records for x in rec['private_failures']]
    encountered=[x['subject_id'] for x in complete]+[x['subject_id'] for x in flagged]
    if len(encountered)!=len(set(encountered)) or set(encountered)!={r['subject_id'] for r in train}:
        raise ValueError('Missing, foreign, or duplicated subject across completed shards')
    mean=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    sd=np.sqrt(np.maximum(0,np.divide(squares,counts,out=np.zeros_like(sums),where=counts>0)-mean*mean))
    out.mkdir(parents=True,exist_ok=True)
    out.chmod(0o700)
    for label,data in [('ixi_train_affine_mean.nii.gz',mean),
                       ('ixi_train_affine_sd.nii.gz',sd),('ixi_train_coverage.nii.gz',counts)]:
        img=sitk.GetImageFromArray(data.astype(np.float32))
        img.CopyInformation(fixed)
        sitk.WriteImage(img,str(out/label))
    aggregate={**meta,'train_subjects_planned':len(train),'subjects_included':len(complete),
               'subjects_flagged':len(flagged),'shards_expected':shards,
               'template_approved':False,'P1_PASS':False,'P2_TRAINING_AUTHORIZED':False,
               'status':'REAL_IXI_TRAIN_ONLY_AFFINE_CANDIDATE_UNREVIEWED',
               'note':'No spatial atlas labels, nonlinear atlas, clinical or dynamical brain simulation.'}
    (out/'PRIVATE_merge_receipt.json').write_text(json.dumps({**aggregate,'excluded':flagged},indent=2))
    print(json.dumps(aggregate,indent=2));print('PRIVATE candidate saved to',out)


def run_plan(args):
    from collections import Counter
    train = manifest_train(Path(args.manifest),approved=args.approved)
    info = metadata(Path(args.manifest),Path(args.pilot_root),args.shard_size)
    result = {**info,'train_count':len(train),
              'shard_count':(len(train)+args.shard_size-1)//args.shard_size,
              'training_site_count':dict(Counter(r['site'] for r in train)),
              'split_status':'draft_candidate' if not args.approved else 'requires_independent_signoff',
              'P1_PASS':False,'P2_TRAINING_AUTHORIZED':False}
    print(json.dumps(result,indent=2))

def parse_args():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','shard','merge'):
        sp=sub.add_parser(name)
        sp.add_argument('--manifest',required=True)
        sp.add_argument('--pilot-root',default='/kaggle/working/brainmirror_ixi_pilot5_v1')
        sp.add_argument('--shard-size',type=int,default=12)
        sp.add_argument('--approved',action='store_true',help='Require qc_approved for every train subject')
        if name != 'plan':sp.add_argument('--out',required=True)
        if name=='shard':
            sp.add_argument('--shard-index',type=int,required=True)
            sp.add_argument('--synthstrip-dir',default='/kaggle/working/brainmirror_ixi_pilot')
            sp.add_argument('--min-dice',type=float,default=.70)
        else:
            sp.add_argument('--shards-root',required=True)
    return p.parse_args()

if __name__=='__main__':
    args=parse_args()
    if args.command=='plan':run_plan(args)
    elif args.command=='shard':run_shard(args)
    else:run_merge(args)