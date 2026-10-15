#!/usr/bin/env python3
"""Single-launch, resumable IXI training-cohort anatomical reference build.

Runs in the researcher's PRIVATE Kaggle notebook with mounted IXI T1 data.
It is a reproducible affine MRI population reference candidate, NOT a
simulation of human neural activity, and not a scientifically approved atlas.

Source images, individual subject names/IDs, derived MRI and receipts stay in
Kaggle private storage, never published to GitHub or the public web site.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

EXPECTED_IXI_COUNT = 581
DEFAULT_MANIFEST = '/kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv'
DEFAULT_PILOT = '/kaggle/working/brainmirror_ixi_pilot5_v1'
DEFAULT_SYNTHSTRIP = '/kaggle/working/brainmirror_ixi_pilot'
DEFAULT_WORK = '/kaggle/working/brainmirror_ixi_train_atlas_run1'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda: inp.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(manifest: Path, *, verify_files: bool = True):
    with manifest.open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        required = {'subject_id','site','image_path','split','qc_approved','used_in_pilot5'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Manifest lacks required fields')
        rows = list(reader)
    if len(rows) != EXPECTED_IXI_COUNT:
        raise ValueError(f'Expected {EXPECTED_IXI_COUNT} subjects, got {len(rows)}')
    if len({r['subject_id'] for r in rows}) != EXPECTED_IXI_COUNT:
        raise ValueError('Duplicate subject IDs')
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'dev']
    test = [r for r in rows if r['split'] == 'test']
    if (len(train), len(dev), len(test)) != (467,57,57):
        raise ValueError('Unexpected split; do not silently change cohort')
    if any(r['split']!='train' for r in rows if r['used_in_pilot5'].strip().lower()=='true'):
        raise ValueError('Pilot training contamination of held-out split')
    if sum(r['used_in_pilot5'].strip().lower()=='true' for r in rows) != 5:
        raise ValueError('Expected five pilot cases')
    if len({r['site'] for r in rows}) != 3:
        raise ValueError('Expected three source sites')
    if verify_files:
        for row in rows:
            src=Path(row['image_path'])
            if not src.is_file() or not src.resolve().is_relative_to(Path('/kaggle/input')):
                raise FileNotFoundError('Source MRI not mounted under /kaggle/input')
    return rows


def mark_receipt(root: Path, data: dict, name: str):
    root.mkdir(parents=True, exist_ok=True)
    dest=root/name
    if dest.exists():
        raise FileExistsError(f'Receipt already exists: {dest}')
    dest.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf8')
    dest.chmod(0o600)


def voxel_preflight(rows, out: Path, manifest_hash: str):
    """Full cohort finite/range+geometry screening (one volume at a time)."""
    import nibabel as nib
    import numpy as np
    receipt_path = out/'PRIVATE_voxel_qc.json'
    if receipt_path.exists():
        prev=json.loads(receipt_path.read_text())
        if prev.get('manifest_sha256')!=manifest_hash or prev.get('scanned')!=len(rows):
            raise ValueError('Existing full-voxel QC does not match this manifest')
        if prev.get('bad'):
            raise ValueError(f'Existing voxel QC flagged {len(prev["bad"])} cases; review required')
        print('Full-cohort voxel QC already passed for same manifest; reusing receipt.',flush=True)
        return
    bad=[]
    for i,r in enumerate(rows,1):
        try:
            im=nib.load(r['image_path'])
            if len(im.shape)!=3 or any(s<32 for s in im.shape):
                raise ValueError('not a plausible 3D MRI')
            aff=im.affine
            if not np.isfinite(aff).all() or abs(np.linalg.det(aff[:3,:3]))<1e-8:
                raise ValueError('nonfinite or singular affine')
            data=np.asarray(im.get_fdata(dtype=np.float32))
            if not np.isfinite(data).all() or float(data.max()-data.min())<1e-4:
                raise ValueError('nonfinite or constant voxels')
        except Exception as ex:
            bad.append({'subject_id':r['subject_id'],'error':type(ex).__name__+': '+str(ex)[:100]})
        if i%40==0 or i==len(rows):
            print(f'[voxel QC] {i}/{len(rows)} scanned; failures={len(bad)}',flush=True)
    receipt={'schema':'brainmirror.ixi.full-voxel-qc.v1',
             'manifest_sha256':manifest_hash,'scanned':len(rows),
             'bad':bad,'passed':len(rows)-len(bad),
             'human_mask_reviewed':False,'P1_PASS':False}
    mark_receipt(out,receipt,'PRIVATE_voxel_qc.json')
    if bad:
        raise ValueError(f'Full-cohort voxel QC found {len(bad)} failures; stop and review')


def completed_shard(folder: Path, *, meta: dict, k: int, planned: int):
    rec=folder/'PRIVATE_shard_receipt.json'
    agg=folder/'PRIVATE_aggregates.npz'
    if not folder.exists(): return False
    if not rec.is_file() or not agg.is_file():
        raise RuntimeError(f'Shard folder is incomplete: {folder}. Preserve and inspect; do not overwrite.')
    data=json.loads(rec.read_text())
    for key in ('manifest_sha256','reference_sha256','reference_mask_sha256','shard_size','schema'):
        if data.get(key)!=meta.get(key):
            raise ValueError(f'Existing shard {k} has different provenance: {key}')
    if data.get('shard_index')!=k or data.get('planned_subjects')!=planned:
        raise ValueError('Existing shard index/count mismatch')
    if data.get('planned_subjects')!=data.get('included_subjects',0)+data.get('flagged_subjects',0):
        raise ValueError('Shard not complete')
    return True


def invoke(argv: list[str]):
    print('\n$ '+ ' '.join(argv),flush=True)
    subprocess.run(argv,check=True)


def make_private_browser_preview(final: Path, count_good: int, exporter: Path):
    import nibabel as nib
    import numpy as np
    mean=final/'ixi_train_affine_mean.nii.gz'
    coverage=final/'ixi_train_coverage.nii.gz'
    if not mean.is_file() or not coverage.is_file():
        raise FileNotFoundError('Missing merged atlas NIfTI outputs')
    count_img=nib.load(str(coverage))
    brain_count=np.asarray(count_img.dataobj)
    # Deliberately conservative, statistical coverage mask. This is NOT an
    # independently anatomical segmented atlas.
    threshold=max(1,int(math.ceil(0.80*count_good)))
    mask=(brain_count>=threshold).astype(np.uint8)
    if int(mask.sum())<10000:
        raise ValueError('Insufficient high-coverage template voxels')
    mask_path=final/'PRIVATE_preview_coverage80_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask,count_img.affine),str(mask_path))
    invoke([sys.executable,str(exporter),'--image',str(mean),
        '--brain-mask',str(mask_path),
        '--output',str(final/'PRIVATE_BrainMirror_IXI_preview.json'),
        '--kind','ixi_affine_group_candidate',
        '--source','IXI T1 train-only exploratory affine population reference; NOT QC APPROVED'])
    print('Browser preview is PRIVATE and UNREVIEWED, do not publish.',flush=True)


def run(args):
    sys.path.insert(0,str(Path(args.engine).resolve().parent))
    import importlib.util
    spec=importlib.util.spec_from_file_location('brainmirror_full_atlas_engine',args.engine)
    engine=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    manifest=Path(args.manifest)
    pilot=Path(args.pilot_root)
    synth=Path(args.synthstrip_dir)
    runner=Path(args.engine)
    browser=Path(args.exporter)
    work=Path(args.work_dir)
    work.mkdir(parents=True,exist_ok=True)
    work.chmod(0o700)
    actual_hash=sha256(manifest)
    if args.expected_manifest_sha256 and actual_hash!=args.expected_manifest_sha256:
        raise ValueError('Manifest checksum changed. Refuse to launch unreviewed cohort.')
    rows=read_manifest(manifest)
    train=engine.manifest_train(manifest,approved=args.approved)
    if len(train)!=467:raise ValueError('Incorrect training cohort')
    if not (synth/'mri_synthstrip').is_file() or not (synth/'synthstrip.1.pt').is_file():
        raise FileNotFoundError('SynthStrip script/weights not found. Reuse earlier pilot location.')
    meta=engine.metadata(manifest,pilot,args.shard_size)
    print('REAL IXI AFFINE CANDIDATE — NOT WHOLE-BRAIN FUNCTION SIMULATION',flush=True)
    print(f'581 sources: 467 train, 57 dev, 57 test; sha256={actual_hash}',flush=True)
    print('P1_PASS=false; model training not authorized; no public output.',flush=True)
    if not args.skip_full_voxel_qc:
        voxel_preflight(rows,work,actual_hash)
    shard_root=work/'shards'
    shard_root.mkdir(exist_ok=True)
    num_shards=math.ceil(len(train)/args.shard_size)
    print(f'Pipeline: {num_shards} independent batches (size {args.shard_size})',flush=True)
    started=time.monotonic()
    engine_command=str(runner.resolve())
    for k in range(num_shards):
        shard=shard_root/f'shard_{k:03d}'
        n=min(args.shard_size,len(train)-k*args.shard_size)
        if completed_shard(shard,meta=meta,k=k,planned=n):
            print(f'[Resume] {k+1}/{num_shards} already complete',flush=True)
            continue
        elapsed=(time.monotonic()-started)/60
        if elapsed>args.max_minutes:
            print('TIME BUDGET REACHED at shard boundary. Save Kaggle version and resume.',flush=True)
            print('No results discarded. Completed shards:',k,'/',num_shards,flush=True)
            return 0
        invoke([sys.executable,engine_command,'shard',
            '--manifest',str(manifest),'--pilot-root',str(pilot),
            '--synthstrip-dir',str(synth),'--shard-index',str(k),
            '--shard-size',str(args.shard_size),'--out',str(shard),
            '--min-dice',str(args.min_dice)] + (['--approved'] if args.approved else []))
    final=work/'final'
    if not (final/'PRIVATE_merge_receipt.json').exists():
        invoke([sys.executable,engine_command,'merge',
            '--manifest',str(manifest),'--pilot-root',str(pilot),
            '--shard-size',str(args.shard_size),
            '--shards-root',str(shard_root),'--out',str(final)] + (['--approved'] if args.approved else []))
    receipt=json.loads((final/'PRIVATE_merge_receipt.json').read_text())
    n_ok=int(receipt['subjects_included'])
    n_bad=int(receipt['subjects_flagged'])
    if n_bad:
        print(f'REVIEW REQUIRED: {n_bad} subjects flagged; mean uses {n_ok} scans, not all 467.',flush=True)
        print('Do not call the output approved or complete; review exclusions.',flush=True)
    if args.make_private_preview and n_ok:
        preview=final/'PRIVATE_BrainMirror_IXI_preview.json'
        if not preview.exists():make_private_browser_preview(final,n_ok,browser)
    status={'schema':'brainmirror.ixi.onecell.run.v1','status':'CANDIDATE_NOT_APPROVED',
        'manifest_sha256':actual_hash,'train_planned':467,'train_included':n_ok,
        'train_flagged':n_bad,'dev_test_excluded':114,
        'private_outputs':str(final),'review_required':True,
        'P1_PASS':False,'P2_TRAINING_AUTHORIZED':False,
        'neural_activity_simulated':False}
    summary=work/'PRIVATE_run_summary.json'
    if not summary.exists():mark_receipt(work,status,'PRIVATE_run_summary.json')
    print('\n3D ANATOMICAL CANDIDATE GENERATED. SCIENTIFIC QC / APPROVAL OUTSTANDING.',flush=True)
    print(json.dumps(status,indent=2),flush=True)
    return 0 if n_bad==0 else 3


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default=DEFAULT_MANIFEST)
    p.add_argument('--pilot-root',default=DEFAULT_PILOT)
    p.add_argument('--synthstrip-dir',default=DEFAULT_SYNTHSTRIP)
    p.add_argument('--engine',default='/kaggle/working/ixi_full_atlas.py')
    p.add_argument('--exporter',default='/kaggle/working/export_anatomy_preview.py')
    p.add_argument('--work-dir',default=DEFAULT_WORK)
    p.add_argument('--expected-manifest-sha256',default='')
    p.add_argument('--shard-size',type=int,default=12)
    p.add_argument('--min-dice',type=float,default=.70)
    p.add_argument('--max-minutes',type=float,default=480)
    p.add_argument('--skip-full-voxel-qc',action='store_true')
    p.add_argument('--approved',action='store_true',help='Require the independently approved cohort manifest')
    p.add_argument('--make-private-preview',action='store_true')
    args=p.parse_args()
    if not 1<=args.shard_size<=60 or not .5<=args.min_dice<=.95 or args.max_minutes<=0:
        p.error('invalid batch / Dice / time parameters')
    return args

if __name__=='__main__':
    try:
        raise SystemExit(run(parse_args()))
    except Exception as ex:
        print(f'PIPELINE STOPPED SAFELY: {type(ex).__name__}: {ex}',file=sys.stderr,flush=True)
        raise