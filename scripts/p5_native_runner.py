#!/usr/bin/env python3
"""Full, fail-closed Docker-based FastSurfer P5 individual-cortex reconstruction.

Runs ONE original IXI TRAIN subject, never the 467-person mean. Requires a
working Docker GPU/CPU host, a FreeSurfer license, and the staged private T1.
Runs a real surface reconstruction, verifies its artifacts, and exports the
browser-native local-only bundle. No GitHub upload unless requested separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SUBJECT = 'brainmirror_case01'
IMAGES = {'gpu':'deepmi/fastsurfer:cu128-v2.5.4', 'cpu':'deepmi/fastsurfer:cpu-v2.5.4'}
REQUIRED_FILES = [
    'surf/lh.pial', 'surf/rh.pial', 'label/lh.aparc.DKTatlas.annot',
    'label/rh.aparc.DKTatlas.annot', 'mri/aseg.mgz'
]

def hash_file(p: Path) -> str:
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe_preflight(t1:Path, license:Path, root:Path, docker_binary:str):
    if not t1.is_file() or not t1.name.lower().endswith(('.nii','.nii.gz')):
        raise ValueError('Missing original-resolution IXI T1 .nii/.nii.gz input')
    if 'ixi_train_affine_mean' in str(t1).lower() or 'group' in t1.stem.lower():
        raise ValueError('Population mean is not an acceptable cortical input')
    if not license.is_file() or not 20<=license.stat().st_size<=8192:
        raise ValueError('FreeSurfer license file missing/invalid length; keep it private')
    if not shutil.which(docker_binary):
        raise RuntimeError('Docker not available. Kaggle notebooks normally lack Docker daemon; use a Docker-capable Linux compute host')
    if root.resolve() in [Path('/'), Path.home().resolve()]:
        raise ValueError('Refusing unsafe output root')
    if t1.resolve().is_relative_to(root.resolve()):
        raise ValueError('Keep original T1 outside the generated output directory')
    return True

def docker_command(t1:Path, license:Path, root:Path, mode:str, threads:int,docker_binary:str='docker'):
    if mode not in IMAGES:raise ValueError('mode must be gpu or cpu')
    if not 1<=threads<=32:raise ValueError('Invalid thread count')
    cmd=[docker_binary,'run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
         '--shm-size','8g']
    if mode=='gpu':cmd.extend(['--gpus','all'])
    cmd.extend([
        '-v',f'{t1.resolve()}:/brainmirror-input/{t1.name}:ro',
        '-v',f'{license.resolve()}:/brainmirror-license/{license.name}:ro',
        '-v',f'{root.resolve()}:/brainmirror-output:rw',
        IMAGES[mode],
        '--t1',f'/brainmirror-input/{t1.name}',
        '--sid',SUBJECT, '--sd','/brainmirror-output/subjects',
        '--fs_license',f'/brainmirror-license/{license.name}',
        '--threads',str(threads),
    ])
    return cmd

def files_complete(subjectdir:Path) -> bool:
    return all((subjectdir/p).is_file() and (subjectdir/p).stat().st_size>0 for p in REQUIRED_FILES)

def verified_private_backup(args):
    if not getattr(args, 'require_backup', False): return
    token=os.environ.get('BRAINMIRROR_BACKUP_TOKEN')
    if not token:raise RuntimeError('Missing BRAINMIRROR_BACKUP_TOKEN: private backup is mandatory')
    # Never back up original MRI, only recon-derived, reviewed provenance.
    import p5_private_backup as backup
    backup.sync(Path(args.out),backup.DEFAULT_REPO,token)

def execute(args, *, runner=subprocess.run):
    t1=args.t1.resolve();license=args.license.resolve();root=args.out.resolve()
    safe_preflight(t1,license,root,args.docker_binary)
    if getattr(args,'require_backup',False):
        token=os.environ.get('BRAINMIRROR_BACKUP_TOKEN')
        if not token:raise RuntimeError('Require-backup enabled but token absent; fail before costly reconstruction')
        import p5_private_backup as backup
        backup.check_repo(backup.DEFAULT_REPO,token)
    subject_dir=root/'subjects'/SUBJECT
    output=root/'private'/f'{SUBJECT}_native.json.gz'
    receipt=root/'private'/'PRIVATE_p5_native_receipt.json'
    if receipt.exists() and output.exists() and files_complete(subject_dir):
        record=json.loads(receipt.read_text())
        if record.get('t1_sha256') != hash_file(t1) or record.get('native_bundle_sha256') != hash_file(output):
            raise RuntimeError('Existing private results do not match input or checksum; refuse overwrite')
        print('ALREADY_COMPLETE_AND_VERIFIED: '+str(output), flush=True)
        verified_private_backup(args)
        return record
    root.mkdir(parents=True,exist_ok=True)
    (root/'subjects').mkdir(exist_ok=True)
    (root/'private').mkdir(exist_ok=True)
    (root/'private').chmod(0o700)
    t1_sha=hash_file(t1)
    if not files_complete(subject_dir):
        # A crashed partial subject is never mistaken for a valid reconstruction.
        if subject_dir.exists() and not args.allow_partial_resume:
            raise RuntimeError('Partial subject directory exists; explicit --allow-partial-resume required. Preserve and review logs first.')
        cmd=docker_command(t1,license,root,args.mode,args.threads,args.docker_binary)
        print('RUNNING_LICENSED_FASTSURFER: '+SUBJECT+'; input='+t1.name+'; device='+args.mode,flush=True)
        # The license is a mounted local file, never printed or passed in plaintext.
        runner(cmd,check=True)
    if not files_complete(subject_dir):
        missing=[p for p in REQUIRED_FILES if not (subject_dir/p).is_file()]
        raise RuntimeError('Native cortical reconstruction incomplete; missing '+', '.join(missing))
    if not output.exists():
        export = Path(__file__).with_name('export_freesurfer_surface.py')
        if not export.is_file():raise FileNotFoundError(export)
        runner([sys.executable,str(export),'--subject-dir',str(subject_dir),'--annotation','aparc.DKTatlas','--out',str(output)],check=True)
    if not output.exists():raise RuntimeError('Native export missing')
    record={
        'schema':'brainmirror.native_run_receipt.v1',
        'scientific_status':'RECONSTRUCTED_UNREVIEWED',
        'subject_alias':SUBJECT,
        't1_sha256':t1_sha,
        'native_bundle_sha256':hash_file(output),
        'bundle_bytes':output.stat().st_size,
        'required_surfaces_and_labels':REQUIRED_FILES,
        'source_type':'single original-resolution IXI TRAIN-only T1',
        'reconstruction_mode':args.mode,
        'container_image':IMAGES[args.mode],
        'scientific_review_completed':False,
        'registered_to_467_person_group_mean':False,
        'brainAE_model_coupled':False,
        'private_only':True,
        'created_unix_s':int(time.time())
    }
    # Do not overwrite an inconsistent previous receipt.
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old.get('native_bundle_sha256')!=record['native_bundle_sha256']:
            raise RuntimeError('Previous scientific receipt differs; refuse overwrite')
    else:
        receipt.write_text(json.dumps(record,indent=2));receipt.chmod(0o600)
    verified_private_backup(args)
    print('P5_NATIVE_EXPORT_COMPLETE: '+str(output),flush=True)
    print('SCIENCE_QC_REQUIRED: cortical topology, segmentation, visual overlays',flush=True)
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1',type=Path,required=True,help='One original IXI TRAIN T1 MRI, never group mean')
    p.add_argument('--license',type=Path,required=True,help='Private FreeSurfer license.txt')
    p.add_argument('--out',type=Path,required=True,help='PRIVATE durable output directory')
    p.add_argument('--mode',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--docker-binary',default='docker')
    p.add_argument('--allow-partial-resume',action='store_true')
    p.add_argument('--require-backup',action='store_true',help='Fail closed unless backup to approved PRIVATE release verifies')
    args=p.parse_args();execute(args)
if __name__=='__main__':main()