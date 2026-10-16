#!/usr/bin/env python3
"""Select a previously-used TRAIN-only IXI T1 for a new, PRIVATE cortical reconstruction.

No public storage, no implicit model training, no new split. The pilot case was
already explored during affine template development, so preserve test isolation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED_SPLIT_SHA256='00f3ec7d490bff06f529640ccfed4bbae02f172c2f92f4e851e756cc1cf4e40a'

def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def stage(manifest:Path,data_root:Path,out:Path,site='GUYS'):
    import nibabel as nib
    manifest=Path(manifest)
    if hashfile(manifest)!=EXPECTED_SPLIT_SHA256:
        raise RuntimeError('Unexpected split manifest: cannot safely assign a test subject')
    with manifest.open(newline='',encoding='utf8') as f: rows=list(csv.DictReader(f))
    if len(rows)!=581 or len({r['subject_id'] for r in rows})!=581:
        raise RuntimeError('Missing or duplicated cohort IDs')
    candidates=sorted(
        (r for r in rows if r['site']==site and r['split']=='train' and r['used_in_pilot5']=='true'),
        key=lambda r:r['subject_id']
    )
    if not candidates:raise RuntimeError('No existing pilot subject in requested TRAIN-only site')
    candidate=candidates[0]
    recorded=Path(candidate['image_path'])
    if recorded.is_file():src=recorded
    else:
        matches=[p for p in Path(data_root).rglob(recorded.name) if p.is_file()]
        if len(matches)!=1:raise RuntimeError('Original IXI MRI not uniquely mounted')
        src=matches[0]
    image=nib.load(str(src))
    zooms=image.header.get_zooms()[:3]
    if len(image.shape)!=3 or min(image.shape)<100 or max(image.shape)>512:
        raise RuntimeError('Input is not a full-resolution 3D T1 MRI')
    if any(not .4<=z<=2.0 for z in zooms):
        raise RuntimeError('Native voxel dimensions suspicious')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite a previous private case')
    out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    target=out/'PRIVATE_ixi_case01_native_T1.nii.gz'
    nib.save(image,str(target))
    receipt={
        'schema':'brainmirror.ixi.native_case.v1','selection':'train-only already used in pilot5',
        'site':site,'case_alias':'brainmirror_case01',
        'full_native_resolution_shape':list(map(int,image.shape)),
        'voxel_spacing_mm':list(map(float,zooms)),
        'source_sha256':hashfile(src),'staged_file_sha256':hashfile(target),
        'anatomy_reconstructed':False,'split_untouched':True,
        'private_only':True,
        'next_action':'Run a licensed FreeSurfer/FastSurfer full cortical reconstruction on the staged individual MRI.'
    }
    (out/'PRIVATE_stage_receipt.json').write_text(json.dumps(receipt,indent=2))
    (out/'PRIVATE_stage_receipt.json').chmod(0o600)
    print(json.dumps(receipt,indent=2))
    return target

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--manifest',type=Path,default=Path('/kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv'))
    a.add_argument('--data-root',type=Path,default=Path('/kaggle/input'))
    a.add_argument('--site',choices=['GUYS','HH','IOP'],default='GUYS')
    a.add_argument('--out',type=Path,default=Path('/kaggle/working/brainmirror_p5_native_case01'))
    v=a.parse_args();stage(v.manifest,v.data_root,v.out,v.site)
if __name__=='__main__':main()