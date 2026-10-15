#!/usr/bin/env python3
"""Explicitly download official IXI T1 to PRIVATE durable storage.

Never send the archive, images, identifiers or demographic data to GitHub.
Only run where source host is allowed and sufficient persistent storage exists.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tarfile
from urllib.request import Request, urlopen

URL='https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar'

def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def download(url:str, dest:Path):
    """Resume only when upstream honours Range; refuse silent truncation."""
    part=dest.with_suffix(dest.suffix+'.part')
    offset=part.stat().st_size if part.exists() else 0
    req=Request(url,headers={'User-Agent':'BrainMirror-Research-P1/1.0',
                             **({'Range':f'bytes={offset}-'} if offset else {})})
    with urlopen(req,timeout=90) as resp:
        code=getattr(resp,'status',200)
        if offset and code!=206:raise RuntimeError('Source did not honor Range; remove partial only after review')
        with part.open('ab' if offset else 'wb') as f:
            shutil.copyfileobj(resp,f,4*1024*1024)
    part.replace(dest)

def extract(archive:Path,dest:Path):
    if dest.exists():raise ValueError('Refusing to overwrite existing data directory')
    dest.mkdir(mode=0o700,parents=True)
    accepted=0
    try:
        with tarfile.open(archive,'r:*') as tf:
            for member in tf:
                if member.isdir():
                    name = Path(member.name)
                    if name.is_absolute() or '..' in name.parts:
                        raise ValueError('Unsafe archive directory')
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError('Archive contains forbidden non-regular file')
                name=Path(member.name)
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('Unsafe archive path')
                if len(name.parts)>3 or member.size > 500_000_000:
                    raise ValueError('Unreasonable archive member')
                if not member.name.lower().endswith(('.nii','.nii.gz')):
                    raise ValueError('Non MRI member in T1 source archive')
                path=(dest/name).resolve()
                if not path.is_relative_to(dest.resolve()):
                    raise ValueError('Path escapes storage directory')
                path.parent.mkdir(parents=True,exist_ok=True)
                stream=tf.extractfile(member)
                if stream is None:raise ValueError('Could not extract member')
                with path.open('xb') as f:
                    shutil.copyfileobj(stream,f,1024*1024)
                accepted+=1
        return accepted
    except BaseException:
        # Fail closed and never leave a partially extracted cohort mistaken as complete.
        shutil.rmtree(dest)
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--private-root',type=Path,required=True)
    p.add_argument('--acknowledge-cc-by-sa-3',action='store_true')
    p.add_argument('--extract',action='store_true')
    args=p.parse_args()
    if not args.acknowledge_cc_by_sa_3:
        p.error('Read https://brain-development.org/ixi-dataset/ license and explicitly acknowledge')
    dest=args.private_root.resolve()
    repo=Path(__file__).resolve().parents[1]
    if dest.is_relative_to(repo):p.error('Private storage must be outside git repo')
    dest.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(dest).free < 10*1024**3:
        p.error('Require at least 10 GiB free PRIVATE durable storage before download')
    archive=dest/'IXI-T1.tar'
    if not archive.exists():download(URL,archive)
    digest=sha256(archive)
    with tarfile.open(archive,'r:*') as tf:
        members=[m for m in tf if m.isfile()]
        if not members or len(members)>1000:
            raise ValueError('Unexpected archive member count')
    count=extract(archive,dest/'T1') if args.extract else None
    receipt={'source':'IXI','source_url':URL,'terms':'CC BY-SA 3.0',
             'official_archive_sha256':digest,'archive_integrity_checked':True,'archive_verified':False,
             'archive_member_files':len(members),'extracted':count,
             'archive_subjects_verified':None,
             'status':'ACQUIRED_NOT_QUALIFIED'}
    rp=dest/'IXI_T1_ACQUISITION_PRIVATE.json'
    if rp.exists():raise ValueError('Refuse receipt overwrite')
    rp.write_text(json.dumps(receipt,indent=2),encoding='utf8');rp.chmod(0o600)
    print('Acquired official archive. SHA256:',digest,'Members:',len(members),
          'Real image QC and scientific source review remain pending.')
if __name__=='__main__':main()
