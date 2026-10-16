#!/usr/bin/env python3
"""Private-only P5 FreeSurfer reconstruction backup/restore via GitHub Releases.

Keep SUBJECT original T1, identities and license OFF GitHub by default.
Upload only allowlisted FreeSurfer files, local browser surface pack and receipt.
Both uploads and restores verify SHA256 against provider asset digest. A digest
omission is an error. No public-repository fallback. Never overwrites mismatch.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPO='changxinjiresearch/BrainMirror-Data-Private'
TAG='ixi-native-p5-case01-v1'
SUBJECT='brainmirror_case01'
FILES=['surf/lh.pial','surf/rh.pial','label/lh.aparc.DKTatlas.annot',
       'label/rh.aparc.DKTatlas.annot','mri/aseg.mgz']


def digest(path:Path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): sha.update(b)
    return 'sha256:'+sha.hexdigest()


def request_json(url,token,method='GET',payload=None):
    base={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
          'X-GitHub-Api-Version':'2022-11-28','User-Agent':'BrainMirror-P5-private-only'}
    b=None
    if payload is not None:
        b=json.dumps(payload).encode()
        base['Content-Type']='application/json'
    req=urllib.request.Request(url,headers=base,method=method,data=b)
    with urllib.request.urlopen(req,timeout=45) as r:return json.load(r)


def check_repo(repo,token):
    if repo!=DEFAULT_REPO:raise ValueError('P5 backup ONLY targets the approved private research repository')
    x=request_json('https://api.github.com/repos/'+repo,token)
    if x.get('private') is not True or x.get('full_name')!=repo:
        raise ValueError('Repository is not the expected PRIVATE research repository')
    return True


def get_release(repo,token,create=False):
    base='https://api.github.com/repos/'+repo
    try:return request_json(base+'/releases/tags/'+TAG,token)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
    if not create:raise RuntimeError('No P5 private release to restore')
    return request_json(base+'/releases',token,method='POST',payload={
        'tag_name':TAG,'name':'P5 individual native cortical anatomy (PRIVATE)',
        'body':'PRIVATE derived native MRI subject. Does not authorize public redistribution. QC status UNREVIEWED.',
        'draft':False,'prerelease':True})


def source_paths(root:Path):
    subject=root/'subjects'/SUBJECT
    return [(subject/f,'native/'+f) for f in FILES]


def package_recovery(root:Path):
    root=root.resolve();target=root/'private'/'PRIVATE_p5_recovery.tar.gz'
    if target.exists():
        # Ensure a prior archive is not silently repacked over different inputs.
        return target
    paths=source_paths(root)
    missing=[str(p) for p,name in paths if not p.is_file()]
    if missing:raise FileNotFoundError('Missing reconstructed native files: '+repr(missing))
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.partial')
    if temporary.exists():raise RuntimeError('Stale recovery archive partial; investigate before retry')
    try:
        with tarfile.open(temporary,'w:gz') as archive:
            for p,name in paths:
                if p.is_symlink():raise ValueError('Refusing linked FreeSurfer source')
                archive.add(p,arcname=name,recursive=False)
        temporary.replace(target)
        target.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def staged(root:Path):
    root=root.resolve()
    bundle=root/'private'/f'{SUBJECT}_native.json.gz'
    receipt=root/'private'/'PRIVATE_p5_native_receipt.json'
    if not bundle.is_file() or not receipt.is_file():
        raise RuntimeError('Run cortical reconstruction, exporter and audit first')
    record=json.loads(receipt.read_text())
    if record.get('native_bundle_sha256')!=digest(bundle).removeprefix('sha256:'):
        raise ValueError('Local native bundle and receipt SHA256 differ')
    return [('PRIVATE_p5_native.json.gz',bundle),('PRIVATE_p5_native_receipt.json',receipt),
            ('PRIVATE_p5_recovery.tar.gz',package_recovery(root))]


def upload(url,token,path:Path):
    # GitHub release asset uploads may be >100 MB. Stream, do not load binary MRI into memory.
    u=urllib.parse.urlparse(url)
    if u.hostname!='uploads.github.com':raise ValueError('Unexpected GitHub upload host')
    conn=http.client.HTTPSConnection(u.hostname,timeout=180)
    try:
        conn.putrequest('POST',u.path+'?'+u.query)
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                 'Content-Type':'application/octet-stream',
                 'Content-Length':str(path.stat().st_size),'User-Agent':'BrainMirror-P5-private-only'}
        for k,v in headers.items():conn.putheader(k,v)
        conn.endheaders()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):conn.send(block)
        response=conn.getresponse()
        data=response.read()
        if response.status not in (200,201):
            raise RuntimeError('GitHub upload failed HTTP '+str(response.status)+': '+data[:450].decode('utf8','replace'))
        return json.loads(data)
    finally:conn.close()


def verify_remote(asset,local_path):
    expected=digest(local_path)
    if asset.get('digest')!=expected or asset.get('size')!=local_path.stat().st_size:
        raise RuntimeError('Remote asset SHA256/size not verified: '+asset.get('name','unknown'))
    return True


def sync(root:Path,repo:str,token:str):
    check_repo(repo,token)
    paths=staged(root)
    release=get_release(repo,token,create=True)
    existing={a['name']:a for a in release.get('assets',[])}
    done=0
    for name,path in paths:
        if name in existing:
            verify_remote(existing[name],path)
            print('PRIVATE P5 VERIFIED EXISTING: '+name,flush=True);continue
        prefix=release['upload_url'].split('{',1)[0]
        sep='&' if '?' in prefix else '?'
        target=prefix+sep+'name='+urllib.parse.quote(name)
        got=upload(target,token,path)
        verify_remote(got,path)
        done+=1
        print('PRIVATE P5 UPLOADED VERIFIED: '+name,flush=True)
    print('P5 PRIVATE BACKUP COMPLETE:',len(paths),'assets;',done,'new; SHA256 verified',flush=True)
    return len(paths)


def retrieve_asset(asset,repo,token,dst:Path):
    if dst.exists():
        if digest(dst)==asset.get('digest') and dst.stat().st_size==asset.get('size'):return
        raise RuntimeError('Local file exists but SHA mismatch. Refusing overwrite: '+str(dst))
    u=asset['url']
    req=urllib.request.Request(u,headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream',
         'X-GitHub-Api-Version':'2022-11-28','User-Agent':'BrainMirror-P5-private-only'})
    dst.parent.mkdir(parents=True,exist_ok=True)
    temp=dst.with_suffix(dst.suffix+'.partial')
    if temp.exists():raise RuntimeError('Incomplete previous download; inspect and remove manually')
    try:
        with urllib.request.urlopen(req,timeout=180) as r,temp.open('xb') as out:
            for block in iter(lambda:r.read(4*1024*1024),b''):out.write(block)
        verify_remote(asset,temp)
        temp.replace(dst);dst.chmod(0o600)
    except Exception:
        temp.unlink(missing_ok=True);raise


def restore(root:Path,repo:str,token:str):
    check_repo(repo,token)
    release=get_release(repo,token)
    names={a['name']:a for a in release.get('assets',[])}
    expected=['PRIVATE_p5_native.json.gz','PRIVATE_p5_native_receipt.json','PRIVATE_p5_recovery.tar.gz']
    if any(n not in names for n in expected):raise RuntimeError('Incomplete remote P5 recovery release')
    for n in expected:
        target=root/'private'/({
            'PRIVATE_p5_native.json.gz':f'{SUBJECT}_native.json.gz',
            'PRIVATE_p5_native_receipt.json':'PRIVATE_p5_native_receipt.json',
            'PRIVATE_p5_recovery.tar.gz':'PRIVATE_p5_recovery.tar.gz'
        }[n])
        retrieve_asset(names[n],repo,token,target)
    # Avoid untrusted tar extraction. Package exists for manual forensic retrieval.
    print('P5 PRIVATE RESTORE VERIFIED: 3 assets; native tar not automatically extracted')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['sync','restore'])
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--repo',default=DEFAULT_REPO)
    args=ap.parse_args()
    token=os.environ.get('BRAINMIRROR_BACKUP_TOKEN')
    if not token:raise RuntimeError('Missing private BRAINMIRROR_BACKUP_TOKEN; never paste into code')
    if args.action=='sync':sync(args.root,args.repo,token)
    else:restore(args.root,args.repo,token)
if __name__=='__main__':main()