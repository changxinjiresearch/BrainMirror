#!/usr/bin/env python3
"""GitHub private-release backup/restore for BrainMirror Kaggle IXI results.

Requires a GitHub *private* artifact repository and fine-grained PAT with
Contents read/write granted ONLY to that repository. Token is loaded from
BRAINMIRROR_BACKUP_TOKEN, never saved in logs or files.

Release assets, rather than git blobs, are used for large scientific binary
artifacts. No files are sent to a public repository. The remote asset's SHA256
is checked against the local file (download verification when no digest).
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path('/kaggle/working')
ALLOWED_TOP = {'brainmirror_ixi_train_atlas_run1', 'brainmirror_ixi_split_v1',
               'brainmirror_ixi_pilot5_v1', 'brainmirror_ixi_bootstrap'}
MANIFEST = 'brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv'
REPO_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
MAX_FILE_BYTES = 2 * 1024**3 - 1  # GitHub release per-asset limit


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def safe_relative(label: str) -> Path:
    parts = PurePosixPath(label)
    if parts.is_absolute() or len(parts.parts) < 2 or '..' in parts.parts or parts.parts[0] not in ALLOWED_TOP:
        raise ValueError('Unsafe or unexpected GitHub asset path')
    if any(part in ('', '.') for part in parts.parts):
        raise ValueError('Unsafe path component')
    return Path(*parts.parts)


def asset_name(label: str) -> str:
    return 'bm-' + hashlib.sha256(label.encode('utf8')).hexdigest()[:20] + '--' + Path(label).name


def candidate_files(root: Path) -> list[Path]:
    """Include only complete shards; never read a half-written shard snapshot."""
    results=[]
    for part in ('brainmirror_ixi_split_v1', 'brainmirror_ixi_pilot5_v1', 'brainmirror_ixi_bootstrap'):
        folder=root/part
        if folder.exists():
            results += [p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()]
    run=root/'brainmirror_ixi_train_atlas_run1'
    for name in ('PRIVATE_voxel_qc.json', 'PRIVATE_run_summary.json'):
        path=run/name
        if path.is_file(): results.append(path)
    shards=run/'shards'
    if shards.is_dir():
        for folder in sorted(shards.glob('shard_*')):
            if not folder.is_dir(): continue
            agg=folder/'PRIVATE_aggregates.npz'
            receipt=folder/'PRIVATE_shard_receipt.json'
            if not agg.is_file() or not receipt.is_file():
                print(f'[SKIP incomplete] {folder.name}', flush=True)
                continue
            rec=json.loads(receipt.read_text(encoding='utf8'))
            if rec.get('planned_subjects') != rec.get('included_subjects',0) + rec.get('flagged_subjects',0):
                print(f'[SKIP inconsistent] {folder.name}',flush=True)
                continue
            # Load npz central directory to catch interrupted writes.
            import numpy as np
            with np.load(agg, allow_pickle=False) as data:
                if not {'sums','squares','count'}.issubset(data.files):
                    raise ValueError(f'Incomplete numerical aggregates: {agg}')
                if data['sums'].shape != data['count'].shape or data['squares'].shape != data['count'].shape:
                    raise ValueError(f'Inconsistent aggregate geometry: {agg}')
            results += [agg, receipt]
    final=run/'final'
    if final.is_dir() and (final/'PRIVATE_merge_receipt.json').is_file():
        results += [p for p in final.rglob('*') if p.is_file() and not p.is_symlink()]
    return sorted(set(results))


class GitHubBackup:
    def __init__(self, repository: str, token: str, *, timeout: int = 120):
        if not REPO_PATTERN.fullmatch(repository):
            raise ValueError('Repository must be owner/name')
        if not token:
            raise ValueError('BRAINMIRROR_BACKUP_TOKEN secret is not available')
        self.repo=repository
        self.token=token
        self.timeout=timeout

    def json_request(self, method: str, endpoint: str, payload=None):
        data=json.dumps(payload).encode('utf8') if payload is not None else None
        headers={'Authorization':f'Bearer {self.token}',
                 'Accept':'application/vnd.github+json',
                 'X-GitHub-Api-Version':'2022-11-28', 'User-Agent':'BrainMirror-IXI-Backup/1'}
        if data is not None: headers['Content-Type']='application/json'
        request=Request('https://api.github.com'+endpoint,data=data,headers=headers,method=method)
        try:
            with urlopen(request,timeout=self.timeout) as stream:
                return json.load(stream)
        except HTTPError as e:
            # Never print request/auth headers or user tokens.
            if e.code==404: return None
            raise RuntimeError(f'GitHub API {method} failed with HTTP {e.code}') from None

    def assert_private(self):
        info=self.json_request('GET',f'/repos/{self.repo}')
        if not info or info.get('private') is not True:
            raise PermissionError('STOP: target repository is missing or NOT PRIVATE; no upload permitted')
        return info

    def release(self, tag: str, create: bool):
        r=self.json_request('GET',f'/repos/{self.repo}/releases/tags/{quote(tag)}')
        if r is None and create:
            r=self.json_request('POST',f'/repos/{self.repo}/releases',{
                'tag_name':tag,'name':f'PRIVATE IXI checkpoint {tag}',
                'body':'Sensitive research-derived MRI artifacts. Private storage only; not clinically approved.',
                'draft':False,'prerelease':True
            })
        if r is None: raise FileNotFoundError('Private GitHub release does not exist')
        return r

    def assets(self, release_id: int):
        result=[]
        for page in range(1,20):
            part=self.json_request('GET',f'/repos/{self.repo}/releases/{release_id}/assets?per_page=100&page={page}')
            result += part or []
            if len(part or []) < 100: return result
        raise RuntimeError('Too many release assets; review capacity')

    def upload(self, url_template: str, path: Path, label: str):
        size=path.stat().st_size
        if size==0 or size>=MAX_FILE_BYTES:
            raise ValueError('Empty or oversized release asset')
        addr=url_template.split('{',1)[0]
        url=urlsplit(addr)
        if url.scheme!='https' or url.hostname!='uploads.github.com':
            raise ValueError('Untrusted GitHub upload endpoint')
        query=urlencode({'name':asset_name(label), 'label':label})
        endpoint=url.path+'?'+query
        con=http.client.HTTPSConnection(url.hostname,timeout=self.timeout)
        try:
            with path.open('rb') as stream:
                con.request('POST',endpoint,body=stream,headers={
                    'Authorization':f'Bearer {self.token}',
                    'Accept':'application/vnd.github+json',
                    'X-GitHub-Api-Version':'2022-11-28',
                    'User-Agent':'BrainMirror-IXI-Backup/1',
                    'Content-Type':'application/octet-stream',
                    'Content-Length':str(size)
                })
                resp=con.getresponse()
                output=resp.read()
                if resp.status not in (200,201):
                    raise RuntimeError(f'GitHub asset upload returned HTTP {resp.status}')
                return json.loads(output)
        finally:
            con.close()

    def asset_bytes(self, asset_id: int):
        req=Request(f'https://api.github.com/repos/{self.repo}/releases/assets/{asset_id}',headers={
            'Authorization':f'Bearer {self.token}', 'Accept':'application/octet-stream',
            'X-GitHub-Api-Version':'2022-11-28','User-Agent':'BrainMirror-IXI-Backup/1'
        })
        with urlopen(req,timeout=self.timeout) as stream:
            if stream.headers.get('Content-Type','').startswith('application/json'):
                raise RuntimeError('GitHub returned metadata instead of asset bytes')
            return stream.read()

    def validate_asset(self, asset: dict, expected_sha: str, expected_size: int, *, verify_bytes: bool):
        if asset.get('size')!=expected_size:
            raise ValueError(f'Remote asset size mismatch: {asset.get("name")}')
        digest=asset.get('digest')
        if digest and digest != 'sha256:'+expected_sha:
            raise ValueError(f'Remote asset digest mismatch: {asset.get("name")}')
        if not digest and verify_bytes:
            fetched=self.asset_bytes(asset['id'])
            if hashlib.sha256(fetched).hexdigest()!=expected_sha:
                raise ValueError('Downloaded GitHub asset hash mismatch')
        if not digest and not verify_bytes:
            raise ValueError('SHA256 digest missing; verification cannot be skipped')
        return True


def get_tag(root: Path):
    manifest=root/MANIFEST
    if not manifest.is_file():raise FileNotFoundError('Private split manifest must be present')
    return 'ixi-t1-'+sha256(manifest)[:16]+'-atlas-v1'


def sync(root: Path, client: GitHubBackup):
    client.assert_private()
    tag=get_tag(root)
    release=client.release(tag,create=True)
    assets={x['name']:x for x in client.assets(release['id'])}
    files=candidate_files(root)
    if not files:raise ValueError('No complete artifacts to back up')
    saved=0;skipped=0
    for i,path in enumerate(files,1):
        rel=path.relative_to(root).as_posix()
        safe_relative(rel)
        name=asset_name(rel)
        local_hash=sha256(path)
        expected_size=path.stat().st_size
        existing=assets.get(name)
        if existing:
            if existing.get('label')!=rel:
                raise ValueError('Asset label collision; stop')
            client.validate_asset(existing,local_hash,expected_size,verify_bytes=True)
            skipped+=1
        else:
            new=client.upload(release['upload_url'],path,rel)
            client.validate_asset(new,local_hash,expected_size,verify_bytes=True)
            assets[name]=new
            saved+=1
        print(f'[{i}/{len(files)}] VERIFIED {rel}',flush=True)
    print(f'BACKUP COMPLETE: uploaded={saved}, already_verified={skipped}, total={len(files)}',flush=True)
    print(f'PRIVATE GitHub release: https://github.com/{client.repo}/releases/tag/{tag}',flush=True)
    return 0


def restore(root: Path, client: GitHubBackup, tag: str):
    client.assert_private()
    release=client.release(tag,create=False)
    assets=client.assets(release['id'])
    restored=0
    for a in assets:
        rel=a.get('label')
        if not rel:raise ValueError('Release asset missing original relative path label')
        path=safe_relative(rel)
        if a['name']!=asset_name(rel):raise ValueError('Remote filename does not match label')
        dest=root/path
        if not dest.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe local restore destination")
        if dest.exists():
            # A same-sized file without GitHub's digest does not count as verified.
            content=a.get('digest')
            if not content:
                remote=client.asset_bytes(a['id'])
                content='sha256:'+hashlib.sha256(remote).hexdigest()
            if dest.stat().st_size!=a.get('size') or sha256(dest)!=content.removeprefix('sha256:'):
                raise FileExistsError(f'Existing restore target differs; will not overwrite {rel}')
            print(f'Already restored: {rel}',flush=True)
            continue
        data=client.asset_bytes(a['id'])
        digest=hashlib.sha256(data).hexdigest()
        if a.get('digest') and a['digest'] != 'sha256:'+digest:
            raise ValueError('Remote asset sha256 inconsistent on download')
        if a.get('size')!=len(data):raise ValueError('Remote asset size inconsistent on download')
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_name(dest.name+'.download_partial')
        with tmp.open('xb') as file: file.write(data)
        os.replace(tmp,dest)
        restored+=1
        print(f'RESTORED {rel}',flush=True)
    print(f'RESTORE COMPLETE: {restored} new files.',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=['sync','restore'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--repo',default=os.getenv('BRAINMIRROR_PRIVATE_REPO',''))
    p.add_argument('--release-tag',default='')
    args=p.parse_args()
    token=os.getenv('BRAINMIRROR_BACKUP_TOKEN','')
    client=GitHubBackup(args.repo,token)
    if args.operation=='sync':return sync(args.root,client)
    if not args.release_tag:raise ValueError('--release-tag is required for restore')
    return restore(args.root,client,args.release_tag)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print(f'BACKUP STOPPED SAFELY: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(2)