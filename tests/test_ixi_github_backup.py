from pathlib import Path
import importlib.util
import hashlib
import json
import pytest
import numpy as np

P=Path(__file__).with_name('ixi_github_backup.py')
spec=importlib.util.spec_from_file_location('backup',P)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def test_safe_path_rejects_traversal():
    for name in ('../secrets.txt','/etc/passwd','brainmirror_ixi_train_atlas_run1/../.env','not_allowlisted/data','brainmirror_ixi_pilot5_v1'):
        with pytest.raises(ValueError): b.safe_relative(name)
    assert b.safe_relative('brainmirror_ixi_train_atlas_run1/shards/shard_000/PRIVATE_aggregates.npz')


def test_private_repo_mandatory():
    client=b.GitHubBackup('user/BrainMirror','FAKE')
    client.json_request=lambda *a,**k: {'private':False}
    with pytest.raises(PermissionError,match='NOT PRIVATE'):
        client.assert_private()


def test_select_only_completed_shards(tmp_path):
    root=tmp_path/'brainmirror_ixi_train_atlas_run1'/'shards'
    done=root/'shard_000';done.mkdir(parents=True)
    arr=np.ones((5,6,7))
    np.savez_compressed(done/'PRIVATE_aggregates.npz',sums=arr,squares=arr,count=arr.astype(np.int16))
    (done/'PRIVATE_shard_receipt.json').write_text(json.dumps({'planned_subjects':12,'included_subjects':12,'flagged_subjects':0}))
    incomplete=root/'shard_001';incomplete.mkdir()
    np.savez_compressed(incomplete/'PRIVATE_aggregates.npz',sums=arr,squares=arr,count=arr)
    results=b.candidate_files(tmp_path)
    assert len(results)==2
    assert all('shard_000' in str(v) for v in results)


def test_digest_validation():
    cli=b.GitHubBackup('u/private','FAKE')
    path=b'hello'
    sha=hashlib.sha256(path).hexdigest()
    assert cli.validate_asset({'id':1,'size':5,'digest':'sha256:'+sha,'name':'hello'},sha,5,verify_bytes=True)
    with pytest.raises(ValueError,match='digest mismatch'):
        cli.validate_asset({'id':1,'size':5,'digest':'sha256:'+'0'*64,'name':'hello'},sha,5,verify_bytes=True)
    cli.asset_bytes=lambda _:path
    assert cli.validate_asset({'id':1,'size':5,'digest':None,'name':'hello'},sha,5,verify_bytes=True)


def test_backup_no_files_aborts(tmp_path):
    manifest=tmp_path/b.MANIFEST
    manifest.parent.mkdir()
    manifest.write_text('private fixture')
    cli=b.GitHubBackup('u/private','FAKE')
    cli.assert_private=lambda : True
    cli.release=lambda *a,**kw: {'id':1,'upload_url':'https://uploads.github.com/a'}
    cli.assets=lambda _:[]
    # The draft manifest itself is backed up; no synthetic scans required.
    class MemoryClient:
        pass
    backed=[]
    cli.upload=lambda url, path, label:(backed.append(label) or {'name':b.asset_name(label),'label':label,'id':1,'size':path.stat().st_size,'digest':'sha256:'+b.sha256(path)})
    assert b.sync(tmp_path,cli)==0
    assert backed==[b.MANIFEST]