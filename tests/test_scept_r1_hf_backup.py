"""Offline remote-store contract checks: no HF credentials or network needed."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from brainmirror.scept.brats_r1_trainer import MirroredCheckpointStore
from brainmirror.scept.hf_backup import HFPrivateBackup


class FakePrivateHub:
    def __init__(self, private=True):
        self.private = private
        self.files = {}
        self.fail_upload = False

    def repo_info(self, repo_id, repo_type, token):
        assert repo_id == 'author/scept-private' and repo_type == 'model' and token == 'secret'
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, repo_id, repo_type, token):
        return sorted(self.files)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, token, commit_message):
        if self.fail_upload:
            raise ConnectionError('simulated upload failure')
        self.files[path_in_repo] = (Path(path_or_fileobj).read_bytes()
                                   if isinstance(path_or_fileobj, Path) else bytes(path_or_fileobj))

    def download(self, *, repo_id, filename, repo_type, token, cache_dir, force_download):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        path = Path(cache_dir) / filename.replace('/', '_')
        path.write_bytes(self.files[filename])
        return str(path)


def client(hub):
    return HFPrivateBackup('author/scept-private', 'secret', api=hub,
                           downloader=hub.download, retries=1)


def test_refuse_public_repository(tmp_path):
    with pytest.raises(ValueError, match='PUBLIC'):
        client(FakePrivateHub(private=False))


def test_verified_two_slots_remote_restore_after_kaggle_session_destroy(tmp_path):
    hub=FakePrivateHub()
    remote=client(hub)
    assert remote.probe_writable()
    split=tmp_path/'split.json'
    split.write_text(json.dumps({'splits': {'train':['a'],'validation':['b'],'test':['c']}}))
    assert remote.pin_split(split) == 'uploaded_verified'
    assert remote.pin_split(split) == 'already_pinned'
    split.write_text(json.dumps({'splits':{'train':['b'],'validation':['a'],'test':['c']}}))
    with pytest.raises(ValueError, match='REMOTE frozen'):
        remote.pin_split(split)

    work=MirroredCheckpointStore(tmp_path/'session1',tmp_path/'session1_backup_stage',prefix='scept_r1_training')
    for step in (1,2):
        work.save({'step':step},metadata={'step':step})
        remote.publish_latest(work)
    assert len([x for x in hub.files if x.endswith('.pt')])==2

    # Kaggle working volume lost entirely. New process needs no old files.
    restored=MirroredCheckpointStore(tmp_path/'session2',tmp_path/'new_backup_stage',prefix='scept_r1_training')
    new_remote=client(hub)
    outcome=new_remote.restore(restored)
    assert outcome['highest_generation']==1
    obj,manifest=restored.load()
    assert obj['step']==2 and manifest['generation']==1
    assert restored.verified() and (tmp_path/'session2'/'scept_r1_training_slot1.pt').exists()


def test_corrupted_latest_remote_falls_back_verified_older_generation(tmp_path):
    hub=FakePrivateHub();remote=client(hub)
    store=MirroredCheckpointStore(tmp_path/'local',tmp_path/'localbackup',prefix='scept_r1_training')
    store.save({'step':11});remote.publish_latest(store)
    store.save({'step':22});remote.publish_latest(store)
    hub.files['scept-v09-r1/scept_r1_training_slot1.pt'] = b'CORRUPT'
    fresh=MirroredCheckpointStore(tmp_path/'newlocal',tmp_path/'newback',prefix='scept_r1_training')
    out=client(hub).restore(fresh)
    assert out['highest_generation']==0
    obj,_=fresh.load()
    assert obj['step']==11


def test_partial_upload_preserves_last_valid_generation(tmp_path):
    hub=FakePrivateHub();remote=client(hub)
    store=MirroredCheckpointStore(tmp_path/'local',tmp_path/'stage',prefix='scept_r1_training')
    store.save({'step':1});remote.publish_latest(store)
    store.save({'step':2});hub.fail_upload=True
    with pytest.raises(ConnectionError):remote.publish_latest(store)
    hub.fail_upload=False
    new=MirroredCheckpointStore(tmp_path/'new',tmp_path/'newstage',prefix='scept_r1_training')
    assert client(hub).restore(new)['highest_generation']==0


def test_bad_token_and_paths_rejected():
    hub=FakePrivateHub()
    for path in ('owner','owner/repo/other','../repo'):
        with pytest.raises(ValueError): HFPrivateBackup(path,'secret',api=hub,downloader=hub.download)
    with pytest.raises(ValueError):HFPrivateBackup('author/scept-private','',api=hub,downloader=hub.download)
