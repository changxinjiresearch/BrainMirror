import importlib.util
from pathlib import Path
import hashlib
import json
import pytest

F=Path(__file__).resolve().parents[1]/'scripts'/'p5_private_backup.py'
spec=importlib.util.spec_from_file_location('p5_backup',F)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_private_repo_guard_refuses_public_even_with_token(monkeypatch):
    with pytest.raises(ValueError,match='approved private'):
        m.check_repo('changxinjiresearch/BrainMirror','fake-token')
    monkeypatch.setattr(m,'request_json',lambda *a,**kw:{'private':False,'full_name':m.DEFAULT_REPO})
    with pytest.raises(ValueError,match='PRIVATE'):
        m.check_repo(m.DEFAULT_REPO,'fake-token')

def test_sha256_guard_and_reuse(tmp_path):
    f=tmp_path/'p5.json.gz';f.write_bytes(b'group geometry not raw patient')
    assert m.verify_remote({'name':f.name,'digest':m.digest(f),'size':f.stat().st_size},f)
    with pytest.raises(RuntimeError,match='SHA256'):
        m.verify_remote({'name':f.name,'digest':'sha256:wrong','size':f.stat().st_size},f)

def test_archive_contains_no_original_mri_or_license(tmp_path):
    subject=tmp_path/'subjects'/m.SUBJECT
    for name in m.FILES:
        p=subject/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'sample')
    (tmp_path/'secret-license.txt').write_bytes(b'private')
    tar=m.package_recovery(tmp_path)
    import tarfile
    with tarfile.open(tar) as f:
        names=f.getnames()
    assert set(names)=={'native/'+p for p in m.FILES}
    assert not any('license' in n or 'T1' in n or '.nii' in n for n in names)