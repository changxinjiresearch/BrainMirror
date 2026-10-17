import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import json

FILE=Path(__file__).resolve().parents[1]/'scripts'/'p5_native_runner.py'
spec=importlib.util.spec_from_file_location('p5_native_runner',FILE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_command_contains_real_input_and_license_read_only(tmp_path,monkeypatch):
    t1=tmp_path/'data'/'subject.nii.gz';t1.parent.mkdir();t1.write_bytes(b'fake-mri')
    lic=tmp_path/'keys'/'license.txt';lic.parent.mkdir();lic.write_text('license contents not printed')
    out=tmp_path/'results';out.mkdir()
    cmd=m.docker_command(t1,lic,out,'gpu',4)
    assert '--gpus' in cmd and 'all' in cmd
    assert 'deepmi/fastsurfer:cu128-v2.5.4' in cmd
    assert '/brainmirror-license/license.txt' in cmd
    assert 'license contents not printed' not in ' '.join(cmd)
    assert str(t1)+':/brainmirror-input/subject.nii.gz:ro' in ' '.join(cmd)
    assert str(lic)+':/brainmirror-license/license.txt:ro' in ' '.join(cmd)
    assert '--sid' in cmd and m.SUBJECT in cmd

def test_refuses_population_mean_and_unlicensed_run(tmp_path,monkeypatch):
    t1=tmp_path/'ixi_train_affine_mean.nii.gz';t1.write_bytes(b'fake')
    lic=tmp_path/'license.txt';lic.write_bytes(b'valid license placeholder text')
    monkeypatch.setattr(m.shutil,'which',lambda exe:'/usr/bin/docker')
    with pytest.raises(ValueError,match='Population mean'):
        m.safe_preflight(t1,lic,tmp_path/'output','docker')
    t1=tmp_path/'native.nii.gz';t1.write_bytes(b'fake')
    lic.unlink()
    with pytest.raises(ValueError,match='license'):
        m.safe_preflight(t1,lic,tmp_path/'output','docker')

def test_guard_partial_subject(tmp_path,monkeypatch):
    t1=tmp_path/'native.nii.gz';t1.write_bytes(b'fake')
    lic=tmp_path/'license.txt';lic.write_bytes(b'valid license placeholder text')
    output=tmp_path/'output';partial=output/'subjects'/m.SUBJECT;partial.mkdir(parents=True)
    monkeypatch.setattr(m.shutil,'which',lambda exe:'/usr/bin/docker')
    opts=SimpleNamespace(t1=t1,license=lic,out=output,mode='cpu',threads=2,docker_binary='docker',allow_partial_resume=False)
    with pytest.raises(RuntimeError,match='Partial subject directory'):
        m.execute(opts,runner=lambda *a,**kw:None)