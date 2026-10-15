"""Security/unit tests only, no IXI data fetched here."""
from io import BytesIO
from pathlib import Path
import importlib.util
import tarfile
import pytest

spec=importlib.util.spec_from_file_location('p1_acquire',Path(__file__).parents[1]/'scripts'/'p1_acquire_ixi.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_tar(path, filename):
    with tarfile.open(path,'w') as tf:
        b=b'abcd';info=tarfile.TarInfo(filename);info.size=len(b)
        tf.addfile(info,BytesIO(b))


def test_archive_path_traversal_rejected_and_private_data_not_left(tmp_path):
    source=tmp_path/'danger.tar';make_tar(source,'../sneaky-T1.nii.gz')
    out=tmp_path/'private'
    with pytest.raises(ValueError,match='Unsafe'):
        module.extract(source,out)
    assert not out.exists()


def test_tar_regular_nifti_safe_extract(tmp_path):
    source=tmp_path/'ok.tar';make_tar(source,'IXI001-Guys-111-T1.nii.gz')
    out=tmp_path/'private'
    assert module.extract(source,out)==1
    assert (out/'IXI001-Guys-111-T1.nii.gz').read_bytes()==b'abcd'
    with pytest.raises(ValueError,match='overwrite'):
        module.extract(source,out)
