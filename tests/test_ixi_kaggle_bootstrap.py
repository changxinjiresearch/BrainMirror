"""Synthetic-only tests for IXI Kaggle bootstrap; no real MRI in Git."""
import importlib.util
from pathlib import Path
import tempfile
import pytest

SRC=Path(__file__).resolve().parents[1]/"scripts"/"ixi_kaggle_bootstrap.py"
spec=importlib.util.spec_from_file_location("ixi_kaggle_bootstrap",SRC)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_mount_discovery_ignores_nii_named_directories(tmp_path):
    ds=tmp_path/"datasets"/"owner"/"brainmirror-ixi-t1-raw"
    d=ds/"IXI002-Guys-0828-T1.nii";d.mkdir(parents=True)
    (d/"IXI002-Guys-0828-s01.nii").write_bytes(b"fixture")
    root,files=mod.discover(tmp_path)
    assert root==ds and len(files)==1 and files[0].is_file()

def test_reject_duplicate_subject_ids(tmp_path):
    ds=tmp_path/"brainmirror-ixi-t1-raw";ds.mkdir()
    (ds/"IXI002-Guys-0828-T1.nii").write_bytes(b"1")
    (ds/"IXI002-Guys-0828-copy.nii").write_bytes(b"2")
    with pytest.raises(RuntimeError,match="Duplicate"):mod.discover(tmp_path)

def test_reject_unexpected_nifti_filename(tmp_path):
    ds=tmp_path/"brainmirror-ixi-t1-raw";ds.mkdir()
    (ds/"unknown.nii").write_bytes(b"1")
    with pytest.raises(RuntimeError,match="Unrecognized"):mod.discover(tmp_path)
