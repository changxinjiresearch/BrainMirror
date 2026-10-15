"""Synthetic fixture tests only; not evidence of real human MRI review."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from brainmirror.p1.intake import (
    inspect_nifti, inventory, parse_ixi_identity, paths_safe, pseudonym, stable_split
)


def test_parse_ixi_t1_only_and_subject_stability():
    assert parse_ixi_identity('IXI002-Guys-0828-T1.nii.gz') == ('IXI002', 'T1')
    assert parse_ixi_identity('IXI002-Guys-0828-T2.nii.gz') is None
    assert parse_ixi_identity('foo-T1.nii.gz') is None
    assert parse_ixi_identity('IXI123-Guys-44-T1.jpg') is None
    assert pseudonym(b'1'*32, 'scan', 'a') != pseudonym(b'1'*32, 'subject', 'a')


def test_stable_subject_split():
    names = [f's{i:03}' for i in range(43)]
    a = stable_split(names)
    b = stable_split(list(reversed(names)))
    assert a == b
    assert len(a) == 43
    assert list(a.values()).count('train') == 35
    assert list(a.values()).count('dev') == 4
    assert list(a.values()).count('test') == 4
    assert len({k for k, v in a.items() if v == 'train'} &
               {k for k, v in a.items() if v == 'test'}) == 0
    with pytest.raises(ValueError):
        stable_split(['sub1','sub2'])


def test_private_paths_avoids_repo_and_input(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    outside = tmp_path / 'raw'
    outside.mkdir()
    result = tmp_path / 'private-results'
    assert paths_safe(outside, result, repo) == (outside, result)
    with pytest.raises(ValueError, match='outside'):
        paths_safe(outside, repo / 'runs', repo)
    with pytest.raises(ValueError, match='outside'):
        paths_safe(repo, result, repo)
    with pytest.raises(ValueError, match='nested'):
        paths_safe(outside, outside / 'out', repo)


class StubHeader:
    def get_zooms(self):
        return (1., 1., 1.)


class StubImg:
    shape = (96, 96, 96)
    affine = np.eye(4)
    header = StubHeader()
    @property
    def dataobj(self):
        return np.ones(self.shape, dtype=np.float32)
    def get_qform(self, coded=True):
        return (np.eye(4), 1)
    def get_sform(self, coded=True):
        return (np.eye(4), 1)


def fake_nib(monkeypatch, factory=None):
    stub = SimpleNamespace(load=lambda fn: factory(fn) if factory else StubImg(),
                           aff2axcodes=lambda mat: ('R','A','S'))
    monkeypatch.setitem(sys.modules, 'nibabel', stub)


def test_3d_headers_can_be_checked_without_voxel_loading(monkeypatch, tmp_path):
    fake_nib(monkeypatch)
    r = inspect_nifti(tmp_path / 'dummy.nii.gz')
    assert r['qc_status'] == 'PASS'
    assert r['voxel_qc_executed'] is False
    assert r['orientation'] == 'RAS'
    r2 = inspect_nifti(tmp_path / 'dummy.nii.gz', voxel_qc=True)
    assert r2['qc_status'] == 'PASS'
    assert r2['finite_fraction'] == 1


def test_conflicting_q_sforms_quarantine(monkeypatch, tmp_path):
    class Bad(StubImg):
        def get_sform(self, coded=True):
            moved = np.eye(4)
            moved[0,3] = 10
            return (moved, 1)
    fake_nib(monkeypatch, lambda _: Bad())
    r = inspect_nifti(tmp_path / 'dummy.nii.gz')
    assert 'QFORM_SFORM_CONFLICT' in r['qc_reasons']


def test_private_inventory_only_aggregate_safe_for_public(monkeypatch, tmp_path):
    fake_nib(monkeypatch)
    repo = tmp_path/'repo'
    repo.mkdir()
    raw = tmp_path/'MRI'
    raw.mkdir()
    for i in range(10):
        (raw/f'IXI{i:03d}-Guys-scan{i:03d}-T1.nii.gz').write_bytes(f'unique{i}'.encode())
    out = tmp_path/'private'
    summary = inventory(raw, out, signing_key=b'x'*32, accepted_terms=True,
                        voxel_qc=True, repo_root=repo)
    assert summary['eligible_subjects'] == 10
    assert summary['split_subject_counts'] == {'train':8,'dev':1,'test':1}
    assert summary['data_gate'] == 'PRELIMINARY_SPLIT_READY_NOT_LOCKED'
    assert summary['split_locked'] is False
    assert summary['real_mri_training_started'] is False
    aggregate_bytes = (out/'aggregate_receipt.json').read_text()
    assert 'IXI000-' not in aggregate_bytes
    assert 'subject_hmac' not in aggregate_bytes
    assert 'relative_path' not in aggregate_bytes
    assert 'private_scans.json' not in aggregate_bytes
    private = json.loads((out/'private_scans.json').read_text())
    assert len(private['records']) == 10
    assert len({r['subject_hmac'] for r in private['records']}) == 10
    assert all(r['file_sha256'] for r in private['records'])
    assert all(r['split'] in ('train', 'dev', 'test') for r in private['records'])


def test_duplicate_bytes_different_subject_quarantine(monkeypatch, tmp_path):
    fake_nib(monkeypatch)
    raw = tmp_path/'raw'; raw.mkdir()
    for i in range(11):
        (raw/f'IXI{i:03d}-Site-01-T1.nii').write_bytes(b'dup' if i in (0,1) else str(i).encode())
    repo = tmp_path/'code'; repo.mkdir()
    summary = inventory(raw, tmp_path/'private', signing_key=b'x'*32,
                        accepted_terms=True, voxel_qc=True, repo_root=repo)
    assert summary['failure_counts']['DUPLICATE_BYTES_ACROSS_SUBJECTS'] == 2
    assert summary['scans_failed'] == 2
    assert summary['data_gate'] == 'BLOCKED_PENDING_COMPLETION_AND_REVIEW'


def test_license_secret_and_missing_data_fail_closed(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    repo = tmp_path / 'repo'; repo.mkdir()
    with pytest.raises(ValueError, match='licen'):
        inventory(raw, tmp_path/'private', signing_key=b'x'*32,
                  accepted_terms=False, voxel_qc=False, repo_root=repo)
    with pytest.raises(ValueError, match='HMAC'):
        inventory(raw, tmp_path/'private', signing_key=b'short',
                  accepted_terms=True, voxel_qc=False, repo_root=repo)
    with pytest.raises(ValueError, match='No NIfTI'):
        inventory(raw, tmp_path/'private', signing_key=b'x'*32,
                  accepted_terms=True, voxel_qc=False, repo_root=repo)


def test_weak_4d_candidate_rejected(monkeypatch, tmp_path):
    class FourD(StubImg):
        shape = (96,96,96,3)
    fake_nib(monkeypatch,lambda _:FourD())
    r=inspect_nifti(tmp_path/'s.nii.gz')
    assert r['qc_status']=='FAIL'
    assert 'NOT_3D_SINGLE_VOLUME' in r['qc_reasons']


def test_actual_nibabel_synthetic_nifti_roundtrip(tmp_path, monkeypatch):
    """Real NiBabel I/O on a synthetic volume; never treated as human MRI."""
    monkeypatch.delitem(sys.modules, 'nibabel', raising=False)
    nib = pytest.importorskip('nibabel')
    voxels = np.zeros((96, 96, 96), dtype=np.float32)
    voxels[18:78, 18:78, 18:78] = 1
    affine = np.diag([1.2, 1.2, 1.2, 1.0])
    img = nib.Nifti1Image(voxels, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    path = tmp_path / 'IXI999-Guys-999-T1.nii.gz'
    nib.save(img, str(path))
    result = inspect_nifti(path, voxel_qc=True)
    assert result['qc_status'] == 'PASS'
    assert result['orientation'] == 'RAS'
    assert result['shape'] == [96,96,96]
    assert result['finite_fraction'] == 1
    assert 0.1 < result['nonzero_fraction'] < 0.5