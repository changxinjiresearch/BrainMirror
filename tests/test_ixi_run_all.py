"""Synthetic-only regression tests: no IXI individual data here."""
import csv, json, importlib.util
from pathlib import Path
import pytest

S=Path(__file__).resolve().parents[1]/'scripts'/'ixi_run_all.py'
spec=importlib.util.spec_from_file_location('ixi_run_all',S)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def make_csv(tmp_path, pilot5_train=True):
    path=tmp_path/'draft.csv'
    fields=['subject_id','site','image_path','split','qc_approved','used_in_pilot5']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i in range(581):
            split='train' if i<467 else 'dev' if i<524 else 'test'
            writer.writerow(dict(subject_id=f'IXI{i:03d}',site=['GUYS','HH','IOP'][i%3],
                image_path='bogus.nii',split=split,qc_approved='false',
                used_in_pilot5='true' if i<5 and (pilot5_train or i!=0) or (i==467 and not pilot5_train) else 'false'))
    return path


def test_581_draft_sizes_and_pilot_held_in_train(tmp_path):
    x=mod.read_manifest(make_csv(tmp_path),verify_files=False)
    assert len(x)==581
    assert len([r for r in x if r['split']=='train'])==467


def test_manifest_rejects_pilot_in_dev(tmp_path):
    with pytest.raises(ValueError, match='Pilot'):
        mod.read_manifest(make_csv(tmp_path,False),verify_files=False)


def test_resume_existing_same_provenance(tmp_path):
    d=tmp_path/'shard_000';d.mkdir()
    d.joinpath('PRIVATE_shard_receipt.json').write_text(json.dumps(dict(
        schema='sc',manifest_sha256='m',reference_sha256='r',reference_mask_sha256='rm',
        shard_size=12,shard_index=0,planned_subjects=12,included_subjects=12,flagged_subjects=0
    )))
    d.joinpath('PRIVATE_aggregates.npz').write_bytes(b'synthetic')
    meta=dict(schema='sc',manifest_sha256='m',reference_sha256='r',reference_mask_sha256='rm',shard_size=12)
    assert mod.completed_shard(d,meta=meta,k=0,planned=12)
    with pytest.raises(ValueError,match='provenance'):
        mod.completed_shard(d,meta={**meta,'manifest_sha256':'changed'},k=0,planned=12)


def test_refuses_incomplete_partial_shard(tmp_path):
    d=tmp_path/'shard_000';d.mkdir();(d/'scratch').mkdir()
    with pytest.raises(RuntimeError,match='incomplete'):
        mod.completed_shard(d,meta={},k=0,planned=12)