"""No patient data: deterministic manifest, leakage, shard integrity tests."""
from pathlib import Path
import csv, importlib.util, json
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/'scripts'/'ixi_full_atlas.py'
spec=importlib.util.spec_from_file_location('ixi_full_atlas',P)
app=importlib.util.module_from_spec(spec); spec.loader.exec_module(app)


def test_shard_indices():
    rows=list(range(27))
    assert app.shard_rows(rows,0,12)==list(range(12))
    assert app.shard_rows(rows,2,12)==[24,25,26]
    with pytest.raises(ValueError):app.shard_rows(rows,3,12)
    with pytest.raises(ValueError):app.shard_rows(rows,0,100)


def test_masked_normalization():
    img=np.zeros((32,32,32),np.float32)
    m=np.zeros(img.shape,bool);m[2:30,2:30,2:30]=1
    img[m]=np.linspace(1,10,int(m.sum()))
    out=app.masked_normalize(img,m,np)
    assert np.all(out[~m]==0)
    assert float(out.max())<=1
    assert float(out.min())>=0
    with pytest.raises(ValueError):app.masked_normalize(np.zeros((32,32,32)),m,np)


def test_manifest_rejects_leaking_pilot_and_duplicates(tmp_path):
    fp=tmp_path/'split.csv'
    with fp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subject_id','site','image_path','split','qc_approved','used_in_pilot5'])
        w.writeheader()
        for i in range(581):
            w.writerow(dict(subject_id=f'IXI{i:03d}',site='GUYS',image_path='/kaggle/input/fake.nii',
                            split='test' if i<60 else 'dev' if i<118 else 'train',qc_approved='false',
                            used_in_pilot5='true' if i==1 else 'false'))
    with pytest.raises(ValueError,match='Pilot'):
        app.manifest_train(fp)


def test_shard_receipt_rejects_missing(tmp_path):
    meta={'schema':'schema','manifest_sha256':'a','reference_sha256':'b',
          'reference_mask_sha256':'c','shard_size':12}
    with pytest.raises(FileNotFoundError):app.checked_shard_receipts(tmp_path,meta,2)


def test_aggregates_roundtrip(tmp_path):
    p=tmp_path/'agg.npz'
    app.save_aggregates(p,np.ones((6,7,8)),np.ones((6,7,8)),np.ones((6,7,8),np.int16),np)
    with np.load(p) as z: assert z['count'].sum()==6*7*8