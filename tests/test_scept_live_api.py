"""Live backend security/API tests; no GPU, uploaded patient data or model weights."""
from fastapi.testclient import TestClient
from services.scept_live import api
from services.scept_live.model import CHANNELS, EXPECTED_SHA256

def test_health_truthful():
    x=TestClient(api.app).get("/health")
    assert x.status_code==200
    assert x.json()["authentication"]=="required"

def test_unconfigured_api_closed(monkeypatch):
    monkeypatch.setattr(api,"SECRET","")
    x=TestClient(api.app).post("/api/v1/runs")
    assert x.status_code==503

def test_bad_token_rejected(monkeypatch):
    monkeypatch.setattr(api,"SECRET","secret")
    x=TestClient(api.app).post("/api/v1/runs",
      headers={"Authorization":"Bearer incorrect"})
    assert x.status_code==401

def test_all_four_channels_required(monkeypatch):
    monkeypatch.setattr(api,"SECRET","secret")
    x=TestClient(api.app).post("/api/v1/runs",
      headers={"Authorization":"Bearer secret"})
    assert x.status_code==422

def test_file_type_rejected_without_starting_gpu(monkeypatch):
    monkeypatch.setattr(api,"SECRET","secret")
    f={k:(k+".txt",b"unsafe","text/plain") for k in CHANNELS}
    x=TestClient(api.app).post("/api/v1/runs",files=f,
      headers={"Authorization":"Bearer secret"})
    assert x.status_code==400

def test_pinned_official_checkpoint():
    assert len(EXPECTED_SHA256)==64
    assert all(k in "0123456789abcdef" for k in EXPECTED_SHA256)
    assert CHANNELS==("flair","t1ce","t1","t2")


def test_four_modality_nifti_alignment_and_label_free_crop(tmp_path):
    import numpy as np
    import nibabel as nib
    from services.scept_live.model import prepare,CHANNELS
    rng=np.random.default_rng(17)
    array=np.zeros((64,64,64),np.float32)
    array[7:58,8:57,5:60]=rng.normal(55,9,size=(51,49,55)).astype(np.float32)
    paths={}
    for mod in CHANNELS:
        p=tmp_path/(mod+".nii.gz")
        nib.save(nib.Nifti1Image(array,np.eye(4)),p)
        paths[mod]=p
    cropped,metadata=prepare(paths)
    assert cropped.shape==(4,64,64,64)
    assert metadata["policy"]=="brain_nonzero_centroid_no_ground_truth"
    assert metadata["start"]==[0,0,0]

def test_reject_cross_modality_affine_mismatch(tmp_path):
    import numpy as np
    import nibabel as nib
    import pytest
    from services.scept_live.model import prepare,CHANNELS
    rng=np.random.default_rng(21)
    arr=rng.normal(10,2,size=(64,64,64)).astype(np.float32)
    paths={}
    for name in CHANNELS:
        p=tmp_path/(name+".nii")
        affine=np.eye(4)
        if name=="t2":affine[0,3]=3.0
        nib.save(nib.Nifti1Image(arr,affine),p)
        paths[name]=p
    with pytest.raises(ValueError,match="matching shape and affine"):
        prepare(paths)
