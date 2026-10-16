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
