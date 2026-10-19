"""A5 research archive: actual Torch+ONNX tensors and cross-runtime ZIP verifier."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pytest
import torch
from torch import nn

from scept import SCEPT, SCEPTError, load_observation
from scept.archive import SCHEMA, MAX_ARCHIVE_BYTES
from test_onnx_adapter import model_bytes

ROOT=Path(__file__).resolve().parents[2]
NODE_CHECK=ROOT/"scept_sdk/tests/verify_python_zip_in_node.cjs"

def _actual_torch():
    torch.manual_seed(131)
    net=nn.Sequential(nn.Conv2d(1,3,1),nn.ReLU())
    observer=SCEPT(net,["0","1"],profile="cnn-nchw",backend="torch")
    source=torch.randn(1,1,6,6)
    capture=observer.observe(source,preprocess_id="actual-frozen-preprocessing-v1",
                             preprocess_config={"source":"independent-PyTorch"})
    assert capture.tensor("0").shape==(1,3,6,6)
    return observer,capture

def test_full_real_torch_zip_binary_and_verified_offline_replay(tmp_path):
    observer,original=_actual_torch()
    archive=tmp_path/"genuine-torch.zip"
    meta=original.export(archive,include_inputs=True)
    assert meta["status"]=="archive_written_and_verified_not_scientifically_authenticated"
    assert meta["outputTensors"]==3
    assert meta["inputsIncluded"] is True
    loaded=load_observation(archive)
    assert loaded.manifest["schema"]==SCHEMA
    assert loaded.record==original.record
    assert loaded.saved_naf["runId"]==original.run_id
    assert len(loaded.tensors)==3
    assert loaded.input_names==("input",)
    for name in original.tensors:
        np.testing.assert_array_equal(loaded.tensor(name),original.tensor(name))
    np.testing.assert_array_equal(loaded.input("input"),original.input("input"))
    old=loaded.tensor("0");old.flat[0]=100000
    assert loaded.tensor("0").flat[0]!=100000
    offline=loaded.recompute_naf()
    assert offline["source"]=="offline_recomputed_from_integrity_checked_archive"
    assert offline["science"]["model_forward_reexecuted"] is False
    assert offline["science"]["causality"]=="NOT_ESTABLISHED"
    report=loaded.verify_model_replay(observer)
    assert report["status"]=="numeric_reexecution_matched"
    assert report["newIndependentRunId"]!=loaded.run_id
    assert report["newModelForwardExecuted"] is True
    assert all(x["matched"] for x in report["perTensor"])
    checked=subprocess.run(["node",str(NODE_CHECK),str(archive)],
        cwd=ROOT,capture_output=True,text=True,check=True)
    assert "PASS A5 JavaScript" in checked.stdout

def test_actual_onnx_zip_replay_fails_if_source_model_differs(tmp_path):
    orig=model_bytes()
    observer=SCEPT(orig,["conv_out","relu_out"],backend="onnx",profile="cnn-nchw")
    tensor=np.arange(1,10,dtype=np.float32).reshape(1,1,3,3)
    run=observer.observe(tensor,preprocess_id="frozen-onnx-v1")
    file=tmp_path/"true-onnx.zip"
    run.export(file,include_inputs=True)
    loaded=load_observation(file)
    np.testing.assert_array_equal(loaded.tensor("conv_out"),run.tensor("conv_out"))
    assert loaded.verify_model_replay(observer)["status"]=="numeric_reexecution_matched"
    assert len(loaded.recompute_naf()["groups"])==2
    diff=SCEPT(nn.Sequential(nn.Conv2d(1,2,1)),["0"],backend="torch",profile="cnn-nchw")
    with pytest.raises(SCEPTError,match="MODEL_MISMATCH"):
        loaded.verify_model_replay(diff)

def test_default_excludes_private_input_and_cannot_exe_model(tmp_path):
    model,run=_actual_torch()
    p=tmp_path/"private-default.zip"
    run.export(p)
    obj=load_observation(p)
    assert obj.manifest["captureScope"]["inputsIncluded"] is False
    assert obj.input_names==()
    with pytest.raises(SCEPTError,match="INPUT_NOT_ARCHIVED"):
        obj.verify_model_replay(model)
    with zipfile.ZipFile(p) as z:
        assert all("tensors/inputs/" not in name for name in z.namelist())
        assert z.getinfo("manifest.json").compress_type==zipfile.ZIP_STORED

def _resign(source:bytes, overrides:dict[str,bytes]) -> bytes:
    """Adversarial mutation: edit bytes and recalculate standard file SHA/CRC.
    An attacker can always re-sign an unsigned ZIP. Scientific numeric validation
    must catch changed raw data even when ZIP CRC + manifest hashes were forged.
    """
    output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        original={n:z.read(n) for n in z.namelist()}
    original.update(overrides)
    manifest=json.loads(original["manifest.json"])
    for entry in manifest["files"]:
        raw=original[entry["path"]]
        entry["bytes"]=len(raw)
        entry["sha256"]=hashlib.sha256(raw).hexdigest()
    original["manifest.json"]=(json.dumps(manifest,separators=(",",":"))+"\n").encode()
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_STORED) as z:
        for n,v in original.items():z.writestr(n,v)
    return output.getvalue()

def test_forged_numeric_naf_file_sha_cannot_pass_validation(tmp_path):
    _,run=_actual_torch()
    p=tmp_path/"plain.zip";run.export(p)
    with zipfile.ZipFile(p) as z:
        naf=json.loads(z.read("naf.json"))
    original=naf["groups"][0]["units"][0]["rms"]
    naf["groups"][0]["units"][0]["rms"]=original+1.0
    changed=_resign(p.read_bytes(),{"naf.json":(json.dumps(naf)+"\n").encode()})
    with pytest.raises(SCEPTError,match="NAF_MISMATCH"):
        load_observation(changed)

def test_wrong_tensor_record_sha_and_truncated_zip_fails_closed(tmp_path):
    _,run=_actual_torch()
    file=tmp_path/"valid.zip";run.export(file)
    with zipfile.ZipFile(file) as z:
        records=json.loads(z.read("record.json"))
    records["outputs"][0]["sha256"]="f"*64
    changed=_resign(file.read_bytes(),{"record.json":(json.dumps(records)+"\n").encode()})
    with pytest.raises(SCEPTError,match="TENSOR_HASH_MISMATCH"):
        load_observation(changed)
    with pytest.raises(Exception):
        load_observation(file.read_bytes()[:80])

def test_import_is_not_authenticated_and_denies_path_traversal_and_zip_bomb(tmp_path):
    _,run=_actual_torch()
    file=tmp_path/"good.zip";run.export(file)
    with zipfile.ZipFile(file) as z:
        allfiles={n:z.read(n) for n in z.namelist()}
    allfiles["../escape.txt"]=b"bad"
    attack=io.BytesIO()
    with zipfile.ZipFile(attack,"w",compression=zipfile.ZIP_STORED) as z:
        for n,d in allfiles.items():z.writestr(n,d)
    with pytest.raises(SCEPTError,match="INVALID_PATH"):
        load_observation(attack.getvalue())
    assert len(file.read_bytes())<MAX_ARCHIVE_BYTES
