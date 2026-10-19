"""A5 import actual SITE-V6 JavaScript ZIP; offline rederive real captured values."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scept import load_observation, SCEPTError

ROOT=Path(__file__).resolve().parents[2]
JS=ROOT/"scept-web/tests/v65_full_archive_contract.cjs"

def test_real_site_javascript_zip_uses_unchanged_browser_science_schema():
    # Execute the ACTUAL original V6 captureOnce, NAF and ZIP writer, rather
    # than synthesizing an imaginary browser-compatible JSON fixture.
    result=subprocess.run(["node",str(JS)],cwd=ROOT,
                          capture_output=True,text=True,check=True)
    assert "PASS full archive" in result.stdout
    source=Path("/tmp/scept-full-archive-contract.zip")
    assert source.is_file()
    run=load_observation(source)
    assert run.record["schema"]=="scept.run-record.v1"
    assert run.manifest["version"]=="6.5.2"
    assert run.manifest["captureScope"]["allDeclaredRuntimeOutputs"] is True
    assert run.manifest["captureScope"]["storedOutputTensors"]==2
    assert run.manifest["captureScope"]["inputsIncluded"] is True
    assert run.input_names==("x",)
    assert set(run.tensors)=={"hidden","logits"}
    np.testing.assert_array_equal(run.tensor("hidden"),
       np.array([1.5,2,3,-1,-2,-4],dtype=np.float32).reshape(1,2,3))
    np.testing.assert_array_equal(run.tensor("logits"),
       np.array([-.2,.8],dtype=np.float32).reshape(1,2))
    np.testing.assert_array_equal(run.input("x"),
       np.array([1,2,3,-1,-2,-3],dtype=np.float32).reshape(1,2,3))
    replay=run.recompute_naf()
    assert replay["source"]=="offline_recomputed_from_integrity_checked_archive"
    assert replay["science"]["model_forward_reexecuted"] is False
    assert replay["science"]["clinical"]=="NOT_ESTABLISHED"
    assert replay["groups"][0]["tensor"]=="hidden"
    assert replay["groups"][0]["units"][0]["count"]==3
    assert replay["groups"][1]["status"]=="not_observed"

def test_tamper_archive_cannot_be_mistaken_for_model_reexecution():
    # A model-free imported raw ZIP cannot claim to have reexecuted weights.
    subprocess.run(["node",str(JS)],cwd=ROOT,check=True,capture_output=True,text=True)
    run=load_observation("/tmp/scept-full-archive-contract.zip")
    assert "modelForwardReexecuted" not in run.manifest
    assert run.saved_naf["science"]["causality"]=="NOT_ESTABLISHED"
    assert run.recompute_naf()["science"]["model_forward_reexecuted"] is False
