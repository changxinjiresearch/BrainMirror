"""Numerical engineering gates for genuine ONNX fixture inference and export-time intervention."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "scept-web" / "fixtures.js").read_text()
X = np.array([[.2, .4, .6, .8]], dtype=np.float32)

def _fixture(name):
    marker = name + ":{"
    assert marker in SOURCE, "fixture not present: " + name
    encoded = SOURCE.split(marker, 1)[1].split('b64:"', 1)[1].split('"', 1)[0]

    raw = base64.b64decode(encoded)
    model = onnx.load_from_string(raw)
    onnx.checker.check_model(model)
    return raw

def test_linear_reference_real_cpu_run():
    raw = _fixture("linear")
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": X})
    names = [v.name for v in sess.get_outputs()]
    np.testing.assert_allclose(out[names.index("hidden")], [[.46,.98,.50]], atol=1e-6)
    np.testing.assert_allclose(out[names.index("y")], [[.46,.98,.50]], atol=1e-6)

def test_actual_internal_ablation_changes_output():
    raw = _fixture("ablation")
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    ones = np.ones((1,3),dtype=np.float32)
    zeros = ones.copy(); zeros[0,0]=0
    a=dict(zip([q.name for q in sess.get_outputs()],sess.run(None,{"x":X,"scept_mask":ones})))
    b=dict(zip([q.name for q in sess.get_outputs()],sess.run(None,{"x":X,"scept_mask":zeros})))
    np.testing.assert_allclose(a["y"], [[-.02,1.05]],atol=1e-6)
    np.testing.assert_allclose(b["y"], [[-.25,.59]],atol=1e-6)
    np.testing.assert_allclose(a["act"],b["act"],atol=0)
    assert not np.allclose(a["patched"], b["patched"])

def test_export_instrumented_onnx_no_training(tmp_path):
    original=tmp_path/"input.onnx";instrumented=tmp_path/"instrumented.onnx"
    original.write_bytes(_fixture("linear"))
    subprocess.run([sys.executable,str(ROOT/"scripts"/"expose_onnx_probes.py"),
                   str(original),str(instrumented),"--ablate","hidden"],check=True)
    model=onnx.load(str(instrumented));onnx.checker.check_model(model)
    session=ort.InferenceSession(instrumented.read_bytes(),providers=["CPUExecutionProvider"])
    out_names=[q.name for q in session.get_outputs()]
    a=session.run(None,{"x":X,"scept_mask":np.ones((1,3),dtype=np.float32)})
    b=session.run(None,{"x":X,"scept_mask":np.array([[0,1,1]],dtype=np.float32)})
    assert not np.allclose(a[out_names.index("y")],b[out_names.index("y")])
    assert instrumented.with_suffix(".onnx.manifest.json").exists()
