"""Static safety/DOM smoke tests for SCEPT browser studio (no browser pretending)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "scept-studio"


def test_web_studio_has_expected_files_and_dom_contract():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "studio.js").read_text(encoding="utf-8")
    css = (ROOT / "studio.css").read_text(encoding="utf-8")
    assert 'type="module" src="./studio.js"' in html
    assert len(css) > 1000
    ids = set(re.findall(r"""el\('([^']+)'\)""", js))
    missing = [name for name in ids if 'id="' + name + '"' not in html]
    assert not missing, "Missing DOM elements: " + ", ".join(missing)


def test_real_forward_and_no_train_endpoint():
    js = (ROOT / "studio.js").read_text(encoding="utf-8")
    assert "InferenceSession.create" in js
    assert "session.run(feeds)" in js
    assert "await runOnce(lib,original,shape" in js
    assert "await runOnce(lib,edited,shape" in js
    assert "source_provenance" not in js
    assert "fetch('/api/train" not in js
    assert "app.training" not in js


def test_observational_limit_is_explicit():
    js = (ROOT / "studio.js").read_text(encoding="utf-8")
    assert "L1_NUMERIC_INPUT_PERTURBATION" in js
    assert "L1_NUMERIC_INTERNAL_INTERVENTION" in js
    assert "export_truncated" in js
    assert "actual_onnx_tensor_dependency" in (
        (ROOT.parents[0] / "scripts" / "scept_instrument_onnx.py").read_text(
            encoding="utf-8"
        )
    )
