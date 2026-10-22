"""A6-IJ-02D static safety and locked research scope checks (no MRI)."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"examples"
SCRIPTS=("a6_ij_02d_measure.py","a6_ij_02d_archive_private.py")

def test_02d_source_syntax_python_310_compatible():
    for filename in SCRIPTS:
        p=ROOT/filename
        assert p.is_file()
        ast.parse(p.read_text(),filename=str(p),feature_version=(3,10))

def test_02d_replay_and_archive_are_privacy_guarded():
    measure=(ROOT/SCRIPTS[0]).read_text()
    arch=(ROOT/SCRIPTS[1]).read_text()
    assert '"BraTS20_Training_022"' in measure
    assert '"BraTS20_Training_040"' in measure
    assert 'PREVIOUS_TAG="scept-a6-ij-02c-ln-position-controls-v1"' in measure
    assert 'MAX_REPLAY_ABS=1e-6' in measure
    assert 'MAX_CROSS_ENGINE_ABS=1e-3' in measure
    assert 'MAX_CROSS_ENGINE_REL=1e-3' in measure
    assert 'pretrained=False' in measure
    assert 'class Resnet18Probe(' in measure
    assert 'direct_vit' in measure and 'direct_cnn' in measure
    assert 'providers=("CPUExecutionProvider",)' in measure
    assert 'include_inputs=False' in measure
    assert 'if info.get("private") is not True' in measure
    assert 'source_zip' in measure
    assert 'report.get("status")!="PASS_ENGINEERING_CROSS_ARCHITECTURE_AND_CROSS_ENGINE"' in arch
    assert 'if info.get("full_name")!=REPO or info.get("private") is not True:' in arch
    assert 'verify_remote(s,asset,original)' in arch
    assert 'private_immutable' not in arch
