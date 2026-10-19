"""Offline CI checks of preregistered A6-IJ-02A operator scripts (no MRI)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "examples" / "a6_ij_02a_measure.py",
    ROOT / "examples" / "a6_ij_02a_archive_private.py",
)

def test_scripts_parse_with_python_310_grammar():
    for path in SCRIPTS:
        assert path.is_file()
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3,10))

def test_frozen_protocol_and_privacy_flags_present():
    measure = SCRIPTS[0].read_text(encoding="utf-8")
    archive = SCRIPTS[1].read_text(encoding="utf-8")
    assert "BraTS20_Training_022" in measure
    assert "BraTS20_Training_040" in measure
    assert 'ORDER=("A1","A2","B1","A3")' in measure
    assert '"repeatMaxAbsoluteTolerance":1e-6' in measure
    assert '"repeatRelativeL2Tolerance":1e-7' in measure
    assert '"sensitivityRelativeL2Threshold":1e-3' in measure
    assert 'include_inputs=False' in measure
    assert 'repo.get("private") is not True' in measure
    assert 'info.get("private") is not True' in archive
    assert 'verify_remote(sess,remote,item["sha256"],item["bytes"])' in archive
    assert 'report.get("testSet")!="NOT_ACCESSED"' in archive
