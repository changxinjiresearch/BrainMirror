"""A6-IJ-02D source-only first-failed-gate diagnosis regression."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"examples"

def test_d01_error_provenance_and_write_guard():
    path=ROOT/"a6_ij_02d_diagnose_replay.py"
    code=path.read_text()
    ast.parse(code,filename=str(path),feature_version=(3,10))
    assert "MAX_REPLAY_ABS" in code
    assert '"DIAGNOSTIC_ONLY_NO_02D_PASS"' in code
    assert '"identicalInputSHA256"' in code
    assert '"identicalModelStateSHA256"' in code
    assert '"failedOriginalThreshold"' in code
    assert 'if repo.get("private") is not True' in code
    assert "private_save_once" in code
    assert 'state_hash=SCEPT(' in code
    assert "direct_vit(model,x)" in code
    assert '"results/scept_a6_ij/02d_cross_arch_runtime"' in code
