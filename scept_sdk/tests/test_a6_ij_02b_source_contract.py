"""A6-IJ-02B source audit without private medical data."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"examples"
PATHS=("a6_ij_02b_measure.py","a6_ij_02b_analyse.py",
       "a6_ij_02b_archive_private.py")

def test_02b_scripts_are_python_310_compatible():
    for p in PATHS:
        source=(ROOT/p).read_text()
        ast.parse(source,filename=p,feature_version=(3,10))

def test_02b_prelocked_scope_metrics_and_private_integrity():
    measure=(ROOT/PATHS[0]).read_text()
    analysis=(ROOT/PATHS[1]).read_text()
    archive=(ROOT/PATHS[2]).read_text()
    assert '"nPatients":55' in measure
    assert 'len(val)!=55' in measure
    assert 'SPLIT_SHA' in measure
    assert 'include_inputs=False' in measure
    assert 'independent_hook_replay' in measure
    assert '"patchEffectiveRankNormalized"' in measure
    assert 'def gram(' in measure and 'def cka(' in measure
    assert '2000' in analysis and '240216' in analysis
    assert '"P4BDescriptiveSpearman"' in analysis
    assert 'range(0,55,5)' in archive
    assert 'verify_remote_asset' in archive
    assert 'info.get("private") is not True' in archive
