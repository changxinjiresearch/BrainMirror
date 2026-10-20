"""03A executable contract — source-only safety checks, no patient bytes."""
import ast
from pathlib import Path

EX=Path(__file__).resolve().parents[1]/"examples"

def test_03a_scripts_parse_on_python_310():
    for p in ("a6_ij_03a_pair_source.py","a6_ij_03a_archive_private.py"):
        source=(EX/p).read_text()
        ast.parse(source,filename=p,feature_version=(3,10))

def test_03a_strict_pairing_no_new_gradcam_and_private_record():
    m=(EX/"a6_ij_03a_pair_source.py").read_text()
    a=(EX/"a6_ij_03a_archive_private.py").read_text()
    assert '"patient_id","slice_rank=1","model=S/J"' in m
    assert '"predictionMaxAbs":0.03' in m
    assert '"ckaMaxAbs":1e-5' in m
    assert '"bootstrapResamples":2000' in m
    assert '"bootstrapSeed":240216' in m
    assert 'if h(original)!=meta.get("csv_sha256"):' in m
    assert 'if len(p4rows)!=1100' in m
    assert 'if len(joined)!=110' in m
    assert '"test_set")!="NOT_ACCESSED"' in m
    assert '02B' in m and '02C' in m
    assert 'r["slice_rank"]!=1' in a
    assert 'info.get("private") is not True' in a
    assert 'write_immutable' in a
    assert 'patientRawTensorsCopied":False' in a
