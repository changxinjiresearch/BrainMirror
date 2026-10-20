"""A6-IJ-03B source contract (synthetic/no medical data); Python 3.10 compatible."""
import ast
from pathlib import Path

EX=Path(__file__).resolve().parents[1]/"examples"
FILES=("a6_ij_03b_measure.py","a6_ij_03b_archive_chunk_private.py",
       "a6_ij_03b_finalize.py","a6_ij_03b_run.py")

def test_all_03b_sources_python_310_parse():
    for item in FILES:
        ast.parse((EX/item).read_text(),filename=item,feature_version=(3,10))

def test_locked_scientific_design_and_sensitive_patient_boundaries():
    main=(EX/FILES[0]).read_text()
    arc=(EX/FILES[1]).read_text()
    fin=(EX/FILES[2]).read_text()
    run=(EX/FILES[3]).read_text()
    assert '"totalNewModelForwards":990' in main
    assert '"reusedOriginal02BModelForwards":110' in main
    assert '"predictionMaxAbsoluteDifference":0.03' in main
    assert '"bootstrapResamples":2000' in main
    assert '"sealedP5Test":"NOT_ACCESSED"' in main
    assert '"externalP6":"NOT_ACCESSED"' in main
    assert '"patient_id"' in main and '"slice_rank"' in main
    assert 'if int(row["axial_index"])!=int(axes[rank-1])' in main
    assert 'if len(indexed)!=100' in main
    assert 'rank in range(1,11)' in main
    assert 'rank==1' in main
    assert 'include_inputs=False' in main
    assert 'recompute_naf()' in main
    assert 'if len(files)!=95' in arc
    assert 'if checked_pairs!=100 or len(originals)!=95:' in arc
    assert 'remote_sha(s,asset,digest_value,shard_path.stat().st_size)' in arc
    assert 'remote_status(s,num)' in arc
    assert 'if len(all_rows)!=55 or set(all_rows)!=set(ids)' in fin
    assert 'n_new!=990 or n_reused!=110' in fin
    assert '2000' in fin and 'patient' in fin.lower()
    assert 'if status is not None:' in run
    assert '"a6_ij_03b_archive_chunk_private.py"' in run
    assert '"a6_ij_03b_finalize.py"' in run
