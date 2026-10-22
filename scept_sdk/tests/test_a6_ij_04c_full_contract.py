"""04C full observational source-final scientific integrity contracts (no patient data)."""
import ast
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/"examples"

def test_python310_and_main_gate():
    for name in ("a6_ij_04c_association_private.py",
                 "a6_ij_04c02_full_tensor_audit_private.py",
                 "a6_ij_04c_finalize_private.py"):
        source=(BASE/name).read_text()
        ast.parse(source,filename=name,feature_version=(3,10))
    s=(BASE/"a6_ij_04c_finalize_private.py").read_text()
    for token in (
        'nAuthenticOriginalViTOutputFullTensorReplays',
        'original04BChunkSHA256',
        'nOriginalSourceRunTensorSets',
        'sourceSummaryNumericallyReverified',
        'original04C01AnalysisSHA256',
        'nOriginalLOPOFolds',
        'for chunk in range(1,8)',
        'len(groups)!=33',
        'len({r["patient_id"] for r in groups})!=33',
        '!=594',
        'commit_once(session,final,raw)',
        'scientificVerdict',
        'OBSERVATIONAL_INTERNAL_REPRESENTATION_VS_DECISION_ONLY_NOT_CAUSAL_FAITHFULNESS',
        '"sealedP5Test":"NOT_ACCESSED"',
        '"independentP6":"NOT_ACCESSED"',
    ):
        assert token in s,token

def test_full_authentic_zip_layer_replay_not_placeholder():
    s=(BASE/"a6_ij_04c02_full_tensor_audit_private.py").read_text()
    for token in (
       'hashfile(tmp)!=receipt["assetSHA256"]',
       'rec=load_observation(tmp.name)',
       'rec.tensor("model.output")',
       'rec.tensor(k)',
       'sourceSummaryNumericallyReverified',
       'layer_metrics(clean[k],newer[k],coord,metric["layerStateDelta"][k])',
       'if checked!=18*len(allcase)',
       'commit_once(session,target,canonical(body))',
    ):
       assert token in s,token
    assert 'checkpoint_model(' not in s
    assert 'model.observe(' not in s
