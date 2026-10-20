"""04A02 formal post-block amendment: zero-patient, zero-model contract tests."""
import ast
from pathlib import Path
import pytest

EX=Path(__file__).resolve().parents[1]/"examples"
SCANNER=EX/"a6_ij_04a02_screen_val_private.py"

def test_python_310_and_preserve_original_04a_failure():
    s=SCANNER.read_text()
    ast.parse(s,filename=str(SCANNER),feature_version=(3,10))
    assert 'PREVIOUS="results/scept_a6_ij/04a_controlled_input_pilot/eligibility_failure.json"' in s
    assert "validate_cohort(records,prior)" in s
    assert "prior.get(\"failedFixedCriterion\")!=original[\"ineligibilityReason\"]" in s
    assert 'locked[0]["patient_id"]!=PILOT' in s
    assert 'len(locked)!=55' in s
    assert 'z.read(item["zip_member"])' in s
    assert 'if original_sha!=item["p1_sha256"]' in s
    assert '"frozenModelsRun":0,"gradcamCalls":0,"sceptCalls":0' in s
    assert 'eligible[0]["patient_id"]' in s
    assert 'commit_once(session,target,orig)' in s
    assert '"sealedP5":"NOT_ACCESSED"' in s
    assert "threshold" not in s[s.index("def screen_record"):s.index("def validate_cohort")]

def obtain_validator():
    tree=ast.parse(SCANNER.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="validate_cohort")
    scope={
       "PILOT":"BraTS20_Training_022",
       "STATUS_INELIGIBLE":"INELIGIBLE_UNDER_UNCHANGED_V1_RULE",
       "fingerprint":lambda:"original-v1-locked-sha"
    }
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SCANNER),"exec"),scope)
    return scope["validate_cohort"]

def synthetic_records():
    arr=[]
    for i in range(1,56):
        p=f"BraTS20_Training_{i:03d}"
        record={"patient_id":p,"grade":"HGG" if i<=44 else "LGG",
                "status":"INELIGIBLE_UNDER_UNCHANGED_V1_RULE",
                "ineligibilityReason":"04A_FIXED_LESION_PATCH_INELIGIBLE",
                "originalFullP1NPZSHA256":f"sha-{i}"}
        if i==22:
            record["ineligibilityReason"]="04A_MATCHING_SCORE_EXCEEDED: score=3.14177 threshold=2.0"
        arr.append(record)
    return arr

def old_failure():
    return {
     "patient_id":"BraTS20_Training_022","sliceRank":1,
     "interventionPlanSha256":"original-v1-locked-sha",
     "status":"BLOCKED_FIXED_PATIENT_ROI_ELIGIBILITY",
     "originalP1PatientNPZFullSHA256":"sha-22",
     "failedFixedCriterion":"04A_MATCHING_SCORE_EXCEEDED: score=3.14177 threshold=2.0"
    }

def test_full_55_screen_preserves_original_failed_case():
    validate=obtain_validator()
    assert validate(synthetic_records(),old_failure()) is True

def test_old_fixed_patient_must_not_relabel_to_eligible():
    validate=obtain_validator()
    rows=synthetic_records()
    rows[21]["status"]="ELIGIBLE_UNDER_UNCHANGED_V1_RULE"
    with pytest.raises(RuntimeError,match="preserve prior fixed-patient"):
        validate(rows,old_failure())

def test_source_checksum_and_old_failed_reason_must_not_change():
    validate=obtain_validator()
    rows=synthetic_records()
    rows[21]["originalFullP1NPZSHA256"]="different"
    with pytest.raises(RuntimeError,match="would rewrite old"):
        validate(rows,old_failure())

def test_55_original_patient_coverage_required():
    validate=obtain_validator()
    with pytest.raises(RuntimeError,match="55 sorted"):
        validate(synthetic_records()[:-1],old_failure())
