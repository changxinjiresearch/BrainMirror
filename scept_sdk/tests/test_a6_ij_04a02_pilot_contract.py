"""A6-IJ-04A02 post-failure exploratory pilot safety contracts, no patient MRI.

Source code is parsed without importing private medical data or reusing
original v1 private outputs. Tests pin exact original 55-person source and
revised-pilot distinct archive destination.
"""
import ast
from pathlib import Path
import pytest

EX=Path(__file__).resolve().parents[1]/"examples"
SCRIPTS=("a6_ij_04a02_pilot.py","a6_ij_04a02_archive_private.py")

def test_python_310_syntax_and_original_source_fingerprint():
    for name in SCRIPTS:
        ast.parse((EX/name).read_text(),filename=name,feature_version=(3,10))
    s=(EX/SCRIPTS[0]).read_text()
    for token in (
        'PILOT="BraTS20_Training_040"',
        'ORIGINAL_SCREEN_GIT_BLOB_SHA="fe39ebcc9bc323aa260c6cec4fd497287971d06c"',
        'PILOT_SOURCE_P1_SHA256="b6737737a34cccfb466f8091b823c98952e7c473f638e3376de93c9ec67e177a"',
        'OUT=ROOT/"a6_ij_04a02_private"',
        '"sameOriginalROIAndInterventionRuleSha256":fingerprint()',
        '"originalROIMatchScoreUpperLimit":2.0',
        'screen_response.json().get("sha")!=ORIGINAL_SCREEN_GIT_BLOB_SHA',
        'screen.get("nScreened")!=55 or screen.get("nEligible")!=33',
        'screen.get("original04APlanSHA256")!=fingerprint()',
        'screen_case["originalFullP1NPZSHA256"]!=PILOT_SOURCE_P1_SHA256',
        'previous.get("patient_id")!="BraTS20_Training_022"',
        'next(q["patient_id"] for q in all_patients',
        'selection["matchScore"]>2.0',
        'prior_roi[arm][key]',
        'write_fixed(result_path,report)',
        'status"]="PASS_AMENDED_INPUT_INTERVENTION_PILOT_ENGINEERING"',
        '"amendmentPlanSha256":amendment_fingerprint()',
        'include_inputs=False',
        'shamExactEquality',
        '"numericScientificConclusions":"NOT_ESTABLISHED"',
    ):
        assert token in s, token

def test_archive_separate_original_and_private_complete():
    p=(EX/SCRIPTS[1]).read_text()
    for token in (
        'from a6_ij_04a02_pilot import',
        'TAG="scept-a6-ij-04a02-amended-val-040-input-intervention-v1"',
        'BASE="results/scept_a6_ij/04a02_amended_040_intervention_pilot"',
        'ASSET="scept_04a02_amended_val_040_10_original_observations.zip"',
        'lock.get("amendment")!=AMENDMENT',
        'data.get("amendmentPlan")!=AMENDMENT',
        '"original04Av1BlockedPreserved":True',
        'len(files)!=12',
        'remote_sha(s,item,got_sha,size)',
        'commit_once(s,f"{BASE}/analysis.json",report_bytes)',
        'commit_once(s,f"{BASE}/archive_receipt.json",rec)',
        'originalPatientMRIStoredInSCEPTZips":False',
    ):
        assert token in p, token
    assert 'TAG="scept-a6-ij-04a-locked-input-intervention-pilot-v1"' not in p
    assert '04a_controlled_input_pilot' not in p

def test_no_new_rule_threshold_and_no_original_case_relabel():
    s=(EX/SCRIPTS[0]).read_text()
    original=(EX/"a6_ij_04a_interventions.py").read_text()
    assert '"controlMaxIntensityMatchScore":2.0' in original
    assert '"sameOriginalROIAndInterventionRuleSha256":fingerprint()' in s
    assert "selection=choose(image,wt,fg)" in s
    assert "selection[arm][key]" in s
    assert 'case["p1_sha256"]!=PILOT_SOURCE_P1_SHA256' in s
    assert 'include_inputs=True' not in s

def test_pilot_does_not_select_max_lesion_response_posthoc():
    s=(EX/SCRIPTS[0]).read_text()
    # Selection must be before ANY model forward; no model-output-gated case
    # choosing or gradcam-gated case choosing.
    assert s.index('selection=choose(image,wt,fg)')<s.index('model=checkpoint_model(name)')
    assert 'screen["patientRecords"]' in s
    assert '"numericScientificConclusions":"NOT_ESTABLISHED"' in s
