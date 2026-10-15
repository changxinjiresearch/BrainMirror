"""P1 completion gate: fail closed if evidence is incomplete or synthetic.

Machine validation cannot substitute for independent real-image inspection.
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Gate:
    pass_gate:bool
    blockers:tuple[str,...]
    evidence:dict


def evaluate(receipt:dict, *, acquisition:dict|None=None,preprocess:dict|None=None,
             duplicates:dict|None=None,independent_review:dict|None=None)->Gate:
    blockers=[]
    if receipt.get('data_tier')!='REAL_MRI_CANDIDATE_NOT_SCIENTIFICALLY_VALIDATED':
        blockers.append('REAL_DATA_PROVENANCE_MISSING')
    if receipt.get('source',{}).get('source')!='IXI' or not receipt.get('source_terms_acknowledged_by_operator'):
        blockers.append('SOURCE_OR_LICENSE_UNVERIFIED')
    if not acquisition or acquisition.get('official_archive_sha256') is None or acquisition.get('archive_verified') is not True:
        blockers.append('OFFICIAL_DATA_ARCHIVE_NOT_VERIFIED')
    if receipt.get('files_seen',0)<500 or receipt.get('eligible_subjects',0)<500:
        blockers.append('FULL_COHORT_NOT_ESTABLISHED')
    if receipt.get('scans_failed',0)>0 or receipt.get('scans_header_pass',0)!=receipt.get('files_seen',0):
        blockers.append('QC_FAILURES_UNRESOLVED')
    if not receipt.get('voxel_qc_executed_for_passing_scans'):
        blockers.append('REAL_VOXEL_QC_NOT_EXECUTED')
    if not preprocess or preprocess.get('all_eligible_scans_completed') is not True or preprocess.get('visual_qc_approved') is not True:
        blockers.append('FULL_PREPROCESS_OR_VISUAL_QC_MISSING')
    if not duplicates or duplicates.get('all_scans_screened') is not True or duplicates.get('all_candidates_adjudicated') is not True:
        blockers.append('NEAR_DUPLICATE_REVIEW_MISSING')
    if not independent_review or independent_review.get('independent_reviewer_approved') is not True or not independent_review.get('review_receipt_sha256'):
        blockers.append('INDEPENDENT_DATA_AND_SPLIT_REVIEW_MISSING')
    if receipt.get('split_locked') is True:
        blockers.append('UNAUTHORIZED_PRIOR_SPLIT_LOCK_IN_INTAKE')
    if receipt.get('real_mri_training_started') is not False:
        blockers.append('TRAINING_ALREADY_STARTED_OR_UNKNOWN')
    # Cohort size must match source source-specific reviewed archive inventory,
    # not assumed from IXI website statement.
    if not acquisition or acquisition.get('archive_subjects_verified')!=receipt.get('eligible_subjects'):
        blockers.append('COHORT_COUNTS_NOT_RECONCILED')
    return Gate(not blockers,tuple(blockers),{'files_seen':receipt.get('files_seen'),
                                'eligible_subjects':receipt.get('eligible_subjects')})


def closeout(receipt_path:Path,output_path:Path,*,acquisition_path:Path|None=None,
             preprocess_path:Path|None=None,duplicate_path:Path|None=None,
             review_path:Path|None=None)->Gate:
    def load(path):return json.loads(path.read_text()) if path and path.exists() else None
    gate=evaluate(load(receipt_path),acquisition=load(acquisition_path),
                  preprocess=load(preprocess_path),duplicates=load(duplicate_path),
                  independent_review=load(review_path))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    if output_path.exists():raise ValueError('Never overwrite a P1 closeout')
    output_path.write_text(json.dumps({'P1_PASS':gate.pass_gate,
        'P2_TRAINING_AUTHORIZED':gate.pass_gate,
        'blockers':gate.blockers,'counts':gate.evidence},indent=2),encoding='utf8')
    output_path.chmod(0o600)
    return gate
