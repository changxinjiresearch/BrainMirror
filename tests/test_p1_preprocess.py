"""Synthetic NIfTI spatial tests; NOT human MRI qualification."""
import json
from pathlib import Path
import numpy as np
import pytest
from brainmirror.p1.preprocess import PreprocessConfig,preprocess_pair
from brainmirror.p1.duplicates import review_candidates
from brainmirror.p1.closeout import evaluate


def _pair(tmp_path, shape=(96,96,96), different_affine=False):
    nib=pytest.importorskip('nibabel')
    dr=tmp_path/'private-input';dr.mkdir(exist_ok=True)
    idx=np.indices(shape,dtype=np.float32)
    mask=(((idx[0]-48)/24)**2+((idx[1]-48)/28)**2+((idx[2]-48)/22)**2)<1
    data=(100+25*np.sin(idx[0]/5)+30*np.cos(idx[1]/6))*mask
    a=np.diag([-1.2,1.3,1.4,1.0]);a[0,3]=120
    b=a.copy();b[0,3]+=2 if different_affine else 0
    for name,arr,aff in [('brain.nii.gz',data,a),('mask.nii.gz',mask.astype(np.uint8),b)]:
        im=nib.Nifti1Image(arr.astype(np.float32),aff)
        im.set_qform(aff,code=1);im.set_sform(aff,code=1)
        nib.save(im,str(dr/name))
    return dr/'brain.nii.gz',dr/'mask.nii.gz'


def test_real_nifti_synthetic_affine_preserved_and_mask_resampled(tmp_path):
    nib=pytest.importorskip('nibabel')
    image,mask=_pair(tmp_path)
    repo=tmp_path/'repo';repo.mkdir()
    output=tmp_path/'private-output'
    token='a'*64
    r=preprocess_pair(image,mask,output,token,PreprocessConfig(target_shape=(96,112,96)),
                      repo_root=repo,mask_provenance='SynthSeg: synthetic verified test mask')
    assert r['preprocess_gate']=='NEEDS_INDEPENDENT_REVIEW'
    assert r['brain_retention']>0.999
    final=nib.load(str(output/(token+'_image.nii.gz')))
    final_mask=nib.load(str(output/(token+'_mask.nii.gz')))
    assert final.shape==(96,112,96)
    assert nib.aff2axcodes(final.affine)==('R','A','S')
    assert np.array_equal(final.affine, final_mask.affine)
    values=np.asarray(final.dataobj)
    maskvox=np.asarray(final_mask.dataobj)>0
    assert np.all(values[~maskvox]==0)
    assert abs(values[maskvox].mean())<0.01
    assert abs(values[maskvox].std()-1)<0.01
    assert not (output/(token+'_image.nii.gz')).is_relative_to(repo)
    with pytest.raises(ValueError,match='overwrite'):
        preprocess_pair(image,mask,output,token,PreprocessConfig(target_shape=(96,112,96)),
                        repo_root=repo,mask_provenance='Synthetic valid test mask')


def test_missing_mask_alignment_must_fail(tmp_path):
    image,mask=_pair(tmp_path,different_affine=True)
    repo=tmp_path/'repo';repo.mkdir()
    with pytest.raises(ValueError,match='affine/shape mismatch'):
        preprocess_pair(image,mask,tmp_path/'out','b'*64,PreprocessConfig(),
                        repo_root=repo,mask_provenance='Synthetic valid test mask')


def test_tight_fov_and_unsupported_n4(tmp_path):
    image,mask=_pair(tmp_path)
    repo=tmp_path/'repo';repo.mkdir()
    with pytest.raises(ValueError,match='field of view'):
        preprocess_pair(image,mask,tmp_path/'out','c'*64,
                        PreprocessConfig(target_shape=(32,32,32)),repo_root=repo,
                        mask_provenance='Synthetic valid test mask')


def test_near_duplicate_candidates_not_automatically_classified(tmp_path):
    image,mask=_pair(tmp_path)
    raw=image.parent
    # compare same physical image bytes with two different pseudonyms (two scans)
    import shutil
    second=raw/'duplicate.nii.gz';shutil.copy2(image,second)
    records=[{'qc_status':'PASS','relative_path':'brain.nii.gz','scan_hmac':'a'*64,'subject_hmac':'1'*64},
             {'qc_status':'PASS','relative_path':'duplicate.nii.gz','scan_hmac':'b'*64,'subject_hmac':'2'*64}]
    output=tmp_path/'private-candidates'
    r=review_candidates(records,raw,output,threshold=0.995)
    assert r['cross_subject_candidates']==1
    assert r['gate']=='BLOCKED_PENDING_REVIEW'
    text=(output/'near_duplicate_candidates_PRIVATE.json').read_text()
    assert 'brain.nii.gz' not in text
    assert json.loads(text)['candidate_pairs'][0]['reviewed'] is False


def test_closeout_fails_closed_on_missing_real_evidence():
    receipt={'source':{'source':'IXI'},'source_terms_acknowledged_by_operator':True,
             'data_tier':'REAL_MRI_CANDIDATE_NOT_SCIENTIFICALLY_VALIDATED',
             'files_seen':580,'eligible_subjects':580,'scans_header_pass':580,'scans_failed':0,
             'voxel_qc_executed_for_passing_scans':True,'split_locked':False,
             'real_mri_training_started':False}
    result=evaluate(receipt)
    assert not result.pass_gate
    assert 'OFFICIAL_DATA_ARCHIVE_NOT_VERIFIED' in result.blockers
    assert 'INDEPENDENT_DATA_AND_SPLIT_REVIEW_MISSING' in result.blockers
    assert 'NEAR_DUPLICATE_REVIEW_MISSING' in result.blockers


def test_duplicate_privacy_path_check(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    raw=tmp_path/'raw';raw.mkdir()
    with pytest.raises(ValueError,match='outside'):
        review_candidates([],raw,repo/'audit',repo_root=repo)


def test_gate_count_must_represent_full_ixicohort():
    receipt={'source':{'source':'IXI'},'source_terms_acknowledged_by_operator':True,
             'data_tier':'REAL_MRI_CANDIDATE_NOT_SCIENTIFICALLY_VALIDATED',
             'files_seen':200,'eligible_subjects':200,'scans_header_pass':200,'scans_failed':0,
             'voxel_qc_executed_for_passing_scans':True,'split_locked':False,
             'real_mri_training_started':False}
    gate=evaluate(receipt,acquisition={'official_archive_sha256':'a'*64,'archive_verified':True,
                'archive_subjects_verified':200},
                preprocess={'all_eligible_scans_completed':True,'visual_qc_approved':True},
                duplicates={'all_scans_screened':True,'all_candidates_adjudicated':True},
                independent_review={'independent_reviewer_approved':True,'review_receipt_sha256':'a'*64})
    assert not gate.pass_gate
    assert 'FULL_COHORT_NOT_ESTABLISHED' in gate.blockers
