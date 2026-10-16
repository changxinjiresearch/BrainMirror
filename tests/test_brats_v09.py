"""CPU controls for source inventory, BraTS labels, security and leakage gates."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from brainmirror.scept.brats2020 import (
  CaseFiles, discover_training_cases,audited_inventory,patient_split,label_regions,
  normalized_modalities,dice_regions,region_reference,roi_patch,
)
from brainmirror.scept.brats_v09_pipeline import (
  attest_metadata,require_clean_splits,patch_for_train,brats_loss,
)
from scripts.scept_brats_v09 import read_split


def _case_tree(tmp_path,n=12):
    out=[]
    for k in range(n):
        cid=f'BraTS20_Training_{k+1:03d}'; d=tmp_path/'training'/cid;d.mkdir(parents=True)
        for mod in ('t1','t1ce','t2','flair','seg'):(d/f'{cid}_{mod}.nii.gz').write_bytes(b'mocked')
        out.append(cid)
    return out


def test_roi_labels_nested_hierarchy():
    seg=np.array([[[0,1,2,4]]]);mask=label_regions(seg)
    assert mask.shape==(3,1,1,4)
    assert mask[:,0,0,:].tolist()==[[0,1,1,1],[0,1,0,1],[0,0,0,1]]
    with pytest.raises(ValueError):label_regions(np.array([3]))


def test_source_discovery_and_integrity(tmp_path):
    ids=_case_tree(tmp_path)
    root=tmp_path/'training';cases=discover_training_cases(root)
    assert len(cases)==len(ids)
    inv=audited_inventory(cases,root=root,expected_count=12)
    assert all(not str(tmp_path) in str(row) for row in inv['cases'])
    assert len(inv['case_manifest_sha256'])==64
    with pytest.raises(ValueError):audited_inventory(cases,root=root,expected_count=369)
    (root/ids[0]/f'{ids[0]}_t2.nii').write_bytes(b'duplicate')
    with pytest.raises(ValueError,match='ambiguous'):discover_training_cases(root)


def test_patient_split_stable_and_disjoint():
    ids=[f'BraTS20_Training_{i:03d}' for i in range(1,370)]
    s=patient_split(ids)
    assert s==patient_split(ids[::-1]);assert require_clean_splits(s)
    assert sorted(sum(s.values(),[]))==sorted(ids)
    assert len(s['train'])==258 and len(s['validation'])==55 and len(s['test'])==56
    s['test'][0]=s['train'][0]
    with pytest.raises(ValueError,match='leakage'):require_clean_splits(s)


def test_normalize_nonbrain_and_degenerate_controls():
    rng=np.random.default_rng(1)
    vols=np.zeros((4,16,16,16),dtype=np.float32)
    vols[:,3:14,3:14,3:14]=rng.normal(200,50,(4,11,11,11))
    x,qc=normalized_modalities(vols)
    assert np.isfinite(x).all() and np.all(x[:,:3]==0)
    assert qc['brain_voxels']>64
    with pytest.raises(ValueError):normalized_modalities(np.ones((4,4,4,4),np.float32))


def test_dice_empty_not_artificial_perfect():
    target=np.zeros((3,2,2,2),np.float32)
    probs=np.zeros_like(target)
    assert all(x['dice'] is None for x in dice_regions(probs,target).values())
    probs[2,0,0,0]=1
    assert dice_regions(probs,target)['ET']['dice']==0.


def test_diagnostic_refs_offtarget_and_qa():
    torch.manual_seed(3)
    x=torch.randn(1,4,32,32,32)
    roi=torch.zeros(32,32,32,dtype=torch.bool);roi[11:18,11:18,11:18]=True
    for policy in ('tissue_median','local_blur'):
        ref,qc=region_reference(x,roi,policy=policy)
        assert torch.equal(ref[...,~roi],x[...,~roi]);assert qc['roi_mean_abs_change']>0
        assert qc['outside_max_abs_change']==0.
    with pytest.raises(ValueError):region_reference(x,roi,policy='unregistered_inpainting')
    with pytest.raises(ValueError):region_reference(x,torch.zeros_like(roi))


def test_roi_patch_and_train_patch_alignment():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(4,50,46,40)).astype(np.float32)
    y=np.zeros((3,50,46,40),np.float32);y[:,20,25,17]=1
    cropped,mask,metadata=roi_patch(torch.from_numpy(x[None]),torch.from_numpy(y[0]>.5),size=32)
    assert cropped.shape==(1,4,32,32,32) and mask.any()
    a,b=patch_for_train(x,y,size=32,seed=3)
    assert a.shape==b.shape[:1]+(4,32,32,32)
    assert b.any()
    with pytest.raises(ValueError):patch_for_train(x,y,size=30)


def test_checkpoint_attestation_rejects_synthetic_and_test_leakage():
    splits={'train':['a','b'],'validation':['c'],'test':['d']}
    meta={'schema':'scept/brats2020-checkpoint/v1','dataset':'BraTS2020_training','monai_version':'1.5.1',
          'optimization_steps':10,'epochs':1,'train_ids':['a','b'],'validation_ids':['c'],
          'inventory_sha256':'foo','seed':3,'final_loss':.8,'synthetic_only':False}
    assert attest_metadata(meta,splits=splits,expected_inventory_digest='foo')
    with pytest.raises(ValueError):attest_metadata(dict(meta,train_ids=['a','d']),splits=splits)
    with pytest.raises(ValueError):attest_metadata(dict(meta,optimization_steps=0),splits=splits)
    with pytest.raises(ValueError):attest_metadata(dict(meta,synthetic_only=True),splits=splits)
    with pytest.raises(ValueError):attest_metadata(meta,splits=splits,expected_inventory_digest='other')


def test_train_loss_grad_and_shape():
    torch.manual_seed(1)
    logits=torch.randn(1,3,4,4,4,requires_grad=True)
    y=torch.randint(0,2,(1,3,4,4,4)).float()
    loss=brats_loss(logits,y);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):brats_loss(logits,y[:,1:])


def test_split_manifest_drift_rejected(tmp_path):
    ids=['a','b','c'];split=patient_split(ids,fractions=(.34,.33,.33))
    # 3 names allow 1,0,2 due floor; manifest-level coverage is still enforced
    p=tmp_path/'split.json'
    p.write_text(json.dumps({'schema':'scept/brats2020-split/v1','inventory_sha256':'abc','splits':split}))
    with pytest.raises(ValueError,match='digest'):read_split(p,ids,'different')


def test_source_audited_355_segment_suffix(tmp_path):
    d=tmp_path/'BraTS20_Training_355';d.mkdir()
    for mod in ('t1','t1ce','t2','flair'):(d/f'{d.name}_{mod}.nii').write_bytes(b'ok')
    (d/f'{d.name}_segm.nii').write_bytes(b'known anomaly')
    cases=discover_training_cases(tmp_path)
    assert len(cases)==1 and cases[0].paths['seg'].name.endswith('_segm.nii')
    (d/f'{d.name}_seg.nii').write_bytes(b'duplicate')
    with pytest.raises(ValueError,match='ambiguous'):discover_training_cases(tmp_path)
