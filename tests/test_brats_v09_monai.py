"""Official MONAI 1.5.1 + true NIfTI I/O, but SYNTHETIC TEST DATA ONLY."""
from pathlib import Path
import numpy as np
import pytest
import torch

pytest.importorskip('monai')
nib=pytest.importorskip('nibabel')

from brainmirror.scept.brats2020 import discover_training_cases, load_case, dice_regions
from brainmirror.scept.brats_v09_pipeline import predict_full,explain_heldout_case
from brainmirror.scept.monai_adapter import build_monai_model


def make_nifti(tmp_path):
    d=tmp_path/'BraTS20_Training_001';d.mkdir()
    rng=np.random.default_rng(240216)
    shp=(32,32,32)
    a=np.zeros(shp,dtype=np.float32)
    a[3:29,3:29,3:29]=rng.normal(100,10,size=(26,26,26))
    gt=np.zeros(shp,dtype=np.int16)
    gt[11:19,11:19,11:19]=2
    gt[13:17,13:17,13:17]=1
    gt[14:16,14:16,14:16]=4
    for i,mod in enumerate(('t1','t1ce','t2','flair')):
        arr=a*(1+0.03*i)
        nib.save(nib.Nifti1Image(arr,np.eye(4)),str(d/f'{d.name}_{mod}.nii.gz'))
    nib.save(nib.Nifti1Image(gt,np.eye(4)),str(d/f'{d.name}_seg.nii.gz'))
    return d


def test_actual_nifti_loader_and_monai_sliding_window(tmp_path):
    make_nifti(tmp_path)
    x,y,qc=load_case(discover_training_cases(tmp_path)[0])
    assert x.shape==(4,32,32,32) and y.shape==(3,32,32,32)
    assert [int(y[c].sum()) for c in range(3)]==[512,72,8]
    torch.manual_seed(22)
    model=build_monai_model(depths=(1,1,1,1),norm_name='batch').eval()
    prob=predict_full(model,x,patch_size=32,device='cpu')
    assert prob.shape==(3,32,32,32)
    assert np.isfinite(prob).all() and 0<=prob.min()<=prob.max()<=1
    assert all(k in dice_regions(prob,y) for k in ('WT','TC','ET'))


def test_actual_monai_scca_local_on_synthetic_roi(tmp_path):
    make_nifti(tmp_path)
    x,y,_=load_case(discover_training_cases(tmp_path)[0])
    torch.manual_seed(44)
    model=build_monai_model(depths=(2,1,1,1),norm_name='batch').eval()
    result=explain_heldout_case(model,x,y,roi_region='WT',patch_size=32,
                               device='cpu',max_exact=3,effect_tol=1e-5)
    assert result['scope']=='local_patch_not_full_volume'
    assert result['path_search']['forward_budget_used']<=3
    assert result['status'] in ('unresolved','conflicting','intervened','cross_reference_supported')
    assert result['reference_qc'][0]['outside_max_abs_change']==0
