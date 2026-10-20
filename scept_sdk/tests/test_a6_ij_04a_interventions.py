"""A6-IJ-04A source syntax + SYNTHETIC MRI lesion/matched healthy intervention tests.

Do not use any patient real MRI or model weights in public GitHub CI.
"""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("scipy")

EX=Path(__file__).resolve().parents[1]/"examples"
SOURCES=("a6_ij_04a_interventions.py","a6_ij_04a_pilot.py",
         "a6_ij_04a_archive_private.py")

def module():
    spec=importlib.util.spec_from_file_location("scept04a_interventions",EX/SOURCES[0])
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def fabricated():
    rng=np.random.default_rng(240216)
    im=rng.normal(loc=0.5,scale=.12,size=(224,224)).astype(np.float32)
    foreground=np.zeros((224,224),dtype=np.uint8)
    foreground[16:208,16:208]=1
    tumor=np.zeros_like(foreground)
    # first main lesion patch has full WT and valid foreground
    tumor[64:80,80:96]=1
    im[64:80,80:96]+=np.float32(.1)
    return im,tumor,foreground

def test_python_310_syntax_and_locked_medical_privacy_bounds():
    for name in SOURCES:
        ast.parse((EX/name).read_text(),filename=name,feature_version=(3,10))
    patient=(EX/SOURCES[1]).read_text()
    private=(EX/SOURCES[2]).read_text()
    assert '"BraTS20_Training_022"' in patient
    assert 'checkpoint_model(name)' in patient
    assert 'manifest_and_zip(s)' in patient
    assert 'strict_image(image,wt,fg)' in patient
    assert 'include_inputs=False' in patient
    assert 'shamExactEquality' in patient
    assert 'nOriginalSCEPTCapturedForwards' in patient
    assert 'status"]="PASS_INPUT_INTERVENTION_PILOT_ENGINEERING"' in patient
    assert 'sourceNPZSha256' in patient
    assert 'remote_sha(s,item,got_sha,size)' in private
    assert 'originalPatientMRIStoredInSCEPTZips":False' in private
    assert 'if len(files)!=12' in private
    assert 'verified' in private.lower()

def test_tumor_selection_is_deterministic_independent_of_model_output():
    m=module()
    x,wt,fg=fabricated()
    a=m.choose(x,wt,fg)
    b=m.choose(x.copy(),wt.copy(),fg.copy())
    assert a==b
    assert [a["lesion"]["row"],a["lesion"]["col"]]==[4,5]
    assert a["lesion"]["wtFraction"]==1.0
    assert a["control"]["wtFraction"]==0.0
    assert a["control"]["dilatedWTFraction"]==0.0
    assert a["matchScore"]<=2.0
    assert a["healthyDonorPixels"]>=512
    assert len(m.fingerprint())==64

@pytest.mark.parametrize("arm",["lesion","control"])
@pytest.mark.parametrize("operator",["gaussian_local_sigma3","healthy_pool_median_fill"])
def test_roi_shape_area_and_no_external_pixel_mutation(arm,operator):
    m=module()
    x,wt,fg=fabricated()
    select=m.choose(x,wt,fg)
    changed,info=m.mutate(x,wt,fg,select,arm,operator)
    assert changed.dtype==np.float32 and changed.shape==x.shape
    assert info["outsideROIBitwiseSame"] is True
    assert info["roiPixels"]==256
    assert info["inputROIChangeL2"]>0
    r,c=info["position"];roi=m.rect(r,c)
    outside=np.ones((224,224),dtype=bool);outside[roi]=False
    np.testing.assert_array_equal(changed[outside],x[outside])
    assert info["changedPixelCount"]<=256
    if arm=="control":
        assert np.count_nonzero(wt[roi])==0
    else:
        assert np.mean(wt[roi])>=.25

def test_mask_corruption_and_no_candidate_fail_closed():
    m=module()
    x,wt,fg=fabricated()
    bad_fg=fg.copy();bad_fg[64:80,80:96]=0
    with pytest.raises(RuntimeError,match="WT is not contained"):
        m.choose(x,wt,bad_fg)
    bad_wt=np.zeros_like(wt)
    with pytest.raises(RuntimeError,match="WT or brain mask"):
        m.choose(x,bad_wt,fg)
    bad_image=np.full_like(x,np.nan)
    with pytest.raises(RuntimeError,match="finite"):
        m.choose(bad_image,wt,fg)

def test_tensor_state_relative_change_zero_and_original_patch_token_mapping():
    m=module()
    x=np.ones((1,197,768),dtype=np.float32)
    y=x.copy()
    # original 14x14 patch token map, CLS excluded
    position=4*14+5
    y[0,position+1,:]=2.0
    r=m.score_tensors(x,y,(4,5))
    assert r["patchDeltaRMS"]>0
    assert r["directTargetedTokenDeltaL2"]==pytest.approx(np.sqrt(768))
    assert r["directTargetedTokenCosineCleanVsPerturbed"]==pytest.approx(1.0)
    assert r["clsTokenDeltaL2"]==0
    identical=m.score_tensors(x,x,(4,5))
    assert identical["patchRelativeL2"]==0
    assert identical["patchDeltaRMS"]==0
