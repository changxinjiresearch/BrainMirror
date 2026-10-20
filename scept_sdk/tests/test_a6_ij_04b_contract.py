"""04B multi-Python synthetic equal-input-L2 scientific invariants; NO REAL MRI.

Inputs fabricated; never instantiate neural model, open patient assets or read
private research GitHub. Source code paths are original SHA-bound and 33-person
inference cannot execute unless the full input-only preflight passed.
"""
from __future__ import annotations
import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
EX=Path(__file__).resolve().parents[1]/"examples"
sys.path.insert(0,str(EX))

FILES=(
 "a6_ij_04b_matched_interventions.py",
 "a6_ij_04b_preflight_private.py",
 "a6_ij_04b_measure.py",
 "a6_ij_04b_archive_private.py",
 "a6_ij_04b_finalize.py",
 "a6_ij_04b_run.py"
)

def module():
 return importlib.import_module("a6_ij_04b_matched_interventions")

def mock():
 rng=np.random.default_rng(240216)
 x=rng.normal(.5,.15,size=(224,224)).astype(np.float32)
 fg=np.zeros((224,224),dtype=np.uint8)
 fg[16:208,16:208]=1
 wt=np.zeros_like(fg)
 wt[64:80,80:96]=1
 x[64:80,80:96]+=np.float32(.20)
 return x,wt,fg

def test_six_real_scripts_python_310_syntax():
 for name in FILES:
  ast.parse((EX/name).read_text(),filename=name,feature_version=(3,10))

def test_prespecified_original_04a_geometry_and_33_complete_input_dose_pairs():
 from a6_ij_04a_interventions import choose
 m=module()
 x,wt,fg=mock()
 sel=choose(x,wt,fg)
 inputs,diag=m.matched_inputs(x,wt,fg,sel)
 assert len(inputs)==9 and len(diag)==8
 assert m.PLAN["nEligible"]==33
 assert m.PLAN["totalRealSCEPTSourceForwards"]==594
 assert m.PLAN["forwardsPerPatient"]==18
 assert m.PLAN["fractionalDoses"]==[.25,.5]
 assert m.PLAN["original04Av1ROISHA256"]
 assert len(m.fingerprint04b())==64
 assert np.array_equal(inputs["clean"],x)
 for op in m.OPS:
  for dose in m.DOSES:
   l=m.condition_name(op,dose,"lesion")
   c=m.condition_name(op,dose,"control")
   dl=diag[l];dc=diag[c]
   assert dl["roiPixels"]==dc["roiPixels"]==256
   assert dl["outsideROIBitwiseSame"] and dc["outsideROIBitwiseSame"]
   assert dl["lesionWTFraction"]>0
   assert dc["lesionWTFraction"]==0
   assert 0<dl["fractionalInterpolationAlpha"]<=1
   assert 0<dc["fractionalInterpolationAlpha"]<=1
   assert dl["actualPixelL2"]>0 and dc["actualPixelL2"]>0
   assert abs(dl["actualPixelL2"]-dc["actualPixelL2"])<=max(1e-5,1e-5*dl["commonSourceMaxL2"])
   assert dl["actualPixelL2"]==pytest.approx(dose*dl["commonSourceMaxL2"],abs=1e-5,rel=1e-5)
   for arm,k in (("lesion",l),("control",c)):
    row,col=sel[arm]["row"],sel[arm]["col"]
    outside=np.ones_like(wt,dtype=bool)
    outside[row*16:(row+1)*16,col*16:(col+1)*16]=False
    assert np.array_equal(inputs[k][outside],x[outside])
    assert inputs[k].shape==(224,224) and inputs[k].dtype==np.float32

def test_input_operator_pair_is_independent_of_any_model_response():
 from a6_ij_04a_interventions import choose
 m=module()
 x,wt,fg=mock()
 selection=choose(x,wt,fg)
 a,stats=m.matched_inputs(x,wt,fg,selection)
 b,stats2=m.matched_inputs(x.copy(),wt.copy(),fg.copy(),selection)
 assert stats==stats2
 for name in a:
  np.testing.assert_array_equal(a[name],b[name])

def test_locked_true_grade_margin_and_sign_of_specificity():
 m=module()
 clean=m.real_output([0.,2.],"HGG")
 les=m.real_output([0.,1.5],"HGG")
 control=m.real_output([0.,1.9],"HGG")
 e=m.contrast_from_outputs(clean,{"lesion":les,"control":control},
                           {"actualPixelL2":1.0},{"actualPixelL2":1.0})
 assert clean["trueGradeMargin"]==2
 assert e["signedMarginChangeLesion"]==pytest.approx(-.5)
 assert e["signedMarginChangeControl"]==pytest.approx(-.1)
 assert e["absMarginSpecificityLesionMinusControl"]==pytest.approx(.4)
 assert e["lesionActualPixelL2"]==e["controlActualPixelL2"]
 lg=m.real_output([3.,1.],"LGG")
 assert lg["trueGradeMargin"]==2

def test_no_partial_or_invalid_source_may_progress_to_frozen_model():
 measure=(EX/"a6_ij_04b_measure.py").read_text()
 pre=(EX/"a6_ij_04b_preflight_private.py").read_text()
 run=(EX/"a6_ij_04b_run.py").read_text()
 archive=(EX/"a6_ij_04b_archive_private.py").read_text()
 final=(EX/"a6_ij_04b_finalize.py").read_text()
 assert 'SCREEN_GIT_BLOB_SHA="fe39ebcc9bc323aa260c6cec4fd497287971d06c"' in measure
 assert 'previous.get("nNewMatchedPass")!=33' in measure
 assert measure.index('previous=json.loads(private_contents(s,preflight_path))')<measure.index('models={k:checkpoint_model(k) for k in names}')
 assert '"originalP1NPZSHA256":item["p1_sha256"]' in pre
 assert '"nNewModelForward":0' in pre
 assert 'if blocked:' in pre
 assert 'if len(originals)!=n*19' in archive
 assert 'remote_sha(s,asset,source_sha,size)' in archive
 assert '"nPatientOriginalRealForwards":18*n' in archive
 assert '"nOriginalFirstSliceEligiblePatients":33' in final
 assert '"nActualGenuineSCEPTSourceRuns":594' in final
 assert 'if ids!=[p["patient_id"] for p in eligible]' in final
 assert '2000' in final and 'size=(NFLIP,len(v))' in final
 assert 'ensure_preflight(s,eligible,lock)' in run
 assert '"a6_ij_04b_finalize.py"' in run
 assert 'include_inputs=False' in measure
 assert 'NOT_ACCESSED' in archive and 'NOT_ACCESSED' in final

def test_nonzero_original_equal_l2_cases_are_mandatory():
 m=module()
 from a6_ij_04a_interventions import choose
 x,wt,fg=mock()
 a=choose(x,wt,fg)
 fake=x.copy()
 # Preserve source masks but change the selected lesion patch to a constant
 # to ensure code never silently changes its selected, original ROI.
 a_wrong=dict(a)
 a_wrong["lesion"]=dict(a["lesion"],row=99)
 with pytest.raises((IndexError,ValueError,RuntimeError,TypeError)):
  m.matched_inputs(x,wt,fg,a_wrong)
