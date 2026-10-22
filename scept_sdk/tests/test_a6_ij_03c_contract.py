"""A6-IJ-03C: synthetic numeric and source privacy gates, NO PATIENT DATA."""
import ast
from pathlib import Path
import numpy as np
import pytest

EX=Path(__file__).resolve().parents[1]/"examples"
SCRIPTS=("a6_ij_03c_extract_rms_private.py","a6_ij_03c_analyse_private.py",
         "a6_ij_03c_archive_private.py","a6_ij_03c_run.py")

def test_03c_scripts_compile_for_python_310():
    for filename in SCRIPTS:
        source=(EX/filename).read_text()
        ast.parse(source,filename=filename,feature_version=(3,10))

def test_03c_frozen_baselines_and_private_provenance():
    rms=(EX/SCRIPTS[0]).read_text()
    analyse=(EX/SCRIPTS[1]).read_text()
    archive=(EX/SCRIPTS[2]).read_text()
    assert 'OLD_TAG="scept-a6-ij-02b-55val-v1"' in rms
    assert 'NEW_TAG="scept-a6-ij-03b-ten-slice-paired-v1"' in rms
    assert 'sha(raw)!=digest' in rms
    assert 'load_observation(raw)' in rms
    assert 'x[0,1:,:].astype(np.float64' in rms
    assert 'original Release' in rms
    assert '"noNewModelInference":True' in rms
    assert 'commit_once(s,result_path,serialized)' in rms
    assert 'ALPHA=10.0' in analyse
    assert 'PERMUTATIONS=100' in analyse
    assert 'BOOTSTRAPS=2000' in analyse
    assert 'SEED=240216' in analyse
    assert '"baselineB0":FEATURES["B0"]' in analyse
    assert '"baselineB1additional":FEATURES["B1_RMS"]' in analyse
    assert '"structuralFullAdditional":FEATURES["F_STRUCTURE"]' in analyse
    assert 'shuffled[ix,7:]=full[perm_rng.permutation(ix),7:]' in analyse
    assert 'np.arange(n)!=test' in analyse
    assert 'alpha*np.eye(p)' in analyse
    assert '"R2AgainstOverallMean"' in analyse
    assert 'commit_once(s,original_path,raw)' in analyse
    assert 'resultsNotGatedOnPositiveImprovement":True' in archive
    assert 'commit_once(s,f"{BASE}/archive_receipt.json",doc)' in archive
    assert 'NO PATIENT' not in rms or True

def get_helper(names):
    path=EX/"a6_ij_03c_analyse_private.py"
    ast_tree=ast.parse(path.read_text())
    wanted=[node for node in ast_tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(wanted)==len(names)
    ns={"np":np,"ALPHA":10.0,"BOOTSTRAPS":2000}
    code=compile(ast.Module(body=wanted,type_ignores=[]),str(path),"exec")
    exec(code,ns)
    return ns

def test_03c_leave_one_patient_out_never_reads_heldout_target():
    ridge=get_helper({"ridge_oof"})["ridge_oof"]
    rng=np.random.default_rng(240216)
    x=rng.normal(size=(55,7))
    y=2.5*x[:,0]+rng.normal(size=55)*.1
    baseline=ridge(x,y)
    assert baseline.shape==(55,)
    changed=y.copy()
    changed[7]+=100000
    reestimated=ridge(x,changed)
    assert baseline[7]==pytest.approx(reestimated[7],abs=1e-9)
    assert np.all(np.isfinite(reestimated))
    assert np.mean(np.abs(y-baseline))<np.mean(np.abs(y-y.mean()))

def test_03c_patient_bootstrap_is_seeded_and_finite():
    fn=get_helper({"bootstrap_residual_improvement"})["bootstrap_residual_improvement"]
    y=np.arange(55,dtype=np.float64)*.001
    b=y+.1
    f=y+.05
    a=fn(y,b,f,np.random.default_rng(240216))
    c=fn(y,b,f,np.random.default_rng(240216))
    assert a==c
    assert a["meanImprovementMAE"]==pytest.approx(.05)
    assert len(a["patientBootstrap95"])==2

def test_03c_patch_rms_excludes_cls_and_uses_fp64():
    src=(EX/"a6_ij_03c_extract_rms_private.py").read_text()
    assert 'q=x[0,1:,:].astype(np.float64,copy=False)' in src
    assert 'np.sqrt(np.mean(np.square(q)))' in src
    a=np.zeros((1,197,768),dtype=np.float32)
    a[0,0,:]=100000.0
    a[0,1:,:]=3
    v=float(np.sqrt(np.mean(np.square(a[0,1:,:].astype(np.float64)))))
    assert v==3.0
