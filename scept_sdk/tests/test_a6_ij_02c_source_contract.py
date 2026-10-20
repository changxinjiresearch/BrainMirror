"""A6-IJ-02C offline source + deterministic synthetic-math QA. No MRI data."""
import ast
from pathlib import Path
import numpy as np

EX=Path(__file__).resolve().parents[1]/"examples"
SCRIPTS=("a6_ij_02c_measure.py","a6_ij_02c_analyse.py",
         "a6_ij_02c_archive_private.py")

def test_02c_source_parses_on_python_310():
    for name in SCRIPTS:
        ast.parse((EX/name).read_text(),filename=name,feature_version=(3,10))

def test_02c_locked_replay_controls_and_privacy_are_explicit():
    m=(EX/SCRIPTS[0]).read_text()
    a=(EX/SCRIPTS[1]).read_text()
    z=(EX/SCRIPTS[2]).read_text()
    assert '"patientSelection":"all 55 lexicographic P0 validation patients"' in m
    assert '"permutationCount":64' in m
    assert '"independentHookReplayMaxAbs":1e-6' in m
    assert '"02BPostCKAMaxAbs":1e-5' in m
    assert 'independent_replay' in m
    assert 'torch.zeros((1,3,224,224)' in m
    assert 'include_inputs=False' in m
    assert 'F.layer_norm(xt,module.normalized_shape,None,None,module.eps)' in m
    assert 'len(val)!=55' in a
    assert '2000' in a and '240216' in a
    assert '02BPostCKAMaxAbs' in a
    assert 'originalMRIIncluded":False' in z
    assert 'range(0,55,5)' in z
    assert 'asset_readback(s,asset,item["sha256"],item["bytes"])' in z
    assert 'repo.get("private") is not True' in z

def _get_fns(fname,wanted,namespace):
    parsed=ast.parse((EX/fname).read_text())
    selected=[node for node in parsed.body if isinstance(node,ast.FunctionDef)
              and node.name in wanted]
    assert len(selected)==len(wanted)
    exec(compile(ast.Module(body=selected,type_ignores=[]),fname,"exec"),namespace)

def test_02c_synthetic_ckas_and_patch_shuffling_null_are_deterministic():
    ns={"np":np, "SEED":240216,"PLAN":{"permutationCount":64}}
    _get_fns("a6_ij_02b_measure.py",{"gram"},ns)
    _get_fns("a6_ij_02c_measure.py",
             {"as_patch_gram","aligned_cka","alignment_null"},ns)
    rng=np.random.default_rng(340216)
    X=rng.normal(size=(196,768))
    Y=X[:,::-1].copy()  # orthogonal feature permutation preserves patch Gram
    value,A,B,denom=ns["aligned_cka"](X,Y)
    assert abs(value-1.0)<1e-9
    baseline=ns["alignment_null"](A,B,denom,patient_index=7,block=11,stage="post")
    duplicate=ns["alignment_null"](A,B,denom,patient_index=7,block=11,stage="post")
    assert baseline==duplicate
    assert baseline["count"]==64
    assert 0<=baseline["mean"]<value
    assert baseline["q95"]<value
    assert ns["alignment_null"](A,B,denom,7,11,"pre")["seed"]!=baseline["seed"]
