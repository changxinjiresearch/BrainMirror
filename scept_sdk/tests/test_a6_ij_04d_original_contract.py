"""04D source-gated 33-real-patient immutable scientific study contracts.

Syntax and invariant-gate tests only. Genuine medical source and S/J ViT are
kept PRIVATE and cannot be simulated as real results in public CI.
"""
import ast
from pathlib import Path
EX=Path(__file__).resolve().parents[1]/"examples"
FILES=(
 "a6_ij_04d_hook.py",
 "a6_ij_04d_measure.py",
 "a6_ij_04d_archive_private.py",
 "a6_ij_04d_finalize_private.py",
 "a6_ij_04d_run.py"
)
def read(file):return (EX/file).read_text()
def test_python_310_and_all_scientific_source_contracts():
 for file in FILES:
  ast.parse(read(file),filename=file,feature_version=(3,10))
 check={
  "a6_ij_04d_hook.py":(
   '"totalActualModelForwardPasses":1914',
   '"forwardPassesPerModelPatient":29',
   'layer_module(model,layer)',
   'mod.register_forward_hook(replace)',
   'cloned[:,idx,:]=src[:,idx,:]',
   'torch.equal(cloned[:,untouched,:],actual[:,untouched,:])',
   'handle.remove()',
   'torch.inference_mode()',
   '"INDETERMINATE_ORIGINAL_OUTPUT_CONTRAST_BELOW_FIXED_GATE"'),
  "a6_ij_04d_measure.py":(
   'original04b_load(s)',
   'status")!="PASS_A6_IJ_04C_33_PATIENT_594_GENUINE_ORIGINAL_FULL_TENSOR_OBSERVATIONAL_AUDIT"',
   'check_mask_selection' # Must NOT occur: old 04A schema is not old 04B case
  ),
 }
 hook=read("a6_ij_04d_hook.py")
 for x in check["a6_ij_04d_hook.py"]:assert x in hook,x
 measure=read("a6_ij_04d_measure.py")
 assert 'check_mask_selection' not in measure
 for x in ('original04b_load(s)','original["patientFixedROIs"]',
           'matched_inputs(image,wt,fg,selection)',
           'for direction in DIRECTIONS:',
           'for selector in SELECTORS:',
           'for arm in ARMS:',
           'for layer in LAYERS:',
           'if selector=="all":',
           'assert_logits(sham,baseline["clean"]["logits"]',
           'case["realForwardCount"]!=58',
           'save_npz(npz,donated)',
           'state_digest(m)!=weight_hashes[k]',
           '"sealedP5":"NOT_ACCESSED"'):
  assert x in measure,x
 archive=read("a6_ij_04d_archive_private.py")
 for x in ('remote_sha(s,asset,checksum,size)',
    'prior_receipt(s,args.chunk,rows)',
    'seal(members,assetpath)',
    'len(a["bidirectionalPostBlockInterventions"])!=12',
    'case.get("realForwardCount")!=58',
    'nGenuineDirectGraphForwards',
    'commit_once(s,prefix+"/archive_receipt.json"'):
  assert x in archive,x
 final=read("a6_ij_04d_finalize_private.py")
 for x in (
   'for chunk in range(1,8)',
   'prior_receipt(s,chunk,cases)',
   'len(rows)!=33',
   'sum(q["realForwardCount"] for q in rows)!=1914',
   'nOriginalPatientFractionIndeterminate',
   '"nTrueFrozenModelPyTorchForwards":1914',
   'RESAMPLES=2000',
   'commit_once(s,f"{BASE}/final/archive_receipt.json"'
 ):
  assert x in final,x
 run=read("a6_ij_04d_run.py")
 for x in ('--pilot','--chunk','--all','prior_receipt(s,i,rows)',
           '"a6_ij_04d_archive_private.py"','"a6_ij_04d_finalize_private.py"'):
  assert x in run,x
