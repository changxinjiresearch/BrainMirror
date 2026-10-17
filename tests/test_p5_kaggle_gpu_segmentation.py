"""Safety and syntax guards for true GPU segmentation of a single IXI TRAIN MRI.

A test is not a real run. Real Kaggle GPU work is user-session gated.
"""
import ast
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/p5_kaggle_segmentation.py'
CODE=SCRIPT.read_text()

def test_gpu_segmentation_script_parses():
    assert ast.parse(CODE)

def test_real_gpu_inference_and_no_claim_of_pial():
    assert '"--seg_only"' in CODE
    assert 'run_fastsurfer.sh' in CODE
    assert 'torch.cuda.is_available()' in CODE
    assert 'pial_surface_reconstructed":False' in CODE
    assert 'PIAL surface reconstruction is still pending' in CODE

def test_safe_private_release_and_reproducible_train_split():
    assert 'EXPECTED_SPLIT' in CODE
    assert 'ixi_train_affine_mean' not in CODE
    assert 'BrainMirror-Data-Private' in CODE
    assert 'module.check_repo(REPO,token)' in CODE
    assert 'module.verify_remote' in CODE
    assert '"PRIVATE_p5_seg_receipt.json"' in CODE
    assert '"PRIVATE_p5_aparc_DKTatlas_aseg_deep.mgz"' in CODE
    assert 'FreeSurfer license' not in CODE.split('def backup_segmentation')[1].split('def main')[0]


def test_restore_on_gpu_session_reset_uses_real_legacy_helper():
    assert 'backup=WORK/"ixi_github_backup.py"' in CODE
    assert 'backup=fetch("ixi_github_backup.py")' not in CODE
    assert 'ddd904035c91bffad459db28ef15495789349dc6' in CODE
