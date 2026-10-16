#!/usr/bin/env python3
"""BrainMirror P5: run a REAL IXI FastSurfer GPU segmentation directly in Kaggle.

This is a segmentation milestone, NOT a cortical pial surface reconstruction.
Uses pinned official FastSurfer v2.5.4, isolated Python 3.12 environment,
original TRAIN-only IXI MRI, and GitHub PRIVATE Release SHA256 verification.
No patient MRI, split manifest or FreeSurfer license is uploaded.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

WORK = Path("/kaggle/working")
T1 = WORK / "brainmirror_p5_native_case01/PRIVATE_ixi_case01_native_T1.nii.gz"
SPLIT = WORK / "brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv"
EXPECTED_SPLIT = "00f3ec7d490bff06f529640ccfed4bbae02f172c2f92f4e851e756cc1cf4e40a"
FS = WORK / "FastSurfer-v2.5.4"
OUT = WORK / "brainmirror_p5_segmentation_run1"
SUBJECT = "brainmirror_case01"
SEG = OUT / SUBJECT / "mri/aparc.DKTatlas+aseg.deep.mgz"
REPO = "changxinjiresearch/BrainMirror-Data-Private"
TAG = "ixi-p5-segmentation-case01-v1"
CODE_REV = "c478b4538f6a64d77b43f96f0f16431b614a3031"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for c in iter(lambda: fp.read(4*1024*1024),b""):h.update(c)
    return h.hexdigest()

def run(args, **kw):
    print("+", " ".join(str(x) for x in args[:8]), "..." if len(args)>8 else "", flush=True)
    return subprocess.run([str(x) for x in args], check=True, **kw)

def fetch(name):
    f=WORK/name
    if not f.is_file():
        url=("https://raw.githubusercontent.com/changxinjiresearch/BrainMirror/"
             +CODE_REV+"/scripts/"+name)
        urllib.request.urlretrieve(url,f)
    return f

def stage_if_needed():
    if T1.is_file():
        if not SPLIT.is_file() or sha(SPLIT)!=EXPECTED_SPLIT:
            raise RuntimeError("Cannot verify original TRAIN-only split; refusing to use staged T1")
        return
    try:
        import nibabel  # image prep and hash
    except ImportError:
        run([sys.executable,"-m","pip","install","-q","nibabel"])
        import nibabel
    if not SPLIT.is_file():
        from kaggle_secrets import UserSecretsClient
        backup=fetch("ixi_github_backup.py")
        # That backup utility lives in another previous frozen revision:
        prev=("https://raw.githubusercontent.com/changxinjiresearch/BrainMirror/"
              "ddd904035c91bffad459db28ef15495789349dc6/scripts/ixi_github_backup.py")
        urllib.request.urlretrieve(prev,backup)
        env=os.environ.copy()
        env["BRAINMIRROR_BACKUP_TOKEN"]=UserSecretsClient().get_secret("BRAINMIRROR_BACKUP_TOKEN")
        run([sys.executable,backup,"restore","--root",WORK,"--repo",REPO,
             "--release-tag","ixi-t1-00f3ec7d490bff06-atlas-v1"],env=env)
    if sha(SPLIT)!=EXPECTED_SPLIT: raise RuntimeError("Manifest SHA256 mismatch")
    stage=fetch("ixi_prepare_native_case.py")
    run([sys.executable,stage,"--manifest",SPLIT,"--data-root","/kaggle/input",
         "--site","GUYS","--out",T1.parent])
    if not T1.is_file():raise RuntimeError("Missing original-resolution train T1")

def backup_segmentation(token):
    import importlib.util
    helper=fetch("p5_private_backup.py")
    spec=importlib.util.spec_from_file_location("bm_private_seg_backup",helper)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_repo(REPO,token)
    relbase="https://api.github.com/repos/"+REPO
    try:
        release=module.request_json(relbase+"/releases/tags/"+TAG,token)
    except __import__("urllib.error",fromlist=["HTTPError"]).HTTPError as e:
        if e.code!=404:raise
        release=module.request_json(relbase+"/releases",token,"POST",{
          "tag_name":TAG,"name":"BrainMirror PRIVATE real IXI FastSurfer segmentation",
          "draft":False,"prerelease":True,
          "body":"Single original IXI train case derived segmentation (PRIVATE). No raw T1/license. Pial cortical reconstruction NOT completed. Pending scientific QC."
        })
    rec=OUT/"PRIVATE_p5_seg_receipt.json"
    expected={"schema":"brainmirror.p5.segmentation.v1",
      "subject_alias":SUBJECT,"t1_sha256":sha(T1),"segmentation_sha256":sha(SEG),
      "status":"REAL_FASTSURFER_SEGMENTATION_UNREVIEWED",
      "pial_surface_reconstructed":False,"individual_data_private_only":True}
    if rec.exists():
        if json.loads(rec.read_text())!=expected:raise RuntimeError("Existing receipt mismatch")
    else:
        rec.write_text(json.dumps(expected,indent=2));rec.chmod(0o600)
    entries=[("PRIVATE_p5_aparc_DKTatlas_aseg_deep.mgz",SEG),("PRIVATE_p5_seg_receipt.json",rec)]
    remote={asset["name"]:asset for asset in release.get("assets",[])}
    for name,path in entries:
        if name in remote:
            module.verify_remote(remote[name],path)
        else:
            prefix=release["upload_url"].split("{",1)[0]
            target=prefix+("&" if "?" in prefix else "?")+"name="+name
            got=module.upload(target,token,path)
            module.verify_remote(got,path)
        print("GITHUB_PRIVATE_SHA256_VERIFIED:",name,flush=True)

def main():
    from kaggle_secrets import UserSecretsClient
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU not enabled. Enable GPU T4 in Notebook Settings and rerun this cell.")
    token=UserSecretsClient().get_secret("BRAINMIRROR_BACKUP_TOKEN")
    if not token:raise RuntimeError("Private GitHub backup token unavailable")
    stage_if_needed()
    receipt=json.loads((T1.parent/"PRIVATE_stage_receipt.json").read_text())
    if receipt.get("split_untouched") is not True or receipt.get("anatomy_reconstructed") is not False:
        raise RuntimeError("Staging provenance invalid")
    print("REAL IXI TRAIN MRI",receipt["full_native_resolution_shape"],"GPU",torch.cuda.get_device_name(0),flush=True)
    if not FS.is_dir():
        run(["git","clone","--depth","1","--branch","v2.5.4","https://github.com/Deep-MI/FastSurfer.git",FS])
    run([sys.executable,"-m","pip","install","-q","uv"])
    uv=shutil.which("uv") or str(Path(sys.executable).parent/"uv")
    envname=FS/".venv"
    if not (envname/"bin/python").exists():
        run([uv,"venv","--python","3.12",envname])
    py=envname/"bin/python"
    # Install requirements in separate env: do not damage Kaggle/PyTorch kernel.
    if not (OUT/"environment_ready.flag").exists():
        resolved=subprocess.run([uv,"pip","compile","--no-build","--torch-backend","auto","requirements.txt"],
                         cwd=FS,check=True,stdout=subprocess.PIPE,text=True).stdout
        run([uv,"pip","sync","--no-build","--torch-backend","auto","-"],
            cwd=FS,env={**os.environ,"VIRTUAL_ENV":str(envname)},input=resolved,text=True)
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"environment_ready.flag").write_text("FastSurfer v2.5.4 uv Python 3.12\n")
    env=os.environ.copy()
    env["VIRTUAL_ENV"]=str(envname)
    env["PATH"]=str(envname/"bin")+os.pathsep+env["PATH"]
    env["FASTSURFER_HOME"]=str(FS)
    env["PYTHONPATH"]=str(FS)+os.pathsep+env.get("PYTHONPATH","")
    # Download only if required; retries on rerun.
    run([py,FS/"FastSurferCNN/download_checkpoints.py","--all"],cwd=FS,env=env)
    OUT.mkdir(parents=True,exist_ok=True)
    if not SEG.is_file():
        command=["bash",FS/"run_fastsurfer.sh","--seg_only",
          "--sid",SUBJECT,"--sd",OUT,"--t1",T1,"--device","cuda"]
        print("START REAL FASTSURFER MRI SEGMENTATION",flush=True)
        with (OUT/"PRIVATE_seg_run.log").open("a") as logfile:
            p=subprocess.Popen([str(x) for x in command],cwd=FS,env=env,
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,bufsize=1)
            for line in p.stdout:
                print(line,end="",flush=True)
                logfile.write(line);logfile.flush()
            if p.wait()!=0:raise RuntimeError("FastSurfer segmentation failed; see preserved private log")
    if not SEG.is_file() or SEG.stat().st_size<100_000:
        raise RuntimeError("Real aseg/DKTatlas segmentation output missing")
    # Validate actual label content, not merely file existence.
    import nibabel as nib
    import numpy as np
    seg_img=nib.load(str(SEG))
    volume=np.asarray(seg_img.dataobj)
    ids=set(np.unique(volume.astype(np.int32)).tolist())
    if volume.ndim!=3 or min(volume.shape)<160 or len(ids)<15 or not ({17,53,10,49}&ids):
        raise RuntimeError("Output failed label content QA; do not treat as completed segmentation")
    print("REAL SEGMENTATION LABEL_QA:",len(ids),"label IDs",flush=True)
    print("REAL FASTSURFER SEGMENTATION FINISHED; backing up private derived labels.",flush=True)
    backup_segmentation(token)
    print("P5 SEGMENTATION COMPLETE AND PRIVATELY BACKED UP.",flush=True)
    print("Note: PIAL surface reconstruction is still pending licensed FreeSurfer.",flush=True)

if __name__=="__main__":main()
