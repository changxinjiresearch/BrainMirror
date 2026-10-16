"""One-case official BraTS21 SwinUNETR -> real BraTS2020 SCEPT diagnostic. NOT independent evaluation."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
from brainmirror.scept.brats2020 import discover_training_cases, audited_inventory, label_regions, dice_regions, region_reference
from brainmirror.scept.monai_adapter import build_monai_model, MonaiSwinObserver, AttentionAtom, atomic_bidirectional_effect
from scripts.scept_brats_v09 import read_split, save_json

URL="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip"
MODALITIES=("flair","t1ce","t1","t2") # official channel order
PREDICTED=("TC","WT","ET") # official output order

def normalize(v):
    v=np.asarray(v,np.float32); mask=v!=0; vals=v[mask]
    if vals.size<64 or vals.std()<1e-6: raise ValueError("Bad MRI modality")
    y=np.zeros_like(v); y[mask]=(vals-vals.mean())/vals.std()
    return y

def load_subject(case):
    import nibabel as nib
    arrays=[]; shape=None; affine=None
    for m in (*MODALITIES,"seg"):
        nii=nib.load(str(case.paths[m])); x=np.asarray(nii.dataobj,dtype=np.float32)
        if x.ndim!=3 or not np.isfinite(x).all():raise ValueError("Invalid MRI")
        if shape is not None and (x.shape!=shape or not np.allclose(affine,nii.affine,atol=1e-4,rtol=0)):
            raise ValueError("MRI spatial mismatch")
        shape,affine=x.shape,nii.affine
        arrays.append(x if m=="seg" else normalize(x))
    return np.stack(arrays[:4]),label_regions(arrays[4])

def download(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    weights=dest/"official_fold1_model.pt"
    if not weights.exists():
        zpath=dest/"official_fold1.zip"
        if not zpath.exists():
            print("Downloading official MONAI Fold 1 checkpoint...",flush=True)
            urllib.request.urlretrieve(URL,zpath)
        with zipfile.ZipFile(zpath) as z:
            names=[n for n in z.namelist() if Path(n).name=="model.pt"]
            if len(names)!=1:raise ValueError("Unexpected official archive")
            with z.open(names[0]) as f,weights.open("wb") as out:shutil.copyfileobj(f,out)
    h=hashlib.sha256()
    with weights.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    if weights.stat().st_size<20_000_000:raise ValueError("Checkpoint too small")
    return weights,h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--split",required=True)
    p.add_argument("--output",required=True);args=p.parse_args()
    import monai
    if monai.__version__.split("+")[0]!="1.5.1":raise ValueError("Need MONAI 1.5.1")
    if not torch.cuda.is_available():raise RuntimeError("GPU required")
    began=time.perf_counter()
    weights,digest=download("/kaggle/working/scept_official_model")
    print("Checkpoint SHA256:",digest,flush=True)
    model=build_monai_model(in_channels=4,out_channels=3,feature_size=48,
            depths=(2,2,2,2),num_heads=(3,6,12,24),window_size=7,
            norm_name="instance",use_checkpoint=False)
    obj=torch.load(weights,map_location="cpu",weights_only=True)
    state=obj["state_dict"]
    if all(k.startswith("module.") for k in state):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True) # MUST NOT silently use random weights
    model.cuda().eval()
    print("OFFICIAL_PRETRAINED_WEIGHTS_STRICT_PASS",flush=True)
    cases=discover_training_cases(Path(args.root))
    inventory=audited_inventory(cases,root=Path(args.root),expected_count=369)
    splits=read_split(args.split,[c.case_id for c in cases],inventory["case_manifest_sha256"])
    chosen=sorted(splits["validation"])[0]
    subject=next(c for c in cases if c.case_id==chosen)
    images,regions=load_subject(subject)
    tumour=np.argwhere(regions[0]>.5)
    start=np.clip(np.rint(tumour.mean(0)).astype(int)-32,0,np.array(regions.shape[1:])-64)
    sl=tuple(slice(int(a),int(a)+64) for a in start)
    patch=np.ascontiguousarray(images[(slice(None),)+sl])
    gt=np.ascontiguousarray(regions[(slice(None),)+sl])
    del images,regions
    x=torch.from_numpy(patch[None]).cuda()
    region=torch.from_numpy(gt[0]>.5).cuda()
    with torch.no_grad():
        logits=model(x)
        official=torch.sigmoid(logits)[0].cpu().numpy()
    pred=official[[1,0,2]] # to SCEPT WT/TC/ET order
    score=dice_regions(pred,gt)
    print("REAL_PATCH_PREDICTION_PASS",json.dumps({k:v["dice"] for k,v in score.items()}),flush=True)
    ref,qc=region_reference(x,region,policy="tissue_median")
    observer=MonaiSwinObserver(model)
    site=list(observer.sites)[-1] # smallest/deepest tensors for fast diagnostic
    atom=AttentionAtom(site,0,0,0,0,"value_message")
    effect=atomic_bidirectional_effect(model,x,ref,atom,class_index=1,roi=region)
    output={"schema":"scept/official-brats21-fast-diagnostic/v1",
            "case_id":chosen,"weights_sha256":digest,"weights_url":URL,
            "official_modality_order":MODALITIES,"official_output_order":PREDICTED,
            "test_independence":"NOT_INDEPENDENT_BraTS2021_BraTS2020_OVERLAP_UNKNOWN",
            "selection":"ground_truth_guided_64cube_from_frozen_BraTS2020_validation",
            "patch_dice_NOT_full_volume":score,"reference_qc":qc,
            "scept_single_atom_intervention":effect,
            "seconds":round(time.perf_counter()-began,2)}
    save_json(args.output,output)
    print("REAL_SCEPT_DIAGNOSTIC_COMPLETE",json.dumps({"case_id":chosen,
           "effect":effect["disruption_effect"],"restore":effect["restoration_effect"],
           "seconds":output["seconds"],"output":args.output}),flush=True)

if __name__=="__main__":main()
