#!/usr/bin/env python3
"""BrainMirror: five-case IXI-T1 registration pilot (PRIVATE data only).

This is a feasibility candidate, NOT an approved 581-person atlas.
Requires Kaggle-mounted IXI T1, SynthStrip script/weights acquired in previous
notebook cell, and CPU Python dependencies nibabel/SimpleITK/surfa/matplotlib.
Never commit brain images, masks, preview PNGs or individual file IDs to public Git.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re
import shutil
import subprocess
import sys

SITE = re.compile(r"^IXI\d{3}-([A-Za-z]+)-", flags=re.I)
ORDER = (("Guys", 2), ("HH", 2), ("IOP", 1))

def select_five(dataset_root: Path) -> list[Path]:
    files = sorted(
        p for p in Path(dataset_root).rglob("*")
        if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
    )
    # IXI all-T1 source received by researcher has 581 files. Don't confuse
    # Kaggle-extracted .nii-named directories with actual scans.
    if len(files) != 581:
        raise ValueError(f"Expected 581 actual IXI-T1 files; got {len(files)}")
    sites = defaultdict(list)
    for p in files:
        match = SITE.match(p.name)
        if not match:
            raise ValueError("Unrecognized IXI filename")
        sites[match.group(1).lower()].append(p)
    selected = []
    for name, n in ORDER:
        pool = sites[name.lower()]
        if len(pool) < n:
            raise ValueError(f"Not enough valid {name} scans")
        selected.extend(pool[:n])
    if len(selected) != 5 or len(set(selected)) != 5:
        raise AssertionError("Pilot selection has duplicated inputs")
    return selected

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def make_reference(image, mm, sitk):
    spacing = image.GetSpacing()
    size = image.GetSize()
    target_size = [max(8, int(round((s-1)*sp/mm))+1) for s,sp in zip(size,spacing)]
    return sitk.Resample(
        image, target_size, sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear, image.GetOrigin(), [mm]*3,
        image.GetDirection(), 0.0, sitk.sitkFloat32
    )

def normalize_brain(data, brain_mask, np):
    x=np.asarray(data,dtype=np.float32)
    m=np.asarray(brain_mask,dtype=bool)
    fg=x[m]
    if fg.size < 10000 or not np.isfinite(fg).all():
        raise ValueError("Insufficient or nonfinite brain voxels")
    lo,hi=np.percentile(fg,[1,99])
    if not np.isfinite([lo,hi]).all() or hi<=lo:
        raise ValueError("Invalid MRI intensity range")
    return np.where(m,np.clip((x-lo)/(hi-lo),0,1),0).astype(np.float32)

def register(moving, fixed, sitk):
    init=sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg=sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.025,seed=240215)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.5,minStep=0.0005,
        numberOfIterations=120,relaxationFactor=0.5
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4,2,1])
    reg.SetSmoothingSigmasPerLevel([2,1,0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init,inPlace=False)
    tx=reg.Execute(fixed,moving)
    if not __import__("numpy").isfinite(reg.GetMetricValue()):
        raise ValueError("Nonfinite registration metric")
    return tx,float(reg.GetMetricValue())

def run(args):
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    data_mount=Path(args.data_root)
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output exists; use new directory, do not overwrite: {out}")
    if out.is_relative_to(data_mount.resolve()):
        raise ValueError("Output cannot be placed in input dataset")
    tool_dir=Path(args.synthstrip_dir)
    script=tool_dir/"mri_synthstrip"
    weights=tool_dir/"synthstrip.1.pt"
    if not script.is_file() or not weights.is_file() or weights.stat().st_size < 20_000_000:
        raise FileNotFoundError("Missing prior SynthStrip script/weights in --synthstrip-dir")
    selected=select_five(data_mount)
    out.mkdir(parents=True,exist_ok=True)
    out.chmod(0o700)
    masks=[];brains=[];private=[]
    for i,source in enumerate(selected,1):
        destination=out/f"case{i:02d}"
        destination.mkdir()
        brain=destination/"brain.nii.gz"
        mask=destination/"mask.nii.gz"
        print(f"[{i}/5] Extracting brain: source site={SITE.match(source.name).group(1)}",flush=True)
        proc=subprocess.run(
            [sys.executable,str(script),"-i",str(source),
             "-o",str(brain),"-m",str(mask),
             "--model",str(weights),"-t","2"],
            check=False,capture_output=True,text=True
        )
        if proc.returncode:
            raise RuntimeError(f"SynthStrip failed for case {i}: {proc.stderr[-1200:]}")
        orig=nib.load(str(source))
        mask_img=nib.load(str(mask))
        if orig.shape!=mask_img.shape or not np.allclose(orig.affine,mask_img.affine,atol=1e-3,rtol=0):
            raise ValueError(f"SynthStrip mask geometry mismatch case {i}")
        if not (np.asarray(mask_img.dataobj)>0).sum()>10000:
            raise ValueError(f"Brain mask too small case {i}")
        brains.append(brain);masks.append(mask)
        private.append({
            "case":i,"source":str(source),
            "source_sha256":sha256(source),
            "site":SITE.match(source.name).group(1),
            "brain_sha256":sha256(brain),
            "mask_sha256":sha256(mask)
        })

    ref_orig=sitk.ReadImage(str(brains[0]),sitk.sitkFloat32)
    reference=make_reference(ref_orig,2.0,sitk)
    ref_mask_orig=sitk.ReadImage(str(masks[0]),sitk.sitkUInt8)
    ref_mask=sitk.Resample(
        ref_mask_orig,reference,sitk.Transform(3,sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,0,sitk.sitkUInt8
    )
    ref_arr=sitk.GetArrayFromImage(reference)
    ref_support=sitk.GetArrayFromImage(ref_mask)>0
    if ref_support.sum()<10000: raise ValueError("Reference brain mask empty after resampling")

    totals=np.zeros_like(ref_arr,dtype=np.float64)
    squares=np.zeros_like(ref_arr,dtype=np.float64)
    count=np.zeros_like(ref_arr,dtype=np.int16)
    qc=[]
    for i,(brain,mask) in enumerate(zip(brains,masks),1):
        print(f"[{i}/5] Aligning real T1 brain to reference...",flush=True)
        if i==1:
            transformed=sitk.Transform(3,sitk.sitkIdentity)
            metric=None
        else:
            im=sitk.ReadImage(str(brain),sitk.sitkFloat32)
            transformed,metric=register(im,reference,sitk)
        moved=sitk.Resample(
            sitk.ReadImage(str(brain),sitk.sitkFloat32),reference,
            transformed,sitk.sitkLinear,0.0,sitk.sitkFloat32
        )
        moved_mask=sitk.Resample(
            sitk.ReadImage(str(mask),sitk.sitkUInt8),reference,
            transformed,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8
        )
        image=sitk.GetArrayFromImage(moved)
        support=sitk.GetArrayFromImage(moved_mask)>0
        intersect=np.count_nonzero(support & ref_support)
        dice=2*intersect/max(1,int(support.sum()+ref_support.sum()))
        print(f"  overlap brain-mask Dice={dice:.4f}",flush=True)
        qc.append({"case":i,"site":private[i-1]["site"],"reference_mask_dice":float(dice),"registration_metric":metric})
        if dice<args.min_dice:
            (out/"PRIVATE_registration_review.json").write_text(json.dumps(qc,indent=2))
            raise ValueError(f"Case {i}: Dice {dice:.3f} below threshold {args.min_dice}; manually review. Candidate NOT accepted.")
        norm=normalize_brain(image,support,np)
        totals[support]+=norm[support]
        squares[support]+=np.square(norm[support])
        count[support]+=1
    mean=np.divide(totals,count,out=np.zeros_like(totals),where=count>0)
    var=np.maximum(0,np.divide(squares,count,out=np.zeros_like(totals),where=count>0)-mean**2)
    sd=np.sqrt(var)
    for filename,vol in (
        ("pilot5_affine_mean.nii.gz",mean),
        ("pilot5_affine_sd.nii.gz",sd),
        ("pilot5_coverage.nii.gz",count.astype(np.float32))
    ):
        img=sitk.GetImageFromArray(vol.astype(np.float32))
        img.CopyInformation(reference)
        sitk.WriteImage(img,str(out/filename))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    views=[mean[mean.shape[0]//2,:,:],mean[:,mean.shape[1]//2,:],mean[:,:,mean.shape[2]//2]]
    for ax,title,v in zip(axes,("Axial","Coronal","Sagittal"),views):
        ax.imshow(np.rot90(v),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        ax.set_title("5-case affine mean: "+title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out/"PRIVATE_pilot5_3views.png",dpi=150)
    plt.close(fig)
    receipt={
        "schema":"brainmirror.ixi.pilot5.v1","status":"EXPLORATORY_CANDIDATE_NOT_APPROVED",
        "source":"IXI T1","subject_count":5,
        "sampling_sites":{"Guys":2,"HH":2,"IOP":1},
        "brain_extractor":"SynthStrip (prior notebook tool and weights)",
        "brain_extractor_script_sha256":sha256(script),
        "brain_extractor_weights_sha256":sha256(weights),
        "alignment":"affine to first pilot subject; 2mm grid",
        "independent_brain_mask_visual_review":False,
        "data_split_locked":False,"P1_PASS":False,
        "full_581_template_built":False,
        "overlap_dice":qc,
        "input_records_PRIVATE":private,
        "output_sha256":{p.name:sha256(p) for p in out.glob("pilot5_*.nii.gz")},
    }
    (out/"PRIVATE_receipt.json").write_text(json.dumps(receipt,indent=2))
    (out/"PRIVATE_receipt.json").chmod(0o600)
    print("PILOT5 CANDIDATE COMPLETE",flush=True)
    print("Outputs:",out,flush=True)
    print("Review PRIVATE_pilot5_3views.png and per-case Dice before continuing.",flush=True)
    return 0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default="/kaggle/input")
    p.add_argument("--synthstrip-dir",default="/kaggle/working/brainmirror_ixi_pilot")
    p.add_argument("--out",default="/kaggle/working/brainmirror_ixi_pilot5_v1")
    p.add_argument("--min-dice",type=float,default=0.65)
    args=p.parse_args()
    if not 0.5<=args.min_dice<=0.95:p.error("min-dice out of range")
    raise SystemExit(run(args))
if __name__=="__main__":main()
