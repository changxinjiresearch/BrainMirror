#!/usr/bin/env python3
"""Prepare an *exploratory, affine-aligned* IXI brain-only population reference.

Run only after P1 QC and prior review. No test subjects and no raw scans are
written to public Git. Depends on SimpleITK and nibabel in a private runner.
Not a replacement for unbiased nonlinear atlas construction. Images must be
pre-reviewed with matching brain masks and valid provenance/permission.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping
import numpy as np

ALLOWED_SPLITS={'train'}  # avoid held-out data contaminating training template

def approved_rows(manifest: Path) -> list[dict[str,str]]:
    """Require explicit independent QC and no accidental test-set reads."""
    with manifest.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        required={'subject_id','image_path','mask_path','split','qc_approved'}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError('Expected subject_id,image_path,mask_path,split,qc_approved')
        rows=[];seen=set()
        for row in reader:
            subject=row['subject_id'].strip()
            if not subject or subject in seen:raise ValueError('Blank/duplicate subject')
            seen.add(subject)
            if row['split'].strip().lower() not in ALLOWED_SPLITS:continue
            if row['qc_approved'].strip().lower() not in ('true','yes','1'):
                raise ValueError('Train subject lacks independent QC approval')
            for key in ('image_path','mask_path'):
                path=Path(row[key]).expanduser()
                if not path.is_file():raise FileNotFoundError(f'{key} does not exist for {subject}')
            rows.append(row)
    if not rows:raise ValueError('No approved train subjects')
    return rows

def summarize_reference(images: list[np.ndarray], masks:list[np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Pure numerics: track coverage and within-mask mean/variance, voxelwise."""
    if len(images)!=len(masks) or not images:raise ValueError('Image/mask count mismatch')
    shape=images[0].shape
    totals=np.zeros(shape,np.float64);squares=np.zeros(shape,np.float64);counts=np.zeros(shape,np.int32)
    for im,mask in zip(images,masks):
        a=np.asarray(im,dtype=np.float64);valid=np.asarray(mask,bool)
        if a.shape!=shape or valid.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid spatial or intensity values')
        totals[valid]+=a[valid];squares[valid]+=np.square(a[valid]);counts[valid]+=1
    mean=np.divide(totals,counts,out=np.zeros_like(totals),where=counts>0)
    var=np.maximum(0,np.divide(squares,counts,out=np.zeros_like(totals),where=counts>0)-mean*mean)
    return mean.astype(np.float32),np.sqrt(var).astype(np.float32),counts

def hash_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def normalize_brain(data:np.ndarray,mask:np.ndarray)->np.ndarray:
    x=np.asarray(data,np.float32);m=np.asarray(mask,bool)
    values=x[m]
    if values.size<1000 or not np.isfinite(values).all():raise ValueError('Invalid foreground voxels')
    lo,hi=np.percentile(values,[1,99]);
    if hi<=lo:raise ValueError('No intensity range')
    return np.where(m,np.clip((x-lo)/(hi-lo),0,1),0).astype(np.float32)

def build(manifest:Path,reference_path:Path,reference_mask_path:Path,output_dir:Path,min_dice:float=.7):
    try:import SimpleITK as sitk
    except ImportError as exc:raise RuntimeError('Install SimpleITK in a private compute environment') from exc
    if output_dir.exists() and any(output_dir.iterdir()):raise FileExistsError('Refusing to overwrite prior results')
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=approved_rows(manifest)
    reference=sitk.ReadImage(str(reference_path),sitk.sitkFloat32)
    reference_mask=sitk.ReadImage(str(reference_mask_path),sitk.sitkUInt8)
    if reference.GetSize()!=reference_mask.GetSize():raise ValueError('Reference mask geometry mismatch')
    refmask=sitk.GetArrayFromImage(reference_mask)>0
    count=np.zeros(refmask.shape,np.int32);total=np.zeros(refmask.shape,np.float64);square=np.zeros(refmask.shape,np.float64);receipt=[]
    for k,row in enumerate(rows,1):
        name=row['subject_id'];im=sitk.ReadImage(row['image_path'],sitk.sitkFloat32);mask=sitk.ReadImage(row['mask_path'],sitk.sitkUInt8)
        if im.GetSize()!=mask.GetSize() or any(abs(a-b)>.001 for a,b in zip(im.GetSpacing(),mask.GetSpacing())):raise ValueError('Image/mask mismatch')
        init=sitk.CenteredTransformInitializer(reference,im,sitk.AffineTransform(3),sitk.CenteredTransformInitializerFilter.GEOMETRY)
        reg=sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
        reg.SetMetricSamplingStrategy(reg.RANDOM);reg.SetMetricSamplingPercentage(.04,seed=777)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0,numberOfIterations=125,convergenceMinimumValue=1e-6,convergenceWindowSize=10)
        reg.SetOptimizerScalesFromPhysicalShift();reg.SetShrinkFactorsPerLevel([4,2,1]);reg.SetSmoothingSigmasPerLevel([2,1,0]);reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init,inPlace=False)
        tx=reg.Execute(reference,im)
        moved=sitk.Resample(im,reference,tx,sitk.sitkLinear,0.,sitk.sitkFloat32)
        moved_mask=sitk.Resample(mask,reference_mask,tx,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        arr=sitk.GetArrayFromImage(moved);ma=sitk.GetArrayFromImage(moved_mask)>0
        intersection=np.logical_and(ma,refmask).sum();dice=2*intersection/max(1,int(ma.sum())+int(refmask.sum()))
        if dice<min_dice:raise ValueError(f'Registration mask Dice below predetermined threshold {min_dice}: {name} ({dice:.4f})')
        normalized=normalize_brain(arr,ma);total[ma]+=normalized[ma];square[ma]+=normalized[ma]**2;count[ma]+=1
        receipt.append({'subject':name,'image_sha256':hash_file(Path(row['image_path'])),'mask_sha256':hash_file(Path(row['mask_path'])),'registration_metric':float(reg.GetMetricValue()),'reference_mask_dice':float(dice)})
        print(f'[{k}/{len(rows)}] registered {name}; dice={dice:.4f}',flush=True)
    mean=np.divide(total,count,out=np.zeros_like(total),where=count>0)
    sd=np.sqrt(np.maximum(0,np.divide(square,count,out=np.zeros_like(total),where=count>0)-mean**2))
    for filename,arr in [('train_affine_mean.nii.gz',mean),('train_affine_sd.nii.gz',sd),('train_coverage.nii.gz',count)]:
        out=sitk.GetImageFromArray(arr.astype(np.float32));out.CopyInformation(reference);sitk.WriteImage(out,str(output_dir/filename))
    result={'schema':'brainmirror.ixi-affine-reference.v1','status':'RESEARCH_CANDIDATE_NOT_QC_APPROVED','subject_count':len(rows),'source':'IXI T1','split':'train_only','min_reference_mask_dice':min_dice,'alignment':'affine_registration_NOT_unbiased_nonlinear_template','reference_sha256':hash_file(reference_path),'reference_mask_sha256':hash_file(reference_mask_path),'manifest_sha256':hash_file(manifest),'files':receipt}
    (output_dir/'PRIVATE_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Averaging completed. Still requires independent visual QC and license/provenance review.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--approved-manifest',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-mask',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--min-dice',type=float,default=.7);args=p.parse_args()
    if not (.1<=args.min_dice<=1):p.error('Invalid min-dice')
    build(args.approved_manifest,args.reference,args.reference_mask,args.output_dir,args.min_dice)
if __name__=='__main__':main()