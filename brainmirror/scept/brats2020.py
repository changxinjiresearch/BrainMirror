"""BraTS2020 v0.9 loader and audit. MRI stays in execution environment.

The official BraTS2020 training labels are 0=background, 1=NCR/NET,
2=edema, 4=enhancing tumor. NEVER re-map 4 to 3 before region conversion.
The three output heads use WT=(1|2|4), TC=(1|4), ET=(4).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ('t1','t1ce','t2','flair')
LABELS = (0,1,2,4)
REGIONS = ('WT','TC','ET')
CASE_ID = re.compile(r'^BraTS20_Training_\d{3}$')


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    paths: dict[str, Path]
    grade: str = 'unknown'


def discover_training_cases(root: Path, *, strict=True) -> list[CaseFiles]:
    """Read local, authorised official-training NIfTI files; never fetch them.

    The original BraTS zip sometimes contains nested training directories.
    Patient ID is inferred from the containing directory, not the filename.
    If a case has mismatched/ambiguous files it is rejected, not guessed.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'No BraTS2020 training directory: {root}')
    folders = sorted(p for p in root.rglob('BraTS20_Training_*') if p.is_dir() and CASE_ID.fullmatch(p.name))
    cases=[]
    used=set()
    for d in folders:
        if d.name in used:
            raise ValueError(f'Duplicate case directory ID {d.name}')
        used.add(d.name)
        files={}
        for suffix in (*MODALITIES,'seg'):
            options = [d / f'{d.name}_{suffix}{ext}' for ext in ('.nii.gz','.nii')]
            # BraTS2020 case 355 has nonstandard mask filenames across mirrors.
            # Accept only the confirmed case-355 alias, never broad filename guesses.
            if suffix=='seg' and d.name=='BraTS20_Training_355':
                options += [d / f'{d.name}_segm{ext}' for ext in ('.nii.gz','.nii')]
                options.append(d / 'W39_1998.09.19_Segm.nii')
            found=[p for p in options if p.is_file()]
            if len(found)!=1:
                if strict:
                    raise ValueError(f'Missing or ambiguous {suffix} for {d.name}: found {len(found)}')
                files={}
                break
            files[suffix]=found[0]
        if files:
            relative=d.relative_to(root)
            grade=next((part for part in relative.parts if part in ('HGG','LGG')), 'unknown')
            cases.append(CaseFiles(d.name,files,grade))
    if not cases:
        raise ValueError('No complete labelled BraTS2020 training cases found')
    return cases


def audited_inventory(cases: Sequence[CaseFiles], *, root: Path, expected_count: int|None=None) -> dict:
    root=Path(root).resolve()
    cases=sorted(cases,key=lambda c:c.case_id)
    if len({c.case_id for c in cases})!=len(cases):
        raise ValueError('Duplicate patients')
    if expected_count is not None and len(cases)!=expected_count:
        raise ValueError(f'Case count mismatch: expected {expected_count}, observed {len(cases)}')
    rows=[]
    for c in cases:
        row={'case_id':c.case_id,'grade':c.grade,'files':{}}
        for m,p in c.paths.items():
            rp=Path(p).resolve()
            if not rp.is_relative_to(root):
                raise ValueError('MRI file outside configured dataset root')
            row['files'][m]={'relative_path':rp.relative_to(root).as_posix(),'bytes':rp.stat().st_size}
        rows.append(row)
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return {'schema':'scept/brats2020-inventory/v1','source':'user-provided-local-Brats2020-training',
            'case_count':len(rows),'modalities':list(MODALITIES), 'case_manifest_sha256':digest,
            'grade_counts':dict(Counter(c.grade for c in cases)),'cases':rows,
            'data_access_note':'Dataset bytes are not embedded or uploaded'}


def patient_split(case_ids:Sequence[str], *, seed=240216, fractions=(.70,.15,.15)) -> dict[str,list[str]]:
    """Deterministic patient-level split; disjoint even with case folder reordering.

    Sorting by seeded subject digest yields reproducibility; all volumes/modalities
    of each subject are assigned jointly. No optimising on held-out test IDs.
    """
    ids=list(case_ids)
    if len(set(ids))!=len(ids) or not ids:
        raise ValueError('Case IDs must be nonempty and unique')
    if len(fractions)!=3 or not np.isclose(sum(fractions),1) or min(fractions)<=0:
        raise ValueError('Provide three positive split fractions summing to 1')
    ids.sort(key=lambda s:hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    n=len(ids); a=int(n*fractions[0]); b=int(n*fractions[1])
    groups={'train':sorted(ids[:a]), 'validation':sorted(ids[a:a+b]), 'test':sorted(ids[a+b:])}
    if len(set(groups['train'])|set(groups['validation'])|set(groups['test']))!=n:
        raise AssertionError('Split integrity error')
    return groups


def label_regions(labels:np.ndarray) -> np.ndarray:
    labels=np.asarray(labels)
    if not np.isfinite(labels).all() or not np.isin(labels,LABELS).all():
        raise ValueError(f'Unexpected BraTS2020 labels; expected {LABELS}')
    wt=np.isin(labels,(1,2,4));tc=np.isin(labels,(1,4));et=labels==4
    return np.stack((wt,tc,et),axis=0).astype(np.float32)


def normalized_modalities(volumes:np.ndarray, *, clip=5.0) -> tuple[np.ndarray,dict]:
    """Per-modality brain-only z-scoring, no anatomy-altering interpolation.

    Input [4,D,H,W] already registered by challenge, and background is zero.
    The returned mask is from nonzero voxels across the four sequences.
    """
    v=np.asarray(volumes,dtype=np.float32)
    if v.ndim!=4 or v.shape[0]!=4 or not np.isfinite(v).all():
        raise ValueError('Expected finite [4,D,H,W] MRI data')
    brain=np.any(v!=0,axis=0)
    if int(brain.sum())<64:
        raise ValueError('Brain foreground mask has too few voxels')
    out=np.zeros_like(v)
    qcs=[]
    for i in range(4):
        tissue=v[i][brain & (v[i]!=0)]
        if tissue.size<64:
            raise ValueError('Missing foreground MRI signal for modality '+MODALITIES[i])
        mean=float(tissue.mean());std=float(tissue.std())
        if std<1e-6:raise ValueError('Degenerate MRI intensity distribution')
        out[i][brain]=np.clip((v[i][brain]-mean)/std,-clip,clip)
        qcs.append({'modality':MODALITIES[i],'mean':mean,'std':std,'clip':clip,
                    'clipped_fraction':float(np.mean(np.abs((v[i][brain]-mean)/std)>clip))})
    return out,{'brain_voxels':int(brain.sum()),'modalities':qcs}


def load_case(case:CaseFiles) -> tuple[np.ndarray,np.ndarray,dict]:
    """Load original NIfTI with affine congruence/shape checks; nibabel required."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError('nibabel>=5 required for real NIfTI MRI') from exc
    images=[];affines=[]
    for suffix in (*MODALITIES,'seg'):
        img=nib.load(str(case.paths[suffix]))
        arr=np.asarray(img.dataobj)
        if arr.ndim!=3 or not np.isfinite(arr).all():
            raise ValueError(f'{case.case_id}/{suffix}: invalid NIfTI volume')
        images.append(arr)
        affines.append(np.asarray(img.affine))
    if len({tuple(x.shape) for x in images})!=1:
        raise ValueError('MRI/seg shape mismatch')
    if any(not np.allclose(affines[0],a,rtol=0,atol=1e-4) for a in affines[1:]):
        raise ValueError('MRI/seg affine mismatch; do not silently reorient')
    seg=images[-1]
    regions=label_regions(seg)
    stacked=np.stack(images[:4],axis=0)
    norm,qc=normalized_modalities(stacked)
    qc.update({'shape':list(seg.shape),'affine_sha256':hashlib.sha256(affines[0].tobytes()).hexdigest(),
               'roi_voxels':{REGIONS[i]:int(regions[i].sum()) for i in range(3)}})
    return norm,regions,qc


def dice_regions(pred:np.ndarray,target:np.ndarray,*,threshold=.5) -> dict:
    """Empty GT + empty prediction is undefined (None), never fake perfect Dice."""
    p=np.asarray(pred);t=np.asarray(target)
    if p.shape!=t.shape or p.ndim!=4 or p.shape[0]!=3:
        raise ValueError('Expected aligned [3,D,H,W] probabilities and masks')
    out={}
    for i,name in enumerate(REGIONS):
        pp=p[i]>=threshold;tt=t[i]>.5
        if not tt.any() and not pp.any():d=None
        elif not tt.any() or not pp.any():d=0.0
        else:d=float(2*(pp&tt).sum()/(int(pp.sum())+int(tt.sum())))
        out[name]={'dice':d,'pred_voxels':int(pp.sum()),'gt_voxels':int(tt.sum())}
    return out


def region_reference(x:torch.Tensor,roi:torch.Tensor,*,policy:str='tissue_median',kernel=5):
    """Return a *diagnostic*, not necessarily in-distribution reference.

    Input [1,4,D,H,W]; ROI [D,H,W]. Exactly zero changes outside ROI.
    tissue_median uses non-target foreground within the same volume.
    local_blur averages a larger context, but still changes ONLY ROI voxels.
    """
    if x.ndim!=5 or x.shape[0]!=1 or x.shape[1]!=4 or roi.shape!=x.shape[-3:]:
        raise ValueError('Expected x [1,4,D,H,W] and matching 3D ROI')
    if not torch.isfinite(x).all() or roi.dtype!=torch.bool or not roi.any():
        raise ValueError('Non-finite volume or empty/nonboolean ROI')
    brain=x[0].abs().sum(dim=0)>0
    donor=brain&~roi
    if int(donor.sum())<64:raise ValueError('Insufficient non-ROI brain tissue')
    out=x.clone();m=roi[None,None]
    if policy=='tissue_median':
        for c in range(4):
            med=x[0,c][donor].median()
            out[:,c]=torch.where(m[0],med,x[:,c])
    elif policy=='local_blur':
        if kernel<3 or kernel%2!=1:raise ValueError('Kernel must be odd and >=3')
        smoothed=F.avg_pool3d(x,kernel_size=kernel,stride=1,padding=kernel//2)
        out=torch.where(m,smoothed,x)
    else:raise ValueError('Unknown MRI reference policy')
    changed=(out-x).abs()
    if changed[...,~roi].abs().max().item()!=0.:
        raise AssertionError('Reference changed off-target pixels')
    return out,{'policy':policy,'roi_voxels':int(roi.sum()),'outside_max_abs_change':0.,
                'roi_mean_abs_change':float(changed[...,roi].mean()),
                'reference_is_diagnostic_not_anatomically_validated':True}


def roi_patch(x:torch.Tensor,region:torch.Tensor,size:int=32):
    """Shared deterministic crop; marks LOCAL-CONTEXT intervention, not full MRI."""
    if x.ndim!=5 or region.ndim!=3 or tuple(region.shape)!=tuple(x.shape[-3:]):
        raise ValueError('Input/ROI dimension mismatch')
    if size%32:raise ValueError('MONAI SwinUNETR test patch must be multiple of 32')
    where=torch.nonzero(region)
    if where.numel()==0:raise ValueError('Empty ROI')
    center=where.float().mean(dim=0).round().long().tolist()
    starts=[max(0,min(int(c)-size//2,max(d-size,0))) for c,d in zip(center,region.shape)]
    ends=[min(start+size,d) for start,d in zip(starts,region.shape)]
    crop=x[:,:,starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]
    rcrop=region[starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]
    pad=[0,size-crop.shape[-1],0,size-crop.shape[-2],0,size-crop.shape[-3]]
    if any(pad):
        crop=F.pad(crop,pad);rcrop=F.pad(rcrop[None,None].float(),pad)[0,0].bool()
    return crop,rcrop,{'starts':starts,'ends':ends,'size':size,
                        'interpretation':'local_patch_context_not_full_volume'}
