"""SCEPT v0.9 BraTS2020 **real-data** pipeline (not an achieved clinical result).

Requires legitimately obtained BraTS2020 training NIfTI and a locally trained,
provenance-attested MONAI 1.5.1 SwinUNETR checkpoint. Synthetic smoke runs are
explicitly separate and never counted as patient evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .brats2020 import (
    MODALITIES, REGIONS, audited_inventory, discover_training_cases, patient_split,
    load_case, dice_regions, roi_patch, region_reference,
)
from .monai_adapter import build_monai_model, require_monai

SCHEMA='scept/brats2020-checkpoint/v1'
PROTOCOL='scept/brats2020-evaluation/v1'
ARCH_DEFAULT={'in_channels':4,'out_channels':3,'feature_size':12,
              'depths':[2,2,2,2],'num_heads':[3,6,12,24],
              'window_size':7,'norm_name':'instance'}


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while (b:=f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def require_clean_splits(splits):
    if set(splits)!={'train','validation','test'}:
        raise ValueError('Splits must be train / validation / test')
    lists=[splits[k] for k in ('train','validation','test')]
    if any(not isinstance(x,list) or not x or len(set(x))!=len(x) for x in lists):
        raise ValueError('Empty/duplicate patient split')
    if len(set().union(*map(set,lists)))!=sum(map(len,lists)):
        raise ValueError('Patient leakage across splits')
    return True


def attest_metadata(meta,*,splits,expected_inventory_digest=None):
    """Fail closed on random/unknown weights and training-test overlap.

    A metadata declaration alone is not proof of actual training; experiment
    reports must include checkable logs and an independently audited checkpoint.
    """
    require_clean_splits(splits)
    if meta.get('schema')!=SCHEMA or meta.get('dataset')!='BraTS2020_training':
        raise ValueError('Unattested or wrong-dataset model checkpoint')
    if meta.get('monai_version')!='1.5.1':
        raise ValueError('Only MONAI 1.5.1 checkpoint schema supported')
    if not isinstance(meta.get('optimization_steps'),int) or meta['optimization_steps']<1:
        raise ValueError('Weights must have undergone nonzero training steps')
    if not isinstance(meta.get('epochs'),int) or meta['epochs']<1:
        raise ValueError('Invalid training epoch count')
    if set(meta.get('train_ids',[]))!=set(splits['train']):
        raise ValueError('Checkpoint training IDs do not match frozen train split')
    if set(meta.get('validation_ids',[]))!=set(splits['validation']):
        raise ValueError('Checkpoint validation IDs mismatch')
    if set(meta.get('train_ids',[]))&set(splits['test']):
        raise ValueError('Test patient was used during training')
    if expected_inventory_digest and meta.get('inventory_sha256')!=expected_inventory_digest:
        raise ValueError('Dataset inventory drift; refusing model reuse')
    if not isinstance(meta.get('seed'),int) or not isinstance(meta.get('final_loss'),(float,int)) or not math.isfinite(meta['final_loss']):
        raise ValueError('Training provenance is incomplete')
    if meta.get('synthetic_only'):
        raise ValueError('Synthetic weights cannot be used for patient evidence')
    return True


def load_attested_model(checkpoint_path, splits, *, sha256=None,inventory_digest=None,device='cpu'):
    require_monai()
    path=Path(checkpoint_path)
    actual=sha256_file(path)
    if sha256 and actual!=sha256:raise ValueError('Checkpoint SHA256 mismatch')
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(obj,dict) or not {'architecture','state_dict','provenance'}<=obj.keys():
        raise ValueError('Expected checkpoint mapping containing architecture/state_dict/provenance')
    arch=obj['architecture']
    if not isinstance(arch,dict) or arch.get('in_channels')!=4 or arch.get('out_channels')!=3:
        raise ValueError('Only 4-channel input, WT/TC/ET 3-head output supported')
    attest_metadata(obj['provenance'],splits=splits,expected_inventory_digest=inventory_digest)
    model=build_monai_model(**arch)
    model.load_state_dict(obj['state_dict'],strict=True)
    model.to(device).eval()
    return model, {'sha256':actual,'architecture':arch,'provenance':obj['provenance']}


def patch_for_train(x:np.ndarray,regions:np.ndarray, *,size=64,seed=0,positive=True):
    """Aligned input/target patch, foreground-biased only on TRAIN subjects."""
    if size%32:raise ValueError('Patch size must be divisible by 32')
    rng=np.random.default_rng(seed)
    tumor=regions[0]>.5
    candidates=np.argwhere(tumor) if positive and tumor.any() else np.argwhere(x[0]!=0)
    if not len(candidates):raise ValueError('Empty candidate sampling mask')
    center=candidates[int(rng.integers(len(candidates)))]
    begins=[max(0,min(int(c)-size//2,max(d-size,0))) for c,d in zip(center,tumor.shape)]
    slices=tuple(slice(s,s+size) for s in begins)
    a=x[(slice(None),)+slices]
    y=regions[(slice(None),)+slices]
    pad=[(0,0)]+[(0,size-a.shape[i+1]) for i in range(3)]
    a=np.pad(a,pad);y=np.pad(y,pad)
    return torch.from_numpy(a[None].copy()).float(),torch.from_numpy(y[None].copy()).float()


def brats_loss(logits, target):
    """Sigmoid BCE + soft Dice loss for nested independent region logits."""
    if logits.shape!=target.shape:raise ValueError('Prediction/target size mismatch')
    bce=F.binary_cross_entropy_with_logits(logits,target)
    p=torch.sigmoid(logits)
    numerator=2*(p*target).sum(dim=(0,2,3,4))+1
    denominator=(p+target).sum(dim=(0,2,3,4))+1
    dice=1-(numerator/denominator).mean()
    return bce+dice


def save_trained_checkpoint(path,model,*,architecture,train_ids,validation_ids,
                            inventory_sha256,seed,epochs,optimization_steps,final_loss):
    provenance={'schema':SCHEMA,'dataset':'BraTS2020_training','monai_version':'1.5.1',
                'train_ids':sorted(train_ids),'validation_ids':sorted(validation_ids),
                'inventory_sha256':inventory_sha256,'seed':int(seed),
                'epochs':int(epochs),'optimization_steps':int(optimization_steps),
                'final_loss':float(final_loss),'synthetic_only':False}
    if optimization_steps<1 or epochs<1 or not math.isfinite(final_loss):
        raise ValueError('Cannot save an untrained checkpoint')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'architecture':architecture,'state_dict':model.cpu().state_dict(),
                'provenance':provenance},p)
    return {'sha256':sha256_file(p),'provenance':provenance,'local_only':True}


def train_brats(cases, splits, *,inventory_digest, output,architecture=None,
                patch_size=64,epochs=1,steps_per_epoch=30,lr=1e-4,seed=240216,device='cpu'):
    """Minimal resumable-target trainer; GPU recommended for practical cohorts.

    No test-set reads, no pretrained claim, no public checkpoint upload.
    Caller must complete training and qualification before science conclusions.
    """
    require_clean_splits(splits)
    require_monai()
    if epochs<1 or steps_per_epoch<1 or lr<=0 or patch_size%32:raise ValueError('Invalid training config')
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    arch=architecture or dict(ARCH_DEFAULT)
    model=build_monai_model(**arch).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    cmap={c.case_id:c for c in cases}
    if any(s not in cmap for s in splits['train']):raise ValueError('Missing training cases')
    rng=random.Random(seed);train_ids=sorted(splits['train'])
    losses=[];steps=0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            case=cmap[rng.choice(train_ids)]
            x,y,_=load_case(case)
            px,py=patch_for_train(x,y,size=patch_size,seed=seed+steps,positive=steps%3!=0)
            optimizer.zero_grad(set_to_none=True)
            logits=model(px.to(device));loss=brats_loss(logits,py.to(device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step()
            losses.append(float(loss.detach()));steps+=1
    stamp=save_trained_checkpoint(output,model,architecture=arch,train_ids=train_ids,
            validation_ids=splits['validation'],inventory_sha256=inventory_digest,seed=seed,
            epochs=epochs,optimization_steps=steps,final_loss=losses[-1])
    stamp['loss_trace']=losses
    return stamp


def predict_full(model,image,*,patch_size=96,overlap=.25,device='cpu'):
    """Full-volume tiled inference; avoid materializing full-graph activations.

    Full-volume prediction is an aggregate of context-dependent windows. v0.9
    SCCA uses a local crop and reports that as a *different estimand*.
    """
    if image.ndim!=4 or image.shape[0]!=4 or patch_size%32:
        raise ValueError('Expected [4,D,H,W] and patch multiple of 32')
    if not 0<=overlap<1:raise ValueError('Invalid overlap')
    from monai.inferers import sliding_window_inference
    x=torch.as_tensor(image[None],dtype=torch.float32,device=device)
    with torch.inference_mode():
        logits=sliding_window_inference(x,(patch_size,)*3,1,model,
                                       overlap=overlap,mode='constant')
        if logits.shape[1]!=3 or tuple(logits.shape[-3:])!=tuple(image.shape[1:]):
            raise ValueError('Unexpected inference output')
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def evaluate_heldout(model,cases,splits,*,checkpoint_sha256,inventory_digest,
                     patch_size=96,overlap=.25,device='cpu',max_cases=None):
    """Freeze ALL configuration before examining test labels/metrics."""
    require_clean_splits(splits)
    cmap={c.case_id:c for c in cases}
    test_ids=sorted(splits['test'])
    if any(s not in cmap for s in test_ids):raise ValueError('Missing test cases')
    if max_cases is not None:
        if max_cases<1:raise ValueError('Invalid case limit')
        test_ids=test_ids[:max_cases]
    rows=[]
    for name in test_ids:
        x,y,qc=load_case(cmap[name]);prediction=predict_full(model,x,patch_size=patch_size,overlap=overlap,device=device)
        rows.append({'case_id':name,'regions':dice_regions(prediction,y),'qc':qc})
    per_region={r:[item['regions'][r]['dice'] for item in rows if item['regions'][r]['dice'] is not None] for r in REGIONS}
    averages={r:float(np.mean(v)) if v else None for r,v in per_region.items()}
    return {'schema':PROTOCOL,'dataset':'BraTS2020_training_heldout_patients',
            'scientific_status':'measured_on_real_patients_unless_input_fixture_is_synthetic',
            'checkpoint_sha256':checkpoint_sha256,'inventory_sha256':inventory_digest,
            'n_test_total':len(splits['test']),'n_evaluated':len(rows),'partial_evaluation':len(rows)<len(splits['test']),
            'metric':'subject_macro_Dice_WT_TC_ET_empty_both_undefined',
            'mean_dice':averages,'patients':rows,'model_is_not_clinically_validated':True}


def explain_heldout_case(model,image:np.ndarray,regions:np.ndarray,*,roi_region='WT',
                         patch_size=64,device='cpu',max_exact=8,effect_tol=1e-5):
    """One local 3D ROI SCCA proof with two diagnostic reference policies.

    This is intentionally local-context, distinct from the full-volume sliding
    window segmentation score; do not extrapolate to whole patient volume.
    """
    if roi_region not in REGIONS:raise ValueError('Unknown ROI label')
    if patch_size<64 and any(isinstance(m,torch.nn.InstanceNorm3d) for m in model.modules()):
        raise ValueError('Official InstanceNorm SwinUNETR needs at least 64-voxel local patches')
    i=REGIONS.index(roi_region)
    if image.shape[0]!=4 or regions.shape[0]!=3:raise ValueError('Bad BraTS input')
    x=torch.as_tensor(image[None],dtype=torch.float32,device=device)
    roi=torch.as_tensor(regions[i]>.5,device=device)
    crop,crop_roi,geom=roi_patch(x,roi,size=patch_size)
    ref0,qc0=region_reference(crop,crop_roi,policy='tissue_median')
    ref1,qc1=region_reference(crop,crop_roi,policy='local_blur',kernel=5)
    model.eval()
    from .monai_crosslayer import discover_monai_crosslayer
    with torch.no_grad():
        result=discover_monai_crosslayer(model,crop,ref0,crop_roi,
                class_index=i,heldout_reference=ref1,
                max_hops=3,max_sites=4,per_step=2,beam_width=2,
                max_exact=max_exact,effect_tol=effect_tol)
    return {'schema':'scept/brats2020-scca-local/v1','roi':roi_region,
            'crop_geometry':geom,'reference_qc':[qc0,qc1],
            'path_search':result,'uncertainty':'Both reference policies are diagnostic; held-out reference is not held-out patient',
            'scope':'local_patch_not_full_volume','status':result['status']}
