"""Pretrained 4-modality SCEPT computation, without training or fabricated traces."""
from __future__ import annotations
from pathlib import Path
import hashlib
import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from brainmirror.scept.monai_adapter import (
    build_monai_model, MonaiSwinObserver, AttentionAtom,
    atomic_bidirectional_effect, decompose_monai_attention
)
from brainmirror.scept.brats2020 import region_reference
from scripts.scept_pretrained_brats21_fast import URL, EXPECTED_SHA256, download, normalize

CHANNELS=("flair","t1ce","t1","t2")
_MODEL=None

def get_model():
    global _MODEL
    if _MODEL is not None:return _MODEL
    import monai
    if monai.__version__.split("+")[0]!="1.5.1":raise RuntimeError("MONAI 1.5.1 required")
    path,sha=download(os.environ.get("SCEPT_MODEL_CACHE","/tmp/scept_official"))
    if sha!=EXPECTED_SHA256:
        raise RuntimeError("Checkpoint checksum mismatch; refuse unsafe deserialization")
    model=build_monai_model(in_channels=4,out_channels=3,feature_size=48,
        depths=(2,2,2,2),num_heads=(3,6,12,24),
        window_size=7,norm_name="instance",use_checkpoint=False)
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    state=checkpoint.get("state_dict")
    if not isinstance(state,dict) or not state:raise RuntimeError("Invalid pretrained model")
    if all(k.startswith("module.") for k in state):
        state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL=model
    return model

def prepare(paths):
    """Validate aligned 3D scans and normalize all nonzero voxels channel-wise."""
    if set(paths)!=set(CHANNELS):raise ValueError("Four MRI modalities required")
    channels=[];shape=None;affine=None
    for key in CHANNELS:
        nii=nib.load(str(paths[key]))
        vol=np.asarray(nii.dataobj,dtype=np.float32)
        if vol.ndim!=3 or min(vol.shape)<64 or max(vol.shape)>512 or not np.isfinite(vol).all():
            raise ValueError("Invalid 3D MRI")
        if shape is not None and (vol.shape!=shape or not np.allclose(nii.affine,affine,atol=1e-4,rtol=0)):
            raise ValueError("MRI modalities must have matching shape and affine")
        shape,affine=vol.shape,nii.affine
        channels.append(normalize(vol))
    arr=np.stack(channels,axis=0)
    positions=np.argwhere(np.any(arr!=0,axis=0))
    if not len(positions):raise ValueError("Empty MRI")
    starts=np.clip(np.rint(positions.mean(0)).astype(int)-32,0,np.array(shape)-64)
    sl=tuple(slice(int(i),int(i)+64) for i in starts)
    patch=np.ascontiguousarray(arr[(slice(None),)+sl])
    return patch,{"full_shape":list(shape),"start":starts.tolist(),
                  "size":64,"policy":"brain_nonzero_centroid_no_ground_truth"}

def screen(model,x,ref,roi,site):
    """Score actual attention message changes at ROI-corresponding query tokens."""
    o=MonaiSwinObserver(model)
    with torch.no_grad():
        with o.recording(selected=[site]):
            model(x)
        a=o.windows[site]["input"];mask=o.windows[site]["mask"];geometry=o.geometry[site]
        with o.recording(selected=[site]):
            model(ref)
        b=o.windows[site]["input"]
        first=decompose_monai_attention(o.sites[site],a,mask)
        second=decompose_monai_attention(o.sites[site],b,mask)
        difference=torch.linalg.vector_norm((first["v"]-second["v"]).float(),dim=-1)
        weights=first["weights"].float()
        pooled=F.adaptive_max_pool3d(roi[None,None].float(),geometry.input_shape)[0,0]>0
        choices=[]
        nwin,nheads,ntokens,_=first["v"].shape
        for wi in range(nwin):
            for qi in range(ntokens):
                pos=geometry.position(wi%geometry.total_windows,qi)
                if pos is None or not bool(pooled[pos]):continue
                for head in range(nheads):
                    strength=weights[wi,head,qi].abs()*difference[wi,head]
                    for val,idx in zip(*torch.topk(strength,k=min(2,ntokens))):
                        ki=int(idx)
                        if mask is not None and float(mask[wi%mask.shape[0],qi,ki])<-50:continue
                        if float(val)>0 and bool(torch.isfinite(val)):
                            choices.append((float(val),AttentionAtom(site,wi,head,qi,ki,"value_message")))
                if len(choices)>4000:break
            if len(choices)>4000:break
        return sorted(choices,key=lambda pair:-pair[0])[:2]

def pack_axial_prediction(prob, crop):
    """Compress one measured full-resolution 64x64 output slice, never raw MRI."""
    import base64
    if tuple(prob.shape)!=(1,3,64,64,64):
        raise ValueError("Unexpected pretrained tensor output")
    # Official channel order TC, WT, ET. Each 64x64 tile uses x-major 2D bytes.
    tiles={}
    for i,name in enumerate(("tc","wt","et")):
        image=(prob[0,i,:,:,32]>0.5).to(dtype=torch.uint8).cpu().numpy()
        tiles[name]=base64.b64encode(image.tobytes(order="C")).decode("ascii")
    return {"kind":"measured_model_prediction_axial_mask",
            "patch_start":crop["start"],"full_z":int(crop["start"][2])+32,
            "tile_width":64,"tile_height":64,"order":"x_major_y_minor",
            "mask_base64":tiles,
            "warning":"Local patch model output; not ground truth and not full MRI segmentation"}


def run(paths,progress=lambda _:None):
    import time
    begin=time.monotonic()
    progress("loading_verified_pretrained_weights")
    model=get_model()
    device=next(model.parameters()).device
    progress("reading_four_modalities")
    patch,crop=prepare(paths)
    x=torch.from_numpy(patch[None]).to(device)
    progress("real_swinunetr_prediction")
    with torch.no_grad():
        logits=model(x)
        prob=torch.sigmoid(logits)
        wt=prob[0,1]
        roi=wt>0.5
        roi_policy="predicted_WT_threshold_0.5"
        if not bool(roi.any()):
            roi=wt>=torch.quantile(wt.float().flatten(),0.99)
            roi_policy="top_1_percent_WT_probability_low_confidence"
        summary={name:float(prob[0,i].mean()) for i,name in enumerate(("TC","WT","ET"))}
        preview=pack_axial_prediction(prob,crop)
        ref,qc=region_reference(x,roi,policy="tissue_median")
    progress("screening_actual_attention_messages")
    obs=MonaiSwinObserver(model)
    sites=[k for k in obs.sites if "layers2.0.blocks.0.attn" in k]
    if not sites:raise RuntimeError("Unsupported official Attention topology")
    candidates=screen(model,x,ref,roi,sites[0])
    progress("real_attention_disruption_and_restoration")
    attempts=[]
    for score,atom in candidates:
        with torch.no_grad():
            value=atomic_bidirectional_effect(model,x,ref,atom,class_index=1,roi=roi)
        attempts.append({"screening_magnitude":score,"effect":value})
    best=max(attempts,key=lambda item:abs(item["effect"]["disruption_effect"]),default=None)
    return {"schema":"scept/live-4modality/v1",
        "case_id":"USER_UPLOAD_UNPAIRED","model":"MONAI SwinUNETR BraTS21 Fold1",
        "checkpoint_sha256":EXPECTED_SHA256,"checkpoint_source":URL,
        "input_modalities":list(CHANNELS),
        "model_outputs":["TC","WT","ET"],
        "cropping":crop,"roi_policy":roi_policy,"roi_voxels":int(roi.sum()),
        "mean_region_probability":summary,
        "prediction_overlay":preview,
        "scept_single_atom_intervention":best["effect"] if best else None,
        "screening_magnitude":best["screening_magnitude"] if best else None,
        "candidate_effects":attempts,
        "reference_qc":qc,
        "patch_dice_NOT_full_volume":None,
        "scientific_status":"Exploratory single-atom effect; not a verified cross-layer circuit, no independent Dice",
        "elapsed_seconds":round(time.monotonic()-begin,2),
        "device":str(device),"raw_mri_uploaded_to_git":False,
        "privacy":"Temporary uploads deleted by server after completion"}
