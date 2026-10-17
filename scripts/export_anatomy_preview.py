#!/usr/bin/env python3
"""Convert reviewed *brain-only* NIfTI to a bounded local-file web preview.

This is NOT permission to publish clinical scans or IXI individual previews.
Output contains voxel data. Keep PRIVATE unless source and privacy review allow release.
"""
from __future__ import annotations
import argparse,base64,json
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom

def bytes_from_volume(data:np.ndarray,max_edge:int=96)->tuple[tuple[int,int,int],str]:
    arr=np.asarray(data,dtype=np.float32)
    if arr.ndim!=3 or any(s<8 for s in arr.shape) or not np.isfinite(arr).all():raise ValueError('Invalid 3D image')
    if not 32<=max_edge<=128:raise ValueError('Invalid preview resolution')
    ratio=min(1,max_edge/max(arr.shape));shape=tuple(max(8,int(round(s*ratio))) for s in arr.shape)
    res=zoom(arr,zoom=tuple(o/i for i,o in zip(arr.shape,shape)),order=1)
    nz=res[res>0]
    if nz.size<100:raise ValueError('Insufficient anatomical signal')
    low,high=np.percentile(nz,[2,98]);
    if high<=low:raise ValueError('Collapsed intensity range')
    scaled=np.uint8(np.round(np.clip((res-low)/(high-low),0,1)*255))
    # numpy defaults to C memory order; client expects x-fastest NIfTI (Fortran).
    return shape,base64.b64encode(scaled.tobytes(order='F')).decode('ascii')

def make_preview(image_path:Path,output_path:Path,kind:str,source:str,brain_mask:Path,atlas_labels:Path|None=None):
    import nibabel as nib
    if output_path.exists():raise FileExistsError(output_path)
    img=nib.load(str(image_path));mask=nib.load(str(brain_mask))
    if img.shape!=mask.shape or not np.allclose(img.affine,mask.affine,atol=.01):raise ValueError('Image and mask geometry mismatch')
    data=np.asarray(img.get_fdata(dtype=np.float32));m=np.asarray(mask.dataobj)>0
    if not m.any():raise ValueError('Empty brain mask')
    data[~m]=0
    shape,encoded=bytes_from_volume(data)
    obj={'schema':'brainmirror.anatomy-preview.v1','shape':list(shape),'intensities_b64':encoded,'affine':img.affine.tolist(),'provenance':{'kind':kind,'source':source,'preview_is_downsampled':True,'brain_mask_applied':True,'source_resampled_for_visualization':True,'human_review_required_before_any_public_deployment':True}}
    if atlas_labels is not None:
        labels_img=nib.load(str(atlas_labels))
        if labels_img.shape!=img.shape or not np.allclose(labels_img.affine,img.affine,atol=.01):
            raise ValueError('Atlas label space/shape mismatch')
        full_labels=np.asarray(labels_img.dataobj)
        if not np.isfinite(full_labels).all() or not np.allclose(full_labels,np.round(full_labels)):
            raise ValueError('Atlas labels must be discrete nonnegative integers')
        if full_labels.min()<0 or full_labels.max()>65535:raise ValueError('Atlas label out of bounds')
        resized=zoom(full_labels,zoom=tuple(o/i for i,o in zip(full_labels.shape,shape)),order=0)
        if resized.max()>255:
            arr=np.asarray(resized,dtype='<u2');obj['label_dtype']='uint16'
        else:
            arr=np.asarray(resized,dtype='uint8');obj['label_dtype']='uint8'
        obj['labels_b64']=base64.b64encode(arr.tobytes(order='F')).decode('ascii')
        obj['provenance']['atlas_labels_provided']=True
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(obj,separators=(',',':')),encoding='utf-8')
    print('Preview exported (PRIVATE; contains voxels):',output_path,shape)

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--brain-mask',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['individual','ixi_affine_group_candidate','external_reference'],required=True);p.add_argument('--source',required=True);p.add_argument('--atlas-labels',type=Path);a=p.parse_args();make_preview(a.image,a.output,a.kind,a.source,a.brain_mask,a.atlas_labels)
if __name__=='__main__':main()