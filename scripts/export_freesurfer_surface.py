#!/usr/bin/env python3
"""Export verified FreeSurfer/FastSurfer native subject anatomy for local-only BrainMirror exploration.

No automatic publication. Cortical pial geometry and aparc annotation must be
from the SAME reconstructed subject. FreeSurfer surfaces are tkregister RAS mm;
segmentation meshes use that subject's MGZ vox2ras_tkr. A 467-subject affine
mean template is *not* a substitute for individual surface reconstruction.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'brainmirror.native-anatomy.v1'
# Common FreeSurfer aseg codes, non-contiguous anatomical labels; do not relabel.
SUBCORTICAL_LABELS = {
    17:'Left Hippocampus', 53:'Right Hippocampus',
    18:'Left Amygdala', 54:'Right Amygdala',
    10:'Left Thalamus', 49:'Right Thalamus',
    11:'Left Caudate', 50:'Right Caudate',
    12:'Left Putamen', 51:'Right Putamen',
    13:'Left Pallidum', 52:'Right Pallidum',
    4:'Left Lateral Ventricle', 43:'Right Lateral Ventricle',
    8:'Left Cerebellum Cortex', 47:'Right Cerebellum Cortex',
    16:'Brainstem',
}


def encoded(arr, dtype):
    array=np.ascontiguousarray(np.asarray(arr,dtype=dtype))
    return base64.b64encode(array.tobytes()).decode('ascii')


def validate_mesh(vertices, faces, *, min_vertices=100, max_vertices=500000):
    v=np.asarray(vertices); f=np.asarray(faces)
    if v.ndim!=2 or v.shape[1]!=3 or not min_vertices<=len(v)<=max_vertices:
        raise ValueError('Mesh vertex count/dimensions out of bounds')
    if f.ndim!=2 or f.shape[1]!=3 or len(f)<100 or len(f)>1_500_000:
        raise ValueError('Mesh face count/dimensions out of bounds')
    if not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(v):
        raise ValueError('Out-of-range face vertex index')
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite vertex coordinates')
    if np.max(np.abs(v))>900:
        raise ValueError('Implausible FreeSurfer tkrRAS physical coordinates')
    a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]]
    zero=(np.linalg.norm(np.cross(b-a,c-a),axis=1)<1e-9).mean()
    if zero>0.015: raise ValueError('Too many degenerate triangles')
    return int(len(v)),int(len(f))


def hemisphere(surf,annotation, name):
    from nibabel.freesurfer.io import read_geometry, read_annot
    coords, faces=read_geometry(str(surf))
    count, face_count=validate_mesh(coords,faces,min_vertices=5000)
    if not annotation.exists(): raise FileNotFoundError(annotation)
    labels, color_lut, names=read_annot(str(annotation),orig_ids=False)
    if len(labels)!=count: raise ValueError('Annotation/surface vertex count mismatch')
    if any(int(x)<-1 or int(x)>65534 for x in labels):
        raise ValueError('Annotation label index outside supported range')
    if len(names)<10 or (labels>=0).mean()<.70:
        raise ValueError('Insufficient valid cortical annotation coverage')
    regions=[name_bytes.decode('utf8','replace') for name_bytes in names]
    label_u16=np.where(labels<0,65535,labels).astype(np.uint16)
    if np.any((label_u16!=65535)&(label_u16>=len(regions))):
        raise ValueError('Invalid region-index-to-name table')
    return {
        'hemisphere':name, 'vertex_count':count,'face_count':face_count,
        'positions_f32_b64':encoded(coords,'<f4'),
        'faces_u32_b64':encoded(faces,'<u4'),
        'region_indices_u16_b64':encoded(label_u16,'<u2'),
        'region_names':regions,
        'atlas':annotation.stem.split('.',1)[1] if '.' in annotation.stem else annotation.stem,
        'coordinate_space':'native_FreeSurfer_tkregister_RAS_mm'
    }


def subcortex(segmentation, minimum_voxels=90):
    import nibabel as nib
    from scipy import ndimage as ndi
    from skimage import measure
    img=nib.load(str(segmentation))
    if not hasattr(img.header,'get_vox2ras_tkr'):
        raise ValueError('Need MGZ aseg segmentation with vox2ras_tkr, not arbitrary NIfTI')
    vox2tkr=np.asarray(img.header.get_vox2ras_tkr(),dtype=np.float64)
    if vox2tkr.shape!=(4,4) or abs(np.linalg.det(vox2tkr[:3,:3]))<1e-5:
        raise ValueError('Invalid MGZ tkregister geometry')
    data=np.asarray(img.dataobj,dtype=np.int16)
    if data.ndim!=3 or np.prod(data.shape)>300_000_000: raise ValueError('Invalid aseg volume')
    outputs=[]
    for idx,label in sorted(SUBCORTICAL_LABELS.items()):
        where=np.argwhere(data==idx)
        if where.shape[0]<minimum_voxels: continue
        lo=np.maximum(where.min(axis=0)-3,0)
        hi=np.minimum(where.max(axis=0)+4,np.array(data.shape))
        bounds=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))
        occ=(data[bounds]==idx).astype(np.float32)
        # Smooth within a fixed label only, not by reassigning MRI/ROI semantics.
        field=ndi.gaussian_filter(np.pad(occ,1),sigma=.65)
        v,f,_,_=measure.marching_cubes(field,level=.45,step_size=1)
        v=v-1+lo[None,:]
        tkr=np.c_[v,np.ones(len(v))]@vox2tkr.T
        coords=tkr[:,:3]
        count,faces=validate_mesh(coords,f,min_vertices=30)
        outputs.append({
            'id':int(idx),'name':label,'vertex_count':count,'face_count':faces,
            'positions_f32_b64':encoded(coords,'<f4'),
            'faces_u32_b64':encoded(f,'<u4'),
            'coordinate_space':'native_FreeSurfer_tkregister_RAS_mm'
        })
    return outputs


def export(subject_dir:Path,out:Path,annotation:str='aparc.DKTatlas',include_subcortex=True):
    root=Path(subject_dir)
    if not root.is_dir(): raise FileNotFoundError(root)
    hm=[]
    for hemi in ['lh','rh']:
        surf=root/'surf'/f'{hemi}.pial'
        ann=root/'label'/f'{hemi}.{annotation}.annot'
        if not surf.exists(): raise FileNotFoundError(surf)
        if not ann.exists(): raise FileNotFoundError(ann)
        hm.append(hemisphere(surf,ann,hemi))
    structures=[]
    if include_subcortex:
        candidate=root/'mri'/'aseg.mgz'
        if not candidate.exists():
            candidate=root/'mri'/'aparc.DKTatlas+aseg.deep.mgz'
        if not candidate.exists():raise FileNotFoundError('No aseg.mgz or aparc.DKTatlas+aseg.deep.mgz')
        structures=subcortex(candidate)
    payload={
        'schema':SCHEMA,
        'provenance':{
            'source':'native individual T1 MRI reconstructed with FreeSurfer/FastSurfer',
            'source_kind':'individual_native_surface',
            'reconstruction_software':'FreeSurfer/FastSurfer; confirm exact version in private run log',
            'coordinate_space':'native_FreeSurfer_tkregister_RAS_mm',
            'hemispheres':['lh','rh'],
            'cortical_parcellation':annotation,
            'is_population_mean':False,
            'registered_to_ixi_population_mean':False,
            'individual_data_private_only':True,
            'scientific_review_completed':False,
            'model_is_coupled':False,
        },
        'hemispheres':hm,'subcortical':structures
    }
    # Output contains individual derived brain anatomy. Do NOT commit publicly.
    out=Path(out)
    if out.exists(): raise FileExistsError('Refusing to overwrite an existing individual surface pack')
    out.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(payload,separators=(',',':'),ensure_ascii=False).encode('utf8')
    if out.name.endswith('.json.gz'):
        with gzip.open(out,'wb',compresslevel=6) as stream:stream.write(raw)
    elif out.name.endswith('.json'):
        out.write_bytes(raw)
    else:
        raise ValueError('Output must end in .json.gz or .json')
    out.chmod(0o600)
    stats={
        'schema':SCHEMA,'hemispheres':{x['hemisphere']:{'vertices':x['vertex_count'],'triangles':x['face_count'],'regions':len(x['region_names'])} for x in hm},
        'subcortical_objects':len(structures),'json_sha256':hashlib.sha256(raw).hexdigest(),
        'file_size_bytes':out.stat().st_size,'reviewed':False,
        'warning':'INDIVIDUAL BRAIN DERIVATIVE; PRIVATE ONLY; never auto-upload to public repository.'
    }
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject-dir',type=Path,required=True,help='One completed FreeSurfer/FastSurfer subject directory')
    p.add_argument('--out',type=Path,required=True,help='PRIVATE native derived surface (.json.gz)')
    p.add_argument('--annotation',choices=['aparc','aparc.a2009s','aparc.DKTatlas'],default='aparc.DKTatlas')
    p.add_argument('--no-subcortex',action='store_true')
    a=p.parse_args()
    print(json.dumps(export(a.subject_dir,a.out,a.annotation,not a.no_subcortex),indent=2))

if __name__=='__main__':main()