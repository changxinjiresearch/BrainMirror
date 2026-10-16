"""End-to-end exporter contract test with synthetic FreeSurfer-format disk fixtures.

This does NOT claim any real patient MRI was reconstructed. It tests I/O,
vertex/annotation consistency, MGZ label geometry, and the gzip browser bundle.
"""
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('p5_export',ROOT/'scripts'/'export_freesurfer_surface.py')
mod=importlib.util.module_from_spec(s)
s.loader.exec_module(mod)

def fake_sphere(side):
    nphi=90;nlat=80
    phi=np.arange(nphi)*2*np.pi/nphi
    lat=np.linspace(.10,np.pi-.10,nlat)
    xyz=[]
    for t in lat:
        for p in phi:
            xyz.append([side*40+55*np.sin(t)*np.cos(p),
                        50*np.sin(t)*np.sin(p),
                        70*np.cos(t)])
    faces=[]
    for i in range(nlat-1):
        for j in range(nphi):
            k=(j+1)%nphi
            a=i*nphi+j;b=i*nphi+k;c=(i+1)*nphi+j;d=(i+1)*nphi+k
            faces.extend([[a,c,b],[b,c,d]])
    return np.asarray(xyz,dtype=np.float32),np.asarray(faces,dtype=np.int32)

def test_native_freesurfer_export_roundtrip(tmp_path):
    import nibabel as nib
    from nibabel.freesurfer.io import write_geometry,write_annot
    subject=tmp_path/'fs_case'
    for d in ['surf','label','mri']:(subject/d).mkdir(parents=True,exist_ok=True)
    names=[f'region_{i:02d}'.encode() for i in range(12)]
    ctab=np.zeros((12,5),dtype=np.int32)
    for i in range(12):
        ctab[i,:4]=[15+i*17,40+i*11,35+i*13,0]
    for hemi,side in [('lh',-1),('rh',1)]:
        verts,faces=fake_sphere(side)
        write_geometry(str(subject/'surf'/f'{hemi}.pial'),verts,faces)
        labels=(np.arange(len(verts))%12).astype(np.int32)
        write_annot(str(subject/'label'/f'{hemi}.aparc.DKTatlas.annot'),labels,ctab,names)
    seg=np.zeros((64,64,64),dtype=np.int16)
    seg[10:24,12:25,12:25]=17
    seg[30:44,12:25,12:25]=53
    seg[10:24,30:43,12:25]=10
    seg[30:44,30:43,12:25]=49
    img=nib.freesurfer.mghformat.MGHImage(seg,np.eye(4))
    nib.save(img,str(subject/'mri'/'aseg.mgz'))
    target=tmp_path/'PRIVATE_real_io_contract.json.gz'
    result=mod.export(subject,target,annotation='aparc.DKTatlas',include_subcortex=True)
    assert result['hemispheres']['lh']['vertices']==7200
    assert result['hemispheres']['rh']['vertices']==7200
    assert result['subcortical_objects']>=4
    assert target.exists()
    with gzip.open(target,'rb') as f:body=json.load(f)
    assert body['schema']=='brainmirror.native-anatomy.v1'
    assert body['provenance']['registered_to_ixi_population_mean'] is False
    assert body['provenance']['individual_data_private_only'] is True
    assert len(body['hemispheres'])==2
    assert 17 in [x['id'] for x in body['subcortical']]
    assert 53 in [x['id'] for x in body['subcortical']]
    assert all('native_FreeSurfer_tkregister_RAS_mm'==x['coordinate_space'] for x in body['hemispheres'])
