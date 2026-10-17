from pathlib import Path
import csv
import numpy as np
import pytest
from scripts.ixi_template_pipeline import approved_rows,summarize_reference,normalize_brain
from scripts.export_anatomy_preview import bytes_from_volume

def test_population_mean_tracks_coverage():
    a=np.ones((4,4,4));b=3*np.ones((4,4,4));m=np.ones((4,4,4),bool);n=m.copy();n[0]=False
    mean,sd,c=summarize_reference([a,b],[m,n]);assert mean[1,1,1]==2 and sd[1,1,1]==1
    assert mean[0,1,1]==1 and sd[0,1,1]==0 and c[0,1,1]==1

def test_missing_coverage_rejected_nonfinite():
    with pytest.raises(ValueError):summarize_reference([np.ones((4,4,4))*np.nan],[np.ones((4,4,4))])

def test_preview_never_exceeds_bound():
    x=np.zeros((80,96,50),np.float32);x[10:70,15:85,8:40]=np.linspace(1,100,60)[:,None,None]
    shape,data=bytes_from_volume(x,64);assert max(shape)<=64;assert len(data)>100

def test_train_only_manifest_requires_review(tmp_path:Path):
    im=tmp_path/'x.nii';mask=tmp_path/'mask.nii';im.write_bytes(b'x');mask.write_bytes(b'm')
    manifest=tmp_path/'m.csv';
    with manifest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject_id','image_path','mask_path','split','qc_approved']);w.writerow(['IXI001',str(im),str(mask),'train','yes']);w.writerow(['IXI002',str(im),str(mask),'test','no'])
    rows=approved_rows(manifest);assert len(rows)==1 and rows[0]['subject_id']=='IXI001'

def test_unreviewed_train_rejected(tmp_path:Path):
    manifest=tmp_path/'m.csv';manifest.write_text('subject_id,image_path,mask_path,split,qc_approved\nIXI001,a,b,train,no\n')
    with pytest.raises(ValueError):approved_rows(manifest)