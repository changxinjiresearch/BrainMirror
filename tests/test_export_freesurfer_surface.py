from pathlib import Path
import importlib.util
import numpy as np
import pytest

SRC=Path(__file__).resolve().parents[1]/'scripts'/'export_freesurfer_surface.py'
spec=importlib.util.spec_from_file_location('surface_export',SRC)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_mesh_validation_geometry():
    coords=np.array([[x,y,z] for z in (0,1,2,3,4,5) for y in range(4) for x in range(4)],dtype=float)
    faces=np.array([[i,i+1,i+4] for i in range(80) if i%4<3 and i+4<len(coords)],dtype=int)
    # Increase triangles for minimum bound
    faces=np.concatenate([faces,faces],axis=0)
    assert m.validate_mesh(coords,faces,min_vertices=20)[0]==96
    with pytest.raises(ValueError): m.validate_mesh(coords, np.array([[0,1,1000]]*100),min_vertices=20)
    with pytest.raises(ValueError):m.validate_mesh(np.array([[np.nan,0,0]]*200),np.array([[0,1,2]]*101),min_vertices=20)

def test_binary_encoding_le():
    import base64
    b=base64.b64decode(m.encoded(np.array([[1.,-2.,3.]]),'<f4'))
    assert np.frombuffer(b,dtype='<f4').tolist()==[1.,-2.,3.]

def test_anatomical_semantics_metadata():
    assert m.SCHEMA=='brainmirror.native-anatomy.v1'
    assert m.SUBCORTICAL_LABELS[17]=='Left Hippocampus'
    assert m.SUBCORTICAL_LABELS[53]=='Right Hippocampus'
    assert m.SUBCORTICAL_LABELS[16]=='Brainstem'