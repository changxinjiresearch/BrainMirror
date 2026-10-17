"""Static scientific/UX guards for real MRI-derived 3D anatomy presentation."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_continuous_surface_is_derived_from_actual_mri_voxels():
    js=(ROOT/'site'/'brain3d.js').read_text(encoding='utf-8')
    assert 'volume.data[srcIx(xx,yy,zz)]' in js
    assert 'volume.affine' in js
    assert 'const TETRAS' in js
    assert 'tri(ac,ad,bc);tri(ad,bd,bc);' in js
    assert 'MeshStandardMaterial' in js
    assert 'new THREE.WebGLRenderer' in js
    assert 'group.rotation.z+' in js
    assert "fallback.style.visibility='hidden'" in js
    assert 'not a pial reconstruction' in js.lower()

def test_mri_physical_planes_and_clear_failure():
    js=(ROOT/'site'/'app.js').read_text(encoding='utf-8')
    page=(ROOT/'site'/'index.html').read_text(encoding='utf-8')
    assert "await import('./brain3d.js')" in js
    assert "rasAxis" in js and "plane.fix" in js
    assert "if(new Set(rasAxis).size!==3)" in js
    assert '3D 表面生成失败' in js
    assert '未完成' in page or '未获独立' in js
    assert 'NOT LINKED' in page
    assert '连续三维大脑' in page
