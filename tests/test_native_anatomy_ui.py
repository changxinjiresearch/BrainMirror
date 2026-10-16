from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_scientific_local_only_contract():
    s = (ROOT/'site'/'native_anatomy.js').read_text()
    assert "brainmirror.native-anatomy.v1" in s
    assert "individual_data_private_only" in s
    assert "registered_to_ixi_population_mean" in s
    assert "positions_f32_b64" in s and "region_indices_u16_b64" in s
    assert "subcortical" in s
    assert "raycaster.intersectObjects" in s
    assert "new THREE.WebGLRenderer" in s
    assert "loadNativePack" in s