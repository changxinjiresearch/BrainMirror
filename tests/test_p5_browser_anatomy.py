from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_native_segmentation_controls_and_data_separation():
    app=(ROOT/'site'/'app.js').read_text(encoding='utf-8')
    html=(ROOT/'site'/'index.html').read_text(encoding='utf-8')
    native=(ROOT/'site'/'native_anatomy.js').read_text(encoding='utf-8')
    for x in ['nativeAnatomyFile','nativeRegion','nativeLeft','nativeRight','nativeDeep','nativePicked','nativeAlignmentNote']:
        assert 'id="'+x+'"' in html
        assert "$('"+x+"')" in app
    assert "await import('./native_anatomy.js')" in app
    assert "DecompressionStream('gzip')" in app
    assert 'registered_to_ixi_population_mean' in native
    assert "individual_data_private_only!==true" in native
    assert 'raycaster.intersectObjects' in native
    assert 'get_subject_data' not in native
    assert 'NOT LINKED' in html
    assert '未配准' in html
