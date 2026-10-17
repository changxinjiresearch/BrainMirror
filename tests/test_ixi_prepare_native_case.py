from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('prep',Path(__file__).resolve().parents[1]/'scripts'/'ixi_prepare_native_case.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_split_lock_kept():
    assert len(m.EXPECTED_SPLIT_SHA256)==64
    assert m.EXPECTED_SPLIT_SHA256=='00f3ec7d490bff06f529640ccfed4bbae02f172c2f92f4e851e756cc1cf4e40a'