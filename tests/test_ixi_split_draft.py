"""Synthetic-only regression tests for site stratification; no patient data."""
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ixi_split_draft",
    Path(__file__).resolve().parents[1] / "scripts" / "ixi_split_draft.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_stratified_splits_keep_pilot_only_in_train():
    groups = {}
    identifier = 1
    for site, n in (("GUYS", 322), ("HH", 185), ("IOP", 74)):
        rows = []
        for i in range(n):
            rows.append((f"IXI{identifier:03d}", Path(f"IXI{identifier:03d}-{site}-{i:04d}-T1.nii")))
            identifier += 1
        groups[site] = rows
    one = module.make_draft(groups)
    two = module.make_draft(groups)
    assert one == two
    assert len(one) == 581
    assert len(set(x["subject_id"] for x in one)) == 581
    assert sum(x["used_in_pilot5"] == "true" for x in one) == 5
    assert all(x["split"] == "train" for x in one if x["used_in_pilot5"] == "true")
    assert set(x["split"] for x in one) == {"train", "dev", "test"}
    assert all(x["qc_approved"] == "false" for x in one)
