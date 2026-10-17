#!/usr/bin/env python3
"""Private, deterministic, site-stratified IXI T1 split DRAFT.

Do not place the private CSV in public GitHub. The five cases used to develop
the affine registration pilot are assigned to TRAIN, not dev or held-out test.
No case becomes QC-approved merely from this split.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import re

RE = re.compile(r"^IXI(\d{3})-([A-Za-z]+)-", re.I)
PER_SITE_PILOT = {"Guys": 2, "HH": 2, "IOP": 1}
SEED = 240215

def collect(dataset_root: Path):
    root = Path(dataset_root)
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
    )
    if len(files) != 581:
        raise RuntimeError(f"IXI T1 cohort expected 581 actual files; found {len(files)}")
    grouped = defaultdict(list)
    seen = set()
    for p in files:
        m = RE.match(p.name)
        if m is None:
            raise RuntimeError("Unrecognized filename; audit privately")
        sid = f"IXI{m.group(1)}"
        if sid in seen:
            raise RuntimeError("Repeated IXI subject number; do not split")
        seen.add(sid)
        grouped[m.group(2).upper()].append((sid, p))
    if set(grouped) != {"GUYS", "HH", "IOP"}:
        raise RuntimeError(f"Unexpected acquisition sites: {sorted(grouped)}")
    return grouped

def make_draft(grouped, seed=SEED):
    rows = []
    for site in sorted(grouped):
        cohort = sorted(grouped[site], key=lambda r: r[1].name)
        pilot_n = PER_SITE_PILOT[{"GUYS": "Guys", "HH": "HH", "IOP": "IOP"}[site]]
        reserved_train = cohort[:pilot_n]
        remaining = cohort[pilot_n:].copy()
        random.Random(f"{seed}:{site}").shuffle(remaining)
        # Test/dev allocated only among previously untouched subjects.
        nt = round(.1 * len(cohort))
        nv = round(.1 * len(cohort))
        if len(remaining) <= nt + nv:
            raise RuntimeError("Not enough untouched subjects for held-out splits")
        splits = (
            [(x, "test", False) for x in remaining[:nt]]
            + [(x, "dev", False) for x in remaining[nt:nt+nv]]
            + [(x, "train", False) for x in remaining[nt+nv:]]
            + [(x, "train", True) for x in reserved_train]
        )
        for (sid, path), split, used_in_pilot in splits:
            rows.append({
                "subject_id": sid, "site": site, "image_path": str(path),
                "mask_path": "", "split": split, "qc_approved": "false",
                "used_in_pilot5": str(used_in_pilot).lower()
            })
    if len(rows) != 581 or len({r["subject_id"] for r in rows}) != 581:
        raise AssertionError("Split is incomplete or leaked across subjects")
    if any(r["split"] != "train" for r in rows if r["used_in_pilot5"] == "true"):
        raise AssertionError("Pilot subject leaked into held-out split")
    return sorted(rows, key=lambda x: x["subject_id"])

def run(data_root: Path, out: Path):
    grouped = collect(data_root)
    rows = make_draft(grouped)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory (no silent overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    private = out / "PRIVATE_ixi_subject_split_DRAFT.csv"
    columns = list(rows[0])
    with private.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    private.chmod(0o600)
    by_site = {
        site: dict(sorted(Counter(r["split"] for r in rows if r["site"] == site).items()))
        for site in sorted(grouped)
    }
    summary = {
        "schema": "brainmirror.ixi.split.draft.v1",
        "cohort": "IXI T1", "subject_count": len(rows),
        "seed": SEED, "selection": "stratified by source acquisition site",
        "counts": dict(sorted(Counter(r["split"] for r in rows).items())),
        "by_site": by_site,
        "pilot5_train_only": True,
        "private_manifest_sha256": hashlib.sha256(private.read_bytes()).hexdigest(),
        "split_locked": False, "P1_PASS": False,
        "P2_TRAINING_AUTHORIZED": False,
        "review_requirements": [
            "whole-cohort voxel QC",
            "source and near-duplicate audit",
            "brain-mask quality review",
            "independent approval before split lock",
        ],
    }
    (out / "aggregate_split_DRAFT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Private manifest path:", private)
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/brainmirror_ixi_split_v1")
    a = p.parse_args()
    run(Path(a.data_root), Path(a.out))
