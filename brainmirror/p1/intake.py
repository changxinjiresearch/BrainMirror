"""Privacy-first, fail-closed NIfTI data intake for P1.

Nothing from patient scans should be stored inside the source repository.
The emitted private manifest includes file paths; only an aggregate receipt is
eligible for human review before publishing. No training occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any

IXI_SUBJECT = re.compile(r"^(IXI\d{3})(?:[-_]|$)", flags=re.I)
IXI_T1 = re.compile(r"(?:^|[-_])T1(?:[-_.]|$)", flags=re.I)
VALID_SOURCE = "IXI"
REGISTRY_URL = "https://brain-development.org/ixi-dataset/"
REGISTRY_LICENSE = "CC BY-SA 3.0"
SCHEMA = "brainmirror.p1.intake.v1"


@dataclass(frozen=True)
class SourceDeclaration:
    source: str
    source_url: str
    source_license: str
    healthy_status: str
    modality: str
    reviewed_on: str


def source_declaration() -> SourceDeclaration:
    return SourceDeclaration(
        source=VALID_SOURCE,
        source_url=REGISTRY_URL,
        source_license=REGISTRY_LICENSE,
        healthy_status="source_describes_healthy_cohort_not_individual_diagnosis",
        modality="T1",
        reviewed_on="2026-10-15",
    )


def parse_ixi_identity(filename: str) -> tuple[str, str] | None:
    """Fail closed rather than guess subject IDs for non-standard filenames."""
    if not (filename.lower().endswith(".nii") or filename.lower().endswith(".nii.gz")):
        return None
    found = IXI_SUBJECT.search(filename)
    if found is None or not IXI_T1.search(filename):
        return None
    return (found.group(1).upper(), "T1")


def pseudonym(key: bytes, label: str, value: str) -> str:
    return hmac.new(key, f"{label}:{value}".encode("utf8"), hashlib.sha256).hexdigest()


def stable_split(subject_ids: list[str], *, seed: str = "brainmirror-2026-p1-v1") -> dict[str, str]:
    """Exact deterministic subject-level 80/10/10 partition, at least 10 subjects.

    Re-running with the same IDs+seed yields the same assignments; adding subjects
    changes the partition and requires a new locked split, never silent extension.
    """
    subjects = sorted(set(subject_ids))
    if len(subjects) < 10:
        raise ValueError("At least ten QC-eligible subjects required for split lock")
    order = sorted(subjects, key=lambda s: hashlib.sha256((seed + ":" + s).encode()).hexdigest())
    n = len(order)
    dev = max(1, round(n * 0.1))
    test = max(1, round(n * 0.1))
    train = n - dev - test
    if train < 1:
        raise ValueError("No training subjects remain")
    return {s: "train" if idx < train else "dev" if idx < train + dev else "test"
            for idx, s in enumerate(order)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_ok(mat: Any) -> bool:
    import numpy as np
    array = np.asarray(mat, dtype=float)
    return bool(array.shape == (4, 4)
                and np.isfinite(array).all()
                and abs(np.linalg.det(array[:3, :3])) > 1e-6)


def inspect_nifti(path: Path, *, voxel_qc: bool = False) -> dict[str, Any]:
    """Real header check through nibabel. By default NO image voxels are read."""
    import numpy as np
    import nibabel as nib

    img = nib.load(str(path))
    h = img.header
    shape = tuple(int(v) for v in img.shape)
    zoom = tuple(float(v) for v in h.get_zooms()[:3])
    qmat, qcode = img.get_qform(coded=True)
    smat, scode = img.get_sform(coded=True)
    orientation = nib.aff2axcodes(img.affine) if _matrix_ok(img.affine) else ("?", "?", "?")
    reasons: list[str] = []
    if len(shape) != 3:
        reasons.append("NOT_3D_SINGLE_VOLUME")
    if any(size < 64 or size > 1024 for size in shape[:3]):
        reasons.append("IMPLAUSIBLE_DIMENSION")
    if len(zoom) != 3 or any(not np.isfinite(z) or not 0.3 <= z <= 3.0 for z in zoom):
        reasons.append("IMPLAUSIBLE_SPACING")
    if not _matrix_ok(img.affine):
        reasons.append("INVALID_AFFINE")
    if int(qcode) == 0 and int(scode) == 0:
        reasons.append("MISSING_SPATIAL_TRANSFORM")
    if int(qcode) > 0 and not _matrix_ok(qmat):
        reasons.append("INVALID_QFORM")
    if int(scode) > 0 and not _matrix_ok(smat):
        reasons.append("INVALID_SFORM")
    if int(qcode) > 0 and int(scode) > 0 and _matrix_ok(qmat) and _matrix_ok(smat):
        if not np.allclose(qmat, smat, atol=0.5, rtol=0):
            reasons.append("QFORM_SFORM_CONFLICT")
    summary: dict[str, Any] = {
        "shape": list(shape),
        "spacing_mm": list(zoom),
        "orientation": "".join(orientation),
        "qform_code": int(qcode),
        "sform_code": int(scode),
        "affine_det": float(np.linalg.det(img.affine[:3, :3])) if _matrix_ok(img.affine) else None,
        "voxel_qc_executed": voxel_qc,
    }
    if voxel_qc and not reasons:
        data = np.asarray(img.dataobj, dtype=np.float32)
        finite = np.isfinite(data)
        foreground = finite & (np.abs(data) > 1e-6)
        summary["finite_fraction"] = float(np.mean(finite))
        summary["nonzero_fraction"] = float(np.mean(foreground))
        summary["foreground_p01_p99"] = (
            [float(v) for v in np.percentile(data[foreground], [1, 99])]
            if foreground.any() else None
        )
        if not finite.all():
            reasons.append("NONFINITE_VOXELS")
        if foreground.mean() < 0.005:
            reasons.append("NEAR_EMPTY_IMAGE")
    summary["qc_reasons"] = reasons
    summary["qc_status"] = "PASS" if not reasons else "FAIL"
    return summary


def paths_safe(data_root: Path, output: Path, repo_root: Path) -> tuple[Path, Path]:
    dr, op, rr = data_root.resolve(strict=True), output.resolve(), repo_root.resolve()
    if not dr.is_dir():
        raise ValueError("--data-dir must be a directory")
    if dr.is_relative_to(rr) or op.is_relative_to(rr):
        raise ValueError("Real MRI and per-subject manifests must be outside the Git repository")
    if op.is_relative_to(dr):
        raise ValueError("Output directory must not be nested inside input MRI directory")
    return dr, op


def inventory(
    data_dir: Path, output_dir: Path, *, signing_key: bytes,
    accepted_terms: bool, voxel_qc: bool, repo_root: Path,
    split_seed: str = "brainmirror-2026-p1-v1",
) -> dict[str, Any]:
    if not accepted_terms:
        raise ValueError("Explicit acknowledgement of IXI license is required")
    if len(signing_key) < 32:
        raise ValueError("Use an HMAC secret containing at least 32 bytes")
    dr, out = paths_safe(data_dir, output_dir, repo_root)
    candidates = sorted(p for p in dr.rglob("*")
                        if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")))
    if not candidates:
        raise ValueError("No NIfTI files found; P1 qualification remains blocked")
    private_rows: list[dict[str, Any]] = []
    hashes_by_content: dict[str, str] = {}
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(dr):
            raise ValueError("Symlinks or path traversal in data directory are forbidden")
        rel = path.relative_to(dr).as_posix()
        parsed = parse_ixi_identity(path.name)
        sid = pseudonym(signing_key, "scan", rel)
        row: dict[str, Any] = {"scan_hmac": sid, "relative_path": rel,
                              "subject_hmac": None, "modality": None,
                              "qc_status": "FAIL", "qc_reasons": []}
        if parsed is None:
            row["qc_reasons"].append("UNRECOGNIZED_IXI_T1_FILENAME")
        else:
            subject_key, modality = parsed
            row["subject_hmac"] = pseudonym(signing_key, "subject", subject_key)
            row["modality"] = modality
            try:
                row.update(inspect_nifti(path, voxel_qc=voxel_qc))
            except Exception as exc:
                row["qc_status"] = "FAIL"
                row["qc_reasons"] = ["HEADER_UNREADABLE:" + type(exc).__name__]
        row["file_sha256"] = sha256_file(path)
        prior_subject = hashes_by_content.get(row["file_sha256"])
        if prior_subject is not None and row["subject_hmac"] != prior_subject:
            row["qc_reasons"].append("DUPLICATE_BYTES_ACROSS_SUBJECTS")
            row["qc_status"] = "FAIL"
            # Also invalidate the original copy, never silently assign the duplicate.
            for old in private_rows:
                if old["file_sha256"] == row["file_sha256"]:
                    old["qc_reasons"].append("DUPLICATE_BYTES_ACROSS_SUBJECTS")
                    old["qc_status"] = "FAIL"
        else:
            hashes_by_content[row["file_sha256"]] = row["subject_hmac"]
        private_rows.append(row)
    # P1 uses only healthy IXI T1 candidates; duplicate scans from same subject
    # are allowed but never cross partitions.
    good = [r for r in private_rows if r["qc_status"] == "PASS"]
    eligible_subjects = sorted({r["subject_hmac"] for r in good})
    split = stable_split(eligible_subjects, seed=split_seed) if len(eligible_subjects) >= 10 else {}
    for row in private_rows:
        row["split"] = split.get(row["subject_hmac"])
    reasons = Counter(reason for r in private_rows for reason in r["qc_reasons"])
    split_subject_counts = Counter(split.values())
    source = asdict(source_declaration())
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "data_tier": "REAL_MRI_CANDIDATE_NOT_SCIENTIFICALLY_VALIDATED",
        "source_terms_acknowledged_by_operator": bool(accepted_terms),
        "files_seen": len(private_rows),
        "scans_header_pass": len(good),
        "scans_failed": len(private_rows) - len(good),
        "eligible_subjects": len(eligible_subjects),
        "split_subject_counts": dict(split_subject_counts),
        "failure_counts": dict(sorted(reasons.items())),
        "voxel_qc_executed_for_passing_scans": bool(voxel_qc),
        "data_gate": ("PRELIMINARY_SPLIT_READY_NOT_LOCKED" if split and voxel_qc and len(good)==len(private_rows)
                      else "BLOCKED_PENDING_COMPLETION_AND_REVIEW"),
        "split_locked": False,
        "preprocessing_complete": False,
        "real_mri_training_started": False,
        "privacy": "public_summary_contains_no_subject_identifiers_or_paths",
    }
    out.mkdir(parents=True, exist_ok=True)
    try:
        out.chmod(0o700)
    except OSError:
        pass
    private = {"schema": SCHEMA, "summary": summary,
               "split_seed": split_seed, "records": private_rows}
    manifest = out / "private_scans.json"
    receipt = out / "aggregate_receipt.json"
    for path, data in [(manifest, private), (receipt, summary)]:
        if path.is_symlink():
            raise ValueError("Output symlink is not allowed")
        with path.open("w", encoding="utf8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return summary


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P1: IXI T1 MRI private intake (NO TRAINING)")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="PRIVATE directory OUTSIDE Git and outside MRI input")
    ap.add_argument("--accept-ixi-terms", action="store_true",
                    help="Operator confirms reviewing IXI CC BY-SA 3.0 terms")
    ap.add_argument("--voxel-qc", action="store_true",
                    help="Explicitly read voxel arrays for finite/foreground QA")
    ap.add_argument("--split-seed", default="brainmirror-2026-p1-v1")
    args = ap.parse_args(argv)
    key = os.environ.get("BRAINMIRROR_P1_HMAC_KEY", "").encode()
    repo = Path(__file__).resolve().parents[2]
    try:
        result = inventory(args.data_dir, args.output_dir, signing_key=key,
                           accepted_terms=args.accept_ixi_terms,
                           voxel_qc=args.voxel_qc, repo_root=repo,
                           split_seed=args.split_seed)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
