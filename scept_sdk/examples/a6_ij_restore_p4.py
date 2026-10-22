"""Restore only A6-IJ P4 frozen checkpoints and ONE authorized validation NPZ.

READ-ONLY GitHub API: no uploads, no P5 test access, no training.
Kaggle Secret GITHUB_TOKEN must have Contents read access to the PRIVATE
research-course-brats2020-ijepa-xai repository and its Release assets.
The P1 ZIP is downloaded in full but ONLY the pinned val member is extracted.
"""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import requests
from kaggle_secrets import UserSecretsClient

REPO = "changxinjiresearch/research-course-brats2020-ijepa-xai"
BASE = "https://api.github.com/repos/" + REPO
ROOT = Path("/kaggle/working")
S_SHA = "2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf"
J_SHA = "719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"
SPLIT_SHA = "1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

try:
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception as exc:
    raise RuntimeError("Enable Kaggle Secret GITHUB_TOKEN before running (do not paste token into notebook).") from exc
if not TOKEN:
    raise RuntimeError("Missing Kaggle Secret GITHUB_TOKEN. Allow notebook access to the existing secret.")
http = requests.Session()
http.headers.update({"Authorization": "Bearer " + TOKEN,
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"})

def get_json(url):
    r = http.get(url, timeout=(30, 180))
    r.raise_for_status()
    return r.json()

def source_bytes(path):
    obj = get_json(f"{BASE}/contents/{path}")
    if obj.get("encoding") != "base64":
        raise RuntimeError("Expected base64 GitHub content: " + path)
    return base64.b64decode(obj["content"])

def one_asset(tag, name):
    rel = get_json(f"{BASE}/releases/tags/{tag}")
    found = [x for x in rel["assets"] if x["name"] == name]
    if len(found) != 1:
        raise RuntimeError("Expected exactly one release asset " + name)
    return found[0]

def download_asset(asset, dest, expected_sha):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == int(asset["size"]) and sha(dest) == expected_sha:
        print("REUSE VERIFIED:", dest, flush=True)
        return dest
    remote_digest = asset.get("digest")
    if remote_digest is not None and remote_digest != "sha256:" + expected_sha:
        raise RuntimeError("GitHub asset digest differs from frozen expected checksum")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    print(f"DOWNLOADING {asset['name']} ({asset['size']/2**20:.1f} MiB)", flush=True)
    with http.get(f"{BASE}/releases/assets/{asset['id']}",
                  headers={"Accept": "application/octet-stream"},
                  stream=True, allow_redirects=True, timeout=(30, 900)) as r:
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type", "").lower():
            raise RuntimeError("GitHub API returned metadata, not binary Release asset")
        with tmp.open("wb") as out:
            for data in r.iter_content(chunk_size=8 * 1024 * 1024):
                if data:
                    out.write(data)
    if tmp.stat().st_size != int(asset["size"]) or sha(tmp) != expected_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded asset failed full SHA256/size check: " + asset["name"])
    os.replace(tmp, dest)
    print("FULL SHA256 VERIFIED:", dest.name, flush=True)
    return dest

# 1. Verify frozen model provenance and restore *best* P4 S/J weights.
s_asset = one_asset("p0-p2-2026-10-13", "model_s_best-2700714f0910.pt")
jmeta = json.loads(source_bytes("results/p3c/model_j/best_model.json"))
if jmeta["best_epoch"] != 16 or jmeta["asset"]["sha256"] != J_SHA:
    raise RuntimeError("P3C best checkpoint manifest mismatch")
j_asset = one_asset("p3c-model-j-2026-10-20", "model_j_best-719df7992216.pt")
if int(j_asset["id"]) != int(jmeta["asset"]["asset_id"]):
    raise RuntimeError("P3C manifest asset ID mismatch")
models = ROOT / "a6_ij_models"
S = download_asset(s_asset, models / "model_s_best.pt", S_SHA)
J = download_asset(j_asset, models / "model_j_best.pt", J_SHA)

# 2. Determine the first lexicographic *val* case from the frozen P0 split.
split_raw = source_bytes("results/p0/patient_split_seed2026.csv")
if hashlib.sha256(split_raw).hexdigest() != SPLIT_SHA:
    raise RuntimeError("Frozen patient split hash mismatch")
split = list(csv.DictReader(io.StringIO(split_raw.decode("utf-8"))))
valid = sorted(row["patient_id"] for row in split if row["split"] == "val")
if len(valid) != 55:
    raise RuntimeError("Locked validation cohort not 55")
patient = valid[0]
manifest = list(csv.DictReader(io.StringIO(source_bytes("results/p1/p1_classification_manifest.csv").decode("utf-8"))))
p1_row = [row for row in manifest if row["patient_id"] == patient]
if len(p1_row) != 1 or p1_row[0]["split"] != "val":
    raise RuntimeError("P1 manifest disagrees with frozen VAL split")
relative = p1_row[0]["npz_relative_path"]
if relative != f"classification_npz/val/{patient}.npz":
    raise RuntimeError("Unexpected P1 cache member; refusing unsupported path")
member = relative.split("classification_npz/", 1)[1]

# 3. Check authoritative ZIP metadata before extracting only this validation case.
inventory = {entry["path"]: entry for entry in json.loads(source_bytes("artifact_integrity/all_files_sha256.json"))}
archive_path = "results/p1/" + relative
if archive_path not in inventory:
    raise RuntimeError("P1 validation NPZ absent from frozen SHA256 inventory")
expected_npz = inventory[archive_path]["sha256"]
p1_release = get_json(f"{BASE}/releases/tags/p0-p2-2026-10-13")
matching = [x for x in p1_release["assets"]
            if x["name"].startswith("p1_classification_npz-") and x["name"].endswith(".zip")]
if len(matching) != 1:
    raise RuntimeError("Expected unique P1 classification NPZ Release archive")
asset = matching[0]
zip_prefix = asset["name"].split("-")[-1][:-4]
if len(zip_prefix) < 12 or any(c not in "0123456789abcdef" for c in zip_prefix):
    raise RuntimeError("Invalid P1 archive SHA prefix")
zip_dest = ROOT / "a6_ij_models" / "p1_classification_npz.zip"
if not (zip_dest.is_file() and zip_dest.stat().st_size == int(asset["size"])
        and sha(zip_dest).startswith(zip_prefix)):
    if zip_dest.exists():
        zip_dest.unlink()
    tmp = zip_dest.with_suffix(".zip.partial")
    tmp.unlink(missing_ok=True)
    print(f"DOWNLOADING P1 ZIP ({asset['size']/2**20:.1f} MiB); extract VAL only", flush=True)
    with http.get(f"{BASE}/releases/assets/{asset['id']}",
                  headers={"Accept": "application/octet-stream"},
                  stream=True, allow_redirects=True, timeout=(30, 900)) as r:
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type", "").lower():
            raise RuntimeError("P1 Release returned JSON instead of ZIP bytes")
        with tmp.open("wb") as f:
            for block in r.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    f.write(block)
    actual_zip_sha = sha(tmp)
    if tmp.stat().st_size != int(asset["size"]) or not actual_zip_sha.startswith(zip_prefix):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("P1 archive size or pinned SHA prefix mismatch")
    remote_zip_digest = asset.get("digest")
    if remote_zip_digest and remote_zip_digest != "sha256:" + actual_zip_sha:
        raise RuntimeError("P1 server full digest mismatch")
    os.replace(tmp, zip_dest)

val_dir = ROOT / "research_course_p1" / "classification_npz" / "val"
val_dir.mkdir(parents=True, exist_ok=True)
target = val_dir / (patient + ".npz")
if not (target.is_file() and sha(target) == expected_npz):
    with ZipFile(zip_dest) as z:
        if member not in z.namelist():
            raise RuntimeError("Declared VAL NPZ missing from Release ZIP")
        with z.open(member) as src, target.with_suffix(".npz.partial").open("wb") as out:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                out.write(block)
    tmp_npz = target.with_suffix(".npz.partial")
    if sha(tmp_npz) != expected_npz:
        tmp_npz.unlink(missing_ok=True)
        raise RuntimeError("Extracted VAL NPZ failed authoritative SHA256")
    os.replace(tmp_npz, target)

print("VAL NPZ FULL SHA256 VERIFIED; sealed test untouched.", flush=True)
report = {"status": "READY_FOR_A6_IJ", "cohort": "val", "patient_id": patient,
          "validation_npz": str(target), "model_s_path": str(S),
          "model_j_path": str(J), "test_set": "NOT_ACCESSED",
          "source": "frozen P4 best checkpoints and SHA256-inventoried P1 VAL"}
report_path = ROOT / "a6_ij_models" / "a6_ij_restore_local.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
print("\nA6-IJ RESTORE: PASS")
print("S =", S)
print("J =", J)
print("VAL =", target)
print("Local-only manifest:", report_path)
