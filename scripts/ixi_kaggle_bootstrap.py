"""IXI Kaggle P1 bootstrap: real MRI intake QC, not an approved atlas."""
from pathlib import Path
from collections import Counter, defaultdict
import json, re, hashlib, argparse
import numpy as np

ID_RE = re.compile(r"^IXI(\d{3})-([A-Za-z]+)-", re.I)

def discover(root):
    root=Path(root)
    datasets=[p for p in root.rglob("brainmirror-ixi-t1-raw") if p.is_dir()] if root.name!="brainmirror-ixi-t1-raw" else [root]
    if len(datasets)!=1: raise RuntimeError(f"Expected one IXI dataset mount, got {len(datasets)}")
    source=datasets[0]
    files=sorted(p for p in source.rglob("*") if p.is_file() and p.name.lower().endswith((".nii",".nii.gz")))
    if not files: raise RuntimeError("No NIfTI files found")
    keys=[]
    for p in files:
        m=ID_RE.match(p.name)
        if not m: raise RuntimeError("Unrecognized source filename; stop and inspect private mount")
        keys.append(m.group(1))
    if len(set(keys))!=len(keys): raise RuntimeError("Duplicate subject IDs need review")
    return source,files

def audit(files,voxel_limit=6):
    import nibabel as nib
    qc=[]; problems=[]
    for i,p in enumerate(files):
        try:
            img=nib.load(str(p))
            affine=np.asarray(img.affine)
            zoom=img.header.get_zooms()[:3]
            reason=[]
            if len(img.shape)!=3: reason.append("not_3D")
            if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-8: reason.append("invalid_affine")
            if any(not np.isfinite(s) or not .3<=s<=5.0 for s in zoom): reason.append("invalid_spacing")
            if not (int(img.header["qform_code"]) or int(img.header["sform_code"])): reason.append("missing_transform_code")
            if i<voxel_limit and len(img.shape)==3:
                data=np.asanyarray(img.dataobj,dtype=np.float32)
                if not np.isfinite(data).all(): reason.append("nonfinite_voxel")
                if float(np.nanmax(data)-np.nanmin(data))<1e-6: reason.append("constant_voxel")
            qc.append({"shape":list(img.shape),"spacing":list(map(float,zoom)),"orientation":"".join(nib.aff2axcodes(affine)),"status":"REVIEW" if reason else "HEADER_OK","reasons":reason})
        except Exception as e:
            qc.append({"status":"REVIEW","reasons":[type(e).__name__]})
        if qc[-1]["status"]=="REVIEW":problems.append(i)
    return qc,problems

def run(root="/kaggle/input",out="/kaggle/working/brainmirror_ixi_bootstrap"):
    source,files=discover(root)
    qc,problems=audit(files)
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Use a fresh output directory for reproducibility")
    out.mkdir(parents=True,exist_ok=True)
    private=[]
    for p,q in zip(files,qc):
        m=ID_RE.match(p.name)
        private.append({"source_file":str(p),"source_subject_id":m.group(1),"site":m.group(2),"qc":q})
    (out/"PRIVATE_subject_qc.json").write_text(json.dumps(private,indent=2))
    summary={"schema":"brainmirror.ixi.bootstrap.v1","mounted_files":len(files),
     "unique_subject_ids":len(files),"headers_ok":sum(q["status"]=="HEADER_OK" for q in qc),
     "requires_review":len(problems),"site_counts":dict(Counter(x["site"] for x in private)),
     "shape_counts":dict(Counter(str(x.get("shape")) for x in qc)),
     "axis_codes":dict(Counter(x.get("orientation","unreadable") for x in qc)),
     "voxel_qc_scope":"first six files only","data_source":"IXI-T1 / PRIVATE KAGGLE",
     "population_template_generated":False,"brain_masks_reviewed":False,
     "split_locked":False,"P1_PASS":False,"P2_TRAINING_AUTHORIZED":False}
    (out/"aggregate_qc.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print("PRIVATE outputs:",out)
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--input",default="/kaggle/input");ap.add_argument("--out",default="/kaggle/working/brainmirror_ixi_bootstrap");ns=ap.parse_args()
    run(ns.input,ns.out)
