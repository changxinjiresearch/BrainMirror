"""Screen near-identical brain MRI scans without leaking images or identifiers.

This is a *candidate review* algorithm; not a clinical/subject match.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def fingerprint(path:Path, *, target:int=24)->np.ndarray:
    import nibabel as nib
    from scipy.ndimage import zoom
    img=nib.as_closest_canonical(nib.load(str(path)))
    arr=np.asarray(img.dataobj,dtype=np.float32)
    if arr.ndim!=3 or not np.isfinite(arr).all():
        raise ValueError('Invalid MRI voxel array')
    values=arr[arr>0]
    if values.size<10000:
        raise ValueError('Insufficient positive foreground')
    threshold=float(np.percentile(values,15))
    xyz=np.argwhere(arr>threshold)
    a,b=xyz.min(axis=0),xyz.max(axis=0)+1
    cropped=arr[tuple(slice(int(x),int(y)) for x,y in zip(a,b))]
    down=zoom(cropped,tuple(target/d for d in cropped.shape),order=1)
    if down.shape!=(target,)*3:raise ValueError('Unexpected fingerprint shape')
    low,high=np.percentile(down,[1,99]);down=np.clip(down,low,high)
    down=(down-float(down.mean()))/(float(down.std())+1e-9)
    vec=down.astype(np.float32).flatten()
    vec/=np.linalg.norm(vec)+1e-9
    return vec


def review_candidates(records:list[dict], data_root:Path, output_dir:Path,
                      *, threshold:float=0.995, repo_root:Path|None=None)->dict:
    """Private pair candidates; IDs only. Reject publication before human review."""
    if not 0.95<=threshold<=1:raise ValueError('Invalid threshold')
    if repo_root is not None and (output_dir.resolve().is_relative_to(repo_root.resolve()) or data_root.resolve().is_relative_to(repo_root.resolve())):
        raise ValueError('Private data and duplicate report must stay outside the source repo')
    if len(records)<2:raise ValueError('Need 2 or more scan records')
    vectors=[]; tokens=[]; subjects=[]
    for row in records:
        if row['qc_status']!='PASS':continue
        p=(data_root/row['relative_path']).resolve()
        if not p.is_relative_to(data_root.resolve()):raise ValueError('Path traversal')
        vectors.append(fingerprint(p));tokens.append(row['scan_hmac']);subjects.append(row['subject_hmac'])
    if not vectors:raise ValueError('No eligible scans')
    vec=np.stack(vectors)
    similarity=vec@vec.T
    candidates=[]
    for i in range(len(tokens)):
        for j in range(i+1,len(tokens)):
            sim=float(similarity[i,j])
            if sim>=threshold:
                candidates.append({'scan_a':tokens[i],'scan_b':tokens[j],
                                   'different_subject_labels':subjects[i]!=subjects[j],
                                   'coarse_similarity':round(sim,7),'reviewed':False})
    output_dir.mkdir(parents=True,exist_ok=True)
    file=output_dir/'near_duplicate_candidates_PRIVATE.json'
    if file.exists() or file.is_symlink():raise ValueError('Refuse to overwrite review')
    file.write_text(json.dumps({'schema':'brainmirror.p1.duplicate.v1','algorithm':'RAS/crop/24cube/coarse pearson',
                                'threshold':threshold,'screened_scans':len(tokens),
                                'candidate_pairs':candidates,'human_signoff':False,
                                'status':'REQUIRES_REVIEW'},indent=2),encoding='utf8')
    file.chmod(0o600)
    return {'screened_scans':len(tokens),'near_duplicate_candidates':len(candidates),
            'cross_subject_candidates':sum(x['different_subject_labels'] for x in candidates),
            'gate':'BLOCKED_PENDING_REVIEW'}
