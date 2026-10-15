"""P1 MRI spatial preprocessing; no learned model or medical diagnosis.

A separately supplied, provenance-audited brain mask is REQUIRED. Intensity
thresholds never masquerade as skull stripping. Every output stays private.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    voxel_mm: float = 1.5
    target_shape: tuple[int, int, int] = (144, 160, 144)
    margin_mm: float = 8.0
    clip_percentiles: tuple[float, float] = (0.5, 99.5)
    min_brain_retention: float = 0.995
    bias_correction: str = 'none'  # 'n4' requires SimpleITK

    def validate(self) -> None:
        if not 0.7 <= self.voxel_mm <= 2.5:
            raise ValueError('Invalid voxel spacing')
        if len(self.target_shape) != 3 or any(v < 32 or v > 512 for v in self.target_shape):
            raise ValueError('Invalid target shape')
        if not 0 <= self.margin_mm <= 40:
            raise ValueError('Invalid margin')
        if not 0.9 <= self.min_brain_retention <= 1.0:
            raise ValueError('Invalid retention')
        if self.bias_correction not in ('none', 'n4'):
            raise ValueError('Unrecognized bias correction')


def _n4(data: np.ndarray, mask: np.ndarray, spacing: float) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError('N4 explicitly requested but SimpleITK is unavailable') from exc
    volume = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)).astype(np.float32))
    volume.SetSpacing((spacing, spacing, spacing))
    mask_image = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)).astype(np.uint8))
    mask_image.CopyInformation(volume)
    filt = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = filt.Execute(volume, mask_image)
    return np.transpose(sitk.GetArrayFromImage(corrected), (2, 1, 0))


def _canonical_pair(image_path: Path, mask_path: Path) -> tuple[Any, Any]:
    import nibabel as nib
    from .intake import inspect_nifti
    for path in (image_path, mask_path):
        result = inspect_nifti(path, voxel_qc=True)
        if result['qc_status'] != 'PASS':
            raise ValueError(f'Invalid input NIfTI: {result["qc_reasons"]}')
    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3, rtol=0):
        raise ValueError('Brain mask affine/shape mismatch with input image')
    return nib.as_closest_canonical(image), nib.as_closest_canonical(mask)


def preprocess_pair(image_path: Path, mask_path: Path, output_dir: Path,
                    sample_token: str, config: PreprocessConfig,
                    *, repo_root: Path, mask_provenance: str) -> dict[str, Any]:
    """Create private, mask-verified RAS normalized NIfTI. No use of subject names.

    Output is never scientific PASS by itself. A qualified mask provenance,
    human visual QC and cohort-wide approval remain mandatory.
    """
    import nibabel as nib
    from nibabel.processing import resample_to_output, resample_from_to
    config.validate()
    if not mask_provenance.strip() or len(mask_provenance.strip()) < 6:
        raise ValueError('Independently sourced brain mask provenance required')
    if not sample_token or any(ch not in '0123456789abcdef' for ch in sample_token.lower()) or len(sample_token) < 32:
        raise ValueError('Opaque HMAC sample token expected')
    repo = repo_root.resolve()
    for path in (image_path, mask_path):
        if path.resolve().is_relative_to(repo):
            raise ValueError('MRI input and masks must remain outside repository')
    out = output_dir.resolve()
    if out.is_relative_to(repo) or out.is_relative_to(image_path.resolve().parent) or out.is_relative_to(mask_path.resolve().parent):
        raise ValueError('Private output must be outside repo and input directories')
    image, mask_img = _canonical_pair(image_path, mask_path)
    original_mask = np.asarray(mask_img.dataobj)
    if not np.isin(np.unique(original_mask), [0, 1]).all():
        raise ValueError('Mask must have binary labels 0/1 only')
    if original_mask.sum() < 10000:
        raise ValueError('Brain mask has insufficient foreground')
    # Resample image with cubic interpolation and discrete binary mask with NN.
    resampled = resample_to_output(image, voxel_sizes=(config.voxel_mm,)*3, order=3)
    mask_res = resample_from_to(mask_img, resampled, order=0)
    data = np.asarray(resampled.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask_res.dataobj) > 0.5
    if not np.isfinite(data).all() or np.count_nonzero(mask_data) < 1000:
        raise ValueError('Invalid resampled data or mask')
    # Compute center on brain support. Cropping is fixed-size; any excessive
    # anatomy is a hard error (no silent trim of brain to fit).
    xyz = np.argwhere(mask_data)
    first, last = xyz.min(axis=0), xyz.max(axis=0)
    required = (last-first+1) + 2*np.ceil(config.margin_mm/config.voxel_mm).astype(int)
    target = np.asarray(config.target_shape)
    if np.any(required > target):
        raise ValueError('Target field of view too small for brain + margin')
    center = (first+last)//2
    start = center-target//2
    stop = start+target
    result = np.zeros(tuple(target), dtype=np.float32)
    roi_mask = np.zeros(tuple(target), dtype=np.uint8)
    src_begin = np.maximum(start, 0)
    src_end = np.minimum(stop, data.shape)
    if np.any(src_end <= src_begin):
        raise ValueError('Crop outside resampled volume')
    dst_begin = src_begin-start
    dst_end = dst_begin+(src_end-src_begin)
    ss = tuple(slice(int(a),int(b)) for a,b in zip(src_begin,src_end))
    dd = tuple(slice(int(a),int(b)) for a,b in zip(dst_begin,dst_end))
    result[dd] = data[ss]
    roi_mask[dd] = mask_data[ss].astype(np.uint8)
    retention = float(roi_mask.sum()/mask_data.sum())
    if retention < config.min_brain_retention:
        raise ValueError(f'Brain retention {retention:.6f} below minimum')
    support = roi_mask.astype(bool)
    if config.bias_correction == 'n4':
        result = _n4(result, support, config.voxel_mm)
    inside = result[support]
    if not np.isfinite(inside).all() or float(inside.std()) < 1e-7:
        raise ValueError('Degenerate intensity inside brain')
    lower, upper = np.percentile(inside, config.clip_percentiles)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError('Degenerate robust intensity range')
    result = np.clip(result, lower, upper)
    mean, std = float(result[support].mean()), float(result[support].std())
    if std <= 1e-6:
        raise ValueError('Degenerate normalization std')
    result = np.where(support, (result-mean)/std, 0).astype(np.float32)
    affine = resampled.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(resampled.affine, start)
    image_out = nib.Nifti1Image(result, affine)
    mask_out = nib.Nifti1Image(roi_mask, affine)
    for volume in (image_out, mask_out):
        volume.set_qform(affine, code=1)
        volume.set_sform(affine, code=1)
    # All QC gates execute before any write.
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    prefix = sample_token.lower()
    image_target = out / (prefix+'_image.nii.gz')
    mask_target = out / (prefix+'_mask.nii.gz')
    for p in (image_target, mask_target):
        if p.is_symlink() or p.exists():
            raise ValueError('Preprocessed output exists; never overwrite silently')
    nib.save(image_out, str(image_target))
    nib.save(mask_out, str(mask_target))
    receipt = {'schema':'brainmirror.p1.preprocess.v1','sample_token':prefix,
               'input_sha256':_hash(image_path),'brain_mask_sha256':_hash(mask_path),
               'brain_mask_provenance':mask_provenance,
               'output_image_sha256':_hash(image_target),'output_mask_sha256':_hash(mask_target),
               'target_shape':list(config.target_shape),'spacing_mm':config.voxel_mm,
               'orientation':'RAS','brain_retention':retention,
               'mask_voxels':int(roi_mask.sum()),'normalization_mean':mean,
               'normalization_std':std,'config':asdict(config),
               'visual_qc_reviewed':False,'preprocess_gate':'NEEDS_INDEPENDENT_REVIEW'}
    receipt_path=out/(prefix+'_preprocess_private.json')
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True),encoding='utf-8')
    receipt_path.chmod(0o600)
    return receipt


def _hash(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()
