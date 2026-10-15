"""3D MRI loading and a deliberately limited, QC-measured local warp.

Scientific caveat: QC metrics do NOT establish anatomical plausibility.
No claims of a valid disease simulation are made for this v0.1 warp.
"""
from io import BytesIO
import gzip
import numpy as np
from scipy.ndimage import zoom, gaussian_filter, distance_transform_edt, map_coordinates
from skimage.measure import marching_cubes

SIDE = 32


def phantom(seed=1):
    """Synthetic ellipsoid test fixture. Never clinical evidence."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[-1:1:complex(SIDE), -1:1:complex(SIDE), -1:1:complex(SIDE)]
    brain = (xx/0.85)**2 + (yy/0.72)**2 + (zz/0.90)**2 < 1
    # Two synthetic ventricular ellipsoids. Artificial on purpose.
    scale = rng.uniform(0.85, 1.16)
    roi = (((xx - 0.15)/(.12*scale))**2 + (yy/.23)**2 + (zz/.37)**2 < 1) | \
          (((xx + 0.15)/(.12*scale))**2 + (yy/.23)**2 + (zz/.37)**2 < 1)
    smooth_brain = gaussian_filter(brain.astype(float), 1.0)
    x = np.clip(.06 + .75 * smooth_brain - .57 * roi.astype(float)
                + .045 * rng.standard_normal(brain.shape), 0, 1).astype(np.float32)
    return x, roi.astype(bool)


def normalize_volume(volume):
    a = np.asarray(volume, dtype=np.float32)
    if a.ndim != 3 or any(n < 8 for n in a.shape):
        raise ValueError('MRI must be a 3D volume with each dimension >= 8')
    if not np.isfinite(a).all():
        raise ValueError('MRI contains NaN or infinite values')
    a = zoom(a, [SIDE/s for s in a.shape], order=1)
    lo, hi = np.percentile(a, [1, 99])
    if hi - lo < 1e-7:
        raise ValueError('MRI is constant intensity')
    return np.clip((a - lo)/(hi-lo), 0, 1).astype(np.float32)


def normalize_mask(mask):
    a = np.asarray(mask)
    if a.ndim != 3:
        raise ValueError('ROI mask must be a 3D volume')
    resampled = zoom(a.astype(np.float32), [SIDE/s for s in a.shape], order=0) > .5
    if resampled.sum() < 16 or resampled.sum() > resampled.size * .6:
        raise ValueError('ROI mask must contain >=16 voxels and <=60% of volume')
    return resampled


def parse_nifti_bytes(data, name):
    """Nibabel is only needed for user-supplied NIfTI; phantom tests run without it."""
    if name.lower().endswith('.npy'):
        return np.load(BytesIO(data), allow_pickle=False), None
    import nibabel as nib
    # Support uncompressed or .gz NIfTI without writing patient data to disk.
    raw = gzip.decompress(data) if name.lower().endswith('.nii.gz') else data
    img = nib.Nifti1Image.from_bytes(raw)
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError('Please upload a single 3D MRI, not a multi-volume series')
    return arr, img.affine


def prepare_pair(volume_bytes, volume_name, mask_bytes, mask_name):
    v, va = parse_nifti_bytes(volume_bytes, volume_name)
    m, ma = parse_nifti_bytes(mask_bytes, mask_name)
    if v.shape != m.shape:
        raise ValueError('MRI and ROI mask shapes differ. Align/resample before uploading.')
    if (va is None) != (ma is None):
        raise ValueError('MRI and ROI mask must use the same file format for alignment checks')
    if va is not None and not np.allclose(va, ma, atol=1e-3):
        raise ValueError('MRI and ROI mask affines do not align; register first.')
    return normalize_volume(v), normalize_mask(m)


def local_expand(volume, mask, strength_voxels):
    """Local signed-distance-normal displacement (approximate, research demo).

    Inverse-sample the image and binary ROI with identical coordinates.
    At strength=0 yields exact identity. Returns QC statistics.
    """
    v = np.asarray(volume, np.float32)
    m = np.asarray(mask, bool)
    if v.shape != (SIDE,)*3 or m.shape != v.shape:
        raise ValueError('Expected matching 32^3 volumes')
    if not (0 <= strength_voxels <= 2.5):
        raise ValueError('Expansion magnitude must be 0..2.5 voxels')
    if strength_voxels == 0:
        return v.copy(), m.copy(), {'target_volume_before': int(m.sum()),
                                    'target_volume_after': int(m.sum()),
                                    'changed_voxel_fraction': 0.,
                                    'non_target_mean_abs_change': 0.,
                                    'warp_limitations': 'Synthetic deformation; no clinical plausibility validation'}
    signed_dist = distance_transform_edt(~m) - distance_transform_edt(m)
    # Smoothed normals reduce discontinuities at the ROI boundary.
    dz, dy, dx = np.gradient(gaussian_filter(signed_dist, 1.0))
    norm = np.sqrt(dz*dz + dy*dy + dx*dx) + 1e-8
    # Gaussian falloff around ROI surface. Sampling opposite the outward normal
    # expands the object's boundary for positive displacement.
    decay = np.exp(-0.5 * (signed_dist / 3.0)**2)
    displacement = float(strength_voxels) * decay
    coords = np.indices(v.shape, dtype=np.float32)
    coords[0] -= displacement * dz / norm
    coords[1] -= displacement * dy / norm
    coords[2] -= displacement * dx / norm
    warped = map_coordinates(v, coords, order=1, mode='nearest').astype(np.float32)
    warped_mask = map_coordinates(m.astype(np.float32), coords, order=0, mode='nearest') > .5
    outside = distance_transform_edt(~m) > 5
    return warped, warped_mask, {
        'target_volume_before': int(m.sum()),
        'target_volume_after': int(warped_mask.sum()),
        'changed_voxel_fraction': float(np.mean(np.abs(warped-v) > .02)),
        'non_target_mean_abs_change': float(np.mean(np.abs(warped-v)[outside])) if outside.any() else 0.,
        'warp_limitations': 'Synthetic displacement, not clinically validated; check spatial anatomy externally'
    }


def to_mesh(mask, max_triangles=12000):
    """Return decimated-ish marching cubes mesh for browser 3D plotting."""
    from scipy.ndimage import binary_dilation
    m = np.asarray(mask, bool)
    if m.all() or not m.any():
        return {'x': [], 'y': [], 'z': [], 'i': [], 'j': [], 'k': []}
    vertices, faces, _, _ = marching_cubes(m.astype(np.float32), level=.5, step_size=1)
    if len(faces) > max_triangles:
        faces = faces[::int(np.ceil(len(faces)/max_triangles))]
    return {'x': vertices[:,2].round(3).tolist(),
            'y': vertices[:,1].round(3).tolist(),
            'z': vertices[:,0].round(3).tolist(),
            'i': faces[:,0].tolist(), 'j': faces[:,1].tolist(), 'k': faces[:,2].tolist()}


def brain_surface(volume):
    # Visualization-only intensity segmentation; not a clinical brain mask.
    threshold = max(.15, float(np.percentile(volume, 65)) * .55)
    return to_mesh(volume > threshold)
