#!/usr/bin/env python3
"""P1 preparation ONLY: inventory NIfTI headers from a local private directory.

No voxel-data materialization, upload or formal cohort qualification performed.
Input paths and original file names are deliberately excluded from output.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True,
                   help='Private local folder outside source repository')
    p.add_argument('--output', type=Path, default=Path('runs/p1_header_inventory.json'))
    p.add_argument('--source-name', required=True,
                   help='Dataset source as registered in approved provenance')
    p.add_argument('--license-reviewed', action='store_true',
                   help='Explicitly confirms source license has been independently reviewed')
    a = p.parse_args()
    if not a.license_reviewed:
        p.error('Data licensing must be reviewed first. Specify --license-reviewed only when true.')
    if not a.data_dir.is_dir():
        p.error('Data directory does not exist')
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise SystemExit('Missing nibabel or numpy. Install requirements.txt.') from exc
    found = sorted(set(a.data_dir.rglob('*.nii')) | set(a.data_dir.rglob('*.nii.gz')))
    if not found:
        raise SystemExit('No NIfTI data found. No inventory written.')
    records = []
    for file in found:
        record = {'opaque_id': hashlib.sha256(str(file.relative_to(a.data_dir)).encode()).hexdigest()[:20]}
        try:
            img = nib.load(str(file))
            record.update({'shape': list(img.shape),
                           'voxel_spacing_mm': [float(v) for v in img.header.get_zooms()[:3]],
                           'spatial_affine_valid': bool(np.isfinite(img.affine).all()),
                           'is_single_3d_volume': bool(len(img.shape) == 3),
                           'header_qc': 'HEADER_READ_ONLY'})
        except Exception as ex:
            record.update({'header_qc': 'FAILED', 'error_type':type(ex).__name__})
        records.append(record)
    output = {'source':a.source_name,'source_tier':'real_healthy_mri_candidate',
              'science_data_gate':'NOT_YET_APPROVED',
              'header_only':True, 'image_voxel_bytes_read':False,
              'count':len(records),'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'count':len(records),'output':str(a.output),
                      'data_gate':'NOT_YET_APPROVED'},ensure_ascii=False))


if __name__=='__main__':
    main()
