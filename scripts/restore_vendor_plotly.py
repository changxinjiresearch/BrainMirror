#!/usr/bin/env python3
"""Restore a pinned legacy Plotly browser bundle from a local trusted P0 ZIP.

Only install from an offline, previously audited archive. No internet/CDN used.
"""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

EXPECTED_SHA256 = "e2b0f77ee8156ff9d525852fe4acb7bfe7d85e8a2e1da0523f6d16209b848968"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p0-zip',type=Path,required=True)
    args = p.parse_args()
    with ZipFile(args.p0_zip) as z:
        members = [x for x in z.namelist() if x.endswith('/web/plotly.min.js') or x == 'web/plotly.min.js']
        if len(members) != 1:
            p.error('Expected exactly one Plotly vendor file in the trusted P0 archive')
        data = z.read(members[0])
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        p.error('Plotly vendor hash does not match accepted P0 source. REFUSING installation.')
    dest = Path(__file__).resolve().parents[1] / 'web' / 'plotly.min.js'
    dest.write_bytes(data)
    print(f'Installed trusted offline legacy asset: {dest} ({len(data)} bytes)')

if __name__=='__main__':
    main()
