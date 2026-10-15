#!/usr/bin/env python3
"""Run spatial preprocessing on a single previously-qualified IXI T1 image+mask."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brainmirror.p1.preprocess import preprocess_pair,PreprocessConfig

def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--brain-mask',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--scan-hmac',required=True)
    p.add_argument('--mask-provenance',required=True)
    p.add_argument('--n4',action='store_true')
    a=p.parse_args()
    receipt=preprocess_pair(a.image,a.brain_mask,a.output_dir,a.scan_hmac,
             PreprocessConfig(bias_correction='n4' if a.n4 else 'none'),
             repo_root=Path(__file__).resolve().parents[1],mask_provenance=a.mask_provenance)
    print('PRIVATE receipt generated; status:',receipt['preprocess_gate'])
if __name__=='__main__':cli()
