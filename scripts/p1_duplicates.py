#!/usr/bin/env python3
"""Private coarse near-duplicate screening (always requires human review)."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brainmirror.p1.duplicates import review_candidates

p=argparse.ArgumentParser()
p.add_argument('--private-intake-manifest',type=Path,required=True)
p.add_argument('--data-dir',type=Path,required=True)
p.add_argument('--private-output-dir',type=Path,required=True)
p.add_argument('--threshold',type=float,default=0.995)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
for folder in (a.private_intake_manifest,a.data_dir,a.private_output_dir):
    if folder.resolve().is_relative_to(repo.resolve()):
        p.error('Source and all private outputs must stay outside repository')
manifest=json.loads(a.private_intake_manifest.read_text(encoding='utf8'))
if manifest.get('schema')!='brainmirror.p1.intake.v1':
    p.error('Expected audited P1 private manifest')
result=review_candidates(manifest['records'],a.data_dir,a.private_output_dir,
                         threshold=a.threshold,repo_root=repo)
print(json.dumps(result,indent=2))
