#!/usr/bin/env python3
"""Only reports P1 PASS if all audit and independent review evidence is present."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brainmirror.p1.closeout import closeout
p=argparse.ArgumentParser()
for flag in ('receipt','output','acquisition','preprocess','duplicates','review'):
    p.add_argument('--'+flag,type=Path,required=flag in ('receipt','output'))
a=p.parse_args()
g=closeout(a.receipt,a.output,acquisition_path=a.acquisition,
           preprocess_path=a.preprocess,duplicate_path=a.duplicates,review_path=a.review)
print('P1_PASS:',g.pass_gate,'blockers:',', '.join(g.blockers))
raise SystemExit(0 if g.pass_gate else 2)
