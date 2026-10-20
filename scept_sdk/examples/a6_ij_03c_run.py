"""A6-IJ-03C — fail-closed runner, source-only in existing Kaggle runtime.

No fresh MRI access or ViT forward: downloads original PRIVATE archived
SCEPT outputs in chunks to reconstruct honest 10-slice RMS baseline, then
performs 55-person cross-validated observational analysis and archive.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from a6_ij_03b_measure import github
from a6_ij_03c_extract_rms_private import BASE,API

HERE=Path(__file__).resolve().parent
def launch(filename,*args):
 cmd=[sys.executable,str(HERE/filename),*map(str,args)]
 print("\n03C EXECUTING:",filename,*args,flush=True)
 subprocess.run(cmd,check=True)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--rms-only",action="store_true",
     help="Run resumable private 11-chunk original raw activation RMS only")
 args=parser.parse_args()
 print("03C original complete SCEPT 03B source and permission preflight",flush=True)
 s=github()
 # Mandatory original 03B qualified private final before consuming raw ZIPs.
 r=s.get(f"{API}/contents/results/scept_a6_ij/03b_full_slice_pairing/final/archive_receipt.json",
         timeout=(30,120))
 r.raise_for_status()
 launch("a6_ij_03c_extract_rms_private.py","--all")
 if args.rms_only:
  print("A6-IJ-03C ORIGINAL RMS: 11/11 PRIVATE CHUNKS COMPLETE; analysis pending",flush=True)
  return
 launch("a6_ij_03c_analyse_private.py")
 launch("a6_ij_03c_archive_private.py")
 print("A6-IJ-03C FULL LOCKED ANALYSIS PIPELINE COMPLETED",flush=True)

if __name__=="__main__":main()
