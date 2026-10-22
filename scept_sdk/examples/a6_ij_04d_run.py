"""04D safe Kaggle operator --pilot / --chunk N / --all / --status.

Private original 33 x2 frozen P4 models, 1914 genuine computational forwards,
no model training. Already SHA-archived chunks are skipped on restart.
Completion is ONLY true after 7 remotely verified shards + final analysis.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from a6_ij_03b_measure import github
from a6_ij_04d_measure import prepare
from a6_ij_04d_archive_private import prior_receipt

EX=Path(__file__).resolve().parent
def launch(script,*args):
 cmd=[sys.executable,str(EX/script),*map(str,args)]
 print("04D ACTUAL KAGGLE EXPERIMENT:",script,*args,flush=True)
 subprocess.run(cmd,check=True)
def main():
 p=argparse.ArgumentParser()
 opt=p.add_mutually_exclusive_group(required=True)
 opt.add_argument("--pilot",action="store_true")
 opt.add_argument("--chunk",type=int,choices=range(1,8))
 opt.add_argument("--all",action="store_true")
 opt.add_argument("--status",action="store_true")
 a=p.parse_args()
 s,rows,_,_=prepare()
 completed=[i for i in range(1,8) if prior_receipt(s,i,rows) is not None]
 print("04D SHA-VERIFIED original private complete chunks:",completed,flush=True)
 if a.status:
  print("33 patient cohort original source verified; no new model training or inference")
  return
 if a.pilot:
  if 1 in completed:
   print("04D pilot original first patient already contained in previously verified complete chunk 01")
   return
  launch("a6_ij_restore_p4.py")
  launch("a6_ij_04d_measure.py","--chunk",1,"--pilot")
  print("04D ONE-PATIENT GRAPH-PATCH PILOT COMPLETE LOCALLY; first chunk must be archived next")
  return
 chunks=([a.chunk] if a.chunk is not None else list(range(1,8)))
 missing=[i for i in chunks if i not in completed]
 if missing:
  launch("a6_ij_restore_p4.py")
  for num in missing:
   launch("a6_ij_04d_measure.py","--chunk",num)
   launch("a6_ij_04d_archive_private.py","--chunk",num)
 if a.all:
  launch("a6_ij_04d_finalize_private.py")
  print("04D ALL 33 PATIENT ORIGINAL 1914-REAL-FORWARD GRAPH INTERVENTION AND PRIVATE AUDIT COMPLETE")
 else:
  print("04D SELECTED ORIGINAL PRIVATE CHUNK COMPLETE; others may remain pending")
if __name__=="__main__":main()
