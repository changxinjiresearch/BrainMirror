"""A6-IJ-03B Kaggle operator: one-patient pilot, one chunk or all 11.

Recover frozen source once; resume from immutable private archived receipts.
No GPU or paid API required, but 990 ViT forwards and ~gigabytes of private
ZIP transfer can take significant free Kaggle resource quota.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from a6_ij_03b_measure import github
from a6_ij_03b_archive_chunk_private import remote_status

EX=Path(__file__).resolve().parent

def run(script,*args):
 cmd=[sys.executable,str(EX/script),*map(str,args)]
 print("\n03B EXECUTING:", " ".join(cmd[1:]),flush=True)
 subprocess.run(cmd,check=True)

def main():
 p=argparse.ArgumentParser()
 group=p.add_mutually_exclusive_group(required=True)
 group.add_argument("--pilot",action="store_true")
 group.add_argument("--chunk",type=int)
 group.add_argument("--all",action="store_true")
 args=p.parse_args()
 s=github()
 if args.pilot:
  selected=[1]
 elif args.chunk is not None:
  if not 1<=args.chunk<=11:raise RuntimeError("Select original chunk 1..11")
  selected=[args.chunk]
 else:selected=list(range(1,12))
 pending=[]
 for chunk in selected:
  status=remote_status(s,chunk)
  if status is not None:
   print(f"03B chunk {chunk:02d} already PRIVATELY VERIFIED; no repeated model inference",flush=True)
  else:pending.append(chunk)
 if args.pilot:
  if not pending:
   print("03B chunk 01 already fully archived; pilot unnecessary")
   return
 else:
  if not pending:
   if args.all:run("a6_ij_03b_finalize.py")
   return
 # Restorer uses the same locked original P4 and SHA256-inventoried P1
 # zip and skips exact files which already exist.
 run("a6_ij_restore_p4.py")
 for chunk in pending:
  if args.pilot:
   run("a6_ij_03b_measure.py","--chunk",chunk,"--pilot")
   print("03B ONE-PATIENT PILOT: local observation complete only. "
         "Run --chunk 1 next to complete and permanently private-archive first batch.")
  else:
   run("a6_ij_03b_measure.py","--chunk",chunk)
   run("a6_ij_03b_archive_chunk_private.py","--chunk",chunk)
 if args.all:
  run("a6_ij_03b_finalize.py")
  print("\nA6-IJ-03B FULL 55-PATIENT WORKFLOW COMPLETED")

if __name__=="__main__":main()
