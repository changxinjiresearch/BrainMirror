"""04B privacy-first operator: source-only preflight, 1 case pilot, chunk, or all.

All --all runs REQUIRE the independently archived original 33-person
input-only magnitude gate; already private-archived chunks skip model work.
Only final --all generates original 33-person patient-clustered science.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_04b_measure import source
from a6_ij_04b_matched_interventions import fingerprint04b
from a6_ij_04b_archive_private import BASE,remote_status

EX=Path(__file__).resolve().parent
PREFLIGHT=f"{BASE}/premodel_input_feasibility.json"
PRESTATUS="PASS_33_OF_33_PATIENTS_4_PAIRED_INPUT_L2_DOSES_PREMODEL"

def run(script,*args):
 cmd=[sys.executable,str(EX/script),*map(str,args)]
 print("A6-IJ-04B RUNNING",script,*args,flush=True)
 subprocess.run(cmd,check=True)
def ensure_preflight(s,selected,lock):
 r=s.get(f"{API}/contents/{PREFLIGHT}",timeout=(30,120))
 if r.status_code==404:
  run("a6_ij_04b_preflight_private.py")
 elif r.status_code!=200:
  raise RuntimeError("04B prior preflight original private record unreadable")
 previous=json.loads(private_contents(s,PREFLIGHT))
 if (previous.get("status")!=PRESTATUS or
     previous.get("planSHA256")!=fingerprint04b() or
     previous.get("sourceLock")!=lock or
     previous.get("nNewMatchedPass")!=33 or
     previous.get("nNewMatchedBlocked")!=0 or
     [r["patient_id"] for r in previous.get("patientRecords",[])]!=
        [r["patient_id"] for r in selected]):
  raise RuntimeError("04B original mandatory 33/33 pre-model input-L2 pilot BLOCKED: do not run models")
 print("A6-IJ-04B COMPLETE 33/33 PRE-MODEL INPUT L2 SOURCE AUDIT: PASS",flush=True)

def main():
 p=argparse.ArgumentParser()
 group=p.add_mutually_exclusive_group(required=True)
 group.add_argument("--preflight-only",action="store_true")
 group.add_argument("--pilot",action="store_true")
 group.add_argument("--chunk",type=int)
 group.add_argument("--all",action="store_true")
 args=p.parse_args()
 if args.chunk is not None and args.chunk not in range(1,8):
  raise RuntimeError("Original predeclared 04B patient chunk must be 1..7")
 s=github()
 eligible,_,lock=source(s)
 ensure_preflight(s,eligible,lock)
 if args.preflight_only:
  print("A6-IJ-04B SOURCE-ONLY PREFLIGHT COMPLETE; ZERO MODEL FORWARDS")
  return
 if args.pilot:
  run("a6_ij_restore_p4.py")
  run("a6_ij_04b_measure.py","--chunk",1,"--pilot")
  print("04B one real patient pilot completed LOCALLY, not yet GitHub-archived full chunk.")
  return
 if args.chunk is not None:
  batches=[args.chunk]
 else:
  batches=list(range(1,8))
 remaining=[n for n in batches if remote_status(s,n,eligible) is None]
 for n in batches:
  if n not in remaining:
   print(f"A6-IJ-04B CHUNK {n:02d} already ORIGINAL PRIVATE SHA VERIFIED; no repeated model calls",flush=True)
 if remaining:
  run("a6_ij_restore_p4.py")
  for n in remaining:
   run("a6_ij_04b_measure.py","--chunk",n)
   run("a6_ij_04b_archive_private.py","--chunk",n)
 if args.all:
  run("a6_ij_04b_finalize.py")
  print("A6-IJ-04B FULL 33-PATIENT 594-REAL-SCEPT MATCHED-INPUT SCIENTIFIC ANALYSIS PIPELINE COMPLETED",flush=True)
 else:
  print("A6-IJ-04B REQUESTED PATIENT CHUNK ARCHIVED; other original batches remain PENDING",flush=True)

if __name__=="__main__":main()
