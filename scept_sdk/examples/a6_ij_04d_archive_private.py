"""04D immutable private source proof: seven complete donor-state+logit shards.

Each selected original patient: 58 true frozen P4 S/J model executions.
Every private release asset SHA256 remotely downloaded/checked before chunk
counts as complete. Idempotent against Kaggle resets. No real MRI pixels added.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_03b_archive_chunk_private import call,commit_once,remote_sha
from a6_ij_04b_archive_private import seal
from a6_ij_02b_measure import sha_file
from a6_ij_04b_measure import canonical,sha
from a6_ij_04d_hook import PLAN,plan_sha,LAYERS,ARMS,SELECTORS,DIRECTIONS
from a6_ij_04d_measure import BASE,OUT,STATUS,prepare

TAG="scept-a6-ij-04d-post-residual-causal-graph-intervention-v1"
ARCHIVED="PASS_04D_PRIVATE_58_PER_PERSON_REAL_FROZEN_GRAPH_AND_DONOR_SHA256"

def subset(num,rows):
 if num not in range(1,8):raise RuntimeError("04D original chunk number invalid")
 sub=rows[(num-1)*5:num*5]
 if len(sub)!=(5 if num<7 else 3):
  raise RuntimeError("04D original 33-person 7-chunk partition differs")
 return sub
def prior_receipt(session,num,rows):
 p=f"{BASE}/chunks/chunk_{num:02d}/archive_receipt.json"
 r=session.get(f"{API}/contents/{p}",timeout=(30,120))
 if r.status_code==404:return None
 r.raise_for_status()
 record=json.loads(private_contents(session,p))
 ids=[x["patient_id"] for x in subset(num,rows)]
 if (record.get("schema")!="scept.a6-ij-04d.chunk-receipt.v1" or
     record.get("status")!=ARCHIVED or
     record.get("chunk")!=num or record.get("patientIds")!=ids or
     record.get("planSHA256")!=plan_sha() or
     record.get("nGenuineDirectGraphForwards")!=58*len(ids) or
     record.get("nOriginalDonorStateNPZ")!=len(ids)):
  raise RuntimeError("04D previously archived source/plan incompatible")
 tag=session.get(f"{API}/releases/tags/{TAG}",timeout=(30,120))
 tag.raise_for_status()
 assets=[v for v in tag.json()["assets"] if v["name"]==record["assetName"]]
 if len(assets)!=1 or assets[0]["size"]!=record["assetBytes"] or (
    assets[0].get("digest") and assets[0]["digest"]!="sha256:"+record["assetSHA256"]):
  raise RuntimeError("04D saved full original private release source asset unverified")
 return record

def verify_local(row,folder,case):
 if (case["patient_id"]!=row["patient_id"] or case["grade"]!=row["grade"] or
     case.get("status")!=STATUS or case.get("planSHA256")!=plan_sha() or
     case.get("realForwardCount")!=58 or
     case["source"]["originalP1NPZSHA256"]!=row["source"]["P1NPZSHA256"] or
     case.get("donorStateTensorCount")!=12 or
     set(case["models"])!={"S","J"}):
  raise RuntimeError("04D patient identity, source, original 58 real forwards mismatch")
 state=folder/"postblock_donor_activations.npz"
 if sha_file(state)!=case["donorStatesNPZSha256"]:
  raise RuntimeError("04D original patient full NPZ true hidden donor tensors SHA256 changed")
 for model in ("S","J"):
  a=case["models"][model]
  if a["nActualFwd"]!=29 or len(a["bidirectionalPostBlockInterventions"])!=12 or \
     set(a["sameSourceSelfPatchShams"])!=set(LAYERS):
   raise RuntimeError("04D original 29 actual model computation records incomplete")
  for arm in ARMS:
   for layer in LAYERS:
    for selector in SELECTORS:
     p=a["bidirectionalPostBlockInterventions"][f"{arm}|{layer}|{selector}"]
     if set(p)!=set(DIRECTIONS):
      raise RuntimeError("04D required opposite patch directions missing")
     for direction,v in p.items():
      if selector=="all" and (v["fullPositiveControlLogitAbsoluteError"] is None or
         v["fullPositiveControlLogitAbsoluteError"]>2e-4):
       raise RuntimeError("04D all-token positive restoration/insertion control BROKEN")
  for val in a["sameSourceSelfPatchShams"].values():
   if val["maxAbsoluteLogitError"]>2e-4:
    raise RuntimeError("04D source-equal sham does not preserve classifier decision")
 return True

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--chunk",type=int,choices=range(1,8),required=True)
 args=p.parse_args()
 s,rows,originalsha,old=prepare()
 prior=prior_receipt(s,args.chunk,rows)
 if prior:
  print(f"04D CHUNK {args.chunk:02d} ALREADY COMPLETE PRIVATE AND REMOTE ASSET SHA VERIFIED",flush=True)
  return
 selected=subset(args.chunk,rows)
 lock_file=OUT/"locked_plan.json"
 if not lock_file.is_file():
  raise RuntimeError("04D source plan not locally saved before archive")
 lock=json.loads(lock_file.read_text())
 if (lock.get("planSHA256")!=plan_sha() or
     lock.get("04BOriginalAnalysisSHA256")!=originalsha or
     lock.get("04COriginalPatientIDs")!=[r["patient_id"] for r in rows]):
  raise RuntimeError("04D original 04B and 04C immutable provenance changed")
 members=[]
 cases=[]
 for item in selected:
  patient=item["patient_id"]
  folder=OUT/"patients"/patient
  path=folder/"case.json"
  if not path.is_file():raise RuntimeError("04D missing patient true model computations "+patient)
  case=json.loads(path.read_text())
  verify_local(item,folder,case)
  cases.append(case)
  members.extend([(f"cases/{patient}.json",path),
                  (f"donors/{patient}.npz",folder/"postblock_donor_activations.npz")])
 if len(cases)!=len(selected) or len(members)!=2*len(selected):
  raise RuntimeError("04D source case/donor original full count incomplete")
 where=OUT/"shards";where.mkdir(parents=True,exist_ok=True)
 assetname=f"scept_04d_original_chunk_{args.chunk:02d}_{len(selected):02d}patients_{58*len(selected):03d}real-forwards.zip"
 assetpath=where/assetname
 member_hashes=seal(members,assetpath)
 size=assetpath.stat().st_size;checksum=sha_file(assetpath)
 route=f"{API}/releases/tags/{TAG}"
 resp=s.get(route,timeout=(30,120))
 if resp.status_code==200:release=resp.json()
 elif resp.status_code==404:
  release=call(s,"POST",f"{API}/releases",json={
   "tag_name":TAG,"target_commitish":"main","prerelease":True,
   "draft":False,"name":"PRIVATE 04D original genuine frozen ViT internal computational interventions",
   "body":"Private patient-derived hidden-state donor tensors, 33 original P0 VAL eligible patients. No medical disease causality, no P5 TEST."})
 else:raise RuntimeError("04D private archival release endpoint unreadable")
 if release.get("prerelease") is not True:
  raise RuntimeError("04D sensitive patient-derived private raw tensor release must be prerelease")
 matched=[a for a in release["assets"] if a["name"]==assetname]
 if len(matched)>1:raise RuntimeError("04D original archive Release duplicate source asset")
 if matched:asset=matched[0]
 else:
  print(f"04D uploading ORIGINAL PRIVATE {args.chunk:02d} donor states and exact intervention outcomes, {size/1048576:.1f}MiB",flush=True)
  with assetpath.open("rb") as fp:
   asset=call(s,"POST",release["upload_url"].split("{",1)[0],
      params={"name":assetname},data=fp,
      headers={"Content-Type":"application/zip","Content-Length":str(size)})
 remote_sha(s,asset,checksum,size)
 report={
  "schema":"scept.a6-ij-04d.chunk.v1",
  "status":"PASS_04D_ALL_ORIGINAL_DIRECT_FROZEN_POSTBLOCK_INTERVENTIONS",
  "chunk":args.chunk,"patientIds":[r["patient_id"] for r in selected],
  "original04BFullAnalysisSHA256":originalsha,
  "lockedSourcePlan":lock,"plan":PLAN,"planSHA256":plan_sha(),
  "patientCases":cases,"realFrozenComputationalForwards":58*len(cases),
  "originalNoTraining":True,"noP5TestOrExternalP6":True}
 record=canonical(report)
 receipt={
  "schema":"scept.a6-ij-04d.chunk-receipt.v1","status":ARCHIVED,
  "chunk":args.chunk,"patientIds":report["patientIds"],
  "planSHA256":plan_sha(),
  "originalCaseBatchFullSHA256":sha(record),
  "nGenuineDirectGraphForwards":58*len(cases),
  "nOriginalDonorStateNPZ":len(cases),
  "assetName":assetname,"assetSHA256":checksum,"assetBytes":size,
  "allOriginalDonorAndCaseFileSHA256":member_hashes,
  "privateRelease":release["html_url"],
  "sourceOriginalPatientMRIAdded":False,"sealedP5":"NOT_ACCESSED"}
 prefix=f"{BASE}/chunks/chunk_{args.chunk:02d}"
 commit_once(s,prefix+"/original_cases.json",record)
 commit_once(s,prefix+"/archive_receipt.json",canonical(receipt))
 print(f"A6-IJ-04D CHUNK {args.chunk:02d} PRIVATE REAL {len(cases)} PATIENTS × 58 FROZEN GRAPH FORWARDS: PASS",flush=True)

if __name__=="__main__":main()
