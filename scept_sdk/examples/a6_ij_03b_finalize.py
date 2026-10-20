"""A6-IJ-03B: finalize all 11 permanently PRIVATE chunks (1100 exact pairs).

Patient-level (not slice-independent) paired analysis, two-thousand seeded
cluster bootstrap replicates, and descriptive first-slice vs all-10-slice
comparisons. Absolutely no new model or Grad-CAM inference.
"""
import base64
import csv
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from a6_ij_03b_measure import (
 PLAN,psha,sha,github,API,PREFIX,RELEASE_TAG,OUT,private_contents,LAYERS
)
from a6_ij_03b_archive_chunk_private import remote_status,commit_once

def interval(values,rng):
 a=np.asarray(values,dtype=np.float64)
 if len(a)!=55 or not np.isfinite(a).all():
  raise RuntimeError("Must bootstrap 55 paired patients, not 550 pseudo-independent slices")
 repl=rng.integers(0,55,size=(2000,55))
 draws=a[repl].mean(axis=1)
 return {"nPatients":55,"mean":float(a.mean()),"median":float(np.median(a)),
         "bootstrapMean95":list(map(float,np.quantile(draws,[0.025,.975])))}
def correlation(x,y):
 v=float(spearmanr(x,y).statistic)
 return {"nPatients":55,"spearmanRho":v if np.isfinite(v) else None,
         "status":"EXPLORATORY_NONCAUSAL" if np.isfinite(v) else "UNDEFINED_CONSTANT"}
def fetch_json(s,path):
 return json.loads(private_contents(s,path))
def main():
 s=github()
 source=fetch_json(s,"results/scept_a6_ij/03a_exact_first_slice_pairing/analysis.json")
 if source.get("status")!="PASS_STRICTLY_PAIRED_55_FIRST_SLICE_VAL" or \
     source["cohort"]["nMatchedModelSliceRecords"]!=110:
  raise RuntimeError("Authenticated source 03A 110 strictly paired first-slice rows unavailable")
 original={(r["patient_id"],r["model"]):r for r in source["exactMatchedPatientRows"]}
 if len(original)!=110:raise RuntimeError("Source first-slice matching inconsistent")
 p0=private_contents(s,"results/p0/patient_split_seed2026.csv")
 if sha(p0)!=PLAN["P0SplitSHA256"]:
  raise RuntimeError("Final 03B original P0 validation split changed")
 val=sorted((r for r in csv.DictReader(io.StringIO(p0.decode("utf-8-sig"))) if r["split"]=="val"),
            key=lambda x:x["patient_id"])
 ids=[r["patient_id"] for r in val]
 grade={r["patient_id"]:r["grade"] for r in val}
 if len(ids)!=55 or sum(v=="HGG" for v in grade.values())!=44 or \
     sum(v=="LGG" for v in grade.values())!=11:
  raise RuntimeError("Final study patient cohort changed")
 release=fetch_json(s,f"releases/tags/{RELEASE_TAG}") if False else None
 resp=s.get(f"{API}/releases/tags/{RELEASE_TAG}",timeout=(30,120))
 resp.raise_for_status()
 release=resp.json()
 if release.get("prerelease") is not True:raise RuntimeError("Private release must remain prerelease")
 report_files=[]
 all_rows={}
 assembled=[]
 original_new_sha=set()
 n_new=0;n_reused=0
 chunk_metadata=[]
 for n in range(1,12):
  receipt=remote_status(s,n)
  if receipt is None:raise RuntimeError(f"03B private source chunk {n:02d}/11 not archived")
  expected_ids=ids[(n-1)*5:n*5]
  if receipt["patientIds"]!=expected_ids:
   raise RuntimeError("03B private chunk patient distribution incorrect")
  matches=[a for a in release["assets"] if a["name"]==receipt["assetName"]]
  if len(matches)!=1 or matches[0]["size"]!=receipt["assetBytes"] or \
     (matches[0].get("digest") and
      matches[0]["digest"]!="sha256:"+receipt["assetSha256"]):
   raise RuntimeError("03B published raw source shard metadata differs: "+str(n))
  record_path=f"{PREFIX}/chunks/chunk_{n:02d}/paired_rows.json"
  raw=private_contents(s,record_path)
  if sha(raw)!=receipt["originalReportSHA256"]:
   raise RuntimeError("03B original patient case report bytes mismatch")
  chunk=json.loads(raw)
  if (chunk.get("status")!="PASS_5_PATIENTS_100_SOURCE_PAIRED_MODEL_SLICES"
       or chunk["chunk"]!=n or chunk["planSHA256"]!=psha()
       or chunk["patientIds"]!=expected_ids or chunk["recordCount"]!=100
       or chunk["newSCEPTForwards"]!=90 or chunk["reused03aCount"]!=10
       or len(chunk["pairedPatientRows"])!=5):
   raise RuntimeError("Not a complete original authenticated private patient chunk")
  stem=f"results/p4b/chunks/chunk_{n:02d}"
  cam_source=private_contents(s,stem+"_metrics.csv")
  if sha(cam_source)!=chunk["originalSources"]["p4bCsvSHA256"]:
   raise RuntimeError("Original frozen Grad-CAM source chunk SHA changed")
  orig_p4={}
  for r in csv.DictReader(io.StringIO(cam_source.decode("utf-8-sig"))):
   key=(r["patient_id"],int(r["slice_rank"]),r["model"])
   if key in orig_p4:raise RuntimeError("Original P4B source CSV duplicated")
   orig_p4[key]=r
  if len(orig_p4)!=100:raise RuntimeError("Original P4B source rows !=100")
  for case in chunk["pairedPatientRows"]:
   patient=case["patient_id"]
   if patient not in expected_ids or patient in all_rows or case["grade"]!=grade[patient] or \
      case["planSHA256"]!=psha() or len(case["slicePairs"])!=10:
    raise RuntimeError("Original 03B matched patient duplicated/incomplete")
   all_rows[patient]=case
   for rank,pair in enumerate(case["slicePairs"],1):
    if (pair["patient_id"]!=patient or pair["slice_rank"]!=rank
         or pair["grade"]!=grade[patient] or
         set(pair["models"])!={"S","J"} or
         set(pair["crossModelCKA"])!=set(LAYERS)):
     raise RuntimeError("Original paired patient/model/ten-slice key mismatched")
    for model in ("S","J"):
     row=pair["models"][model]
     src=orig_p4.get((patient,rank,model))
     if src is None or int(src["axial_index"])!=pair["axial_index"]:
      raise RuntimeError("P4B original axial patient-rank join altered")
     for metric,rawcol in (("tap","tap"),("iou10","iou10"),
                           ("pHgg","p_hgg"),("referenceFraction","reference_fraction")):
      if abs(float(row["GradCAM"][metric])-float(src[rawcol]))>1e-12:
       raise RuntimeError("Stored Grad-CAM metric not original source bytes")
     if abs(float(row["SCEPT"]["pHgg"])-float(src["p_hgg"]))>PLAN["predictionMaxAbsoluteDifference"]:
      raise RuntimeError("SCEPT same-patient probability verification exceeded original tolerance")
     if set(row["SCEPT"]["layers"])!=set(LAYERS):
      raise RuntimeError("Stored SCEPT all three layer metrics incomplete")
     if rank==1:
      first=original.get((patient,model))
      if first is None or row["SCEPT"]["runSource"]!="REUSED_02B_WITH_03A_SHA_MATCHED_PRIVATE_RAW_ZIP":
       raise RuntimeError("Original 02B 110 observations not reused")
      if abs(row["SCEPT"]["pHgg"]-first["pHggSCEPT"])>1e-12:
       raise RuntimeError("Source 03A reused SCEPT slice changed")
      n_reused+=1
     else:
      if row["SCEPT"]["runSource"]!="NEW_03B_GENUINE_SCEPT_FORWARD":
       raise RuntimeError("Additional 03B slice is not actual original forward")
      zipname=row["SCEPT"]["originalZip"]["name"]
      if zipname in original_new_sha:
       raise RuntimeError("Duplicate new actual source ZIP filename")
      original_new_sha.add(zipname)
      n_new+=1
     assembled.append(row)
  chunk_metadata.append({"chunk":n,"patientIds":expected_ids,
    "originalReportSha256":receipt["originalReportSHA256"],
    "sourceRawShardSHA256":receipt["assetSha256"],
    "remoteAssetSize":receipt["assetBytes"],"previousRemoteVerified":True})
 if len(all_rows)!=55 or set(all_rows)!=set(ids) or len(assembled)!=1100 or \
    n_new!=990 or n_reused!=110 or len(original_new_sha)!=990:
  raise RuntimeError("Final 03B complete 550-slice × 2 model pair inventory FAILED")
 rng=np.random.default_rng(PLAN["bootstrapSeed"])
 outcomes={}
 feature_diffs={}
 for patient in ids:
  case=all_rows[patient]
  D={}
  for item in ("tap","iou10"):
   D[item]=float(np.mean([p["models"]["J"]["GradCAM"][item]-
          p["models"]["S"]["GradCAM"][item] for p in case["slicePairs"]]))
  for layer in LAYERS:
   for metric in ("patchEffectiveRankNormalized","patchMeanOffDiagonalCosine"):
    D[layer+"_"+metric]=float(np.mean([
      p["models"]["J"]["SCEPT"]["layers"][layer][metric]-
      p["models"]["S"]["SCEPT"]["layers"][layer][metric] for p in case["slicePairs"]]))
   D[layer+"_crossModelCKA"]=float(np.mean([
       p["crossModelCKA"][layer] for p in case["slicePairs"]]))
  outcomes[patient]=D
 stats={}
 for key in outcomes[ids[0]]:
  vals=[outcomes[patient][key] for patient in ids]
  stats[key]=interval(vals,rng)
 for layer in LAYERS:
  for metric in ("patchEffectiveRankNormalized","patchMeanOffDiagonalCosine"):
   key=layer+"_"+metric
   for cam in ("tap","iou10"):
    label="spearman_patientMeanDelta_"+key+"_vs_"+cam
    stats[label]=correlation([outcomes[p][key] for p in ids],
                             [outcomes[p][cam] for p in ids])
 first_slice={
   "tap":interval([original[(p,"J")]["GradCAM"]["tap"]-
                        original[(p,"S")]["GradCAM"]["tap"] for p in ids],rng),
   "iou10":interval([original[(p,"J")]["GradCAM"]["iou10"]-
                          original[(p,"S")]["GradCAM"]["iou10"] for p in ids],rng)}
 report={"schema":"scept.a6-ij-03b.full-val.v1",
 "status":"PASS_EXACT_55_PATIENTS_1100_PAIRED_MODELSLICE_RECORDS",
 "plan":PLAN,"planSHA256":psha(),
 "nPatients":55,"nSlicesPerPatient":10,"nMatchedModelSliceRecords":1100,
 "originalFirstSlice02BReusedModelRuns":n_reused,
 "new03BRealSCEPTModelRuns":n_new,
 "privateOriginalFullRawZipsInRelease":990,
 "all11RemoteSourceShardsFullSha256CheckedOnChunkUpload":True,
 "sourceChunks":chunk_metadata,
 "patientClusteredBootstrapResamples":2000,
 "gradeCounts":{"HGG":44,"LGG":11},
 "firstSlice03ABaseline":first_slice,
 "patientLevelTenSliceSummaries":stats,
 "privatePatientAggregates":outcomes,
 "interpretation":[
   "Matched original patient, same original P1 slice rank and frozen S/J model; SCEPT forward-only, Grad-CAM true-grade attribution.",
   "Mean within-patient slice-level metrics before between-patient estimates to avoid pseudoreplication.",
   "Original 02B slice1 records reused, not counted as new model inference.",
   "Original 03B raw numerical tensors remain in private GitHub Release; 11 chunk receipts attest full prior remote re-download.",
   "All paired associations descriptive validation-set data, NOT independent clinical explanation, causality, tumor evidence dependence or superiority to Grad-CAM.",
   "No P5 sealed test, no P6 independent external validation, no model training."
 ]}
 raw=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 # Publish immutable original report. Raw paired patient-slice rows stay in
 # 11 original per-chunk patient tables, referenced by SHA only.
 path=f"{PREFIX}/final/analysis.json"
 commit_once(s,path,raw)
 receipt={"schema":"scept.a6-ij-03b.final-receipt.v1",
  "status":"PRIVATE_COMPLETE_1100_PAIRED_SLICES_990_RAW_VERIFIED",
  "analysisSHA256":sha(raw),"planSHA256":psha(),
  "nOriginalPatients":55,"nP4BmatchedModelSliceRecords":1100,
  "nNewGenuineSCEPTRecords":990,"n02bSourceReusedRecords":110,
  "nVerifiedRemotePrivateShards":11,"sourceRelease":release["html_url"],
  "sealedTest":"NOT_ACCESSED","causalOrMedicalEvidence":"NOT_ESTABLISHED"}
 rec=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 commit_once(s,f"{PREFIX}/final/archive_receipt.json",rec)
 print("A6-IJ-03B FULL 55 × 10 × TWO FROZEN MODELS: PASS",flush=True)
 print("1100 exact original paired slice-model rows; 990 new real SCEPT forwards; 110 original verified reused")
 for metric in ("tap","iou10","blocks.11.norm1_patchMeanOffDiagonalCosine",
                "blocks.11.norm1_patchEffectiveRankNormalized"):
  print(f"  paired patient {metric} J-S mean={stats[metric]['mean']:+.6f} "
        f"95% patient-bootstrap={stats[metric]['bootstrapMean95']}")
 print("PRIVATE final:",f"https://github.com/{REPO}/blob/main/{PREFIX}/final/analysis.json")

if __name__=="__main__":main()
