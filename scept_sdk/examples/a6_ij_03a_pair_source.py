"""A6-IJ-03A: VERIFY/PAIR original SCEPT 02B slice0 with P4B Grad-CAM rank1.

NO new model inference; no MRI source cache, Grad-CAM reconstruction, or test.
55 patients x2 frozen S/J = 110 exact source-matched rows. All patient-level
results and medical-derived indicators are PRIVATE and auditable.
"""
from __future__ import annotations
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from scipy.stats import spearmanr
from kaggle_secrets import UserSecretsClient

REPO="changxinjiresearch/research-course-brats2020-ijepa-xai"
API="https://api.github.com/repos/"+REPO
DEST=Path("/kaggle/working/a6_ij_03_private")
P0_SHA="1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352"
LAYERS=("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")
PLAN={
 "schema":"scept.a6-ij-03a.plan.v1",
 "cohort":"P0 locked validation 55 patients (44 HGG, 11 LGG)",
 "join":["patient_id","slice_rank=1","model=S/J"],
 "sources":["P4B exact 11 SHA256-verified CSV chunks (each 100 records)",
            "02B 55 patient authentic SCEPT raw-ZIP-derived analysis",
            "02C original 55 patient post-LN CKA control report",
            "P4B original 55-person patient-level metrics and report"],
 "predictionMaxAbs":0.03,
 "ckaMaxAbs":1e-5,
 "patientMetricReaggregationMaxAbs":1e-9,
 "bootstrapResamples":2000,"bootstrapSeed":240216,
 "interpretation":"Exploratory single slice exact-key pairing, NOT causal evidence or clinical validity",
 "sealedTest":"NOT_ACCESSED","modelTraining":"NONE"
}
def h(x):return hashlib.sha256(x).hexdigest()
def plan_sha():return h(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode())
def gh_bytes(s,path):
 r=s.get(f"{API}/contents/{path}",timeout=(30,180))
 r.raise_for_status()
 v=r.json()
 if v.get("encoding")!="base64":raise RuntimeError("GitHub content must be original base64 bytes: "+path)
 return base64.b64decode(v["content"])
def csv_bytes(b):return list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
def checked_float(value,key):
 try:x=float(value)
 except (ValueError,TypeError):raise RuntimeError("Missing numeric "+key)
 if not np.isfinite(x):raise RuntimeError("Nonfinite "+key)
 return x
def atomic_json(fp,data):
 fp.parent.mkdir(parents=True,exist_ok=True)
 raw=(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 if fp.exists():
  if fp.read_bytes()!=raw:raise RuntimeError("Existing 03A artifact differs; original immutable record preserved")
  return
 tmp=fp.with_name(fp.name+".partial")
 tmp.write_bytes(raw)
 os.replace(tmp,fp)
def paired_interval(v,rng):
 a=np.asarray(v,dtype=np.float64)
 if len(a)!=55 or not np.isfinite(a).all():raise RuntimeError("Patient bootstrap requires 55 patients")
 selected=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(axis=1)
 return {"nPatients":55,"mean":float(a.mean()),"median":float(np.median(a)),
         "mean95PatientBootstrap":list(map(float,np.quantile(selected,[.025,.975])))}
def finite_rho(x,y):
 a=np.asarray(x,dtype=np.float64);b=np.asarray(y,dtype=np.float64)
 if len(a)!=55 or len(b)!=55 or not np.isfinite(a).all() or not np.isfinite(b).all():
  raise RuntimeError("Spearman source must be 55 complete patient rows")
 obj=spearmanr(a,b)
 rho=float(obj.statistic)
 return {"nPatients":55,"rho":rho if np.isfinite(rho) else None,
         "status":"EXPLORATORY_NONCAUSAL" if np.isfinite(rho) else "UNDEFINED_CONSTANT_VECTOR"}

def main():
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Enable already used Kaggle GITHUB_TOKEN for private repo")
 s=requests.Session()
 s.headers.update({"Authorization":"Bearer "+token,
  "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 info=s.get(API,timeout=(30,120));info.raise_for_status()
 if info.json().get("private") is not True or info.json().get("full_name")!=REPO:
  raise RuntimeError("Authoritative research repo must be exactly named and PRIVATE")
 split=gh_bytes(s,"results/p0/patient_split_seed2026.csv")
 if h(split)!=P0_SHA:raise RuntimeError("Locked P0 patient allocation altered")
 cohort=sorted([x for x in csv_bytes(split) if x["split"]=="val"],key=lambda x:x["patient_id"])
 if len(cohort)!=55 or sum(x["grade"]=="HGG" for x in cohort)!=44:
  raise RuntimeError("Expected original 55 (44 HGG/11 LGG) P0 validation subjects")
 ids=[r["patient_id"] for r in cohort]
 grade={r["patient_id"]:r["grade"] for r in cohort}
 raw2b=gh_bytes(s,"results/scept_a6_ij/02b_55_patient_representations/analysis.json")
 two_b=json.loads(raw2b)
 if two_b.get("status")!="PASS_55_VAL_EXPLORATORY_ANALYSIS" or two_b.get("nPatients")!=55:
  raise RuntimeError("02B original full VAL qualification missing")
 bb={r["patient_id"]:r for r in two_b["patientRows"]}
 if len(bb)!=55 or set(bb)!=set(ids):raise RuntimeError("02B cohort differs from P0")
 raw2c=gh_bytes(s,"results/scept_a6_ij/02c_ln_position_controls/analysis.json")
 two_c=json.loads(raw2c)
 if (two_c.get("status")!="PASS_LOCKED_55_PATIENT_CONTROL_ENGINEERING" or
     two_c.get("nPatients")!=55 or two_c.get("original02BPrivateAnalysisSHA256")!=h(raw2b)):
  raise RuntimeError("02C real post-LN report or original 02B byte SHA differs")
 cc={r["patient_id"]:r for r in two_c["patientRows"]}
 if len(cc)!=55 or set(cc)!=set(ids):raise RuntimeError("02C cohort differs from P0")
 report_raw=gh_bytes(s,"results/p4b/p4b_report.json")
 p4report=json.loads(report_raw)
 if (p4report.get("status")!="PASS" or p4report.get("patient_count")!=55 or
     p4report.get("slices_per_patient")!=10 or p4report.get("explanation_count")!=1100
     or p4report.get("valid_explanations")!=1100 or
     p4report.get("test_set")!="NOT_ACCESSED"):
  raise RuntimeError("P4B complete frozen validation source not authenticated")
 p4patient=gh_bytes(s,"results/p4b/p4b_patient_metrics.csv")
 p4rec={(r["patient_id"],r["model"]):r for r in csv_bytes(p4patient)}
 if len(p4rec)!=110:raise RuntimeError("Original P4B 55-patient metric index invalid")
 p4rows={}
 sources={}
 for chunk in range(1,12):
  stem=f"results/p4b/chunks/chunk_{chunk:02d}"
  manifest_bytes=gh_bytes(s,stem+"_manifest.json")
  meta=json.loads(manifest_bytes)
  original_ids=ids[(chunk-1)*5:chunk*5]
  if meta.get("chunk_number")!=chunk or meta.get("patient_ids")!=original_ids:
   raise RuntimeError("Original P4B frozen 5-patient chunk order changed")
  original=gh_bytes(s,stem+"_metrics.csv")
  if h(original)!=meta.get("csv_sha256"):
   raise RuntimeError("P4B source CSV fails original SHA-256 manifest: "+stem)
  rows=csv_bytes(original)
  if len(rows)!=100 or int(meta.get("valid",-1))!=100 or meta.get("test_set")!="NOT_ACCESSED":
   raise RuntimeError("P4B chunk missing 100 genuine valid model-slice rows")
  for row in rows:
   patient=row["patient_id"];model=row["model"]
   k=int(row["slice_rank"])
   key=(patient,k,model)
   if (key in p4rows or patient not in original_ids or not 1<=k<=10 or
       model not in ("S","J") or row["status"]!="VALID" or
       row["cohort"]!="VALIDATION" or row["true_grade"]!=grade[patient] or
       int(row["class_idx"])!=(1 if grade[patient]=="HGG" else 0)):
    raise RuntimeError("Bad/duplicated P4B exact patient-slice-model primary key")
   for col in ("tap","iou10","p_hgg","reference_fraction"):
    v=checked_float(row[col],col)
    if not 0<=v<=1+1e-7:raise RuntimeError("Invalid P4B normalized outcome "+col)
   p4rows[key]=row
  sources[f"chunk_{chunk:02d}"]={"manifestSha256":h(manifest_bytes),
                                "csvSha256":h(original),"rows":len(rows)}
 if len(p4rows)!=1100:raise RuntimeError("P4B authentic exact 1100 slice records incomplete")
 # Independent, non-imputed patient TAP and IoU10 aggregation against old P4B report.
 for patient in ids:
  for model in ("S","J"):
   row=p4rec[(patient,model)]
   if row["true_grade"]!=grade[patient] or row["complete"]!="True" or int(row["n_valid_slices"])!=10:
    raise RuntimeError("P4B patient-level coverage/grade mismatch")
   for metric in ("tap","iou10"):
    actual=np.mean([float(p4rows[(patient,k,model)][metric]) for k in range(1,11)])
    if abs(actual-float(row[metric]))>PLAN["patientMetricReaggregationMaxAbs"]:
     raise RuntimeError("P4B 10-slice patient metric does not equal original slice-level results")
 joined=[]
 for patient in ids:
  b=bb[patient];c=cc[patient]
  if b["grade"]!=grade[patient] or c["grade"]!=grade[patient]:
   raise RuntimeError("SCEPT source patient class labels not aligned")
  for layer in LAYERS:
   target=c["matched"][layer.split(".")[1]]["post"]["alignedCKA"]
   if abs(target-b["crossModelCKA"][layer])>PLAN["ckaMaxAbs"]:
    raise RuntimeError("Source 02B/02C actual cross-model CKA disagreement")
  for model in ("S","J"):
   cam=p4rows[(patient,1,model)]
   obs=b[model]
   delta=abs(checked_float(cam["p_hgg"],"p_hgg")-checked_float(obs["pHggSlice0"],"SCEPT p_hgg"))
   if delta>PLAN["predictionMaxAbs"]:
    raise RuntimeError(f"Same MRI/model/slice predictions DIFFER: {patient}/{model}, delta={delta:g}")
   data={"patient_id":patient,"grade":grade[patient],
      "slice_rank":1,"axial_index":int(cam["axial_index"]),"model":model,
      "SCEPTsource":"02B original P1 FLAIR slice index0",
      "GradCAMsource":"P4B original P1 FLAIR slice_rank1 true-grade logit",
      "pHggSCEPT":obs["pHggSlice0"],"pHggP4B":float(cam["p_hgg"]),
      "pHggAbsoluteError":delta,
      "GradCAM":{"tap":float(cam["tap"]),"iou10":float(cam["iou10"]),
                 "referenceFraction":float(cam["reference_fraction"])},
      "SCEPT":{"layers":{
          name:{"patchEffectiveRankNormalized":obs["layers"][name]["patchEffectiveRankNormalized"],
                "patchMeanOffDiagonalCosine":obs["layers"][name]["patchMeanOffDiagonalCosine"]}
          for name in LAYERS}},
      "SCEPTcrossModelCKA":{name:b["crossModelCKA"][name] for name in LAYERS}}
   joined.append(data)
 if len(joined)!=110 or len({(r["patient_id"],r["slice_rank"],r["model"]) for r in joined})!=110:
  raise RuntimeError("Not exactly 110 uniquely matched S/J first-slice records")
 rng=np.random.default_rng(PLAN["bootstrapSeed"])
 keys={(r["patient_id"],r["model"]):r for r in joined}
 summaries={}
 for field in ("tap","iou10"):
  d=[keys[(p,"J")]["GradCAM"][field]-keys[(p,"S")]["GradCAM"][field] for p in ids]
  summaries["GradCAM_"+field+"_JminusS"]=paired_interval(d,rng)
 for layer in LAYERS:
  for field in ("patchEffectiveRankNormalized","patchMeanOffDiagonalCosine"):
   d=[keys[(p,"J")]["SCEPT"]["layers"][layer][field]-
      keys[(p,"S")]["SCEPT"]["layers"][layer][field] for p in ids]
   summaries[layer+"_"+field+"_JminusS"]=paired_interval(d,rng)
 # Correlate exactly matched *first-slice* deltas, not 10-slice P4B patient averages.
 base="blocks.11.norm1"
 for field in ("tap","iou10"):
  dcam=[keys[(p,"J")]["GradCAM"][field]-keys[(p,"S")]["GradCAM"][field] for p in ids]
  for metric in ("patchEffectiveRankNormalized","patchMeanOffDiagonalCosine"):
   drep=[keys[(p,"J")]["SCEPT"]["layers"][base][metric]-
         keys[(p,"S")]["SCEPT"]["layers"][base][metric] for p in ids]
   summaries["Spearman_"+metric+"_JminusS_vs_firstSlice_"+field+"_JminusS"]=finite_rho(drep,dcam)
 report={
  "schema":"scept.a6-ij-03a.analysis.v1",
  "status":"PASS_STRICTLY_PAIRED_55_FIRST_SLICE_VAL",
  "plan":PLAN,"planSha256":plan_sha(),
  "sourceChecksums":{"P0SplitSHA256":h(split),"P4BReportSHA256":h(report_raw),
    "P4BPatientCSVsha256":h(p4patient),"02BAnalysisSHA256":h(raw2b),
    "02CAnalysisSHA256":h(raw2c),"P4BChunks":sources},
  "cohort":{"nPatients":55,"HGG":44,"LGG":11,
            "nMatchedModelSliceRecords":110,"nAvailableOriginalP4BSliceRecords":1100},
  "maxSCEPTvsP4BPredictionError":max(r["pHggAbsoluteError"] for r in joined),
  "pairedResults":summaries,"exactMatchedPatientRows":joined,
  "scientificLimits":[
    "Same first P1 FLAIR slice, frozen model, and patient. GradCAM targets true grade, SCEPT merely observes.",
    "Single first-slice matching is 03A ONLY. It does not qualify 10-slice 03B.",
    "P4B original validation cohort may have influenced checkpoint selection; no independent clinical outcome or causal evidence.",
    "Spearman association is exploratory and not a comparative performance/saliency claim.",
    "No P5 sealed test, no external P6, no retraining, no newly calculated Grad-CAM maps."]}
 atomic_json(DEST/"locked_plan.json",{"plan":PLAN,"planSha256":plan_sha()})
 atomic_json(DEST/"a6_ij_03a_analysis.json",report)
 print("A6-IJ-03A EXACT 55-PATIENT FIRST-SLICE P4B/SCEPT PAIRING: PASS")
 print("Matched original rows: 110; P4B slice-level source audited: 1100")
 print("Maximum same-input prediction probability absolute difference:",
       round(report["maxSCEPTvsP4BPredictionError"],9))
 for label,result in summaries.items():
  if "Spearman" in label:
   print(label,":",result["rho"])
 print("PRIVATE LOCAL REPORT:",DEST/"a6_ij_03a_analysis.json")
 print("NEXT: private 03A GitHub archiver; 03B and 03C remain pending")

if __name__=="__main__":main()
