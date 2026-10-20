"""A6-IJ-02C: rederive locked 55-patient confound controls from raw ZIPs.

No trained model forward in the analysis step; algebraic reconstruction
of LayerNorm uses frozen model parameters. Statistical bootstrap is
descriptive, not an XAI fidelity or pathology test.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
import requests
import torch
from kaggle_secrets import UserSecretsClient
from scept import load_observation
from a6_ij_02b_measure import private_contents,get_json,API,sha_file
from a6_ij_02c_measure import (
 PLAN,ROOT,BLOCKS,STAGES,PRE,POST,plan_sha,
 checkpoint_model,features,aligned_cka,alignment_null,token_metrics,cka,
)

PATH=Path("/kaggle/working/a6_ij_02c_private")
SEED=240216

def patient_interval(values,rng):
 values=np.asarray(values,dtype=np.float64)
 if not len(values) or not np.isfinite(values).all():
  raise RuntimeError("Patient bootstrap requires real finite patient-level values")
 b=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
 return {"n":int(len(values)),"mean":float(values.mean()),
         "median":float(np.median(values)),
         "bootstrap95":list(map(float,np.quantile(b,[0.025,0.975])))}

def verify_archive(path,entry,model_state):
 if path.name!=entry["zipName"] or not path.is_file() or sha_file(path)!=entry["zipSha256"]:
  raise RuntimeError("Tampered/incomplete real patient archive: "+path.name)
 obs=load_observation(str(path))
 if obs.run_id!=entry["runId"] or obs.record["provenance"]["model"]["sha256"]!=model_state:
  raise RuntimeError("ZIP source model/run provenance disagreement")
 if obs.manifest["captureScope"]["inputsIncluded"] is not False:
  raise RuntimeError("Original patient input appeared in SCEPT ZIP")
 return obs

def close(a,b,tol=1e-8):
 if abs(float(a)-float(b))>tol:raise RuntimeError("Recomputed metric differs from preserved original")
 
def main():
 locked=json.loads((PATH/"locked_plan.json").read_text())
 if locked["plan"]!=PLAN or locked["planSha256"]!=plan_sha() or len(locked["cohortPatientIds"])!=55:
  raise RuntimeError("Not the full preregistered 02C 55-patient lock")
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Private research repo source secret required")
 client=requests.Session()
 client.headers.update({"Authorization":"Bearer "+token,
   "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 if get_json(client,API).get("private") is not True:raise RuntimeError("Research source repository must be private")
 p0=private_contents(client,"results/p0/patient_split_seed2026.csv")
 if hashlib.sha256(p0).hexdigest()!="1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352":
  raise RuntimeError("P0 locked split SHA mismatch")
 val=sorted((r for r in csv.DictReader(io.StringIO(p0.decode())) if r["split"]=="val"),
            key=lambda r:r["patient_id"])
 if len(val)!=55 or [r["patient_id"] for r in val]!=locked["cohortPatientIds"]:
  raise RuntimeError("Full 55 VAL source not matched")
 orig=private_contents(client,"results/scept_a6_ij/02b_55_patient_representations/analysis.json")
 if hashlib.sha256(orig).hexdigest()!=json.loads((PATH/"02b_source_sha.json").read_text())["02BAnalysisBytesSHA256"]:
  raise RuntimeError("Frozen 02B private source checksum drift")
 prev=json.loads(orig)
 if prev["status"]!="PASS_55_VAL_EXPLORATORY_ANALYSIS" or len(prev["patientRows"])!=55:
  raise RuntimeError("Original independent 02B source not complete")
 prior={r["patient_id"]:r["crossModelCKA"] for r in prev["patientRows"]}
 models={name:checkpoint_model(name) for name in ("S","J")}
 entries=[]
 raw_file_count=0
 for index,item in enumerate(val):
  pid=item["patient_id"]
  saved=json.loads((PATH/"cases"/f"{pid}.json").read_text())
  if (saved.get("patient_id")!=pid or saved.get("grade")!=item["grade"] or
      saved.get("planSha256")!=plan_sha() or saved.get("sliceIndex")!=0):
   raise RuntimeError("Patient ID/grade/protocol mismatch")
  stages={}
  summaries={}
  for name in ("S","J"):
   entry=saved["models"][name]
   obs=verify_archive(PATH/"raw"/entry["zipName"],entry,
                      entry["modelStateSha256"])
   stage,lnerr=features(models[name],obs)
   stages[name]=stage
   summaries[name]={}
   for b in BLOCKS:
    row=entry["layers"][str(b)]
    close(lnerr[str(b)],row["layerNormMaxAbsError"])
    close(cka(stage[str(b)]["pre"],stage[str(b)]["post"]),row["prePostCKA"])
    close(cka(stage[str(b)]["post"],stage[str(b)]["affinefree"]),row["postVsAffinefreeCKA"])
    for st in STAGES:
     d=token_metrics(stage[str(b)][st])
     for k,v in d.items():
      close(v,row["stages"][st][k])
    summaries[name][str(b)]={"prePostCKA":row["prePostCKA"],
                             "postVsAffinefreeCKA":row["postVsAffinefreeCKA"],
                             "layerNormMaxAbsError":lnerr[str(b)]}
   if index==0:
    replay=entry.get("independentHookReplay")
    if replay is None or replay.get("status")!="PASS" or len(replay.get("maxAbsByTensor",{}))!=7:
     raise RuntimeError("First-patient direct PyTorch replay missing")
    if any(t>PLAN["independentHookReplayMaxAbs"] for t in replay["maxAbsByTensor"].values()):
     raise RuntimeError("First-patient independent raw Hook replay failed")
   raw_file_count+=1
  match={}
  for b in BLOCKS:
   match[str(b)]={}
   saved_b=saved["matched"][str(b)]
   for st in STAGES:
    observed,ka,kb,den=aligned_cka(stages["S"][str(b)][st],stages["J"][str(b)][st])
    close(observed,saved_b[st]["alignedCKA"])
    valx={"alignedCKA":observed}
    if st in ("pre","post"):
     null=alignment_null(ka,kb,den,index,b,st)
     saved_null=saved_b[st]["permutedPatchNull"]
     if null["seed"]!=saved_null["seed"] or null["count"]!=saved_null["count"]:
      raise RuntimeError("Deterministic permutation null provenance changed")
     for key in ("mean","std","max","q95"):
      close(null[key],saved_null[key])
     close(observed-null["mean"],saved_b[st]["alignedMinusNullMean"])
     valx["patchCorrespondenceNullMean"]=null["mean"]
     valx["alignedMinusPermutationMean"]=observed-null["mean"]
    match[str(b)][st]=valx
   original=prior[pid][f"blocks.{b}.norm1"]
   drift=abs(match[str(b)]["post"]["alignedCKA"]-original)
   if drift>PLAN["02BPostCKAMaxAbs"]:
    raise RuntimeError("02C cannot reproduce 02B original S/J post-norm1 CKA")
   close(drift,saved_b["02BpostCKAAbsoluteDeviation"])
  entries.append({"patient_id":pid,"grade":item["grade"],"models":summaries,"matched":match})
 # Baseline all-zero image is out-of-distribution, NOT a clinical control.
 blank={}
 for name in ("S","J"):
  desc=json.loads((PATH/f"blank_{name}.json").read_text())
  path=PATH/"raw"/desc["zipName"]
  if path.name!=f"BLANK_{name}_private.zip" or sha_file(path)!=desc["zipSha256"]:
   raise RuntimeError("Locked synthetic blank reference missing or corrupted")
  obs=load_observation(str(path))
  if obs.run_id!=desc["runId"] or obs.record["provenance"]["model"]["sha256"]!=desc["modelStateSha256"]:
   raise RuntimeError("Blank SCEPT source provenance incorrect")
  blank[name],errors=features(models[name],obs)
  for b in BLOCKS:close(errors[str(b)],desc["layerNormMaxAbs"][str(b)])
  raw_file_count+=1
 blank_ckas={str(b):{st:cka(blank["S"][str(b)][st],blank["J"][str(b)][st])
                        for st in STAGES} for b in BLOCKS}
 rng=np.random.default_rng(SEED)
 stats={}
 for b in BLOCKS:
  key=str(b)
  out={}
  for st in STAGES:
   raw=np.array([r["matched"][key][st]["alignedCKA"] for r in entries])
   out[st]={"alignedCKA":patient_interval(raw,rng)}
   if st in ("pre","post"):
    null=np.array([r["matched"][key][st]["patchCorrespondenceNullMean"] for r in entries])
    out[st]["patchPermutationNullMean"]=patient_interval(null,rng)
    out[st]["alignmentAboveNull"]=patient_interval(raw-null,rng)
  pre=np.array([r["matched"][key]["pre"]["alignedCKA"] for r in entries])
  free=np.array([r["matched"][key]["affinefree"]["alignedCKA"] for r in entries])
  post=np.array([r["matched"][key]["post"]["alignedCKA"] for r in entries])
  out["pairedPostMinusPre"]=patient_interval(post-pre,rng)
  out["pairedPostMinusAffinefree"]=patient_interval(post-free,rng)
  out["pairedAffinefreeMinusPre"]=patient_interval(free-pre,rng)
  for grade in ("HGG","LGG"):
   mask=np.array([r["grade"]==grade for r in entries])
   out[f"{grade}_pairedPostMinusPre"]=patient_interval((post-pre)[mask],rng)
  for name in ("S","J"):
   out[f"{name}_prePostWithinModelCKA"]=patient_interval(
      [r["models"][name][key]["prePostCKA"] for r in entries],rng)
   out[f"{name}_postVsAffinefreeWithinModelCKA"]=patient_interval(
      [r["models"][name][key]["postVsAffinefreeCKA"] for r in entries],rng)
  out["blankInputAlignedCKA"]=blank_ckas[key]
  stats[key]=out
 report={"schema":"scept.a6-ij-02c.analysis.v1",
    "status":"PASS_LOCKED_55_PATIENT_CONTROL_ENGINEERING",
    "planSha256":plan_sha(),"nPatients":55,"HGG":44,"LGG":11,
    "original02BPrivateAnalysisSHA256":hashlib.sha256(orig).hexdigest(),
    "realPatientCaptures":110,"blankOODCaptures":2,"archiveCount":raw_file_count,
    "firstPatientIndependentDirectHookReplay":"PASS_BOTH_FROZEN_MODELS",
    "all110x3LayerNormAlgebraAnd02BPostCKA":"PASS",
    "metrics":stats,"patientRows":entries,
    "limitations":[
     "Numerical comparisons of pre-LN, affinefree-LN and post-LN are not a causal decomposition of training mechanisms.",
     "Direct LayerNorm affine-free control is a mathematical recalculation, not new model inference.",
     "Permutation null randomizes patch correspondence, not actual image patches or learned positional embeddings.",
     "All-zero input is out-of-distribution and cannot isolate positional encoding alone.",
     "Patient CKA relates one P1 slice; this is a validation-set exploratory analysis, not external validation.",
     "No lesion importance, faithfulness proof, clinical interpretation, test data or training."
     ]}
 target=PATH/"a6_ij_02c_analysis.json"
 tmp=target.with_suffix(".json.partial")
 tmp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
 os.replace(tmp,target)
 print("A6-IJ-02C 55-PATIENT CONTROLS: ENGINEERING PASS")
 for b in BLOCKS:
  row=stats[str(b)]
  print(f"Block {b}: mean cross-model pre={row['pre']['alignedCKA']['mean']:.5f} "
    f"affinefree={row['affinefree']['alignedCKA']['mean']:.5f} "
    f"post={row['post']['alignedCKA']['mean']:.5f} "
    f"post-minus-pre={row['pairedPostMinusPre']['mean']:+.5f}")
  print(f"   patch alignment excess above permutation null (post): "
    f"{row['post']['alignmentAboveNull']['mean']:.5f}; blank-input post CKA="
    f"{row['blankInputAlignedCKA']['post']:.5f}")
 print("DIRECT independent Torch replay: PASS on fixed first patient S/J")
 print("No clinical/causal claims. PRIVATE report:",target)

if __name__=="__main__":main()
