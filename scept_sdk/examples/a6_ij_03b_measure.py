"""A6-IJ-03B: 990 NEW genuine SCEPT ViT observations, 11 resumable five-patient chunks.

Each P4B Grad-CAM record is matched by (P0 VAL patient ID, P1 slice index,
frozen P4 S/J model). Reuse EXACTLY 110 03A-authenticated first-slice SCEPT
records and original 02B private archives. Keep all fresh raw tensor ZIPs
private. No retraining, no sealed P5 test, no P6 external cohort.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
import requests
from kaggle_secrets import UserSecretsClient
from scept import SCEPT,load_observation
from a6_ij_02b_measure import (
  manifest_and_zip,checkpoint_model,sha_file,token_metrics,cka,
  private_contents,get_json,API,REPO,ROOT,LAYERS
)

OUT=ROOT/"a6_ij_03b_private"
RELEASE_TAG="scept-a6-ij-03b-ten-slice-paired-v1"
PREFIX="results/scept_a6_ij/03b_full_slice_pairing"
S_SHA="2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf"
J_SHA="719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"
P0_SHA="1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352"
PLAN={
 "schema":"scept.a6-ij-03b.plan.v1",
 "cohort":"P0 locked validation only, 55 patients 44HGG+11LGG",
 "totalPatientSlices":550,"models":["S","J"],
 "newSlicesPerPatient":[2,3,4,5,6,7,8,9,10],
 "reusedSliceRank":1,
 "totalNewModelForwards":990,"reusedOriginal02BModelForwards":110,
 "target":"P4B original Grad-CAM true-grade logit, original slice CSV unmodified",
 "frozenCheckpoints":{"S":S_SHA,"J":J_SHA},
 "source":"P1 SHA256-verified original classification_npz VAL FLAIR cache",
 "P0SplitSHA256":P0_SHA,"layers":list(LAYERS),
 "predictionMaxAbsoluteDifference":0.03,
 "metrics":["patchEffectiveRankNormalized","patchMeanOffDiagonalCosine",
            "crossModelLinearCKA"],
 "privateChunkPatientCount":5,"totalChunks":11,
 "statisticalUnit":"patient (10 slices nested within person)",
 "bootstrapResamples":2000,"bootstrapSeed":240216,
 "interpretation":"observational exploratory correspondence, no causal or clinical claims",
 "sealedP5Test":"NOT_ACCESSED","externalP6":"NOT_ACCESSED"
}
def sha(raw):
 return hashlib.sha256(raw).hexdigest()
def psha():
 return sha(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode())
def save_once(fp,val):
 raw=(json.dumps(val,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 fp.parent.mkdir(parents=True,exist_ok=True)
 if fp.is_file():
  if fp.read_bytes()!=raw:raise RuntimeError("Immutable local 03B record differs: "+str(fp))
  return
 tmp=fp.with_name(fp.name+".partial")
 tmp.write_bytes(raw)
 os.replace(tmp,fp)
def finite(x):
 v=float(x)
 if not np.isfinite(v):raise RuntimeError("Nonfinite source/observed numeric value")
 return v
def binary_model_prob(logits):
 z=np.asarray(logits,dtype=np.float64)
 if z.shape!=(2,) or not np.isfinite(z).all():
  raise RuntimeError("Frozen ViT binary logits shape or finiteness changed")
 return float(1/(1+np.exp(np.clip(z[0]-z[1],-500,500))))
def github():
 secret=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not secret:raise RuntimeError("Enable private research GitHub token in Kaggle Secrets")
 s=requests.Session()
 s.headers.update({"Authorization":"Bearer "+secret,
  "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 owner=get_json(s,API)
 if owner.get("private") is not True or owner.get("full_name")!=REPO:
  raise RuntimeError("Private authoritative research repo not verified")
 return s
def sources(s,chunk_num,cohort):
 if chunk_num not in range(1,12):raise RuntimeError("Chunks must be numbered 1..11")
 patients=cohort[(chunk_num-1)*5:chunk_num*5]
 if len(patients)!=5:raise RuntimeError("Incomplete original VAL patient chunk")
 prior_bytes=private_contents(s,"results/scept_a6_ij/03a_exact_first_slice_pairing/analysis.json")
 prior=json.loads(prior_bytes)
 receipt=json.loads(private_contents(s,"results/scept_a6_ij/03a_exact_first_slice_pairing/archive_receipt.json"))
 if prior.get("status")!="PASS_STRICTLY_PAIRED_55_FIRST_SLICE_VAL" or \
     prior.get("cohort",{}).get("nMatchedModelSliceRecords")!=110 or \
     receipt.get("status")!="PASS_110_EXACT_MATCHED_MODEL_SLICE_RECORDS_PRIVATE" or \
     receipt.get("sourceAnalysisOriginalSHA256")!=sha(prior_bytes):
  raise RuntimeError("03A source prior/immutable archive receipt not genuine")
 original02b=json.loads(private_contents(s,"results/scept_a6_ij/02b_55_patient_representations/archive_receipt.json"))
 if original02b.get("status")!="ARCHIVED_55_PATIENTS_110_ZIPS_VERIFIED" or \
    original02b.get("nRealModelRuns")!=110 or len(original02b.get("shards",[]))!=11:
  raise RuntimeError("Original 110 first-slice SCEPT raw ZIPs not verified in private 02B release")
 orig={(r["patient_id"],r["model"]):r for r in prior["exactMatchedPatientRows"]}
 if len(orig)!=110:raise RuntimeError("Duplicated or incomplete 03A original rows")
 original_p1=list(csv.DictReader(io.StringIO(private_contents(s,
  "results/p1/p1_classification_manifest.csv").decode("utf-8-sig"))))
 p1_by={r["patient_id"]:r for r in original_p1}
 if len(p1_by)!=len(original_p1):raise RuntimeError("Duplicate P1 patient manifest")
 stem=f"results/p4b/chunks/chunk_{chunk_num:02d}"
 manifest_bytes=private_contents(s,stem+"_manifest.json")
 manifest=json.loads(manifest_bytes)
 chunk_csv=private_contents(s,stem+"_metrics.csv")
 if (sha(chunk_csv)!=manifest.get("csv_sha256") or
     manifest.get("patient_ids")!=[r["patient_id"] for r in patients] or
     manifest.get("chunk_number")!=chunk_num or manifest.get("valid")!=100 or
     manifest.get("test_set")!="NOT_ACCESSED"):
  raise RuntimeError("Original P4B chunk patient keys/manifest integrity mismatch")
 indexed={}
 for row in csv.DictReader(io.StringIO(chunk_csv.decode("utf-8-sig"))):
  key=(row["patient_id"],int(row["slice_rank"]),row["model"])
  if key in indexed or row["patient_id"] not in manifest["patient_ids"] or \
     row["model"] not in ("S","J") or not 1<=key[1]<=10 or \
     row["true_grade"] not in ("HGG","LGG") or row["status"]!="VALID" or \
     row["cohort"]!="VALIDATION" or int(row["class_idx"])!=(1 if row["true_grade"]=="HGG" else 0):
   raise RuntimeError("P4B invalid / duplicate authentic matched model-slice key")
  for column in ("p_hgg","tap","iou10","reference_fraction"):
   val=finite(row[column])
   if val<0 or val>1+1e-7:raise RuntimeError("Invalid original P4B metric: "+column)
  indexed[key]=row
 if len(indexed)!=100:raise RuntimeError("P4B original chunk does not have precisely 100 paired records")
 for item in patients:
  pid=item["patient_id"]
  man=p1_by.get(pid)
  if (man is None or man["split"]!="val" or man["grade"]!=item["grade"] or
      int(man["num_classification_slices"])!=10 or
      man["npz_relative_path"]!=f"classification_npz/val/{pid}.npz"):
   raise RuntimeError("Original P1 VAL patient/slice source mismatch")
  axes=json.loads(man["axial_indices_json"])
  if len(axes)!=10:raise RuntimeError("Original P1 source does not contain 10 axial indices")
  for rank in range(1,11):
   for model in ("S","J"):
    row=indexed[(pid,rank,model)]
    if int(row["axial_index"])!=int(axes[rank-1]) or row["true_grade"]!=item["grade"]:
     raise RuntimeError("Original P4B axial slice does NOT equal P1 exact slice: "+pid)
   if rank==1:
    for model in ("S","J"):
     r=orig.get((pid,model))
     if r is None or r["slice_rank"]!=1 or r["grade"]!=item["grade"] or \
        r["axial_index"]!=int(axes[0]):
      raise RuntimeError("Original 03A first-slice join cannot be reused")
 return {"03aSHA256":sha(prior_bytes),
         "02bOriginalArchiveSHA256":sha(json.dumps(original02b,sort_keys=True).encode()),
         "p4bCsvSHA256":sha(chunk_csv),
         "p4bManifestSHA256":sha(manifest_bytes)},indexed,orig
def original_row(patient,grade,rank,model,axes,cam,source=None,metrics=None,runmeta=None):
 base={"patient_id":patient,"grade":grade,"slice_rank":rank,"axial_index":int(axes[rank-1]),
  "model":model,
  "GradCAM":{"tap":finite(cam["tap"]),"iou10":finite(cam["iou10"]),
             "pHgg":finite(cam["p_hgg"]),"referenceFraction":finite(cam["reference_fraction"])}}
 if rank==1:
  row=source
  if row["model"]!=model or row["patient_id"]!=patient or row["slice_rank"]!=1:
   raise RuntimeError("03A original row identity mismatch")
  base["SCEPT"]={"layers":row["SCEPT"]["layers"],
     "pHgg":finite(row["pHggSCEPT"]),
     "runSource":"REUSED_02B_WITH_03A_SHA_MATCHED_PRIVATE_RAW_ZIP",
     "original02BArchiveReference":"scept-a6-ij-02b-55val-v1"}
  if abs(base["SCEPT"]["pHgg"]-base["GradCAM"]["pHgg"])>PLAN["predictionMaxAbsoluteDifference"]:
   raise RuntimeError("First-slice original 03A prediction guard no longer holds")
 else:
  base["SCEPT"]={"layers":metrics,"pHgg":finite(runmeta["pHgg"]),
    "runSource":"NEW_03B_GENUINE_SCEPT_FORWARD",
    "originalZip":runmeta["originalZip"],
    "runId":runmeta["runId"],"modelStateSHA256":runmeta["modelStateSHA256"]}
  if abs(base["SCEPT"]["pHgg"]-base["GradCAM"]["pHgg"])>PLAN["predictionMaxAbsoluteDifference"]:
   raise RuntimeError("Actual SCEPT vs original P4B prediction mismatch "+patient+" rank "+str(rank))
 base["pHggAbsoluteDifference"]=abs(base["SCEPT"]["pHgg"]-base["GradCAM"]["pHgg"])
 return base
def validate_complete(fp,patient,grade,source_sha):
 if not fp.exists():return None
 row=json.loads(fp.read_text())
 if (row.get("schema")!="scept.a6-ij-03b.patient.v1" or
     row.get("planSHA256")!=psha() or row.get("patient_id")!=patient or
     row.get("grade")!=grade or row.get("originalSources")!=source_sha or
     len(row.get("slicePairs",[]))!=10):
  raise RuntimeError("Existing patient report disagrees with locked source/protocol")
 for pair in row["slicePairs"]:
  if pair["slice_rank"] not in range(1,11) or pair["patient_id"]!=patient:
   raise RuntimeError("Existing original patient-slice numbering incomplete")
  if set(pair["models"])!={"S","J"}:raise RuntimeError("Existing patient S/J model pair incomplete")
  if pair["slice_rank"]>1:
   for model in ("S","J"):
    meta=pair["models"][model]["SCEPT"]["originalZip"]
    path=OUT/"raw"/meta["name"]
    if not path.is_file() or path.stat().st_size!=meta["bytes"] or sha_file(path)!=meta["sha256"]:
     raise RuntimeError("Previously captured actual SCEPT ZIP changed")
    v=load_observation(str(path))
    if v.run_id!=pair["models"][model]["SCEPT"]["runId"] or \
       v.manifest["captureScope"]["inputsIncluded"] is not False:
     raise RuntimeError("Previously captured source original run/privacy changed")
 return row
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--chunk",type=int,required=True)
 p.add_argument("--pilot",action="store_true",help="First patient of chunk ONLY, not completed chunk")
 args=p.parse_args()
 if not 1<=args.chunk<=11:raise RuntimeError("Select 1..11")
 if args.pilot and args.chunk!=1:raise RuntimeError("Only predefined very first patient pilot permitted")
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/"cases").mkdir(exist_ok=True)
 (OUT/"raw").mkdir(exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 s=github()
 cohort,zip_file=manifest_and_zip(s)
 if len(cohort)!=55 or sum(x["grade"]=="HGG" for x in cohort)!=44:
  raise RuntimeError("The locked 55-patient validation cohort changed")
 subset=cohort[(args.chunk-1)*5:args.chunk*5]
 original_checks,cams,first=sources(s,args.chunk,cohort)
 lock={"plan":PLAN,"planSHA256":psha(),
       "cohortPatients":[r["patient_id"] for r in cohort],
       "03aSourceSHA256":original_checks["03aSHA256"]}
 save_once(OUT/"locked_plan.json",lock)
 models={name:checkpoint_model(name) for name in ("S","J")}
 observers={name:SCEPT(models[name],list(LAYERS),backend="torch",
              profile="vit-token-channel",max_capture_bytes=16_000_000) for name in ("S","J")}
 count=0
 start=time.monotonic()
 patients=subset[:1] if args.pilot else subset
 with ZipFile(zip_file) as z:
  for i,item in enumerate(patients,1):
   patient=item["patient_id"]
   prior=validate_complete(OUT/"cases"/f"{patient}.json",
                           patient,item["grade"],original_checks)
   if prior is not None:
    print(f"03B [{i}/{len(patients)}] REUSE original patient SHA-verified: {patient}",flush=True)
    continue
   raw=z.read(item["zip_member"])
   if sha(raw)!=item["p1_sha256"]:
    raise RuntimeError("Full original selected P1 NPZ SHA256 differs")
   with np.load(io.BytesIO(raw),allow_pickle=False) as npz:
    if "flair" not in npz.files or npz["flair"].shape!=(10,224,224):
     raise RuntimeError("Locked P1 ten-slice FLAIR original geometry changed")
    images=np.asarray(npz["flair"],dtype=np.float32).copy()
   if not np.isfinite(images).all():raise RuntimeError("P1 FLAIR contains nonfinite values")
   manifest=list(csv.DictReader(io.StringIO(private_contents(s,
      "results/p1/p1_classification_manifest.csv").decode("utf-8-sig"))))
   manifest_rows={m["patient_id"]:m for m in manifest}
   axes=json.loads(manifest_rows[patient]["axial_indices_json"])
   pairs=[]
   for rank in range(1,11):
    records={}
    activations={}
    x=torch.from_numpy(images[rank-1]).view(1,1,224,224).repeat(1,3,1,1).contiguous()
    for name in ("S","J"):
     cam=cams[(patient,rank,name)]
     if rank==1:
      records[name]=original_row(patient,item["grade"],rank,name,axes,cam,source=first[(patient,name)])
      continue
     run=observers[name].observe(x,model_id=f"P4-frozen-{name}",
       preprocess_id="P1_FLAIR_224_locked_v1",
       preprocess_config={"patient_id":patient,"sliceIndex":rank-1,
         "split":"val","channels":"FLAIRx3","study":"03B"})
     if {g["tensor"] for g in run.fingerprint()["groups"] if g["status"]=="measured"}!=set(LAYERS):
      raise RuntimeError("SCEPT actual three-layer observation incomplete")
     if run.tensor("model.output").shape!=(1,2) or any(run.tensor(k).shape!=(1,197,768) for k in LAYERS):
      raise RuntimeError("P4 ViT original tensor/output shapes changed")
     patches={namekey:run.tensor(namekey)[0,1:,:] for namekey in LAYERS}
     metrics={namekey:token_metrics(patches[namekey]) for namekey in LAYERS}
     output=OUT/"raw"/f"{patient}_rank{rank:02d}_{name}_private.zip"
     tmp=OUT/"raw"/f"{patient}_rank{rank:02d}_{name}_private.zip.partial"
     run.export(str(tmp),include_inputs=False)
     again=load_observation(str(tmp))
     if again.run_id!=run.run_id or again.manifest["captureScope"]["inputsIncluded"] is not False:
      raise RuntimeError("Original SCEPT ZIP run privacy/source identity mismatched")
     if again.record["provenance"]["model"]["sha256"]!=observers[name].model_sha256:
      raise RuntimeError("Captured ZIP source model byte state differs")
     for tensor in (*LAYERS,"model.output"):
      np.testing.assert_array_equal(again.tensor(tensor),run.tensor(tensor))
     if again.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
      raise RuntimeError("Offline arithmetic must not claim genuine new model inference")
     os.replace(tmp,output)
     meta={"runId":run.run_id,"pHgg":binary_model_prob(run.tensor("model.output")[0]),
       "modelStateSHA256":observers[name].model_sha256,
       "originalZip":{"name":output.name,"sha256":sha_file(output),
                      "bytes":output.stat().st_size}}
     records[name]=original_row(patient,item["grade"],rank,name,axes,cam,
                               metrics=metrics,runmeta=meta)
     activations[name]=patches
    if rank==1:
     cross={layer:first[(patient,"S")]["SCEPTcrossModelCKA"][layer] for layer in LAYERS}
     for layer in LAYERS:
      if abs(cross[layer]-first[(patient,"J")]["SCEPTcrossModelCKA"][layer])>1e-12:
       raise RuntimeError("Original first-slice S/J CKA duplicate inconsistent")
    else:
     cross={layer:cka(activations["S"][layer],activations["J"][layer]) for layer in LAYERS}
    pairs.append({"patient_id":patient,"grade":item["grade"],"slice_rank":rank,
                  "axial_index":int(axes[rank-1]),"models":records,"crossModelCKA":cross})
   row={"schema":"scept.a6-ij-03b.patient.v1",
        "planSHA256":psha(),"patient_id":patient,"grade":item["grade"],
        "originalSources":original_checks,
        "originalP1NPZSHA256":item["p1_sha256"],
        "slicePairs":pairs}
   save_once(OUT/"cases"/f"{patient}.json",row)
   count+=1
   print(f"03B [{i}/{len(patients)}] COMPLETE {patient}: 18 new S/J forwards; "
         f"wall={time.monotonic()-start:.1f}s",flush=True)
 complete=sum(validate_complete(OUT/"cases"/f"{p['patient_id']}.json",
                    p["patient_id"],p["grade"],original_checks) is not None for p in patients)
 print(f"A6-IJ-03B CHUNK {args.chunk:02d} {'PILOT' if args.pilot else 'MEASURED'}: "
       f"{complete}/{len(patients)} patient cases; new-case count {count}")
 print("Pilot is NOT full batch; archive only after full five original patients.")

if __name__=="__main__":main()
