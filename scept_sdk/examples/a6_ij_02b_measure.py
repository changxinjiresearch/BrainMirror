"""A6-IJ-02B: 55-patient preregistered REAL SCEPT ViT representation cohort.

Only first frozen P1 FLAIR slice of each of 55 P0 validation patients.
Two frozen P4 S/J models, 3 layers, patch-only Gram structure metrics.
No P5 test, segmentation maps, Grad-CAM recomputation, retraining or spatial
attribution. Raw ZIPs and case-level results stay PRIVATE.
Re-runnable: verifies existing case records and raw ZIPs before skipping.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
import timm
import requests
from kaggle_secrets import UserSecretsClient
from scept import SCEPT,load_observation

REPO="changxinjiresearch/research-course-brats2020-ijepa-xai"
API="https://api.github.com/repos/"+REPO
ROOT=Path("/kaggle/working")
S_SHA="2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf"
J_SHA="719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"
SPLIT_SHA="1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352"
PROTOCOL_SHA="309010d59747eb655f4e5474d4451ec5e9068f67d713f6f48671187f3abcebb0"
LAYERS=("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")
STUDY={
 "schema":"scept.a6-ij-02b.plan.v1",
 "cohort":"locked P0 validation only", "nPatients":55,
 "sampling":"all VAL patients in lexicographic patient ID order; slice index zero of pinned P1 classification cache",
 "labels":"P0 HGG=1 LGG=0; 44 HGG and 11 LGG required",
 "models":{"S":{"checkpointSha256":S_SHA,"epoch":8},
           "J":{"checkpointSha256":J_SHA,"epoch":16}},
 "layerProbes":list(LAYERS),
 "features":"196 patch tokens x768 channels per layer; CLS excluded from topology",
 "metrics":["patchMeanOffDiagonalCosine","patchEffectiveRankNormalized","linearCenteredCKA"],
 "explanationScope":"NONE; no saliency, spatial dependence or medical validity",
 "independentReplay":"same source MRI on first lexicographic VAL patient with direct torch hooks for both models",
 "internalReplayMaxAbsoluteTolerance":1e-6,
 "statistics":"patient-level paired bootstrap (2000, seed 240216), descriptive exploratory intervals, no confirmatory significance tests",
 "P4Comparisons":"optional existing 10-slice patient-level TAP/IoU10; noncausal exploratory Spearman",
 "modelTraining":"NEVER",
 "sealedTest":"NEVER",
}
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def get_json(session,url):
 r=session.get(url,timeout=(30,180));r.raise_for_status();return r.json()
def private_contents(session,path):
 data=get_json(session,f"{API}/contents/{path}")
 if data.get("encoding")!="base64":raise RuntimeError("Unexpected GitHub content encoding")
 return base64.b64decode(data["content"])
def safe_json_write(path,data):
 tmp=path.with_suffix(".json.partial")
 tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 os.replace(tmp,path)
def canonical_plan_hash():
 return hashlib.sha256(json.dumps(STUDY,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def manifest_and_zip(session):
 if get_json(session,API).get("private") is not True:
  raise RuntimeError("Refusing provenance from nonprivate research repository")
 raw=private_contents(session,"results/p0/patient_split_seed2026.csv")
 if hashlib.sha256(raw).hexdigest()!=SPLIT_SHA:raise RuntimeError("Frozen P0 split SHA mismatch")
 rows=list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
 val=sorted((r for r in rows if r["split"]=="val"),key=lambda r:r["patient_id"])
 if len(val)!=55 or sum(r["grade"]=="HGG" for r in val)!=44 or sum(r["grade"]=="LGG" for r in val)!=11:
  raise RuntimeError("Locked 55-patient VAL grade cohort unexpected")
 m=list(csv.DictReader(io.StringIO(private_contents(session,"results/p1/p1_classification_manifest.csv").decode("utf-8"))))
 if len(m)!=len({r["patient_id"] for r in m}):raise RuntimeError("P1 duplicate patient IDs")
 rows_by={r["patient_id"]:r for r in m}
 inventory=json.loads(private_contents(session,"artifact_integrity/all_files_sha256.json"))
 inv={r["path"]:r["sha256"] for r in inventory}
 release=get_json(session,f"{API}/releases/tags/p0-p2-2026-10-13")
 assets=[a for a in release["assets"] if a["name"].startswith("p1_classification_npz-") and a["name"].endswith(".zip")]
 if len(assets)!=1:raise RuntimeError("No uniquely locked source P1 ZIP")
 asset=assets[0]
 prefix=asset["name"].rsplit("-",1)[1][:-4]
 zip_path=ROOT/"a6_ij_models"/"p1_classification_npz.zip"
 if not zip_path.is_file() or zip_path.stat().st_size!=asset["size"] or not sha_file(zip_path).startswith(prefix):
  raise RuntimeError("SHA-verified P1 ZIP missing: first rerun existing a6_ij_restore_p4.py")
 checks=[]
 for r in val:
  patient=r["patient_id"]
  row=rows_by.get(patient)
  rel=f"classification_npz/val/{patient}.npz"
  digest=inv.get("results/p1/"+rel)
  if row is None or row["split"]!="val" or row["grade"]!=r["grade"] or row["npz_relative_path"]!=rel:
   raise RuntimeError("P1 patient classification mismatch: "+patient)
  if not digest or len(digest)!=64:raise RuntimeError("P1 VAL patient absent from SHA inventory")
  checks.append({"patient_id":patient,"grade":r["grade"],
                 "p1_sha256":digest,"zip_member":rel.split("classification_npz/",1)[1]})
 with ZipFile(zip_path) as z:
  members=set(z.namelist())
  if any(r["zip_member"] not in members for r in checks):
   raise RuntimeError("Pre-verified P1 ZIP missing a locked VAL patient")
 return checks,zip_path

def x_from_npz(z,item):
 raw=z.read(item["zip_member"])
 if hashlib.sha256(raw).hexdigest()!=item["p1_sha256"]:
  raise RuntimeError("Frozen P1 patient exact NPZ SHA mismatch: "+item["patient_id"])
 with np.load(io.BytesIO(raw),allow_pickle=False) as npz:
  if "flair" not in npz.files or npz["flair"].shape!=(10,224,224):
   raise RuntimeError("Unrecognized preprocessed P1 validation slice geometry")
  flair=np.asarray(npz["flair"][0],dtype=np.float32).copy()
 if not np.isfinite(flair).all():raise RuntimeError("Nonfinite source patient data")
 x=torch.from_numpy(flair)[None,None].repeat(1,3,1,1).contiguous()
 return x

def gram(patch):
 # 196 samples (spatial PATCHES), 768 feature channels, no CLS.
 x=np.asarray(patch,dtype=np.float64)
 if x.shape!=(196,768) or not np.isfinite(x).all():raise RuntimeError("Invalid patch feature geometry")
 centered=x-x.mean(axis=0,keepdims=True)
 k=centered@centered.T
 if float(np.trace(k))<=1e-20:raise RuntimeError("No nondegenerate patch representation")
 return k

def token_metrics(patch):
 x=np.asarray(patch,dtype=np.float64)
 g=gram(x)
 # Mean off-diagonal cosine between patch features; magnitude-normalized.
 norm=np.linalg.norm(x,axis=1)
 if np.min(norm)<=1e-12:raise RuntimeError("Zero-norm patch token")
 unit=x/norm[:,None]
 cosine=unit@unit.T
 offdiag=float((cosine.sum()-np.trace(cosine))/(196*195))
 vals=np.maximum(np.linalg.eigvalsh(g),0)
 positive=vals[vals>max(vals.max()*1e-12,1e-20)]
 prob=positive/positive.sum()
 erank=float(np.exp(-np.sum(prob*np.log(prob))))
 # Centering yields <=195 nonzero effective dimensions, not 196.
 return {"patchMeanOffDiagonalCosine":offdiag,
         "patchEffectiveRank":erank,
         "patchEffectiveRankNormalized":erank/195}
def cka(X,Y):
 a=gram(X);b=gram(Y)
 den=np.sqrt(np.sum(a*a)*np.sum(b*b))
 if den<=1e-30:raise RuntimeError("CKA zero denominator")
 v=float(np.sum(a*b)/den)
 if not -1e-10<=v<=1+1e-10:raise RuntimeError("Invalid centered CKA result")
 return max(0.,min(1.,v))
def checkpoint_model(name):
 expected,epoch=(S_SHA,8) if name=="S" else (J_SHA,16)
 path=ROOT/"a6_ij_models"/f"model_{name.lower()}_best.pt"
 if sha_file(path)!=expected:raise RuntimeError("Frozen checkpoint SHA mismatch: "+name)
 ck=torch.load(path,map_location="cpu",weights_only=True)
 if (ck.get("epoch")!=epoch or ck.get("protocol_hash")!=PROTOCOL_SHA or
     ck.get("p0_split_sha256")!=SPLIT_SHA or ck.get("label_map")!={"LGG":0,"HGG":1}):
  raise RuntimeError("P4 checkpoint provenance rejected")
 model=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=2,in_chans=3)
 model.load_state_dict(ck["model"],strict=True)
 del ck
 model.eval()
 return model

def independent_hook_replay(model,x,observed):
 # A DIFFERENT measurement implementation, without calling SCEPT.observe.
 got={}
 handles=[]
 for layer in LAYERS:
  module=dict(model.named_modules())[layer]
  def keep(_m,_in,out,key=layer):got[key]=out.detach().cpu().numpy().copy()
  handles.append(module.register_forward_hook(keep))
 try:
  with torch.inference_mode():pred=model(x)
  got["model.output"]=pred.detach().cpu().numpy().copy()
 finally:
  for h in handles:h.remove()
 measurements={}
 for key in (*LAYERS,"model.output"):
  a=got[key];b=observed[key]
  if a.shape!=b.shape:raise RuntimeError("Independent Hook output shape mismatch")
  diff=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
  if diff>STUDY["internalReplayMaxAbsoluteTolerance"]:
   raise RuntimeError(f"INDEPENDENT TORCH HOOK REPLAY FAIL {key} max absolute difference {diff}")
  measurements[key]={"maxAbsoluteDifference":diff,"exactEqual":bool(np.array_equal(a,b))}
 return {"status":"PASS","method":"separate PyTorch forward and direct named module forward hooks (without SCEPT)", "tensors":measurements}

def validate_existing(root,item):
 patient=item["patient_id"]
 file=root/"cases"/f"{patient}.json"
 if not file.exists():return None
 row=json.loads(file.read_text())
 if (row.get("schema")!="scept.a6-ij-02b.case.v1" or row.get("planSha256")!=canonical_plan_hash()
   or row.get("patient_id")!=patient or row.get("grade")!=item["grade"]
   or row.get("sourceNPZsha256")!=item["p1_sha256"]):
  raise RuntimeError("Existing case manifest differs; refuse rerun/overwrite: "+patient)
 for name in ("S","J"):
  x=row["modelResults"][name]
  if x["checkpointSha256"] != (S_SHA if name=="S" else J_SHA):raise RuntimeError("Existing checkpoint differs")
  zip_file=root/"raw"/x["zipName"]
  if not zip_file.exists() or sha_file(zip_file)!=x["zipSha256"]:
   raise RuntimeError("Existing raw archive missing/different: "+patient+"/"+name)
  o=load_observation(str(zip_file))
  if o.run_id!=x["runId"] or o.manifest["captureScope"]["inputsIncluded"] is not False:
   raise RuntimeError("Existing raw archive provenance/privacy differs")
 return row

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--out",default="/kaggle/working/a6_ij_02b_private")
 parser.add_argument("--limit",type=int,default=55,help="optional first N rows for operational pilot only; NOT official cohort PASS")
 args=parser.parse_args()
 if not 1<=args.limit<=55:raise RuntimeError("--limit must be 1 to 55")
 root=Path(args.out);(root/"cases").mkdir(parents=True,exist_ok=True)
 (root/"raw").mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Kaggle private GitHub read token not enabled")
 session=requests.Session()
 session.headers.update({"Authorization":"Bearer "+token,
  "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 cohort,source_zip=manifest_and_zip(session)
 cfg={"plan":STUDY,"planSha256":canonical_plan_hash(),"cohortPatients":[r["patient_id"] for r in cohort],
      "scope":"P1 VAL ONLY","testAccess":"NEVER","targetN":55}
 config=root/"locked_plan.json"
 if config.exists() and json.loads(config.read_text())!=cfg:
  raise RuntimeError("Local study plan drift; do not silently change settings")
 if not config.exists():safe_json_write(config,cfg)
 pending=[item for item in cohort[:args.limit] if validate_existing(root,item) is None]
 if not pending:
  print("A6-IJ-02B: preexisting first",args.limit,"cases verified; no new inference")
  return
 models={name:checkpoint_model(name) for name in ("S","J")}
 observers={name:SCEPT(models[name],list(LAYERS),backend="torch",profile="vit-token-channel",max_capture_bytes=16_000_000)
            for name in ("S","J")}
 with ZipFile(source_zip) as z:
  for i,item in enumerate(cohort[:args.limit],1):
   if validate_existing(root,item) is not None:
    print(f"[{i}/{args.limit}] REUSE VERIFIED {item['patient_id']}",flush=True)
    continue
   patient=item["patient_id"]
   x=x_from_npz(z,item)
   result={"schema":"scept.a6-ij-02b.case.v1","planSha256":canonical_plan_hash(),
           "patient_id":patient,"grade":item["grade"],"sliceIndex":0,"sourceNPZsha256":item["p1_sha256"],
           "modelResults":{},"crossModelLinearCKA":{}}
   features={}
   for name in ("S","J"):
    run=observers[name].observe(x,model_id=f"P4-frozen-{name}",
       preprocess_id="P1_FLAIR_224_locked_v1",
       preprocess_config={"patient_id":patient,"sliceIndex":0,"split":"val","channels":"FLAIRx3"})
    exact={key:run.tensor(key) for key in (*LAYERS,"model.output")}
    if any(exact[l].shape!=(1,197,768) for l in LAYERS) or exact["model.output"].shape!=(1,2):
     raise RuntimeError("Captured ViT layer or classifier geometry incorrect")
    naf=run.fingerprint()
    if len(naf["groups"])!=3 or any(g["status"]!="measured" or len(g["units"])!=197 for g in naf["groups"]):
     raise RuntimeError("Original SCEPT NAF incomplete")
    path=root/"raw"/f"{patient}_{name}_local_private.zip"
    run.export(str(path),include_inputs=False)
    verified=load_observation(str(path))
    if verified.run_id!=run.run_id or verified.manifest["captureScope"]["inputsIncluded"] is not False:
     raise RuntimeError("Capture-archive identity/privacy mismatch")
    for key in (*LAYERS,"model.output"):
     np.testing.assert_array_equal(verified.tensor(key),exact[key])
    if verified.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
     raise RuntimeError("Offline NAF incorrectly claims independent model replay")
    features[name]={layer:exact[layer][0,1:,:] for layer in LAYERS}
    logits=exact["model.output"][0].astype(np.float64)
    p_hgg=float(1/(1+np.exp(np.clip(logits[0]-logits[1],-500,500))))
    row={"runId":run.run_id,
      "checkpointSha256":S_SHA if name=="S" else J_SHA,
      "modelStateSha256":observers[name].model_sha256,
      "zipName":path.name,"zipSha256":sha_file(path),"zipBytes":path.stat().st_size,
      "pHggFirstSlice":p_hgg,
      "logitsLggHgg":logits.tolist(),
      "layerMetrics":{layer:token_metrics(features[name][layer]) for layer in LAYERS},
      "withinModelCKA":{
       "block0_to_5":cka(features[name][LAYERS[0]],features[name][LAYERS[1]]),
       "block5_to_11":cka(features[name][LAYERS[1]],features[name][LAYERS[2]]),
       "block0_to_11":cka(features[name][LAYERS[0]],features[name][LAYERS[2]])}}
    if patient==cohort[0]["patient_id"]:
     row["independentPyTorchHookReplay"]=independent_hook_replay(models[name],x,exact)
    result["modelResults"][name]=row
   result["crossModelLinearCKA"]={layer:cka(features["S"][layer],features["J"][layer]) for layer in LAYERS}
   safe_json_write(root/"cases"/f"{patient}.json",result)
   print(f"[{i}/{args.limit}] DONE {patient}: block11 CKA(S,J)={result['crossModelLinearCKA'][LAYERS[-1]]:.4f}; "
         "two actual ZIPs verified",flush=True)
 print("A6-IJ-02B MEASURE: COMPLETE FIRST",args.limit,"VALIDATION PATIENTS.")
 print("Full 55 cohort required for official analysis; raw patient observations are PRIVATE.")

if __name__=="__main__":main()
