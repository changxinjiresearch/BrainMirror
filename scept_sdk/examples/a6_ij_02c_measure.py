"""A6-IJ-02C real SCEPT 55-patient LayerNorm/confound measurement.

Pre/post paired true model outputs in the SAME forward, 64 deterministic
patch-correspondence nulls, blank-input OOD reference, independent direct
PyTorch Hook replay, locked 02B patient-by-patient post-CKA replication.
No test set/training/saliency. Local files contain derived patient activations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
import torch.nn.functional as F
import requests
from kaggle_secrets import UserSecretsClient
from scept import SCEPT,load_observation
from a6_ij_02b_measure import (manifest_and_zip,x_from_npz,checkpoint_model,
  sha_file,token_metrics,gram,cka,private_contents,get_json,REPO,API,ROOT)

STAGES=("pre","affinefree","post")
BLOCKS=(0,5,11)
PRE={0:"norm_pre",5:"blocks.4",11:"blocks.10"}
POST={i:f"blocks.{i}.norm1" for i in BLOCKS}
PROBES=tuple(x for i in BLOCKS for x in (PRE[i],POST[i]))
SEED=240216
PLAN={
 "schema":"scept.a6-ij-02c.plan.v1",
 "base":"a6-ij-02b-v1",
 "patientSelection":"all 55 lexicographic P0 validation patients",
 "sliceIndex":0,"models":["S","J"],"device":"CPU FP32",
 "probePrePost":{str(i):{"pre":PRE[i],"post":POST[i]} for i in BLOCKS},
 "affinefreeMethod":"F.layer_norm(actual_pre, eps, no weight/bias)",
 "layerNormNumericalMaximumAbsDifference":1e-5,
 "independentHookReplayMaxAbs":1e-6,
 "02BPostCKAMaxAbs":1e-5,
 "permutationCount":64,
 "permutationSeed":SEED,
 "blankControl":"all zero FLAIR 224x224 x3; out of distribution",
 "bootstrapResamples":2000,"bootstrapSeed":SEED,
 "scope":"numerical representation controls; NOT lesion importance or causality",
 "inputsIncludedInZIP":False,
 "sealedTest":"NOT_ACCESSED"
}

def plan_sha():
 return hashlib.sha256(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def save_json(path,value):
 tmp=path.with_suffix(".json.partial")
 tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
 os.replace(tmp,path)

def features(model,capture):
 stages={}
 errors={}
 for b in BLOCKS:
  x=capture.tensor(PRE[b])
  y=capture.tensor(POST[b])
  if x.shape!=(1,197,768) or y.shape!=x.shape:
   raise RuntimeError("Invalid residual and LN tensor geometry")
  module=model.blocks[b].norm1
  assert module.normalized_shape==(768,)
  xt=torch.from_numpy(x)
  predicted=F.layer_norm(xt,module.normalized_shape,module.weight.detach(),
                         module.bias.detach(),module.eps).numpy()
  diff=float(np.max(np.abs(predicted.astype(np.float64)-y.astype(np.float64))))
  if diff>PLAN["layerNormNumericalMaximumAbsDifference"]:
   raise RuntimeError(f"Actual named pre/post LayerNorm mismatch at block {b}: {diff}")
  plain=F.layer_norm(xt,module.normalized_shape,None,None,module.eps).numpy()
  stages[str(b)]={"pre":x[0,1:].copy(),"affinefree":plain[0,1:].copy(),
                  "post":y[0,1:].copy()}
  errors[str(b)]=diff
 return stages,errors

def independent_replay(model,x,run):
 # SCEPT's output is already captured; this reruns model with separate direct hooks.
 capture={}
 handles=[]
 for b in BLOCKS:
  layer=model.blocks[b].norm1
  def pre_hook(mod,inputs,index=b):
   capture[f"pre_{index}"]=inputs[0].detach().cpu().numpy().copy()
  def post_hook(mod,inputs,output,index=b):
   capture[f"post_{index}"]=output.detach().cpu().numpy().copy()
  handles.append(layer.register_forward_pre_hook(pre_hook))
  handles.append(layer.register_forward_hook(post_hook))
 try:
  with torch.inference_mode():
   logits=model(x).detach().cpu().numpy()
 finally:
  for h in handles:h.remove()
 diffs={}
 for b in BLOCKS:
  for stage,key in (("pre",PRE[b]),("post",POST[b])):
   a=capture[f"{stage}_{b}"]
   r=run.tensor(key)
   if a.shape!=r.shape:raise RuntimeError("Independent direct PyTorch Hook tensor shape mismatch")
   delta=float(np.max(np.abs(a.astype(np.float64)-r.astype(np.float64))))
   if delta>PLAN["independentHookReplayMaxAbs"]:
    raise RuntimeError(f"Direct Hook reexecution failed {stage} Block {b}: {delta}")
   diffs[f"{stage}_{b}"]=delta
 expected=run.tensor("model.output")
 if logits.shape!=expected.shape:raise RuntimeError("Independent logits shape mismatch")
 delta=float(np.max(np.abs(logits.astype(np.float64)-expected.astype(np.float64))))
 if delta>PLAN["independentHookReplayMaxAbs"]:
  raise RuntimeError("Independent direct-hook logits mismatch")
 diffs["logits"]=delta
 return {"status":"PASS","method":"independent extra PyTorch forward with direct pre/post hooks, NOT SCEPT",
         "maxAbsByTensor":diffs}

def true_observation(model,observer,input_tensor,output_zip,metadata):
 capture=observer.observe(input_tensor,model_id=metadata["modelId"],
    preprocess_id="P1_FLAIR_224_locked_v1",preprocess_config=metadata)
 if len(capture.fingerprint()["groups"])!=6 or any(
     g["status"]!="measured" for g in capture.fingerprint()["groups"]):
  raise RuntimeError("A SCEPT named pre/post probe was not really observed")
 logits=capture.tensor("model.output")
 if logits.shape!=(1,2):raise RuntimeError("Classifier logits shape invalid")
 stage,ln_errors=features(model,capture)
 capture.export(str(output_zip),include_inputs=False)
 reopened=load_observation(str(output_zip))
 if reopened.run_id!=capture.run_id or reopened.manifest["captureScope"]["inputsIncluded"] is not False:
  raise RuntimeError("Full raw SCEPT ZIP run or input privacy invalid")
 if reopened.record["provenance"]["model"]["sha256"]!=observer.model_sha256:
  raise RuntimeError("Actual model fingerprint differs in source archive")
 for p in (*PROBES,"model.output"):
  np.testing.assert_array_equal(reopened.tensor(p),capture.tensor(p))
 if reopened.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
  raise RuntimeError("Offline NAF must not claim another model inference")
 return capture,stage,ln_errors

def as_patch_gram(p):
 return gram(p)

def aligned_cka(a,b):
 A=as_patch_gram(a);B=as_patch_gram(b)
 denominator=np.sqrt(np.sum(A*A)*np.sum(B*B))
 if denominator<=1e-20:raise RuntimeError("Degenerate CKA Gram")
 return float(np.sum(A*B)/denominator),A,B,denominator

def alignment_null(A,B,denom,patient_index,block,stage):
 stage_index=0 if stage=="pre" else 1
 seed=SEED+patient_index*1000+block*10+stage_index
 rng=np.random.default_rng(seed)
 n=A.shape[0]
 values=[]
 for i in range(PLAN["permutationCount"]):
  idx=rng.permutation(n)
  values.append(float(np.sum(A*B[np.ix_(idx,idx)])/denom))
 return {"seed":seed,"count":len(values),"mean":float(np.mean(values)),
         "std":float(np.std(values)),"max":float(np.max(values)),
         "q95":float(np.quantile(values,0.95))}

def summarize_patient(patient,index,grade,sources,model_runs,raw_results,reference):
 result={"schema":"scept.a6-ij-02c.patient.v1","patient_id":patient,
         "grade":grade,"planSha256":plan_sha(),"sliceIndex":0,
         "sourceNPZsha256":sources["p1_sha256"],"models":{},"matched":{}}
 for name in ("S","J"):
  capture,stages,errors=model_runs[name]
  raw=raw_results[name]
  layer_metrics={}
  for b in BLOCKS:
   layer_metrics[str(b)]={"layerNormMaxAbsError":errors[str(b)],
      "prePostCKA":cka(stages[str(b)]["pre"],stages[str(b)]["post"]),
      "postVsAffinefreeCKA":cka(stages[str(b)]["post"],stages[str(b)]["affinefree"]),
      "stages":{stage:token_metrics(stages[str(b)][stage])
                for stage in STAGES}}
  result["models"][name]={
    "runId":capture.run_id,"modelStateSha256":reference[name],
    "zipName":raw.name,"zipSha256":sha_file(raw),"zipBytes":raw.stat().st_size,
    "layers":layer_metrics,
    "logits":capture.tensor("model.output")[0].tolist()}
  if index==0:
   result["models"][name]["independentHookReplay"]=independent_replay(
      model_runs[name][3],model_runs[name][4],capture) if len(model_runs[name])>4 else None
 for b in BLOCKS:
  pair={}
  for st in STAGES:
   a=model_runs["S"][1][str(b)][st]
   v=model_runs["J"][1][str(b)][st]
   score,G,H,denom=aligned_cka(a,v)
   matched={"alignedCKA":score}
   if st in ("pre","post"):
    null=alignment_null(G,H,denom,index,b,st)
    matched["permutedPatchNull"]=null
    matched["alignedMinusNullMean"]=score-null["mean"]
   pair[st]=matched
  delta_02b=abs(pair["post"]["alignedCKA"]-reference["02B"][patient][f"blocks.{b}.norm1"])
  if delta_02b>PLAN["02BPostCKAMaxAbs"]:
   raise RuntimeError(f"02B source CKA is not reproducible for {patient} block {b}: {delta_02b}")
  pair["02BpostCKAAbsoluteDeviation"]=delta_02b
  result["matched"][str(b)]=pair
 return result

def inspect_existing(root,source):
 pid=source["patient_id"]
 fp=root/"cases"/f"{pid}.json"
 if not fp.is_file():return None
 case=json.loads(fp.read_text())
 if (case.get("schema")!="scept.a6-ij-02c.patient.v1" or
     case.get("planSha256")!=plan_sha() or case.get("patient_id")!=pid or
     case.get("grade")!=source["grade"] or case.get("sourceNPZsha256")!=source["p1_sha256"]):
  raise RuntimeError("Existing 02C case provenance differs; no silent overwrites")
 for name in ("S","J"):
  rec=case["models"][name]
  fp=root/"raw"/rec["zipName"]
  if (fp.name!=f"{pid}_{name}_private.zip" or not fp.is_file()
      or sha_file(fp)!=rec["zipSha256"] or fp.stat().st_size!=rec["zipBytes"]):
   raise RuntimeError("Existing case source ZIP incomplete or changed")
  obs=load_observation(str(fp))
  if obs.run_id!=rec["runId"] or obs.record["provenance"]["model"]["sha256"]!=rec["modelStateSha256"]:
   raise RuntimeError("Existing real SCEPT source fingerprint differs")
 return case

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--out",default="/kaggle/working/a6_ij_02c_private")
 p.add_argument("--limit",type=int,default=55,help="operational first-N pilot; NOT official 55-person outcome")
 args=p.parse_args()
 if not 1<=args.limit<=55:raise RuntimeError("Only first 1..55 validation cases are allowed")
 root=Path(args.out)
 (root/"cases").mkdir(parents=True,exist_ok=True)
 (root/"raw").mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Required existing Kaggle PRIVATE GitHub secret not enabled")
 session=requests.Session()
 session.headers.update({"Authorization":"Bearer "+token,
  "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 cohort,zip_path=manifest_and_zip(session)
 study={"schema":"scept.a6-ij-02c.lock.v1","plan":PLAN,"planSha256":plan_sha(),
        "cohortPatientIds":[r["patient_id"] for r in cohort]}
 lock=root/"locked_plan.json"
 if lock.exists() and json.loads(lock.read_text())!=study:
  raise RuntimeError("Frozen 02C study plan changed")
 if not lock.exists():save_json(lock,study)
 prior=json.loads(private_contents(session,
   "results/scept_a6_ij/02b_55_patient_representations/analysis.json"))
 if prior.get("status")!="PASS_55_VAL_EXPLORATORY_ANALYSIS" or len(prior["patientRows"])!=55:
  raise RuntimeError("Pinned completed 02B patient-level source missing")
 prior_by={r["patient_id"]:{k:r["crossModelCKA"][k] for k in
           (f"blocks.{n}.norm1" for n in BLOCKS)} for r in prior["patientRows"]}
 if set(prior_by)!={r["patient_id"] for r in cohort}:
  raise RuntimeError("02B patient cohort differs from 02C")
 old_sha=hashlib.sha256(json.dumps(prior,sort_keys=True).encode()).hexdigest()
 # Audit provenance uses content SHA of original remote bytes, not reconstructed JSON.
 original_bytes=private_contents(session,
   "results/scept_a6_ij/02b_55_patient_representations/analysis.json")
 source_record={"02BAnalysisBytesSHA256":hashlib.sha256(original_bytes).hexdigest()}
 pin=root/"02b_source_sha.json"
 if pin.exists() and json.loads(pin.read_text())!=source_record:
  raise RuntimeError("Previous 02B source changed mid-study")
 if not pin.exists():save_json(pin,source_record)
 if all(inspect_existing(root,r) is not None for r in cohort[:args.limit]):
  print("A6-IJ-02C: all",args.limit,"prior patient artifacts SHA-verified; no rerun")
  return
 models={name:checkpoint_model(name) for name in ("S","J")}
 observers={name:SCEPT(models[name],list(PROBES),backend="torch",
                       profile="vit-token-channel",max_capture_bytes=16_000_000)
            for name in ("S","J")}
 modelsha={name:observers[name].model_sha256 for name in ("S","J")}
 reference={**modelsha,"02B":prior_by}
 with ZipFile(zip_path) as source:
  for index,item in enumerate(cohort[:args.limit]):
   if inspect_existing(root,item) is not None:
    print(f"[{index+1}/{args.limit}] REUSE verified {item['patient_id']}",flush=True)
    continue
   pid=item["patient_id"]
   x=x_from_npz(source,item)
   observations={}
   zipfiles={}
   for name in ("S","J"):
    zip_file=root/"raw"/f"{pid}_{name}_private.zip"
    cap,stage,err=true_observation(models[name],observers[name],x,zip_file,
       {"modelId":f"P4-frozen-{name}","sliceIndex":0,"split":"val",
        "source":"02C-first-P1-FLAIR","patient_id":pid})
    observations[name]=(cap,stage,err)
    zipfiles[name]=zip_file
   row={"schema":"scept.a6-ij-02c.patient.v1","patient_id":pid,
        "grade":item["grade"],"planSha256":plan_sha(),
        "sourceNPZsha256":item["p1_sha256"],"sliceIndex":0,
        "models":{},"matched":{}}
   for name in ("S","J"):
    cap,stages,errs=observations[name]
    entry={"runId":cap.run_id,"modelStateSha256":modelsha[name],
       "zipName":zipfiles[name].name,"zipSha256":sha_file(zipfiles[name]),
       "zipBytes":zipfiles[name].stat().st_size,
       "logits":cap.tensor("model.output")[0].tolist(),
       "layers":{}}
    for b in BLOCKS:
     p=stages[str(b)]
     entry["layers"][str(b)]={
      "layerNormMaxAbsError":errs[str(b)],
      "prePostCKA":cka(p["pre"],p["post"]),
      "postVsAffinefreeCKA":cka(p["post"],p["affinefree"]),
      "stages":{stage:token_metrics(p[stage]) for stage in STAGES}}
    if index==0:entry["independentHookReplay"]=independent_replay(models[name],x,cap)
    row["models"][name]=entry
   for b in BLOCKS:
    row["matched"][str(b)]={}
    for st in STAGES:
     val,K,L,den=aligned_cka(observations["S"][1][str(b)][st],
                              observations["J"][1][str(b)][st])
     result={"alignedCKA":val}
     if st in ("pre","post"):
      null=alignment_null(K,L,den,index,b,st)
      result["permutedPatchNull"]=null
      result["alignedMinusNullMean"]=val-null["mean"]
     row["matched"][str(b)][st]=result
    target=prior_by[pid][f"blocks.{b}.norm1"]
    drift=abs(row["matched"][str(b)]["post"]["alignedCKA"]-target)
    if drift>PLAN["02BPostCKAMaxAbs"]:
     raise RuntimeError(f"02B/02C raw post-normalisation CKA mismatched: {pid} block {b}: {drift}")
    row["matched"][str(b)]["02BpostCKAAbsoluteDeviation"]=drift
   save_json(root/"cases"/f"{pid}.json",row)
   print(f"[{index+1}/{args.limit}] PASS {pid}: independent="
         f"{'YES' if index==0 else 'first-person only'}; "
         f"block11 pre/post S-J CKA="
         f"{row['matched']['11']['pre']['alignedCKA']:.5f}/"
         f"{row['matched']['11']['post']['alignedCKA']:.5f}",flush=True)
 # Fixed first-patient blank baseline OOD forward is executed once per model.
 if args.limit==55:
  blank=torch.zeros((1,3,224,224),dtype=torch.float32)
  for name in ("S","J"):
   path=root/"raw"/f"BLANK_{name}_private.zip"
   if path.exists() and (root/f"blank_{name}.json").exists():
    data=json.loads((root/f"blank_{name}.json").read_text())
    if sha_file(path)!=data["zipSha256"]:
     raise RuntimeError("Existing blank source ZIP SHA mismatch")
    continue
   cap,stages,errs=true_observation(models[name],observers[name],blank,path,
       {"modelId":f"P4-frozen-{name}","control":"OOD_BLANK_ZERO_FLAIR",
        "sliceIndex":0,"split":"synthetic-control"})
   save_json(root/f"blank_{name}.json",{"runId":cap.run_id,"zipName":path.name,
       "zipSha256":sha_file(path),"zipBytes":path.stat().st_size,
       "modelStateSha256":modelsha[name],"layerNormMaxAbs":errs})
 print("A6-IJ-02C MEASUREMENT:",args.limit,"locked validation patients completed/verified")
 print("Next: run 02C analyse and private archive; PASS is a scientific-measurement engineering gate only")

if __name__=="__main__":main()
