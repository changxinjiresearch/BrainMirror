"""A6-IJ-02D real MRI: frozen ViT independent source replay, ResNet-18 Torch/ONNX.

Two SHA256-verified original P0 VAL patients. No model training, TEST access,
clinical assessment, attribution or inferred importance. ResNet weights are
deterministic RANDOM (untrained), only for genuine architecture/engine QA.
All derived-patient tensors stay private and input ZIPs exclude original MRI.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import io
import json
import os
import platform
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import requests
import torch
import timm
from kaggle_secrets import UserSecretsClient
from scept import SCEPT,load_observation
from a6_ij_02b_measure import (
  manifest_and_zip,x_from_npz,checkpoint_model,private_contents,get_json,
  sha_file,REPO,API,ROOT
)
from a6_ij_02c_measure import PRE,POST,BLOCKS,PROBES

OUT=Path("/kaggle/working/a6_ij_02d_private")
COHORT=("BraTS20_Training_022","BraTS20_Training_040")
PREVIOUS_TAG="scept-a6-ij-02c-ln-position-controls-v1"
MAX_REPLAY_ABS=1e-6
MAX_CNN_HOOK_ABS=1e-6
MAX_CROSS_ENGINE_ABS=1e-3
MAX_CROSS_ENGINE_REL=1e-3
PLAN={"schema":"scept.a6-ij-02d.plan.v1",
 "cohort":list(COHORT),"source":"locked P0 VAL P1 first FLAIR slice (one per patient)",
 "vit":["S","J"],"vitIndependentReplayProbeNames":list(PROBES),
 "vitDirectHookMaxAbs":MAX_REPLAY_ABS,
 "cnnArchitecture":"actual timm ResNet-18 randomly initialized, NOT medically trained",
 "cnnSeed":240216,"cnnOutputs":["net.layer1","net.layer3","model.output.2"],
 "cnnHookMaxAbs":MAX_CNN_HOOK_ABS,
 "cnnTorchVsONNXMaxAbs":MAX_CROSS_ENGINE_ABS,
 "cnnTorchVsONNXRelativeL2":MAX_CROSS_ENGINE_REL,
 "onnxOpset":17,"onnxProvider":"CPUExecutionProvider",
 "engines":["PyTorch CPU","ONNX Runtime CPU"],"crossOS":"separate clean GitHub runners",
 "sensitiveRawData":"PRIVATE ONLY; include_inputs=False",
 "test":"NOT_ACCESSED","scientificValidity":"NOT_ESTABLISHED"}

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def plan_sha():return sha_bytes(json.dumps(PLAN,sort_keys=True,separators=(",",":")).encode())
def persist(name,data):
 path=OUT/name
 raw=(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
 tmp=path.with_suffix(".partial")
 tmp.write_bytes(raw)
 os.replace(tmp,path)
 return path
def relative(a,b):
 a=np.asarray(a,dtype=np.float64)
 b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape:raise RuntimeError("Cross-run tensor shape discrepancy")
 delta=b-a
 return {"maxAbs":float(np.max(np.abs(delta))),
         "relativeL2":float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(a.ravel()),1e-12)),
         "exactEqual":bool(np.array_equal(a,b))}

def request_asset(session,asset,target,digest):
 if target.is_file() and sha_file(target)==digest:return
 temp=target.with_suffix(".partial")
 with session.get(f"{API}/releases/assets/{asset['id']}",
   headers={"Accept":"application/octet-stream"},stream=True,allow_redirects=True,
   timeout=(30,900)) as result:
  result.raise_for_status()
  if "json" in result.headers.get("Content-Type","").lower():
   raise RuntimeError("Private Release fetch returned metadata instead of source ZIP")
  with temp.open("wb") as output:
   for part in result.iter_content(chunk_size=4*1024*1024):
    if part:output.write(part)
 if sha_file(temp)!=digest or temp.stat().st_size!=asset["size"]:
  temp.unlink(missing_ok=True)
  raise RuntimeError("Authoritative old 02C GitHub source shard SHA mismatch")
 os.replace(temp,target)

def private_02c_cases(session):
 receipt=json.loads(private_contents(session,
   "results/scept_a6_ij/02c_ln_position_controls/archive_receipt.json"))
 if receipt.get("status")!="PRIVATE_55_PATIENT_112_REAL_SCEPT_OBSERVATIONS_ARCHIVED_AND_VERIFIED":
  raise RuntimeError("A6-IJ-02C private reference archive not complete")
 src=[x for x in receipt["remoteShards"] if x["filename"]=="scept_02c_val_00_patients_01-05.zip"]
 if len(src)!=1:raise RuntimeError("First-five reference cohort shard not uniquely found")
 entry=src[0]
 release=get_json(session,f"{API}/releases/tags/{PREVIOUS_TAG}")
 if release.get("prerelease") is not True:raise RuntimeError("Original source release changed visibility/policy")
 assets=[x for x in release["assets"] if x["name"]==entry["filename"]]
 if len(assets)!=1:raise RuntimeError("Original private patient shard missing")
 candidate=assets[0]
 if candidate["size"]!=entry["bytes"]:raise RuntimeError("Original release asset size differs from receipt")
 target=OUT/entry["filename"]
 request_asset(session,candidate,target,entry["sha256"])
 row={}
 with ZipFile(target) as archive:
  for patient in COHORT:
   cpath=f"cases/{patient}.json"
   metadata=json.loads(archive.read(cpath))
   if metadata["patient_id"]!=patient or metadata["sliceIndex"]!=0:
    raise RuntimeError("Previous patient source record identity mismatched")
   sources={}
   for name in ("S","J"):
    e=metadata["models"][name]
    path=f"raw/{patient}_{name}_private.zip"
    if path!=f"raw/{e['zipName']}":
     raise RuntimeError("Original patient/model SCEPT archive missing")
    raw=archive.read(path)
    if sha_bytes(raw)!=e["zipSha256"]:
     raise RuntimeError("Original inner SCEPT ZIP changed relative to case record")
    original=load_observation(raw)
    if original.run_id!=e["runId"] or original.manifest["captureScope"]["inputsIncluded"] is not False:
     raise RuntimeError("Original SCEPT run identity or privacy changed")
    if {g["tensor"] for g in original.recompute_naf()["groups"] if g["status"]=="measured"}!=set(PROBES):
     raise RuntimeError("Original SCEPT LN pre/post probes missing")
    sources[name]={"caseMeta":e,"observation":original,"sourceZipSha256":e["zipSha256"]}
   row[patient]=sources
 return row,{"shardFile":target.name,"shardSha256":entry["sha256"],
             "originalReceiptStatus":receipt["status"],"cohort":list(COHORT)}

def direct_vit(model,x):
 data={}
 layers=dict(model.named_modules())
 handles=[]
 for b in BLOCKS:
  module=layers[POST[b]]
  def pre(_m,inputs,idx=b):
   data[PRE[idx]]=inputs[0].detach().cpu().numpy().copy()
  def post(_m,inputs,result,idx=b):
   data[POST[idx]]=result.detach().cpu().numpy().copy()
  handles.append(module.register_forward_pre_hook(pre))
  handles.append(module.register_forward_hook(post))
 try:
  model.eval()
  with torch.inference_mode():
   output=model(x)
  data["model.output"]=output.detach().cpu().numpy().copy()
 finally:
  for h in handles:h.remove()
 if set(data)!=set(PROBES)|{"model.output"}:
  raise RuntimeError("Independent ViT direct Hook missed original probe")
 return data

class Resnet18Probe(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.net=timm.create_model("resnet18",pretrained=False,num_classes=2,in_chans=3)
 def forward(self,x):
  n=self.net
  x=n.conv1(x)
  x=n.bn1(x)
  x=n.act1(x)
  x=n.maxpool(x)
  a=n.layer1(x)
  x=n.layer2(a)
  b=n.layer3(x)
  x=n.layer4(b)
  logits=n.fc(n.global_pool(x))
  return a,b,logits

def direct_cnn(net,x):
 found={}
 names=("net.layer1","net.layer3")
 lookup=dict(net.named_modules())
 hooks=[]
 for name in names:
  def get(_module,_inputs,outputs,key=name):
   found[key]=outputs.detach().cpu().numpy().copy()
  hooks.append(lookup[name].register_forward_hook(get))
 try:
  net.eval()
  with torch.inference_mode():outs=net(x)
 finally:
  for h in hooks:h.remove()
 found["model.output.0"]=outs[0].detach().cpu().numpy().copy()
 found["model.output.1"]=outs[1].detach().cpu().numpy().copy()
 found["model.output.2"]=outs[2].detach().cpu().numpy().copy()
 return found

def onnx_export(net,x,path):
 import onnx
 kwargs={"opset_version":17,"input_names":["flair_rgb"],
         "output_names":["layer1","layer3","logits"],"do_constant_folding":True}
 if "dynamo" in inspect.signature(torch.onnx.export).parameters:
  kwargs["dynamo"]=False
 torch.onnx.export(net,x,str(path),**kwargs)
 model=onnx.load(str(path))
 onnx.checker.check_model(model)
 outputs={entry.name for entry in model.graph.output}
 if outputs!={"layer1","layer3","logits"}:
  raise RuntimeError("Actual exported ONNX output names differ")
 if path.stat().st_size>150_000_000:
  raise RuntimeError("Exported CNN unexpectedly too large for SCEPT original graph budget")
 return path

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--out",default=str(OUT))
 args=p.parse_args()
 # The directory is pre-registered; reject arbitrary redirects for leakage prevention.
 if Path(args.out).resolve()!=OUT.resolve():
  raise RuntimeError("Private archive destination is locked; no arbitrary output path allowed")
 OUT.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Private research source Kaggle secret GITHUB_TOKEN missing")
 s=requests.Session()
 s.headers.update({"Authorization":"Bearer "+token,
    "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 info=get_json(s,API)
 if info.get("private") is not True or info.get("full_name")!=REPO:
  raise RuntimeError("Required PRIVATE research repository not verified")
 cohort,source_zip=manifest_and_zip(s)
 if tuple(item["patient_id"] for item in cohort[:2])!=COHORT:
  raise RuntimeError("Pinned first two VAL patients changed")
 original,receipt=private_02c_cases(s)
 prev=OUT/"a6_ij_02d_report.json"
 if prev.exists():
  source=json.loads(prev.read_text())
  if source.get("planSha256")!=plan_sha():raise RuntimeError("Existing immutable 02D report has incompatible protocol")
  raise RuntimeError("02D previous complete report exists; run archive-only, do not overwrite source")
 with ZipFile(source_zip) as source:
  inputs={item["patient_id"]:x_from_npz(source,item) for item in cohort[:2]}
  input_sha={item["patient_id"]:item["p1_sha256"] for item in cohort[:2]}
 if torch.equal(inputs[COHORT[0]],inputs[COHORT[1]]):
  raise RuntimeError("First two pinned MRI inputs are identical; test invalid")
 result={
  "schema":"scept.a6-ij-02d.real-mri.v1",
  "plan":PLAN,"planSha256":plan_sha(),
  "status":"INCOMPLETE",
  "cohort":"P1 locked VAL first slice, two patients",
  "patientNPZsha256":input_sha,
  "source02C":receipt,
  "environment":{"python":platform.python_version(),"torch":torch.__version__,
                 "timm":timm.__version__,"platform":platform.platform(),
                 "torchThreads":torch.get_num_threads()},
  "frozenViTReplay":{},"crossArchitectureCNN":{},
  "scope":"ENGINEERING ONLY - CNN RANDOM UNTRAINED; NONCLINICAL; NO TEST SET"}
 vit_models={name:checkpoint_model(name) for name in ("S","J")}
 for name,model in vit_models.items():
  checks={}
  for patient in COHORT:
   fresh=direct_vit(model,inputs[patient])
   source=original[patient][name]["observation"]
   comparison={}
   for key in (*PROBES,"model.output"):
    comparison[key]=relative(source.tensor(key),fresh[key])
    if comparison[key]["maxAbs"]>MAX_REPLAY_ABS:
     raise RuntimeError(f"02D independent ViT replay original 02C archive differs: {name}/{patient}/{key}")
   checks[patient]={"status":"PASS","directHookErrors":comparison,
                    "source02COriginalZipSha256":original[patient][name]["sourceZipSha256"],
                    "sourceModelStateSha256":source.record["provenance"]["model"]["sha256"]}
   print(f"ViT {name}/{patient}: independent 02C ZIP direct Hook replay PASS",flush=True)
  result["frozenViTReplay"][name]=checks
 del vit_models
 # Actual timm ResNet-18: random deterministic weights, NOT medically trained.
 torch.manual_seed(PLAN["cnnSeed"])
 cnn=Resnet18Probe().eval()
 probe=("net.layer1","net.layer3")
 observe=SCEPT(cnn,probe,backend="torch",profile="cnn-nchw",max_capture_bytes=32_000_000)
 resnet_sha=observe.model_sha256
 weights_path=OUT/"resnet18_seed240216_untrained_state.pt"
 torch.save(cnn.state_dict(),weights_path)
 npz_results={}
 real_zip_meta=[]
 for patient in COHORT:
  x=inputs[patient]
  captured=observe.observe(x,model_id="timm-resnet18-random-seed240216",
    preprocess_id="P1_FLAIR_224_locked_v1",
    preprocess_config={"patient_id":patient,"slice":0,"split":"val","architecture":"ResNet18","weightTraining":"NONE"})
  for group in captured.fingerprint()["groups"]:
   if group.get("status")!="measured":raise RuntimeError("Genuine CNN NAF group was not measured")
  expected=direct_cnn(cnn,x)
  errors={}
  for key in expected:
   errors[key]=relative(captured.tensor(key),expected[key])
   if errors[key]["maxAbs"]>MAX_CNN_HOOK_ABS:
    raise RuntimeError("Direct ResNet-18 forward Hook not equal to SCEPT: "+key)
  zip_path=OUT/f"resnet18_torch_{patient}_private.zip"
  captured.export(str(zip_path),include_inputs=False)
  imported=load_observation(str(zip_path))
  if imported.run_id!=captured.run_id or imported.manifest["captureScope"]["inputsIncluded"] is not False:
   raise RuntimeError("CNN Torch ZIP privacy/identity mismatch")
  for key in captured.tensors:np.testing.assert_array_equal(imported.tensor(key),captured.tensor(key))
  if imported.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
   raise RuntimeError("Offline NAF should never claim independent model reexecution")
  npz_results[patient]={"torchRunId":captured.run_id,
   "torchDirectHooks":errors,
   "torchZip":{"name":zip_path.name,"sha256":sha_file(zip_path),"bytes":zip_path.stat().st_size},
   "torchTensors":{k:captured.tensor(k) for k in (*probe,"model.output.2")}}
  print(f"CNN ResNet-18 PyTorch {patient}: two conv groups, logits + direct Hook PASS",flush=True)
 onnx_path=OUT/"resnet18_seed240216_untrained.onnx"
 onnx_export(cnn,inputs[COHORT[0]],onnx_path)
 onnx_sha=sha_file(onnx_path)
 import onnxruntime as ort
 onnx_model=SCEPT(onnx_path,("layer1","layer3"),backend="onnx",
    profile="cnn-nchw",providers=("CPUExecutionProvider",),max_capture_bytes=32_000_000)
 onnx_results={}
 for patient in COHORT:
  x=inputs[patient]
  run=onnx_model.observe({"flair_rgb":x.numpy()},model_id="resnet18-frozen-ONNX-opset17",
   preprocess_id="P1_FLAIR_224_locked_v1",
   preprocess_config={"patient_id":patient,"slice":0,"split":"val",
                      "architecture":"ResNet18","weights":"random-untrained"})
  measured={g["tensor"] for g in run.fingerprint()["groups"] if g["status"]=="measured"}
  if measured!={"layer1","layer3"}:
   raise RuntimeError("ONNX SCEPT did not capture two real CNN graph tensors")
  errors={}
  for one,two in (("net.layer1","layer1"),("net.layer3","layer3"),
                  ("model.output.2","logits")):
   err=relative(npz_results[patient]["torchTensors"][one],run.tensor(two))
   if err["maxAbs"]>MAX_CROSS_ENGINE_ABS or err["relativeL2"]>MAX_CROSS_ENGINE_REL:
    raise RuntimeError(f"ResNet18 PyTorch vs ONNX numeric mismatch {patient} {one}: {err}")
   errors[one]=err
  fp=OUT/f"resnet18_onnx_{patient}_private.zip"
  run.export(str(fp),include_inputs=False)
  v=load_observation(str(fp))
  if v.run_id!=run.run_id or v.manifest["captureScope"]["inputsIncluded"] is not False:
   raise RuntimeError("ONNX original archive privacy/run mismatch")
  for name in run.tensors:
   np.testing.assert_array_equal(v.tensor(name),run.tensor(name))
  if v.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
   raise RuntimeError("ONNX offline numeric recomputation incorrectly claims new inference")
  onnx_results[patient]={"runId":run.run_id,"engineNumericErrors":errors,
    "archive":{"name":fp.name,"sha256":sha_file(fp),"bytes":fp.stat().st_size}}
  print(f"CNN ResNet18 cross-engine PyTorch/ONNX {patient}: PASS",flush=True)
 output_diffs={}
 for name in (*probe,"model.output.2"):
  x=npz_results[COHORT[0]]["torchTensors"][name]
  y=npz_results[COHORT[1]]["torchTensors"][name]
  d=relative(x,y)
  if d["relativeL2"]<1e-5:
   raise RuntimeError("Real CNN internal response failed A/B validation smoke in "+name)
  output_diffs[name]=d
 for patient in COHORT:
  del npz_results[patient]["torchTensors"]
 report={"status":"PASS_ENGINEERING_CROSS_ARCHITECTURE_AND_CROSS_ENGINE",
         "schema":"scept.a6-ij-02d.real-mri.v1","planSha256":plan_sha(),
         "plan":PLAN,"source02C":receipt,"patientNPZsha256":input_sha,
         "environment":result["environment"],
         "frozenViTReplay":result["frozenViTReplay"],
         "cnn":{
           "architecture":"timm resnet18 genuine CNN",
           "weightTraining":"NONE - deterministic random initialization, NOT medically trained",
           "randomSeed":PLAN["cnnSeed"],"stateSha256":resnet_sha,
           "torchModelWeights":{"name":weights_path.name,"sha256":sha_file(weights_path),"bytes":weights_path.stat().st_size},
           "onnxFile":{"name":onnx_path.name,"sha256":onnx_sha,"bytes":onnx_path.stat().st_size},
           "torchObservations":npz_results,"onnxObservations":onnx_results,
           "twoPatientInputDifferenceByLayer":output_diffs,
           "crossBackendMaxAbsTolerance":MAX_CROSS_ENGINE_ABS,
           "crossBackendRelativeL2Tolerance":MAX_CROSS_ENGINE_REL,
           "onnxRuntimeVersion":ort.__version__},
         "scope":"NO PATIENT CLINICAL CLAIMS, NO TEST SET, NO SCIENTIFIC XAI FAITHFULNESS"}
 persist("a6_ij_02d_report.json",report)
 persist("locked_plan.json",{"plan":PLAN,"planSha256":plan_sha(),"source02C":receipt})
 print("A6-IJ-02D REAL MRI CROSS-ARCH/ENGINE REPLAY: PASS")
 print("PRIVATE:",OUT/"a6_ij_02d_report.json")
 print("Archive all derived-patient artefacts to private research repo before Kaggle session ends.")

if __name__=="__main__":main()
