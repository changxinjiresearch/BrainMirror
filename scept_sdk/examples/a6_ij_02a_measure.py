"""A6-IJ-02A: preregistered SCEPT repeatability and input-sensitivity smoke.

A,A,B,A same-model forwards per frozen P4 ViT S/J. A and B are the first
two lexicographic VAL patients in the FROZEN P0 cohort. Not a P5 test.
Full raw activation ZIPs are sensitive: local/private GitHub release ONLY.
No model retraining, spatial maps, attribution or hypothesis tuning.
"""
from __future__ import annotations
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import requests
import torch
import timm
from kaggle_secrets import UserSecretsClient
from scept import SCEPT, load_observation

ROOT=Path("/kaggle/working")
REPO="changxinjiresearch/research-course-brats2020-ijepa-xai"
API="https://api.github.com/repos/"+REPO
S_SHA="2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf"
J_SHA="719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"
SPLIT_SHA="1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352"
PROTOCOL_SHA="309010d59747eb655f4e5474d4451ec5e9068f67d713f6f48671187f3abcebb0"
LAYERS=("blocks.0.norm1","blocks.5.norm1","blocks.11.norm1")
ORDER=("A1","A2","B1","A3")
PRELOCK={
 "schema":"scept.a6-ij-02a.plan.v1",
 "patientSelection":"first_two_lexicographic_VAL_from_frozen_P0",
 "sliceIndex":0,
 "runOrder":list(ORDER),
 "models":{"S":{"epoch":8,"sha256":S_SHA},"J":{"epoch":16,"sha256":J_SHA}},
 "layers":list(LAYERS),
 "repeatMaxAbsoluteTolerance":1e-6,
 "repeatRelativeL2Tolerance":1e-7,
 "sensitivityRelativeL2Threshold":1e-3,
 "sensitivityMinimumChangedLayers":2,
 "profile":"vit-token-channel",
 "input":"P1 original 224x224 FP32 FLAIR repeated to 3 channels",
 "device":"CPU",
 "privacy":"never publish patient-derived ZIPs; no inputs in ZIP",
 "interpretation":"raw internal numerical state and input responsiveness, not saliency"
}

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as src:
  for part in iter(lambda:src.read(8*1024*1024),b""):
   h.update(part)
 return h.hexdigest()

def http_get_json(client,url):
 r=client.get(url,timeout=(30,180))
 r.raise_for_status()
 return r.json()

def private_source(client,path):
 payload=http_get_json(client,f"{API}/contents/{path}")
 if payload.get("encoding")!="base64":raise RuntimeError("Unexpected GitHub content encoding")
 return base64.b64decode(payload["content"])

def restore_two_val(client):
 split_raw=private_source(client,"results/p0/patient_split_seed2026.csv")
 if hashlib.sha256(split_raw).hexdigest()!=SPLIT_SHA:
  raise RuntimeError("Frozen P0 cohort SHA256 mismatch")
 patients=sorted(r["patient_id"] for r in csv.DictReader(io.StringIO(split_raw.decode("utf-8"))) if r["split"]=="val")
 if len(patients)!=55 or patients[:2]!=["BraTS20_Training_022","BraTS20_Training_040"]:
  raise RuntimeError("Prerecorded first two VAL patient IDs not matched")
 m=list(csv.DictReader(io.StringIO(private_source(client,"results/p1/p1_classification_manifest.csv").decode("utf-8"))))
 by_id={r["patient_id"]:r for r in m}
 inventory=json.loads(private_source(client,"artifact_integrity/all_files_sha256.json"))
 lookup={r["path"]:r["sha256"] for r in inventory}
 release=http_get_json(client,f"{API}/releases/tags/p0-p2-2026-10-13")
 matches=[x for x in release["assets"] if x["name"].startswith("p1_classification_npz-") and x["name"].endswith(".zip")]
 if len(matches)!=1:raise RuntimeError("No unique P1 frozen ZIP release asset")
 asset=matches[0]
 prefix=asset["name"].rsplit("-",1)[-1].removesuffix(".zip")
 archive=ROOT/"a6_ij_models"/"p1_classification_npz.zip"
 if not archive.is_file() or archive.stat().st_size!=asset["size"] or not file_sha(archive).startswith(prefix):
  raise RuntimeError("Original SHA-verified P1 ZIP absent; run existing a6_ij_restore_p4.py first")
 paths={}
 with ZipFile(archive) as z:
  for patient in patients[:2]:
   row=by_id.get(patient)
   rel=f"classification_npz/val/{patient}.npz"
   if not row or row["split"]!="val" or row["npz_relative_path"]!=rel:
    raise RuntimeError("VAL P1 metadata mismatch: "+patient)
   expected=lookup.get("results/p1/"+rel)
   if not expected or len(expected)!=64:raise RuntimeError("NPZ not in frozen source inventory")
   dest=ROOT/"research_course_p1"/rel
   dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.is_file() or file_sha(dest)!=expected:
    member=rel.split("classification_npz/",1)[1]
    if member not in z.namelist():raise RuntimeError("Missing P1 VAL ZIP member: "+member)
    tmp=dest.with_suffix(".npz.partial")
    with z.open(member) as src,tmp.open("wb") as out:
     for part in iter(lambda:src.read(8*1024*1024),b""):
      out.write(part)
    if file_sha(tmp)!=expected:
     tmp.unlink(missing_ok=True)
     raise RuntimeError("P1 extracted patient fails full SHA256")
    os.replace(tmp,dest)
   paths[patient]=dest
 return patients[:2],paths

def load_x(path):
 with np.load(path,allow_pickle=False) as z:
  if "flair" not in z.files or z["flair"].shape!=(10,224,224):
   raise RuntimeError("Frozen P1 source must be exactly 10 FLAIR slices")
  a=np.array(z["flair"][0],dtype=np.float32,copy=True)
 if not np.isfinite(a).all():raise RuntimeError("Nonfinite P1 original slice")
 return torch.from_numpy(a)[None,None].repeat(1,3,1,1)

def deltas(a,b):
 a=np.asarray(a,dtype=np.float64)
 b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape:raise RuntimeError("Captured layer tensor shapes changed")
 d=b-a
 return {"maxAbs":float(np.max(np.abs(d))),
         "relativeL2":float(np.linalg.norm(d.reshape(-1))/max(float(np.linalg.norm(a.reshape(-1))),1e-12)),
         "exactEqual":bool(np.array_equal(a,b))}

def observed_model(name,path,epoch,inputs,out):
 checkpoint=torch.load(path,map_location="cpu",weights_only=True)
 if (checkpoint.get("epoch")!=epoch or checkpoint.get("protocol_hash")!=PROTOCOL_SHA or
     checkpoint.get("p0_split_sha256")!=SPLIT_SHA or checkpoint.get("label_map")!={"LGG":0,"HGG":1}):
  raise RuntimeError("Wrong frozen P4 best model checkpoint")
 model=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=2,in_chans=3)
 model.load_state_dict(checkpoint["model"],strict=True)
 model.eval()
 del checkpoint
 observer=SCEPT(model,list(LAYERS),backend="torch",profile="vit-token-channel",max_capture_bytes=16000000)
 records=[]
 anchors=None
 for step in ORDER:
  x=inputs["A"] if step.startswith("A") else inputs["B"]
  capture=observer.observe(x,model_id=f"P4-best-{name}",preprocess_id="P1_FLAIR_224_locked_v1",
      preprocess_config={"source":"P1 VAL frozen","slice":0,"inputAlias":step[:1]})
  exact={key:capture.tensor(key) for key in [*LAYERS,"model.output"]}
  if exact["model.output"].shape!=(1,2) or any(exact[k].shape!=(1,197,768) for k in LAYERS):
   raise RuntimeError("Actual ViT output or probe geometry invalid")
  naf=capture.fingerprint()
  if len(naf["groups"])!=3 or any(g["status"]!="measured" or len(g["units"])!=197 for g in naf["groups"]):
   raise RuntimeError("Real NAF could not measure all preregistered tokens")
  zip_file=out/f"{name}_{step}_private.zip"
  capture.export(str(zip_file),include_inputs=False)
  checked=load_observation(str(zip_file))
  if checked.run_id!=capture.run_id or checked.manifest["captureScope"]["inputsIncluded"] is not False:
   raise RuntimeError("ZIP capture ID / input privacy mismatch")
  for key in exact:np.testing.assert_array_equal(checked.tensor(key),exact[key])
  if checked.recompute_naf()["science"]["model_forward_reexecuted"] is not False:
   raise RuntimeError("Offline NAF improperly claims model reexecution")
  if anchors is None:anchors={key:v.copy() for key,v in exact.items()}
  layer_stats={}
  for layer in LAYERS:
   grp=next(g for g in naf["groups"] if g["tensor"]==layer)
   rms=np.array([u["rms"] for u in grp["units"]],dtype=np.float64)
   layer_stats[layer]={"patchRmsMean":float(rms[1:].mean()),"patchRmsStd":float(rms[1:].std())}
  diffs={key:deltas(anchors[key],exact[key]) for key in exact}
  records.append({"step":step,"inputAlias":step[0],"runId":capture.run_id,
                  "archiveName":zip_file.name,"archiveSha256":file_sha(zip_file),
                  "archiveBytes":zip_file.stat().st_size,"logitsLggHgg":exact["model.output"][0].tolist(),
                  "NAF":layer_stats,"vsA1":diffs})
  print(f"{name}/{step}: measured 3 real layers + logits, SHA-verified ZIP; "
        f"Block11 relative L2={diffs['blocks.11.norm1']['relativeL2']:.9f}",flush=True)
 repeat=[r for r in records if r["step"] in ("A2","A3")]
 repeated_keys=[*LAYERS,"model.output"]
 repeated_pass=all(r["vsA1"][k]["maxAbs"]<=PRELOCK["repeatMaxAbsoluteTolerance"] and
                   r["vsA1"][k]["relativeL2"]<=PRELOCK["repeatRelativeL2Tolerance"]
                   for r in repeat for k in repeated_keys)
 comparison=next(r for r in records if r["step"]=="B1")
 changed=[k for k in LAYERS if comparison["vsA1"][k]["relativeL2"]>=PRELOCK["sensitivityRelativeL2Threshold"]]
 sensitivity_pass=len(changed)>=PRELOCK["sensitivityMinimumChangedLayers"]
 return {"model":name,"modelCheckpointSha256":file_sha(path),
         "modelStateSha256":observer.model_sha256,"records":records,
         "repeatability":{"passed":repeated_pass,"testedSteps":["A2","A3"],
                          "thresholds":{"maxAbs":PRELOCK["repeatMaxAbsoluteTolerance"],
                                        "relativeL2":PRELOCK["repeatRelativeL2Tolerance"]}},
         "inputSensitivity":{"passed":sensitivity_pass,"changedLayers":changed,
                             "relativeL2Threshold":PRELOCK["sensitivityRelativeL2Threshold"]},
         "gate":"PASS" if repeated_pass and sensitivity_pass else "FAIL"}
    
def main():
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument("--out",default="/kaggle/working/a6_ij_02a_private")
 args=parser.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 token=UserSecretsClient().get_secret("GITHUB_TOKEN")
 if not token:raise RuntimeError("Enable private research GitHub read Secret GITHUB_TOKEN")
 client=requests.Session()
 client.headers.update({"Authorization":"Bearer "+token,
   "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
 repo=http_get_json(client,API)
 if repo.get("private") is not True:raise RuntimeError("Research repo must remain private")
 patients,files=restore_two_val(client)
 xA,xB=[load_x(files[p]) for p in patients]
 if torch.equal(xA,xB):raise RuntimeError("Different patient inputs identical; no sensitivity experiment")
 input_difference=deltas(xA.numpy(),xB.numpy())
 inputs={"A":xA,"B":xB}
 model_files=ROOT/"a6_ij_models"
 for label,expected in (("s",S_SHA),("j",J_SHA)):
  p=model_files/f"model_{label}_best.pt"
  if not p.is_file() or file_sha(p)!=expected:
   raise RuntimeError("Missing SHA-verified best checkpoint "+label+"; run existing restorer")
 report={
  "schema":"scept.a6-ij-02a.report.v1","plan":PRELOCK,
  "planSha256":hashlib.sha256(json.dumps(PRELOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
  "patientIds":patients,"patientNPZSha256":{p:file_sha(files[p]) for p in patients},
  "inputDifference":input_difference,
  "cohort":"P1 VAL ONLY","testSet":"NOT_ACCESSED",
  "nPatients":2,"nSlicesPerPatient":1,"runsPerModel":4,
  "science":{"spatialEvidence":"NOT_EVALUATED","causalDependence":"NOT_ESTABLISHED",
             "clinicalValidity":"NOT_ESTABLISHED","generalization":"NOT_ESTABLISHED"},
 }
 results=[]
 for name,filename,epoch in (("S",model_files/"model_s_best.pt",8),
                              ("J",model_files/"model_j_best.pt",16)):
  results.append(observed_model(name,filename,epoch,inputs,out))
 report["models"]=results
 report["status"]="PASS_ENGINEERING_RELIABILITY_AND_INPUT_SENSITIVITY" if all(x["gate"]=="PASS" for x in results) else "FAIL_OR_NOT_DETECTED"
 target=out/"a6_ij_02a_report.json"
 target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print("\nA6-IJ-02A:",report["status"])
 print("Model gates:",{r["model"]:r["gate"] for r in results})
 print("Saved PRIVATE locally:",target)
 print("NO spatial attribution, no population-level inference, no P5 test use.")
 print("Archive private data before ending Kaggle session.")

if __name__=="__main__":main()
