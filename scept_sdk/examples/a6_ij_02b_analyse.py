"""A6-IJ-02B: private 55-patient real ZIP reanalysis, exploratory statistics."""
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
from scept import load_observation
from a6_ij_02b_measure import (
    LAYERS, STUDY, canonical_plan_hash, sha_file, token_metrics, cka,
    private_contents, get_json, API
)

ROOT=Path("/kaggle/working/a6_ij_02b_private")
def confidence(x,rng):
    a=np.asarray(x,dtype=np.float64)
    if len(a)==0 or not np.isfinite(a).all():raise RuntimeError("Invalid bootstrap sample")
    means=a[rng.integers(len(a),size=(2000,len(a)))].mean(axis=1)
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),
            "mean95Bootstrap":list(map(float,np.quantile(means,[0.025,0.975])))}

def verified_case(pid,grade):
    fp=ROOT/"cases"/f"{pid}.json"
    saved=json.loads(fp.read_text())
    if saved["patient_id"]!=pid or saved["grade"]!=grade or saved["planSha256"]!=canonical_plan_hash():
        raise RuntimeError("Patient-level source mismatch "+pid)
    data={"patient_id":pid,"grade":grade}
    patches={}
    for name in ("S","J"):
        entry=saved["modelResults"][name]
        zp=ROOT/"raw"/entry["zipName"]
        if zp.name!=f"{pid}_{name}_local_private.zip" or sha_file(zp)!=entry["zipSha256"]:
            raise RuntimeError("Authentic local ZIP missing/changed")
        obs=load_observation(str(zp))
        if obs.run_id!=entry["runId"] or obs.manifest["captureScope"]["inputsIncluded"] is not False:
            raise RuntimeError("Archived SCEPT run identity/privacy invalid")
        measured={g["tensor"] for g in obs.recompute_naf()["groups"] if g["status"]=="measured"}
        if measured!=set(LAYERS):raise RuntimeError("NAF not measured in all three layers")
        patches[name]={}
        metrics={}
        for layer in LAYERS:
            x=obs.tensor(layer)
            if x.shape!=(1,197,768):raise RuntimeError("Unexpected ViT tensor shape")
            patches[name][layer]=x[0,1:]
            fresh=token_metrics(patches[name][layer])
            if any(abs(fresh[k]-entry["layerMetrics"][layer][k])>1e-10 for k in fresh):
                raise RuntimeError("Saved metrics do not match original raw tensor")
            metrics[layer]=fresh
        logits=obs.tensor("model.output")
        if logits.shape!=(1,2) or not np.array_equal(logits[0],np.asarray(entry["logitsLggHgg"],dtype=logits.dtype)):
            raise RuntimeError("Output logits differ from saved original")
        depth={
          "block0_to_5":cka(patches[name][LAYERS[0]],patches[name][LAYERS[1]]),
          "block5_to_11":cka(patches[name][LAYERS[1]],patches[name][LAYERS[2]]),
          "block0_to_11":cka(patches[name][LAYERS[0]],patches[name][LAYERS[2]])}
        if any(abs(depth[k]-entry["withinModelCKA"][k])>1e-10 for k in depth):
            raise RuntimeError("Within-model depth CKA differs from original archive")
        data[name]={"layers":metrics,"pHggSlice0":entry["pHggFirstSlice"],
                    "withinModelCKA":depth}
        if "independentPyTorchHookReplay" in entry:
            replay=entry["independentPyTorchHookReplay"]
            if replay["status"]!="PASS" or any(x["maxAbsoluteDifference"]>1e-6 for x in replay["tensors"].values()):
                raise RuntimeError("Original independently reexecuted model failed Hook comparison")
            data[name]["independentHookReplay"]="PASS"
    cross={}
    for layer in LAYERS:
        value=cka(patches["S"][layer],patches["J"][layer])
        if abs(value-saved["crossModelLinearCKA"][layer])>1e-10:
            raise RuntimeError("S/J CKA differs from raw source archives")
        cross[layer]=value
    data["crossModelCKA"]=cross
    return data

def main():
    plan=json.loads((ROOT/"locked_plan.json").read_text())
    if plan["plan"]!=STUDY or plan["planSha256"]!=canonical_plan_hash():
        raise RuntimeError("Study plan changed")
    token=UserSecretsClient().get_secret("GITHUB_TOKEN")
    if not token:raise RuntimeError("Private GitHub source secret missing")
    s=requests.Session()
    s.headers.update({"Authorization":"Bearer "+token,
                      "Accept":"application/vnd.github+json",
                      "X-GitHub-Api-Version":"2022-11-28"})
    if get_json(s,API).get("private") is not True:raise RuntimeError("Research repo privacy not verified")
    raw=private_contents(s,"results/p0/patient_split_seed2026.csv")
    if hashlib.sha256(raw).hexdigest()!="1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352":
        raise RuntimeError("Authoritative P0 split SHA mismatch")
    val=sorted([r for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))) if r["split"]=="val"],
               key=lambda x:x["patient_id"])
    ids=[r["patient_id"] for r in val]
    if len(ids)!=55 or ids!=plan["cohortPatients"]:raise RuntimeError("Full locked VAL cohort required")
    records=[verified_case(r["patient_id"],r["grade"]) for r in val]
    if any(records[0][m].get("independentHookReplay")!="PASS" for m in ("S","J")):
        raise RuntimeError("Independent non-SCEPT PyTorch Hook replay not validated for both models")
    p4raw=private_contents(s,"results/p4b/p4b_patient_metrics.csv")
    p4rows=list(csv.DictReader(io.StringIO(p4raw.decode())))
    p4={(r["patient_id"],r["model"]):r for r in p4rows}
    if len(p4rows)!=110 or len(p4)!=110:raise RuntimeError("P4B duplicated or incomplete")
    for row in val:
        for model in ("S","J"):
            r=p4.get((row["patient_id"],model))
            if r is None or r["true_grade"]!=row["grade"] or r["complete"]!="True" or int(r["n_valid_slices"])!=10:
                raise RuntimeError("P4B validation outcome mismatch")
    p4report=private_contents(s,"results/p4b/p4b_report.json")
    original=json.loads(p4report)
    if original["status"]!="PASS" or original["patient_count"]!=55:raise RuntimeError("P4B original PASS missing")
    rng=np.random.default_rng(240216)
    aggregate={}
    for layer in LAYERS:
        aggregate[layer]={}
        for key in ("patchEffectiveRankNormalized","patchMeanOffDiagonalCosine"):
            S=np.array([r["S"]["layers"][layer][key] for r in records])
            J=np.array([r["J"]["layers"][layer][key] for r in records])
            d=J-S
            h=np.array([r["grade"]=="HGG" for r in records])
            aggregate[layer][key]={"S":confidence(S,rng),"J":confidence(J,rng),
              "pairedJminusS":confidence(d,rng),
              "HGGpairedJminusS":confidence(d[h],rng),"LGGpairedJminusS":confidence(d[~h],rng)}
        aggregate[layer]["crossModelCKA"]=confidence([r["crossModelCKA"][layer] for r in records],rng)
    for model in ("S","J"):
        aggregate[model+"_withinModelCKA"]={
          k:confidence([r[model]["withinModelCKA"][k] for r in records],rng)
          for k in ("block0_to_5","block5_to_11","block0_to_11")}
    pid=ids
    tap=[float(p4[(p,"J")]["tap"])-float(p4[(p,"S")]["tap"]) for p in pid]
    iou=[float(p4[(p,"J")]["iou10"])-float(p4[(p,"S")]["iou10"]) for p in pid]
    last=LAYERS[-1]
    def delta(key):
        return [r["J"]["layers"][last][key]-r["S"]["layers"][last][key] for r in records]
    def corr(a,b):
        rho=float(spearmanr(a,b).statistic)
        if not np.isfinite(rho):raise RuntimeError("Nonfinite Spearman rank correlation")
        return rho
    d_rank=delta("patchEffectiveRankNormalized")
    d_cos=delta("patchMeanOffDiagonalCosine")
    spearman={
      "block11RankDelta_vs_P4B_TAP_Delta":corr(d_rank,tap),
      "block11RankDelta_vs_P4B_IoU10_Delta":corr(d_rank,iou),
      "block11CosineDelta_vs_P4B_TAP_Delta":corr(d_cos,tap),
      "block11CosineDelta_vs_P4B_IoU10_Delta":corr(d_cos,iou)}
    report={"schema":"scept.a6-ij-02b.analysis.v1","status":"PASS_55_VAL_EXPLORATORY_ANALYSIS",
      "planSha256":canonical_plan_hash(),"nPatients":55,"nSlicesPerPatient":1,
      "grades":{"HGG":44,"LGG":11},"independentReplay":"PASS_S_AND_J",
      "P4BPatientCsvSha256":hashlib.sha256(p4raw).hexdigest(),
      "P4BReportSha256":hashlib.sha256(p4report).hexdigest(),
      "aggregate":aggregate,"P4BDescriptiveSpearman":spearman,"patientRows":records,
      "limits":["Validation set may have influenced P4 model selection; exploratory only.",
        "SCEPT uses first selected P1 slice; P4B patient TAP and IoU10 aggregate 10 slices.",
        "Positional embeddings, learned LayerNorm parameters and anatomy are confounders.",
        "CKA, effective rank and token cosine are NOT causal, spatial or clinical importance.",
        "No test set, P6 external validation, patient-level clinical conclusion or superiority claim."]}
    output=ROOT/"a6_ij_02b_analysis.json"
    tmp=output.with_suffix(".json.partial")
    tmp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,output)
    print("A6-IJ-02B 55-PATIENT EXPLORATORY ANALYSIS: PASS")
    for l in LAYERS:
        v=aggregate[l]["patchEffectiveRankNormalized"]["pairedJminusS"]
        c=aggregate[l]["crossModelCKA"]["mean"]
        print(f"{l}: paired effective-rank J-S={v['mean']:+.6f}, bootstrap95={v['mean95Bootstrap']}; cross-model CKA={c:.6f}")
    print("P4B correlation exploratory only:",spearman)
    print("Independent direct PyTorch Hook replay: PASS S/J")
    print("PRIVATE patient-level result:",output)

if __name__=="__main__":main()
