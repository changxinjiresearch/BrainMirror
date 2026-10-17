#!/usr/bin/env python3
"""Independent U11 provenance/actual U10 input-matching audit (no re-training).

Checks whether every original selected numeric P,Q ROI still obeys frozen U10
mean/std tolerances after the shared seeded nuisance is applied to B/C.
Nonzero failures are SCIENTIFIC DEVIATIONS; preserve them and do not hide.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def load_script(path):
    spec = importlib.util.spec_from_file_location("u11_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit(module, cases, images, labels):
    r = {"schema": "scept.u11.matching-parity-audit.v1",
         "test": "exact U10 target/control ROI mean/std matching after nuisance assignment",
         "thresholds": {"mean": module.CTRL_MEAN_TOL, "sd": module.CTRL_SD_TOL},
         "seeds": list(module.PRE_REGISTERED_SEEDS),
         "n_images": len(cases),
         "paired_cells_checked": 0,
         "violations": [],
         "contrasts_checked": 0,
         "null_failures": [],
         "anatomical_matching": "NOT_ESTABLISHED"}
    for row in cases:
        original_index = row["case_id"]
        if int(labels[original_index]) != row["label"]:
            raise ValueError(f"LABEL_PROVENANCE_MISMATCH {original_index}")
        x=np.asarray(images[original_index].reshape(-1),np.float32)/np.float32(255)
        pairs={m["p"]:m["q"] for m in row["matching_log"] if m["q"] is not None}
        if len(pairs)!=row["matched_n"]:
            raise ValueError("LOGGED_MASK_COVERAGE_CONTRADICTION")
        for seed in module.PRE_REGISTERED_SEEDS:
            z=module.u10_noise(x,seed)
            for p,q in pairs.items():
                a,b=module.stat(z,module.TILES[p]),module.stat(z,module.TILES[q])
                dm=abs(a[0]-b[0]);ds=abs(a[1]-b[1])
                r["paired_cells_checked"]+=1
                if dm>module.CTRL_MEAN_TOL or ds>module.CTRL_SD_TOL:
                    r["violations"].append({"case_id":original_index,"seed":seed,
                                            "target":p,"control":q,
                                            "mean_diff":dm,"std_diff":ds})
        for item in row["scept_measures"]:
            r["contrasts_checked"]+=1
            if item["zero_dose_null"]>1e-5:
                r["null_failures"].append({"case_id":original_index,
                                          "patch":item["p"],"effect":item["zero_dose_null"]})
    r["n_violations"]=len(r["violations"])
    r["n_null_failures"]=len(r["null_failures"])
    r["status"]="PASS" if not r["violations"] and not r["null_failures"] else "SCIENTIFIC_PROTOCOL_DEVIATION"
    return r


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=Path,required=True)
    ap.add_argument("--assets",type=Path,required=True)
    ap.add_argument("--script",type=Path,default=Path("scripts/u11_real_mnist_benchmark.py"))
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    mod=load_script(args.script)
    source=mod.read_idx(args.assets/mod.FILES[0],2051)
    labels=mod.read_idx(args.assets/mod.FILES[1],2049)
    cases=json.loads(args.cases.read_text())
    result=audit(mod,cases,source,labels)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("U11 U10 MATCHING PARITY:",json.dumps({
        "status":result["status"],"checked":result["paired_cells_checked"],
        "violations":result["n_violations"],"null_failures":result["n_null_failures"]}))
    if result["n_violations"] or result["n_null_failures"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
