#!/usr/bin/env python3
"""Publication-style visual summaries of *actual* preregistered U11 data.

No generated/computed result is substituted for authentic summary/case JSON.
Negative effects are shown, not clipped or recolored to appear favorable.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["scept","occlusion","rise","random","scept_no_nuisance"]
TITLES = ["SCEPT v1","Occlusion","RISE-like","Random","SCEPT no noise"]


def make_figures(summary, cases, folder):
    if summary["schema"] != "scept.u11.research.benchmark.v0.1":
        raise ValueError("Unknown scientific results schema")
    if len(cases) != summary["data"]["n"]:
        raise ValueError("Case/result count mismatch")
    folder.mkdir(parents=True,exist_ok=True)
    metrics=summary["statistics"]["methods"]

    for field,title,better in (
        ("deletion_auc","Deletion AUC: lower is better",False),
        ("insertion_auc","Insertion AUC: higher is better",True)
    ):
        scores=[metrics[k][field] for k in METHODS]
        if any(v is None or not np.isfinite(v) for v in scores):
            raise ValueError("Cannot plot absent or nonfinite actual scores")
        fig,ax=plt.subplots(figsize=(9.6,5.2))
        ax.bar(np.arange(len(METHODS)),scores)
        ax.set_xticks(np.arange(len(METHODS)),TITLES,rotation=12,ha="right")
        ax.set_ylabel("Mean true-class probability AUC")
        ax.set_ylim(0,1)
        ax.set_title(f"SCEPT U11 / authentic MNIST-12\n{title} | n={summary['statistics']['n_primary']} paired correct images")
        for i,v in enumerate(scores):
            ax.text(i,v+.017,f"{v:.3f}",ha="center",va="bottom",fontsize=10)
        ax.text(.01,-.22,"Real pretrained CNN / fixed zero-fill procedure; one MNIST task. NOT proof of clinical relevance.",
                transform=ax.transAxes,fontsize=8)
        fig.tight_layout()
        fig.savefig(folder/f"{field}.png",dpi=180,bbox_inches="tight")
        fig.savefig(folder/f"{field}.svg",bbox_inches="tight")
        plt.close(fig)

    comparisons=summary["statistics"]["comparisons"]
    if len(comparisons)!=6:
        raise ValueError("Expected six original Holm-corrected comparisons")
    labels=[x["baseline"].title()+" / "+x["outcome"].replace("_auc","") for x in comparisons]
    effects=np.array([c["signed_improvement_mean"] for c in comparisons],np.float64)
    ci=np.array([c["bootstrap_95ci"] for c in comparisons],np.float64)
    fig,ax=plt.subplots(figsize=(10,6))
    ys=np.arange(len(comparisons))
    ax.errorbar(effects,ys,xerr=np.vstack([effects-ci[:,0],ci[:,1]-effects]),fmt="o",capsize=4)
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(ys,labels)
    ax.invert_yaxis()
    ax.set_xlabel("SCEPT improvement versus baseline (positive = better)")
    ax.set_title("U11: exact paired bootstrap 95% intervals\nNegative results retained; six registered Holm comparisons")
    ax.grid(axis="x",alpha=.2)
    for i,x in enumerate(comparisons):
        ax.text(.995,1-(i+.85)/len(comparisons),
                f"Holm p={x['holm_adjusted_p']:.4f}",transform=ax.transAxes,
                ha="right",fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/"paired_fidelity_forest.png",dpi=180,bbox_inches="tight")
    fig.savefig(folder/"paired_fidelity_forest.svg",bbox_inches="tight")
    plt.close(fig)

    observed=[int(x["dose_violations"]) for x in cases]
    if sum(observed)!=summary["quality"]["dose_violations"]:
        raise ValueError("Failed scientific dose-violation sum cross-check")
    fig,ax=plt.subplots(figsize=(8.5,5))
    ax.hist(observed,bins=min(20,max(5,len(set(observed)))))
    ax.set_xlabel("Observed within-seed monotonicity violations per original image")
    ax.set_ylabel("Number of original test images (not independent patches)")
    ax.set_title(f"U11 preregistered graded-sensitivity falsification\n"
                 f"{sum(observed)} total patch/seed violations across {len(cases)} images")
    fig.tight_layout()
    fig.savefig(folder/"dose_violations_by_image.png",dpi=180,bbox_inches="tight")
    fig.savefig(folder/"dose_violations_by_image.svg",bbox_inches="tight")
    plt.close(fig)
    return sorted(x.name for x in folder.iterdir())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--cases",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    summary=json.loads(args.summary.read_text())
    cases=json.loads(args.cases.read_text())
    created=make_figures(summary,cases,args.out)
    print("U11 REAL FIGURES:",", ".join(created),flush=True)
