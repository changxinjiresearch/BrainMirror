"""A6-IJ: one frozen matched ViT S/J real SCEPT observation (no spatial attribution).

Prerequisites: locally restored SHA-verified P4 best model checkpoints and one
authorized P1 validation NPZ. No test patients, training, model download, or upload.
Usage: python scept_sdk/examples/a6_ijepa_p4_observe.py --s S.pt --j J.pt --npz VAL.npz --out /kaggle/working/a6_ij
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

S_SHA = "2700714f09102fa27f3ad5846c7000da87e99dcf086159f9170be48920229ccf"
J_SHA = "719df79922161ad706634f963a11760d5f5a380356d63ab1ef501520c4d53640"
SPLIT_SHA = "1d7b423759bd11771235e2cee670d19476ac22b855e26190c2e3a1338a576352"
PROTOCOL_SHA = "309010d59747eb655f4e5474d4451ec5e9068f67d713f6f48671187f3abcebb0"
LAYERS = ("blocks.0.norm1", "blocks.5.norm1", "blocks.11.norm1")

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s",required=True,help="P4 frozen best Model S checkpoint")
    parser.add_argument("--j",required=True,help="P4 frozen best Model J checkpoint")
    parser.add_argument("--npz",required=True,help="authorized P1 VAL classification NPZ")
    parser.add_argument("--out",required=True,help="local private directory (not GitHub public)")
    parser.add_argument("--slice-index",type=int,default=0)
    args=parser.parse_args()
    import torch
    import timm
    from scept import SCEPT,load_observation
    if not 0<=args.slice_index<10:
        raise ValueError("Slice index must be in [0,9]")
    if sha(args.s)!=S_SHA or sha(args.j)!=J_SHA:
        raise RuntimeError("Frozen P4 best checkpoint SHA256 mismatch")
    with np.load(args.npz,allow_pickle=False) as source:
        if "flair" not in source.files or source["flair"].shape!=(10,224,224):
            raise RuntimeError("P1 classification NPZ must have 10 FLAIR slices of 224x224")
        flair=np.array(source["flair"][args.slice_index],dtype=np.float32,copy=True)
    if not np.isfinite(flair).all():
        raise RuntimeError("Nonfinite input")
    # P4 uses 3 identical FLAIR channels without input augmentation.
    x=torch.from_numpy(flair)[None,None].repeat(1,3,1,1)
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name,filename,epoch in (("S",args.s,8),("J",args.j,16)):
        checkpoint=torch.load(filename,map_location="cpu",weights_only=True)
        if checkpoint.get("epoch")!=epoch or checkpoint.get("protocol_hash")!=PROTOCOL_SHA or checkpoint.get("p0_split_sha256")!=SPLIT_SHA or checkpoint.get("label_map")!={"LGG":0,"HGG":1}:
            raise RuntimeError("Frozen P4 checkpoint provenance rejected: "+name)
        model=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=2,in_chans=3)
        model.load_state_dict(checkpoint["model"],strict=True)
        model.eval()
        # CPU runs to avoid silent CUDA-dependent numeric differences; no gradients/training.
        observer=SCEPT(model,list(LAYERS),backend="torch",profile="vit-token-channel",max_capture_bytes=16_000_000)
        run=observer.observe(x,preprocess_id="P1_FLAIR_224_locked_v1",preprocess_config={"sliceIndex":args.slice_index,"P4":"validation-only","channels":"FLAIR-repeated-3"},model_id="P4-frozen-ViT-"+name)
        logits=run.tensor("model.output")
        if logits.shape!=(1,2) or not np.isfinite(logits).all():
            raise RuntimeError("True classifier logits absent")
        naf=run.fingerprint()
        for group in naf["groups"]:
            if group["status"]!="measured" or group["shape"]!=[1,197,768] or len(group["units"])!=197:
                raise RuntimeError("ViT NAF geometry / authenticity check failed")
        archive=output/("A6_IJ_"+name+"_local_private.zip")
        run.export(str(archive),include_inputs=False)
        loaded=load_observation(str(archive))
        for tensor in run.tensors:
            np.testing.assert_array_equal(loaded.tensor(tensor),run.tensor(tensor))
        assert loaded.recompute_naf()["science"]["model_forward_reexecuted"] is False
        reports.append({"model":name,"checkpointSha256":sha(filename),"sourceModelStateSha256":observer.model_sha256,
                        "sourceInputSha256":sha(args.npz),"selectedSlice":args.slice_index,
                        "runId":run.run_id,"tensorCount":len(run.tensors),
                        "probedLayers":list(LAYERS),"logits":logits[0].tolist(),
                        "zipSha256":sha(archive),"status":"PASS_REAL_OBSERVATION_AND_OFFLINE_RECOMPUTE",
                        "modelRerun":"NOT_PERFORMED","causalAttribution":"NOT_ESTABLISHED"})
        del observer,model,checkpoint
    summary={"schema":"scept.a6-ij.v1","status":"PASS_REAL_OBSERVATION_ONLY",
             "dataset":"P1 authorized validation slice; never test","modelSAndJ":reports,
             "spatialEvidence":"NOT_EVALUATED","clinicalEvidence":"NOT_ESTABLISHED"}
    (output/"a6_ij_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":summary["status"],"models":[r["model"] for r in reports],
                      "recorded":str(output/"a6_ij_summary.json"),"privacy":"keep ZIP local"},indent=2))

if __name__=="__main__":
    main()
