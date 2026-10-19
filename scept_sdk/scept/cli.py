"""Read-only SCEPT scientific archive verifier. Never loads executable code from ZIP."""
import argparse
import json
import sys
from .archive import load_observation
from .contracts import SCEPTError

def main(argv=None):
    parser=argparse.ArgumentParser(
        prog="scept-archive",
        description="Verify real binary SHA/shape/NAF; this does NOT authenticate the claimed model.")
    parser.add_argument("action",choices=("verify","recompute"))
    parser.add_argument("zip",help="A5 Python or SCEPT V6.5.2 full-observation ZIP")
    args=parser.parse_args(argv)
    try:
        observed=load_observation(args.zip)
        result={"status":"archive_file_integrity_and_NAF_verified",
                "modelOriginIndependentlyAuthenticated":False,
                "modelForwardReexecuted":False,
                "runId":observed.run_id,
                "recordSchema":observed.record["schema"],
                "modelSha256Claimed":observed.record["provenance"]["model"]["sha256"],
                "outputTensors":len(observed.tensors),
                "inputIncluded":bool(observed.input_names)}
        if args.action=="recompute":
            naf=observed.recompute_naf()
            result["replay"]="offline_NAF_numeric_recalculation_only"
            result["measuredGroups"]=sum(x["status"]=="measured" for x in naf["groups"])
            result["observedUnits"]=sum(len(x["units"]) for x in naf["groups"])
        print(json.dumps(result,ensure_ascii=False,sort_keys=True))
        return 0
    except (SCEPTError,ValueError,OSError,KeyError,TypeError) as e:
        print("Archive validation failed: "+str(e),file=sys.stderr)
        return 2

if __name__=="__main__":raise SystemExit(main())
