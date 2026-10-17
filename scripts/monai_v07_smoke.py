"""Synthetic engineering smoke test against official MONAI 1.5.1, no MRI data."""
import argparse
import json
from pathlib import Path
import torch
from brainmirror.scept.monai_adapter import (
    AttentionAtom, MonaiSwinObserver, MonaiDecoderTrace,
    decompose_monai_attention, atomic_bidirectional_effect, build_monai_model)
from brainmirror.scept.monai_vib import MonaiSwinVIBBridge


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--json-out',default='runs/scept_monai_v07_synthetic.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(240216)
    model=build_monai_model(depths=(2,1,1,1)).eval()
    x=torch.randn(1,4,32,32,32)
    altered=x.clone();altered[:,:,:6,:6,:6]*=-1
    obs=MonaiSwinObserver(model)
    with torch.no_grad(),obs.recording(capture_tensors=True):
        normal=model(x)
    diffs=[]
    for name,site in obs.windows.items():
        m=obs.sites[name]
        with torch.no_grad():
            out=m(site['input'],mask=site['mask'])
            replay=decompose_monai_attention(m,site['input'],site['mask'])['output']
        diffs.append({'module':name,'max_abs_replay_error':float((out-replay).abs().max()),
                      'shifted':site['mask'] is not None,
                      'window_geometry':obs.geometry.get(name).__dict__ if name in obs.geometry else None})
    first=next(iter(obs.sites))
    effect=atomic_bidirectional_effect(model,x,altered,AttentionAtom(first,0,0,0,0),class_index=0)
    bridge=MonaiSwinVIBBridge(model,feature_size=12,enabled=False).eval()
    with torch.no_grad():
        neutral=bridge(x)
    eq=float((normal-neutral).abs().max())
    report={'type':'synthetic_monai_151_engineering_only','input_shape':list(x.shape),
            'output_shape':list(normal.shape),'num_window_modules':len(diffs),
            'max_replay_error':max(v['max_abs_replay_error'] for v in diffs),
            'shifted_modules':sum(bool(v['shifted']) for v in diffs),
            'vib_disabled_equivalence_max_abs_error':eq,'atomic_effect':effect,
            'sites':diffs}
    dest=Path(args.json_out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'max_replay_error':report['max_replay_error'],
                      'vib_disabled_equivalence_error':eq,
                      'shifted_modules':report['shifted_modules']},indent=2))
    if report['max_replay_error']>2e-5 or eq>1e-6 or report['shifted_modules']<1:
        raise SystemExit('MONAI integration numeric audit failed')


if __name__=='__main__':
    main()
