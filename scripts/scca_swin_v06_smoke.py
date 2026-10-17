"""Deterministic synthetic, NON-medical Swin v0.6 mechanism experiment.

Usage: python scripts/scca_swin_v06_smoke.py --output runs/scca_swin_v06_synthetic.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brainmirror.scept.swin3d import MultiLayerSwinVIB3D, execution_topology
from brainmirror.scept.swin_discovery import SwinDiscoveryConfig,discover_scca_swin


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/scca_swin_v06_synthetic.json')
    args=parser.parse_args()
    torch.manual_seed(260616)
    torch.set_num_threads(2)
    model=MultiLayerSwinVIB3D(dim=12,heads=3,patch=2,window=2,
                              fine_depth=2,coarse_depth=2,max_tokens=512).eval()
    x=torch.randn(1,4,16,16,16)*0.3
    # Contrived positive foreground ROI, not a tumour or in-distribution MRI.
    source=torch.zeros(16,16,16,dtype=torch.bool)
    source[4:10,4:10,3:9]=True
    x[:,1,source]+=1.2
    target=torch.zeros_like(source)
    target[4:10,4:10,3:9]=True
    cfg=SwinDiscoveryConfig(beam_width=3,max_source_tokens=3,
             per_state_proposals=8,max_exact_paths=8,validate_top=3)
    result=discover_scca_swin(model,x,source,target_roi=target,channel=2,config=cfg)
    topo=execution_topology(model,x)
    params=b''.join(t.detach().cpu().contiguous().numpy().tobytes()
                    for t in model.state_dict().values())
    result['provenance']={'seed':260616,'pytorch':torch.__version__,
                          'weights_sha256':hashlib.sha256(params).hexdigest(),
                          'data_kind':'fully_synthetic_not_medical',
                          'signal':'constructed_nonmedical_channel1_ROI_intensity',
                          'model_training':'random_initialized_untrained',
                          'artifact_kind':'engineering_numeric_smoke'}
    result['topology_summary']={
        'fine_grid':topo['stages']['fine']['grid'],
        'coarse_grid':topo['stages']['coarse']['grid'],
        'fine_shifts':[s['effective_shift'] for s in topo['stages']['fine']['blocks']],
        'coarse_shifts':[s['effective_shift'] for s in topo['stages']['coarse']['blocks']],
        'attention_edges_including_heads':topo['expanded_head_attention_edge_count'],
        'patch_merge_links':len(topo['patch_merge_receptive_links']),
        'skip_bypasses_VIB':True}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','source_candidates','screened_atomic_count',
                                            'exact_tested_count','reserved_validation')},ensure_ascii=False))
    print('saved',dest)

if __name__=='__main__':main()
