"""Reproducible CPU-only synthetic cross-layer path intervention smoke.

NOT anatomical, not pretraining results, no patient validation and no Grad-CAM
comparative claim. Save actual measured values and explicit limits.
"""
import argparse
import json
from pathlib import Path
import torch

from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.explain import Region
from brainmirror.scept.paths import PathHop, PathSearchConfig, evaluate_path, explain_scca_paths
from brainmirror.scept.scca import reference_volume


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', default='runs/scca_cross_layer_smoke.json')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(240216)
    model=TransformerVIB3D(dim=16,heads=4,depth=3,patch=4,latent_dim=8).eval()
    x=torch.randn(1,4,8,8,8)
    region=Region(2,6,2,6,2,6)
    roi=torch.zeros(8,8,8,dtype=torch.bool)
    roi[2:6,2:6,2:6]=True
    ref=reference_volume(x,region)
    chosen=(PathHop('blocks.0.attn_to_residual'),
            PathHop('blocks.1.attn_to_residual','q'),
            PathHop('blocks.2.mlp_to_residual'))
    measured=evaluate_path(model,x,ref,chosen,roi)
    search=explain_scca_paths(model,x,region,roi,
        PathSearchConfig(max_candidates=6,max_hops=3,max_tested_paths=12,
                         attention_roles=('q','k','v'),min_input_effect=1e-12))
    artifact={
       'status':'ENGINEERING_SYNTHETIC_ONLY',
       'method':'SCCA controlled cross-layer path patching v0.3',
       'seed':240216,'data_source':'torch.randn synthetic voxel tensor',
       'model':'untrained 3D TransformerVIB3D (depth 3; 16 dim; 4 heads)',
       'path_semantics':'frozen-offpath residual propagation with Q/K/V or MLP site recomputation',
       'path_test':measured,'candidate_search':search,
       'not_claimed':['patient validation','real MRI','complete multihead circuits',
                       'clinical counterfactual MRI','Grad-CAM superiority',
                       'uniqueness of the causal route'],
    }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(artifact,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'saved':str(output), 'selected':(search.get('selected') or {}).get('path'),
                      'tested':len(search['tested_paths']),
                      'restoration':measured['restoration'],
                      'disruption':measured['disruption'],
                      'evidence':'synthetic engineering control'}))


if __name__=='__main__': main()
