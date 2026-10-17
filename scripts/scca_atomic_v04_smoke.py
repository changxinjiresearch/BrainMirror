"""Reproducible CPU-only SCEPT v0.4 engineering smoke; synthetic tensors only."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from brainmirror.scept.model import TransformerVIB3D
from brainmirror.scept.attention_atomic import AtomicAttentionSite, intervene_atomic_attention
from brainmirror.scept.latent_bridge import LatentSite, intervene_spatial_vib
from brainmirror.scept.atomic_chain import (AtomicPathHop,intervene_atomic_chain,
                                            search_atomic_chains)
from brainmirror.scept.paths import PathHop
from brainmirror.scept.multiscale import (
    MultiScaleTransformerVIB3D,CrossScaleSite,
    intervene_merge_path,intervene_cross_scale_atomic)


def sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def tensor_metric(t):
    return {'l2':float(t.norm()),'max_abs':float(t.abs().max()),'all_finite':bool(torch.isfinite(t).all())}


def execute(seed=240216):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    single=TransformerVIB3D(dim=16,heads=4,depth=3,patch=4,latent_dim=8).eval()
    ref=torch.randn(1,4,8,8,8)
    clean=ref.clone();clean[:,:,:4,:4,:4]+=1.3
    roi=torch.zeros(8,8,8,dtype=torch.bool);roi[:4,:4,:4]=True
    atomic=intervene_atomic_attention(single,clean,ref,
                AtomicAttentionSite(1,1,0,0,'weighted_value'))
    score=intervene_atomic_attention(single,clean,ref,
                AtomicAttentionSite(1,1,0,0,'logit'))
    latent=intervene_spatial_vib(single,clean,ref,LatentSite(0,1))
    chain=intervene_atomic_chain(single,clean,ref,[
            PathHop('blocks.0.mlp_to_residual'),AtomicPathHop(1,1,0,0),
            PathHop('blocks.2.mlp_to_residual')],latent_token=0,latent_channel=1)
    search=search_atomic_chains(single,clean,ref,'blocks.0.mlp_to_residual',1,
        queries=[0,1],keys=[0,1],heads=[0,1],roi=roi,max_tested=5)
    multi=MultiScaleTransformerVIB3D(dim=16,heads=4,patch=2).eval()
    ref2=torch.randn(1,4,8,8,8);clean2=ref2.clone()
    clean2[:,:,:2,:2,:2]+=1.3
    merge=intervene_merge_path(multi,clean2,ref2,CrossScaleSite(0))
    coarse=intervene_cross_scale_atomic(multi,clean2,ref2,CrossScaleSite(0),
                AtomicAttentionSite(0,1,0,0,'weighted_value'))
    return {
      'schema':'brainmirror.scca_atomic_smoke.v0.4',
      'data':'fully_synthetic_tensors_not_mri',
      'seed':seed,'dtype':'torch.float32','device':'cpu',
      'model_a':{'class':'TransformerVIB3D','depth':3,'patch':4,
          'weights_sha256':sha(next(single.parameters())),'input_sha256':sha(clean)},
      'model_b':{'class':'MultiScaleTransformerVIB3D','scales':2,'patch':2,
          'weights_sha256':sha(next(multi.parameters())),'input_sha256':sha(clean2)},
      'single_scale':{
          'weighted_value':tensor_metric(atomic['branch_delta']),
          'single_logit':tensor_metric(score['branch_delta']),
          'latent_oracle_error':float((latent['intervened_logits']-
                 latent['reference_logits']-latent['expected_decoder_delta']).abs().max()),
          'isolated_chain_final_gate_delta':tensor_metric(chain['route_delta']),
          'chain_z_delta':tensor_metric(chain['latent_delta']),
          'search_status':search['status'],'tested_candidates':search['tested_count'],
          'search_top':search['best']},
      'multi_scale':{
          'fine_to_coarse_token':merge['coarse_token'],
          'merge_delta':tensor_metric(merge['merge_delta']),
          'merge_z_delta':tensor_metric(merge['latent_mu_delta']),
          'coarse_atomic_delta':tensor_metric(coarse['atomic_branch_delta']),
          'coarse_atomic_z_delta':tensor_metric(coarse['latent_mu_delta']),
          'fine_skip_unchanged':merge['fine_skip_unchanged']},
      'scope':'same_case_synthetic_engineering_validation_only',
      'not_verified':['independent_patients','BraTS2020','clinical_safety',
                      'method_novelty','GradCAM_superiority','unique_complete_causal_path']
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='runs/scca_atomic_v04_synthetic_smoke.json')
    a=p.parse_args()
    result=execute()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'seed':result['seed'],
         'latent_oracle_error':result['single_scale']['latent_oracle_error'],
         'tested_atomic_candidates':result['single_scale']['tested_candidates'],
         'multi_scale_coarse_token':result['multi_scale']['fine_to_coarse_token']},indent=2))

if __name__=='__main__':main()
