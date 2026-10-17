"""SCEPT v0.8 official MONAI cross-block controlled-cascade smoke (synthetic)."""
import argparse
import json
from pathlib import Path
import torch
import monai
from brainmirror.scept.monai_adapter import build_monai_model,AttentionAtom
from brainmirror.scept.monai_crosslayer import (
    capture_monai_reference,run_isolated_monai_chain,discover_monai_crosslayer)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--json-out',default='runs/scept_monai_v08_synthetic.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(240216)
    model=build_monai_model(depths=(2,1,1,1)).eval()
    clean=torch.randn(1,4,32,32,32)
    reference=clean.clone();reference[:,:,:12,:12,:12]*=-0.65
    alternate=clean.clone();alternate[:,:,:12,:12,:12]*=0.3
    roi=torch.zeros((32,32,32),dtype=torch.bool);roi[:12,:12,:12]=True
    chain=[AttentionAtom('swinViT.layers1.0.blocks.0.attn',0,0,0,0),
           AttentionAtom('swinViT.layers1.0.blocks.1.attn',0,0,0,0),
           AttentionAtom('swinViT.layers2.0.blocks.0.attn',0,0,0,0)]
    with torch.no_grad():
        snapshot=capture_monai_reference(model,clean,reference,roi=roi)
        fixed=run_isolated_monai_chain(model,reference,snapshot,chain,roi=roi)
        severed=run_isolated_monai_chain(model,reference,snapshot,chain,roi=roi,cut_at=2)
        discovery=discover_monai_crosslayer(model,clean,reference,roi,
            heldout_reference=alternate,max_hops=3,max_sites=3,
            per_step=2,beam_width=1,max_exact=5,kinds=('value_message',),effect_tol=1e-8)
    cut_error=float((severed['logits']-snapshot.reference_logits).abs().max())
    report={
        'version':'scept_monai_v08_controlled_cascade',
        'data':'fixed_seed_synthetic_only_no_training_no_patient',
        'seed':240216,'monai':monai.__version__,'torch':torch.__version__,
        'input_shape':list(clean.shape),'roi_voxels':int(roi.sum()),
        'model_architecture':'official_monai_swinunetr_depths_2_1_1_1',
        'fixed_three_block_path':[a.__dict__ for a in chain],
        'fixed_effect':fixed['effect_from_reference'],
        'hop_deltas':fixed['hops'],
        'cut_at_last_max_logit_error':cut_error,
        'automatic_search':discovery,
        'limitation':'controlled-path intervention, not unique natural causal circuit; model random weights'
    }
    Path(args.json_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cut_error':cut_error,'search_status':discovery['status'],
                      'exact_budget':discovery['forward_budget_used'],
                      'path_count':len(discovery.get('candidate_paths',[]))},indent=2))
    if cut_error>3e-5 or discovery['forward_budget_used']>5:
        raise SystemExit('SCEPT v0.8 controlled-path smoke validation failed')


if __name__=='__main__':main()
