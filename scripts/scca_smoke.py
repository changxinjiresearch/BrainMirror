"""CPU-only reproducibility smoke for bounded SCCA on synthetic 3D voxels.

Writes measurements, never invented explanation values. Not a medical study.
"""
import argparse
import json
from pathlib import Path

import torch
from brainmirror.scept import TransformerVIB3D, Region
from brainmirror.scept.scca import SCCAConfig, explain_scca


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='runs/scca_synthetic_smoke.json')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(240216)
    model = TransformerVIB3D(dim=16, heads=4, depth=2,
                             patch=4, latent_dim=8).eval()
    x = torch.randn(1, 4, 16, 16, 16)
    roi = torch.zeros(16,16,16,dtype=torch.bool)
    roi[4:12,4:12,4:12] = True
    result = explain_scca(model, x, Region(4,12,4,12,4,12), roi,
                          SCCAConfig(max_candidates=4, max_circuit_size=2,
                                     beam_width=3))
    artifact = {
        'data': 'random_synthetic_tensor_no_mri_or_patient_data',
        'seed': 240216,
        'model': 'untrained_3d_TransformerVIB3D',
        'interpretation_limit': 'edge-set interventions not complete causal path',
        'patient_independent_validation': False,
        'SCCA': result,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, ensure_ascii=False),
                      encoding='utf-8')
    print(json.dumps({'status':'ENGINEERING_SMOKE_COMPLETED',
                      'candidate_count':len(result['candidates']),
                      'tested_edge_sets':len(result['tested_edge_sets']),
                      'selected':(result['selected'] or {}).get('edges'),
                      'file':str(target)}))


if __name__ == '__main__':
    main()
