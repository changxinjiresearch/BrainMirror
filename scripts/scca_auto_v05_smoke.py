"""Execute reproducible synthetic diagnostic run and record *measured* results."""
import argparse
import json
import platform
from pathlib import Path

import torch
from brainmirror.scept.auto_discovery import AutoDiscoveryConfig, discover_scca_multiscale
from brainmirror.scept.multiscale import MultiScaleTransformerVIB3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/scca_auto_v05_synthetic_smoke.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2025)
    model = MultiScaleTransformerVIB3D(dim=8, heads=1, patch=2,
                                       latent_dim=8, max_tokens=512).eval()
    x = torch.randn(1, 4, 8, 8, 8)
    region = torch.zeros(8, 8, 8, dtype=torch.bool)
    region[:4, :4, :4] = True
    config = AutoDiscoveryConfig(max_fine_candidates=8, per_fine_screen=16,
                                 max_exact_tests=20, validate_top=5,
                                 min_supported_fraction=.001)
    report = discover_scca_multiscale(model, x, region, config=config)
    artifact = {
        'seed': 2025, 'torch': torch.__version__,
        'python': platform.python_version(),
        'data_kind': 'synthetic_random_3d_tensor_NOT_BraTS',
        'protocol': 'same_case_two_diagnostic_references_no_heldout_patients',
        'report': report,
    }
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(f"status={report['status']} screened={report['screened_atomic_count']} "
          f"tested={report['exact_tested_count']} validated={len(report['reserved_validation'])}")
    print(out)


if __name__ == '__main__':
    main()
