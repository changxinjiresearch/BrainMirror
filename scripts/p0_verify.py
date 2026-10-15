#!/usr/bin/env python3
"""Run honest engineering-only numerical QA. Never claim scientific validation."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainmirror.experiment import smoke_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/p0_smoke.json')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    record = smoke_experiment(args.seed)
    assert record['neuron']['max_absolute_arithmetic_error'] < 2e-5
    assert record['scientific_validation'] == 'NOT_STARTED'
    assert record['input_sha256'] != record['edited_input_sha256']
    dest = Path(args.output)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'status': 'ENGINEERING_NUMERIC_QA_PASS',
        'scientific_validation': 'NOT_STARTED',
        'model_sha256_prefix': record['model_sha256'][:16],
        'incoming_edge_count': record['neuron']['incoming_edge_count'],
        'max_arithmetic_error': record['neuron']['max_absolute_arithmetic_error'],
        'output': str(dest),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
