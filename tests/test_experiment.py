import json
import numpy as np
import pytest
import torch

from brainmirror.experiment import smoke_experiment, trace_pair, tensor_sha, weights_sha
from brainmirror.model import BrainAE
from brainmirror.mri import phantom


def test_p0_trace_real_numeric_edges_and_provenance():
    r = smoke_experiment(seed=17)
    assert r['scientific_validation'] == 'NOT_STARTED'
    assert r['source_tier'] == 'synthetic_debug'
    assert r['evidence_level'] == 'L1_REAL_NUMERIC_SYNTHETIC_INPUT'
    assert r['intervention']['anatomical_plausibility_validated'] is False
    assert 1 <= r['neuron']['incoming_edge_count'] <= 216
    assert r['neuron']['max_absolute_arithmetic_error'] < 2e-5
    assert r['input_sha256'] != r['edited_input_sha256']
    assert np.isfinite(r['model_output']['mean_absolute_ablation_effect_on_edited_reconstruction'])
    assert r['debug_training']['trained_on_real_mri'] is False
    json.dumps(r)


def test_no_fake_real_mri_claim_allowed():
    volume, roi = phantom(10)
    with pytest.raises(ValueError, match='restricted'):
        trace_pair(BrainAE(), volume, roi, 0.5, source_tier='real_healthy_mri')


def test_unit_and_weight_hash_are_stable_for_same_tensor():
    torch.manual_seed(99)
    m = BrainAE()
    hash_before = weights_sha(m)
    assert weights_sha(m) == hash_before
    assert tensor_sha(torch.ones((2, 3))) == tensor_sha(torch.ones((2, 3)))
    with torch.no_grad():
        m.conv1.weight[0, 0, 0, 0, 0] += 1
    assert weights_sha(m) != hash_before


def test_invalid_nonfinite_volume_rejected():
    volume, roi = phantom(2)
    volume[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Non-finite'):
        trace_pair(BrainAE(), volume, roi)
