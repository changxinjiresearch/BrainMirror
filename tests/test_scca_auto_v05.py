"""Automatic discovery controls: numerical oracles, no selection leak, budgets."""
import json
import pytest
import torch
from torch.nn import functional as F

from brainmirror.scept.attention_atomic import AtomicAttentionSite, _delta_one_head_query, attention_parts
from brainmirror.scept.auto_discovery import (
    AutoDiscoveryConfig, _attention_message_proxies, _candidate_sources, _check_inputs,
    _merge_delta_for_token, _screen, _state, discover_scca_multiscale, reference_from_region,
)
from brainmirror.scept.explain import fixed_target_score
from brainmirror.scept.multiscale import (
    CrossScaleSite, MultiScaleTransformerVIB3D, intervene_cross_scale_atomic,
)

torch.set_num_threads(2)


@pytest.fixture
def case():
    torch.manual_seed(4107)
    model = MultiScaleTransformerVIB3D(dim=12, heads=3, patch=2, latent_dim=6,
                                       max_tokens=512).eval()
    x = torch.randn(1, 4, 8, 8, 8)
    mask = torch.zeros(8, 8, 8, dtype=torch.bool)
    mask[0:3, 0:3, 0:3] = True
    return model, x, mask


def config(**kwargs):
    return AutoDiscoveryConfig(max_fine_candidates=3, per_fine_screen=4,
                               max_exact_tests=5, validate_top=2, **kwargs)


def test_references_modify_only_selected_roi(case):
    _, x, mask = case
    for mode in ('blur', 'outside_mean'):
        ref = reference_from_region(x, mask, mode)
        assert torch.equal(ref[..., ~mask], x[..., ~mask])
        assert not torch.allclose(ref[..., mask], x[..., mask])
        assert ref.shape == x.shape


def test_uniform_reference_is_unmodified(case):
    _, _, mask = case
    x = torch.ones(1, 4, 8, 8, 8)
    assert torch.equal(reference_from_region(x, mask, 'blur'), x)
    assert torch.equal(reference_from_region(x, mask, 'outside_mean'), x)


def test_full_auto_discovers_without_input_site_ids(case):
    m, x, mask = case
    output = discover_scca_multiscale(m, x, mask, config=config())
    assert output['screened_atomic_count'] > 0
    assert output['exact_tested_count'] == 5
    assert len(output['candidates']) == 5
    assert len(output['reserved_validation']) == 2
    assert output['best']['head'] in range(3)
    assert output['best']['query'] in range(8)
    assert output['best']['key'] in range(8)
    assert 0 <= output['best']['fine_token'] < 64
    assert not output['patient_independent_validation']
    assert output['reference_validity'].startswith('unverified')
    json.dumps(output, allow_nan=False)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in m.modules())


def test_exact_budget_and_determinism(case):
    m, x, mask = case
    cfg = AutoDiscoveryConfig(max_fine_candidates=2, per_fine_screen=3,
                              max_exact_tests=2, validate_top=1)
    r1 = discover_scca_multiscale(m, x, mask, config=cfg)
    r2 = discover_scca_multiscale(m, x, mask, config=cfg)
    assert r1 == r2
    assert r1['exact_tested_count'] == 2
    assert len(r1['reserved_validation']) == 1


def test_zero_input_signal_must_abstain(case):
    m, _, mask = case
    x = torch.ones(1, 4, 8, 8, 8)
    out = discover_scca_multiscale(m, x, mask, config=config())
    assert out['status'] == 'unresolved_low_primary_input_effect'
    assert out['best'] is None and out['exact_tested_count'] == 0


def test_roi_can_be_separate_from_source_mask(case):
    m, x, mask = case
    target = torch.zeros_like(mask)
    target[-3:, -3:, -3:] = True
    r = discover_scca_multiscale(m, x, mask, target_roi=target, channel=0,
                                  config=config())
    assert r['channel'] == 0
    assert not r['patient_independent_validation']


def test_merge_only_gradient_excludes_skip_path(case):
    m, x, mask = case
    reference = reference_from_region(x, mask, 'blur')
    r = _state(m, reference, mask, 2, gradients=True)
    # Verify gradient through patch merging directly by intervening at that
    # input in an otherwise reference run. The skip route must stay frozen.
    t = r['fine']
    assert r['grad_fine'].shape == t.shape
    # finite-difference source gradient w.r.t. the *merge module input*
    index = (0, 0, 0, 0, 0)
    eps = 1e-3
    delta = torch.zeros_like(t)
    delta[index] = eps
    handle = m.patch_merge.register_forward_pre_hook(
        lambda _mod, inp: (inp[0] + delta,))
    try:
        with torch.no_grad():
            score = float(fixed_target_score(m(reference), 2, mask))
    finally:
        handle.remove()
    base_score = float(fixed_target_score(r['logits'], 2, mask))
    estimate = float(r['grad_fine'][index]) * eps
    assert abs((score - base_score) - estimate) < 2e-5


def test_ranking_uses_full_mediator_not_only_local_patch(case):
    m, x, mask = case
    altered = reference_from_region(x, mask, 'blur')
    clean = _state(m, x, mask, 2)
    reference = _state(m, altered, mask, 2, gradients=True)
    cfg = config()
    screened, cache, n = _screen(m, clean, reference, cfg, mask)
    assert n == 3 * 2 * 3 * 8 * 8  # source * mode * heads * Q * K
    assert len(screened) <= 3 * 4
    for c in screened:
        assert c['fine_token'] in cache
        assert cache[c['fine_token']]['coarse_index'] == m.fine_to_coarse(
            c['fine_token'], (4, 4, 4))


def test_exact_effect_matches_independent_v04_oracle(case):
    m, x, mask = case
    reference = reference_from_region(x, mask, 'blur')
    auto = discover_scca_multiscale(m, x, mask, config=config())
    candidate = auto['best']
    assert candidate is not None
    site = AtomicAttentionSite(0, candidate['head'], candidate['query'],
                               candidate['key'], candidate['mode'])
    oracle = intervene_cross_scale_atomic(m, x, reference,
                                          CrossScaleSite(candidate['fine_token']), site)
    reference_score = float(fixed_target_score(oracle['reference_logits'], 2, mask))
    restored_score = float(fixed_target_score(oracle['isolated_logits'], 2, mask))
    assert abs(restored_score - reference_score - candidate['restoration']) < 3e-6


def test_all_attention_message_proxies_match_atomic_math(case):
    m, x, mask = case
    reference = reference_from_region(x, mask, 'blur')
    ref = _state(m, reference, mask, 2, gradients=True)
    clean = _state(m, x, mask, 2)
    sources, _, _ = _candidate_sources(m, clean, ref, config())
    fine_ref_merge = m.patch_merge(ref['fine'])
    delta, _ = _merge_delta_for_token(m, ref['fine'], clean['fine'],
                                      sources[0], fine_ref_merge)
    block = m.coarse_blocks[0]
    base = attention_parts(block, ref['coarse_input'])
    hybrid = attention_parts(block, ref['coarse_input'] + delta)
    proxy = _attention_message_proxies(base, hybrid, ref['grad_gate'],
                                       block.attn.out_proj.weight)
    for mode in ('weighted_value', 'logit'):
        for head, query, key in [(0, 0, 0), (2, 7, 3)]:
            site = AtomicAttentionSite(0, head, query, key, mode)
            expected = _delta_one_head_query(hybrid, base, site,
                                             block.attn.out_proj.weight)
            actual = (expected * ref['grad_gate'][:, query, :]).sum()
            assert torch.allclose(proxy[mode][head, query, key], actual,
                                  rtol=5e-5, atol=2e-7)


def test_validation_inputs_not_used_for_first_round_selection(case, monkeypatch):
    import brainmirror.scept.auto_discovery as module
    m, x, mask = case
    baseline = discover_scca_multiscale(m, x, mask, config=config())
    original = module.reference_from_region
    def altered_secondary(x, mask, strategy):
        if strategy == 'outside_mean':
            # Valid but different held-out reference, no change to first round.
            return x.clone()  # no signal means reserved evidence must abstain
        return original(x, mask, strategy)
    monkeypatch.setattr(module, 'reference_from_region', altered_secondary)
    r = discover_scca_multiscale(m, x, mask, config=config())
    assert r['candidates'] == baseline['candidates']
    assert r['best'] == baseline['best']
    assert r['status'] == 'unresolved_low_reserved_input_effect'


def test_rejects_unsupported_models_shapes_and_nonfinite(case):
    m, x, mask = case
    with pytest.raises(ValueError):
        discover_scca_multiscale(m, x[:, :3], mask)
    with pytest.raises(ValueError):
        discover_scca_multiscale(m, x, mask.to(torch.float32))
    with pytest.raises(ValueError):
        discover_scca_multiscale(m, x, torch.ones_like(mask))
    with pytest.raises(ValueError):
        discover_scca_multiscale(m, x, mask, channel=5)
    with pytest.raises(ValueError):
        discover_scca_multiscale(m, x.clone().fill_(float('nan')), mask)
    with pytest.raises(ValueError):
        AutoDiscoveryConfig(reference_strategy='blur', validation_strategy='blur').validate()
    with pytest.raises(ValueError):
        AutoDiscoveryConfig(max_exact_tests=0).validate()
    deeper = MultiScaleTransformerVIB3D(dim=12, heads=3, patch=2, coarse_depth=2)
    with pytest.raises(ValueError, match='ONE coarse'):
        discover_scca_multiscale(deeper, x, mask)


def test_zeroed_attention_output_abstains_even_with_nonzero_input_effect(case):
    m, x, mask = case
    with torch.no_grad():
        m.coarse_blocks[0].attn.out_proj.weight.zero_()
        m.coarse_blocks[0].attn.out_proj.bias.zero_()
    r = discover_scca_multiscale(m, x, mask, config=config())
    assert abs(r['input_effect_primary']) > 1e-6  # skip and residual may still work
    assert r['exact_tested_count'] == 0
    assert r['status'] == 'unresolved_no_measurable_atomic_candidate'
    assert r['best'] is None


def test_automatic_source_mask_preserves_untouched_voxels_both_directions(case):
    m, x, mask = case
    original = x.clone()
    out = discover_scca_multiscale(m, x, mask, config=config())
    assert torch.equal(x, original)
    assert all(c['reference_name'] == 'blur' for c in out['candidates'])
    assert all(v['reference_name'] == 'outside_mean' for v in out['reserved_validation'])


def test_positive_cross_reference_synthetic_control():
    torch.manual_seed(2025)
    m = MultiScaleTransformerVIB3D(dim=8, heads=1, patch=2,
                                   latent_dim=8, max_tokens=512).eval()
    x = torch.randn((1, 4, 8, 8, 8))
    mask = torch.zeros(8, 8, 8, dtype=torch.bool)
    mask[:4, :4, :4] = True
    cfg = AutoDiscoveryConfig(max_fine_candidates=8, per_fine_screen=16,
                              max_exact_tests=20, validate_top=5,
                              min_supported_fraction=.001)
    r = discover_scca_multiscale(m, x, mask, config=cfg)
    assert r['status'] == 'same_case_cross_reference_supported'
    assert any(c['cross_reference_direction_supported'] for c in r['reserved_validation'])
    assert 'patch_merge_to_coarse_token_' in r['best']['declared_route'][2]
    assert r['exact_tested_count'] <= cfg.max_exact_tests
    assert r['screened_atomic_count'] >= r['exact_tested_count']
