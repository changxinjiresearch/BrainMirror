"""SCCA v0.5 automatic *reference-conditioned* multiscale mechanism discovery.

Given an MRI mask, NO fine token / Attention head / Query / Key is requested.
The source MRI region is changed before a real forward pass. Fine features
potentially mix information globally; selected *mediator tokens* are therefore
not presented as locally originating MRI patches. Ranking uses reference-run
first-order scores; selection uses actual signed bidirectional interventions;
a SECOND reference strategy is evaluated only after candidate selection.

Scope: exactly one coarse full-attention Transformer block (NO Swin/window or
multi-coarse-block path semantics). Atomic weighted-Value or QK-logit changes
are well-defined surgical interventions, not unique natural causal pathways.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from .attention_atomic import AtomicAttentionSite, _delta_one_head_query, attention_parts
from .explain import fixed_target_score
from .multiscale import MultiScaleTransformerVIB3D, _flat_to_zyx


@dataclass(frozen=True)
class AutoDiscoveryConfig:
    max_fine_candidates: int = 6
    per_fine_screen: int = 12
    max_exact_tests: int = 12
    validate_top: int = 3
    min_input_effect: float = 1e-6
    min_source_delta: float = 1e-9
    min_proxy_effect: float = 1e-11
    min_supported_fraction: float = 0.01
    off_target_weight: float = 0.1
    modes: tuple[str, ...] = ('weighted_value', 'logit')
    reference_strategy: str = 'blur'
    validation_strategy: str = 'outside_mean'

    def validate(self):
        for k in ('max_fine_candidates', 'per_fine_screen', 'max_exact_tests', 'validate_top'):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f'{k} must be positive integer')
        for k in ('min_input_effect', 'min_source_delta', 'min_proxy_effect', 'min_supported_fraction', 'off_target_weight'):
            v = getattr(self, k)
            if not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{k} must be finite and nonnegative')
        if not self.modes or len(set(self.modes)) != len(self.modes) or any(
                m not in ('weighted_value', 'logit') for m in self.modes):
            raise ValueError('Unsupported or repeated atomic intervention mode')
        if self.reference_strategy == self.validation_strategy:
            raise ValueError('Primary and reserved validation references must differ')
        if set((self.reference_strategy, self.validation_strategy)) != {'blur', 'outside_mean'}:
            raise ValueError('Supported diagnostic references: blur and outside_mean')


def _check_inputs(model, x, mask, target_roi, channel):
    if not isinstance(model, MultiScaleTransformerVIB3D) or len(model.coarse_blocks) != 1:
        raise ValueError('v0.5 supports MultiScaleTransformerVIB3D with ONE coarse block only')
    if not torch.is_tensor(x) or x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 4:
        raise ValueError('Expected a single [1,4,D,H,W] volume')
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError('Input must be finite floating-point MRI-like tensor')
    model.grids(x)
    for name, roi in (('source MRI mask', mask), ('target ROI', target_roi)):
        if not torch.is_tensor(roi) or roi.dtype != torch.bool or roi.device != x.device or roi.shape != x.shape[-3:]:
            raise ValueError(f'{name} must be a matching boolean 3D mask')
        if not roi.any() or not (~roi).any():
            raise ValueError(f'{name} must be nonempty and not cover the entire volume')
    if channel not in (0, 1, 2):
        raise ValueError('Segmentation channel must be WT=0, TC=1 or ET=2')
    if next(model.parameters()).device != x.device or next(model.parameters()).dtype != x.dtype:
        raise ValueError('MRI dtype/device must match model parameters')


def reference_from_region(x, source_mask, strategy):
    """Deterministic diagnostic references; NOT clinically plausible counterfactuals."""
    if (not torch.is_tensor(x) or x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 4 or
            not torch.is_tensor(source_mask) or source_mask.dtype != torch.bool or
            source_mask.shape != x.shape[-3:] or source_mask.device != x.device or
            not source_mask.any() or not (~source_mask).any()):
        raise ValueError('Expected matched [1,4,D,H,W] input and nontrivial 3D mask')
    if strategy == 'blur':
        smooth = F.avg_pool3d(F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), 3, stride=1)
        replacement = smooth
    elif strategy == 'outside_mean':
        replacement = x[..., ~source_mask].mean(dim=-1).reshape(1, 4, 1, 1, 1)
    else:
        raise ValueError('Unknown diagnostic reference method')
    return torch.where(source_mask[None, None], replacement, x)


@contextmanager
def _capture(model):
    """Collect exact tensors from the executable model; remove every hook on exit."""
    record = {}
    block = model.coarse_blocks[0]
    def fine_hook(_module, inputs):
        record['fine'] = inputs[0]
    def merged_hook(_module, _inputs, output):
        record['merged'] = output
    def coarse_hook(_module, inputs):
        record['coarse_input'] = inputs[0]
    def gate_hook(_module, _inputs, output):
        record['gate'] = output
    handles = [
        model.patch_merge.register_forward_pre_hook(fine_hook),
        model.patch_merge.register_forward_hook(merged_hook),
        block.norm1.register_forward_pre_hook(coarse_hook),
        block.attn_to_residual.register_forward_hook(gate_hook),
    ]
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


def _state(model, x, target_roi, channel, gradients=False):
    with _capture(model) as rec:
        with torch.enable_grad() if gradients else torch.no_grad():
            logits = model(x, sample=False)
            if gradients:
                score = fixed_target_score(logits, channel, target_roi)
                gm, gg = torch.autograd.grad(score, [rec['merged'], rec['gate']])
                # Only the merge branch: the direct fine skip must NOT influence
                # candidate-ranking gradients for a merge-mediated explanation.
                gf = F.conv_transpose3d(gm, model.patch_merge.weight, stride=2)
                rec['grad_fine'], rec['grad_gate'] = gf.detach(), gg.detach()
        result = {k: v.detach() for k, v in rec.items()}
    result['logits'] = logits.detach()
    return result


def _attention_message_proxies(baseline, hybrid, grad_gate, out_proj):
    """Actual attention algebra + first-order downstream sensitivity [H,Q,K]."""
    h, n, dh = baseline['v'].shape[1:]
    grad = grad_gate[0]
    projected = torch.stack([
        grad @ out_proj[:, i * dh:(i + 1) * dh]
        for i in range(h)
    ], dim=0)  # [H,Q,dh]
    difference = (hybrid['weights'][0][..., None] * hybrid['v'][0][:, None, :, :] -
                  baseline['weights'][0][..., None] * baseline['v'][0][:, None, :, :])
    weighted = (difference * projected[:, :, None, :]).sum(-1)
    # Exact algebra for replacing ONE QK logit in an otherwise base query row.
    w = baseline['weights'][0]
    logit_delta = (hybrid['logits'][0] - baseline['logits'][0]).clamp(-35, 35)
    rminus = torch.expm1(logit_delta)
    factor = w * rminus
    denom = (1 + factor).clamp_min(1e-10)
    logit_head_delta = ((factor / denom)[..., None] *
                        (baseline['v'][0][:, None, :, :] - baseline['head_output'][0][:, :, None, :]))
    logit = (logit_head_delta * projected[:, :, None, :]).sum(-1)
    return {'weighted_value': weighted, 'logit': logit}


def _candidate_sources(model, clean, ref, config):
    delta = clean['fine'] - ref['fine']
    relevance = (delta * ref['grad_fine']).sum(dim=1).abs().reshape(-1)
    amplitudes = delta.square().sum(dim=1).sqrt().reshape(-1)
    # Sort is deterministic; even if local gradients vanish, changed features
    # remain eligible through magnitude fallback. Never invent a nonzero delta.
    scores = relevance + 1e-8 * amplitudes
    valid = [i for i in range(scores.numel()) if float(amplitudes[i]) > config.min_source_delta]
    valid.sort(key=lambda i: (-float(scores[i]), i))
    return valid[:config.max_fine_candidates], delta, relevance


def _merge_delta_for_token(model, ref_fine, clean_fine, fine_index, ref_merge):
    fine_grid = tuple(ref_fine.shape[-3:])
    z, y, x = _flat_to_zyx(fine_index, fine_grid)
    patched = ref_fine.clone()
    patched[:, :, z, y, x] = clean_fine[:, :, z, y, x]
    difference = model.patch_merge(patched) - ref_merge
    cidx = model.fine_to_coarse(fine_index, fine_grid)
    grid = tuple(d // 2 for d in fine_grid)
    # One kernel=2/stride=2 patch merge cell is affected by one fine source.
    other = difference.flatten(2).clone()
    other[:, :, cidx] = 0
    if other.abs().max() > 5e-7:
        raise AssertionError('Unselected patch-merge cells changed')
    return difference.flatten(2).transpose(1, 2), cidx


def _screen(model, clean, ref, config, source_mask):
    sources, _, relevance = _candidate_sources(model, clean, ref, config)
    if not sources:
        return [], {}, 0
    block = model.coarse_blocks[0]
    baseline = attention_parts(block, ref['coarse_input'])
    if not torch.allclose(baseline['output'], ref['gate'], rtol=2e-5, atol=2e-6):
        raise AssertionError('Attention decomposition differs from actual branch')
    ref_merge = model.patch_merge(ref['fine'])
    fine_grid = tuple(ref['fine'].shape[-3:])
    # Direct local patch overlap is metadata, not a locality-based filter:
    # global fine attention may route evidence to distant fine mediators.
    direct = F.max_pool3d(source_mask[None, None].to(ref['fine'].dtype),
                          model.patch, model.patch).flatten().bool()
    chosen = []
    source_cache = {}
    count = 0
    for index in sources:
        delta, coarse_index = _merge_delta_for_token(model, ref['fine'], clean['fine'], index, ref_merge)
        hybrid = attention_parts(block, ref['coarse_input'] + delta)
        prox = _attention_message_proxies(baseline, hybrid, ref['grad_gate'], block.attn.out_proj.weight)
        source_cache[index] = {'merge_delta': delta, 'coarse_index': coarse_index,
                               'direct_overlap': bool(direct[index]),
                               'fine_zyx': list(_flat_to_zyx(index, fine_grid)),
                               'source_proxy': float(relevance[index])}
        per_source = []
        for mode in config.modes:
            values = prox[mode]
            for h in range(values.shape[0]):
                for q in range(values.shape[1]):
                    for k in range(values.shape[2]):
                        value = float(values[h, q, k].detach())
                        per_source.append((abs(value), mode, h, q, k, value))
                        count += 1
        per_source.sort(key=lambda v: (-v[0], v[1], v[2], v[3], v[4]))
        for magnitude, mode, h, q, k, signed in per_source[:config.per_fine_screen]:
            if magnitude <= config.min_proxy_effect:
                continue
            chosen.append({'fine_token': index, 'head': h, 'query': q, 'key': k,
                           'mode': mode, 'proxy_signed': signed, 'proxy_abs': magnitude})
    chosen.sort(key=lambda v: (-v['proxy_abs'], v['fine_token'], v['mode'],
                               v['head'], v['query'], v['key']))
    return chosen, source_cache, count


def _patched_score(model, x, site, delta, roi, channel):
    block = model.coarse_blocks[0]
    def hook(_mod, _args, out):
        patched = out.clone()
        patched[:, site.query, :] += delta
        return patched
    handle = block.attn_to_residual.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(x, sample=False)
    finally:
        handle.remove()
    return logits, float(fixed_target_score(logits, channel, roi))


def _evaluate(model, clean, reference, base, candidate, cache, roi, channel, main_effect,
              config, directional=True):
    block = model.coarse_blocks[0]
    source = candidate['fine_token']
    site = AtomicAttentionSite(0, candidate['head'], candidate['query'],
                               candidate['key'], candidate['mode'])
    md = cache[source]['merge_delta']
    ref_before = attention_parts(block, reference['coarse_input'])
    ref_after = attention_parts(block, reference['coarse_input'] + md)
    delta_restore = _delta_one_head_query(ref_after, ref_before, site, block.attn.out_proj.weight)
    restored_logits, restored_score = _patched_score(model, base, site, delta_restore, roi, channel)
    # Symmetric disruption: reference the clean context but REPLACE one fine
    # source with the corresponding altered fine token, then isolate the SAME atom.
    clean_before = attention_parts(block, clean['coarse_input'])
    clean_after = attention_parts(block, clean['coarse_input'] - md)
    delta_disrupt = _delta_one_head_query(clean_after, clean_before, site, block.attn.out_proj.weight)
    disrupted_logits, disrupted_score = _patched_score(model, clean['input'], site, delta_disrupt, roi, channel)
    ref_score = float(fixed_target_score(reference['logits'], channel, roi))
    clean_score = float(fixed_target_score(clean['logits'], channel, roi))
    restoration = restored_score - ref_score
    disruption = clean_score - disrupted_score
    denom = max(abs(main_effect), config.min_input_effect)
    # Off-target term is measured independently of the on-target score.
    outside = ~roi
    off_target = float((restored_logits[:, channel, outside] - reference['logits'][:, channel, outside]).abs().mean())
    objective = ((abs(main_effect - restoration) + abs(main_effect - disruption)) / denom +
                 config.off_target_weight * off_target / denom)
    if not math.isfinite(objective):
        raise ArithmeticError('Nonfinite candidate objective')
    return {'restoration': restoration, 'disruption': disruption,
            'restoration_fraction': restoration / main_effect if abs(main_effect) >= config.min_input_effect else None,
            'disruption_fraction': disruption / main_effect if abs(main_effect) >= config.min_input_effect else None,
            'signed_reference_effect': main_effect,
            'direction_aligned': main_effect * restoration > 0 and main_effect * disruption > 0,
            'off_target_mean_abs': off_target, 'objective': float(objective),
            'reference_name': candidate.get('reference_name', config.reference_strategy)}


def discover_scca_multiscale(model, x, selected_mri_region, target_roi=None, channel=2,
                              config=None):
    """Automatic ROI->fine-mediator->merge->coarse-head-message->VIB->decoder discovery.

    All source tokens, head/query/key and mode are chosen automatically. The
    selected second reference is RESERVED and never used for candidate ranking.
    Cross-reference checks are same-patient sensitivity checks, NOT independent
    patient validation or clinical / path-specific causal uniqueness evidence.
    """
    cfg = config if config is not None else AutoDiscoveryConfig()
    if not isinstance(cfg, AutoDiscoveryConfig):
        raise TypeError('config must be AutoDiscoveryConfig')
    cfg.validate()
    target_roi = selected_mri_region if target_roi is None else target_roi
    _check_inputs(model, x, selected_mri_region, target_roi, channel)
    model.eval()
    with torch.no_grad():
        primary = reference_from_region(x, selected_mri_region, cfg.reference_strategy)
        secondary = reference_from_region(x, selected_mri_region, cfg.validation_strategy)
        clean = _state(model, x, target_roi, channel)
        ref2 = _state(model, secondary, target_roi, channel)
    ref = _state(model, primary, target_roi, channel, gradients=True)
    clean['input'] = x
    ref['input'] = primary
    ref2['input'] = secondary
    d1 = float(fixed_target_score(clean['logits'], channel, target_roi) -
               fixed_target_score(ref['logits'], channel, target_roi))
    d2 = float(fixed_target_score(clean['logits'], channel, target_roi) -
               fixed_target_score(ref2['logits'], channel, target_roi))
    summary = {'method': 'SCCA', 'version': '0.5-auto-multiscale',
               'model': 'two_scale_full_attention_one_coarse_block',
               'input_shape': list(x.shape),
               'source_voxel_count': int(selected_mri_region.sum()),
               'target_voxel_count': int(target_roi.sum()),
               'execution_mode': 'eval_deterministic_vib_mu',
               'status': 'unresolved', 'channel': channel,
               'reference_primary': cfg.reference_strategy,
               'reference_reserved': cfg.validation_strategy,
               'input_effect_primary': d1, 'input_effect_reserved': d2,
               'patient_independent_validation': False,
               'reference_validity': 'unverified_synthetic_diagnostics_not_medical_counterfactuals',
               'origin_claim': 'region-conditioned globally mixed fine feature mediator, NOT a strict local input-to-token path',
               'counterfactual_semantics': 'explicit_fine_feature_patch_merge_single_coarse_atomic_message_downstream_vib_decoder',
               'config': asdict(cfg), 'screened_atomic_count': 0,
               'exact_tested_count': 0, 'candidates': [], 'best': None,
               'reserved_validation': []}
    if abs(d1) < cfg.min_input_effect:
        summary['status'] = 'unresolved_low_primary_input_effect'
        return summary
    with torch.no_grad():
        screened, caches, screened_count = _screen(model, clean, ref, cfg, selected_mri_region)
    summary['screened_atomic_count'] = screened_count
    if not screened:
        summary['status'] = 'unresolved_no_measurable_atomic_candidate'
        return summary
    subset = screened[:cfg.max_exact_tests]
    evaluated = []
    with torch.no_grad():
        for candidate in subset:
            measured = _evaluate(model, clean, ref, primary, candidate, caches,
                                 target_roi, channel, d1, cfg)
            public_cache = {k: v for k, v in caches[candidate['fine_token']].items()
                            if k != 'merge_delta'}
            evaluated.append({**candidate, **public_cache, **measured,
                'declared_route': [
                    'selected_MRI_region_input_intervention',
                    f'globally_mixed_fine_feature_token_{candidate["fine_token"]}',
                    f'patch_merge_to_coarse_token_{public_cache["coarse_index"]}',
                    f'coarse_blocks.0.attn.head_{candidate["head"]}.q_{candidate["query"]}.k_{candidate["key"]}',
                    'coarse_residual_to_spatial_vib_to_decoder_with_reference_skip',
                ]})
    evaluated.sort(key=lambda row: (row['objective'], -row['proxy_abs'], row['fine_token'],
                                    row['mode'], row['head'], row['query'], row['key']))
    summary['exact_tested_count'] = len(evaluated)
    summary['candidates'] = evaluated
    summary['best'] = evaluated[0]
    # Reserved reference is used *only after* selection on the primary reference.
    selected = evaluated[:min(cfg.validate_top, len(evaluated))]
    if abs(d2) < cfg.min_input_effect:
        summary['status'] = 'unresolved_low_reserved_input_effect'
        return summary
    selected_sources = {c['fine_token'] for c in selected}
    with torch.no_grad():
        ref2_merge = model.patch_merge(ref2['fine'])
        reserve_caches = {i: {'merge_delta': _merge_delta_for_token(
            model, ref2['fine'], clean['fine'], i, ref2_merge)[0]} for i in selected_sources}
    validation = []
    with torch.no_grad():
        for chosen in selected:
            c = dict(chosen)
            c['reference_name'] = cfg.validation_strategy
            tested = _evaluate(model, clean, ref2, secondary, c, reserve_caches,
                               target_roi, channel, d2, cfg)
            fraction = min(abs(tested['restoration']), abs(tested['disruption'])) / max(abs(d2), cfg.min_input_effect)
            passes = bool(tested['direction_aligned'] and
                          fraction >= cfg.min_supported_fraction and
                          chosen['direction_aligned'])
            validation.append({'fine_token': c['fine_token'], 'head': c['head'],
                               'query': c['query'], 'key': c['key'], 'mode': c['mode'],
                               'reference_name': cfg.validation_strategy,
                               'effect': tested, 'cross_reference_direction_supported': passes})
    summary['reserved_validation'] = validation
    summary['status'] = ('same_case_cross_reference_supported' if any(
        v['cross_reference_direction_supported'] for v in validation)
        else 'unresolved_or_conflicting_references')
    return summary
