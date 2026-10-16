"""SCCA alpha: cross-level region/declared-edge *intervention* search.

An edge means an explicitly wired, single-consumer branch-to-residual
Identity site, NOT attention weights or a discovered semantic neural circuit.
The search finds small intervention *edge sets*, not unique causal paths.
This implementation does NOT establish patient-independent validation.
"""
from dataclasses import asdict, dataclass
import math
from typing import Optional
import torch
from torch.nn import functional as F
from .explain import Region, fixed_target_score, local_reference
from .trace import allowed_modules, captured_forward, instrument, numeric_trace_equivalence


@dataclass(frozen=True)
class SCCAConfig:
    channel: int = 2
    max_candidates: int = 8
    max_circuit_size: int = 2
    beam_width: int = 3
    sparsity_penalty: float = 0.04
    off_target_penalty: float = 0.02
    disruption_penalty: float = 0.15
    min_input_effect: float = 1e-6

    def validate(self):
        if (self.channel not in (0, 1, 2) or self.max_candidates < 1 or
                self.max_circuit_size < 1 or self.beam_width < 1 or
                not 0 <= self.sparsity_penalty <= 10 or
                not 0 <= self.off_target_penalty <= 10 or
                not 0 <= self.disruption_penalty <= 10 or
                not math.isfinite(self.min_input_effect) or self.min_input_effect <= 0):
            raise ValueError('Invalid SCCA configuration')


def valid_declared_edges(model):
    """Return only documented single-consumer intervention gates.

    This checks registration/type but cannot mechanically prove consumer
    exclusivity: source wiring is verified in the model implementation/tests.
    """
    if not hasattr(model, 'scca_edge_routes'):
        raise TypeError('Model must declare scca_edge_routes()')
    metadata = model.scca_edge_routes()
    sites = allowed_modules(model)
    for name, spec in metadata.items():
        if name not in sites or not isinstance(sites[name], torch.nn.Identity):
            raise ValueError(f'Not a registered Identity edge gate: {name}')
        if not all(k in spec for k in ('source', 'consumer', 'intervention_unit')):
            raise ValueError(f'Incomplete edge metadata: {name}')
    if not metadata:
        raise ValueError('No declared edge intervention gates')
    return metadata


def reference_volume(x, region: Region, strategy='channel_mean'):
    """Synthetic-reference utilities. Neither represents clinical counterfactual MRI."""
    if strategy == 'channel_mean':
        return local_reference(x, region)
    if strategy != 'local_blur':
        raise ValueError('Unknown reference strategy')
    if x.ndim != 5 or x.shape[1] != 4:
        raise ValueError('Expected [B,4,D,H,W]')
    region.validate(x.shape[-3:])
    blurred = F.avg_pool3d(F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), 3, stride=1)
    alt = x.clone()
    alt[..., region.z0:region.z1, region.y0:region.y1,
        region.x0:region.x1] = blurred[..., region.z0:region.z1, region.y0:region.y1,
                                 region.x0:region.x1]
    return alt


def gradient_screen(model, altered, clean_trace, candidates, roi, channel):
    """First-order screening only. Never report scores as causal effects."""
    activations = {}
    handles = []
    modules = allowed_modules(model)
    try:
        for name in candidates:
            def save(_module, _input, output, name=name):
                activations[name] = output[0] if isinstance(output, tuple) else output
            handles.append(modules[name].register_forward_hook(save))
        # Require input grad so screening works even with a frozen model.
        with torch.enable_grad():
            logits = model(altered.detach().requires_grad_(True), sample=False)
            score = fixed_target_score(logits, channel, roi)
            grads = torch.autograd.grad(
                score, [activations[name] for name in candidates], allow_unused=True)
    finally:
        for handle in handles:
            handle.remove()
    scores = {}
    for name, g in zip(candidates, grads):
        if g is None:
            scores[name] = 0.0
        else:
            # Exact clean-corrupt activation difference times local gradient.
            gap = clean_trace[name] - activations[name].detach()
            scores[name] = float((g.detach() * gap).sum())
    return scores


def _score(model, x, roi, channel, patches=None):
    logits, _, _ = captured_forward(model, x, patch=patches)
    return logits, float(fixed_target_score(logits, channel, roi))


def evaluate_edge_set(model, clean_x, altered_x, clean_trace, ref_trace,
                      edge_names, roi, channel, score_clean, score_ref,
                      sparsity_penalty=0.04, off_target_penalty=0.02,
                      disruption_penalty=0.15):
    """Measured two-direction interventions; no gradient approximation here."""
    sites = tuple(sorted(edge_names))
    declared = valid_declared_edges(model)
    if not set(sites).issubset(declared):
        raise ValueError('Only declared single-consumer edge gates may be patched')
    if not sites:
        raise ValueError('Empty edge set')
    disruption_patches = {name: ref_trace[name] for name in sites}
    restoration_patches = {name: clean_trace[name] for name in sites}
    _, score_d = _score(model, clean_x, roi, channel, disruption_patches)
    restored_logits, score_r = _score(model, altered_x, roi, channel, restoration_patches)
    reference_logits, _ = _score(model, altered_x, roi, channel)
    d_input = score_clean - score_ref
    d_disrupt = score_clean - score_d
    d_restore = score_r - score_ref
    target_delta = abs(d_input)
    relative_error = abs(d_input - d_restore) / max(target_delta, 1e-8)
    disruption_error = abs(d_input - d_disrupt) / max(target_delta, 1e-8)
    other = ~roi
    off_target = float((restored_logits[:, channel, other] -
                        reference_logits[:, channel, other]).abs().mean()) if other.any() else 0.0
    objective = (relative_error ** 2 + disruption_penalty * disruption_error ** 2 +
                 sparsity_penalty * len(sites) +
                 off_target_penalty * off_target / max(target_delta, 1e-8))
    if not all(math.isfinite(y) for y in (d_disrupt, d_restore, objective, off_target)):
        raise ArithmeticError('Non-finite intervention effect')
    aligned = (d_input * d_restore > 0 and d_input * d_disrupt >= 0)
    return dict(edges=list(sites), input_effect=d_input, disruption=d_disrupt,
                restoration=d_restore, fractional_recovery=d_restore / d_input if d_input else None,
                relative_restoration_error=relative_error,
                relative_disruption_error=disruption_error, off_target_change=off_target,
                objective=objective, direction_aligned=bool(aligned),
                evidence_status='intervened_selection_only')


@torch.no_grad()
def _baseline(model, x, reference, roi, channel):
    clean_logits, clean_trace, _ = captured_forward(model, x)
    ref_logits, ref_trace, _ = captured_forward(model, reference)
    return (float(fixed_target_score(clean_logits, channel, roi)),
            float(fixed_target_score(ref_logits, channel, roi)), clean_trace, ref_trace)


def explain_scca(model, x, region: Region, roi, config: Optional[SCCAConfig] = None,
                 reference_strategy='channel_mean', validation_strategy='local_blur'):
    """Bounded forward-only circuit search over *declared* computation edges.

    The validation reference is not independent patient validation and must
    NEVER be used to select a circuit. No patient- or clinical-level claims.
    """
    config = config or SCCAConfig()
    config.validate()
    region.validate(x.shape[-3:])
    if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 4:
        raise ValueError('Prototype accepts one 4-channel 3D image at a time')
    if roi.dtype != torch.bool or roi.shape != x.shape[-3:] or not roi.any() or not (~roi).any():
        raise ValueError('Need nonempty, nonfull spatial Boolean ROI')
    if reference_strategy == validation_strategy:
        raise ValueError('Validation strategy must differ from discovery strategy')
    meta = valid_declared_edges(model)
    model.eval()
    if not numeric_trace_equivalence(model, x):
        raise RuntimeError('Tracing is not numerically transparent')
    reference = reference_volume(x, region, reference_strategy)
    score_clean, score_ref, clean, ref = _baseline(model, x, reference, roi, config.channel)
    input_effect = score_clean - score_ref
    base = dict(method='SCCA', stage='declared_edge_set_intervention_alpha',
                region=asdict(region), target_channel=config.channel,
                reference=reference_strategy, validation_reference=validation_strategy,
                clean_score=score_clean, reference_score=score_ref,
                input_effect=input_effect, edge_annotations=meta,
                interpretation='Single-consumer residual branch edge-set interventions; not complete neural circuits',
                independent_patient_validation=False)
    if abs(input_effect) < config.min_input_effect:
        return {**base, 'status':'unresolved_low_input_effect',
                'candidates': [], 'tested_edge_sets': [], 'selected': None,
                'validation': {'status':'not_run'}}
    screen = gradient_screen(model, reference, clean, list(meta), roi, config.channel)
    names = sorted(screen, key=lambda n: (-abs(screen[n]), n))[:config.max_candidates]
    evaluated = {}
    def evaluate(names_tuple):
        key = tuple(sorted(names_tuple))
        if key not in evaluated:
            evaluated[key] = evaluate_edge_set(
                model, x, reference, clean, ref, key, roi, config.channel,
                score_clean, score_ref, config.sparsity_penalty,
                config.off_target_penalty, config.disruption_penalty)
        return evaluated[key]
    beam = [()]
    for size in range(1, min(config.max_circuit_size, len(names)) + 1):
        expanded = set()
        for prefix in beam:
            for name in names:
                if name not in prefix:
                    candidate = tuple(sorted(prefix + (name,)))
                    if len(candidate) == size:
                        expanded.add(candidate)
        if not expanded:
            break
        ranked = sorted(expanded, key=lambda key: (evaluate(key)['objective'], key))
        beam = ranked[:config.beam_width]
    best = min(evaluated, key=lambda key: (evaluated[key]['objective'], key))
    selected = evaluated[best]
    if selected['direction_aligned']:
        selection_status = 'candidate_selected_not_independently_validated'
    else:
        selection_status = 'unresolved_conflicting_intervention_directions'
    # Independent *reference strategy*, but the same image and model. No
    # independent patient/seed inference is claimed here.
    validation_ref = reference_volume(x, region, validation_strategy)
    score_clean_v, score_ref_v, clean_v, ref_v = _baseline(
        model, x, validation_ref, roi, config.channel)
    validation = {'strategy': validation_strategy,
                  'same_case_different_reference': True,
                  'independent_patient_validation': False}
    effect_v = score_clean_v - score_ref_v
    validation['input_effect'] = effect_v
    if abs(effect_v) < config.min_input_effect:
        validation['status'] = 'unresolved_low_validation_effect'
    else:
        verified = evaluate_edge_set(
            model, x, validation_ref, clean_v, ref_v, best, roi,
            config.channel, score_clean_v, score_ref_v,
            config.sparsity_penalty, config.off_target_penalty,
            config.disruption_penalty)
        validation.update({'status':'evaluated_on_reserved_reference',
                           'disruption':verified['disruption'],
                           'restoration':verified['restoration'],
                           'relative_restoration_error':verified['relative_restoration_error'],
                           'direction_aligned':verified['direction_aligned']})
    return {**base, 'status':selection_status,
            'gradient_screen':screen,
            'candidates':names,
            'tested_edge_sets':sorted(evaluated.values(),key=lambda r:r['objective']),
            'selected':selected, 'validation':validation}
