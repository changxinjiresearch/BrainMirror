"""SCEPT alpha: measured spatial effects + measured component patch effects.

This does NOT yet implement faithful edge/path patching, independent clinical
validation, or the full proposed SCCA cross-level sparse joint optimizer.
"""
from dataclasses import asdict, dataclass
import torch
from .trace import captured_forward, allowed_modules, numeric_trace_equivalence


@dataclass(frozen=True)
class Region:
    z0: int; z1: int; y0: int; y1: int; x0: int; x1: int

    def validate(self, spatial):
        for lo, hi, limit in zip((self.z0, self.y0, self.x0),
                                 (self.z1, self.y1, self.x1), spatial):
            if not 0 <= lo < hi <= limit:
                raise ValueError('Invalid region bounds')


def local_reference(x, region):
    """Channel-wise volume mean fill; SYNTHETIC smoke reference only, not an MRI counterfactual."""
    if x.ndim != 5 or x.shape[1] != 4:
        raise ValueError('Expected [B,4,D,H,W]')
    region.validate(x.shape[-3:])
    altered = x.clone()
    fill = x.mean(dim=(-3, -2, -1), keepdim=True)
    altered[..., region.z0:region.z1, region.y0:region.y1,
            region.x0:region.x1] = fill
    return altered


def fixed_target_score(logits, channel, roi):
    if logits.ndim != 5 or channel not in (0, 1, 2) or roi.shape != logits.shape[-3:]:
        raise ValueError('Expected [B,3,D,H,W], channel in 0..2, fixed [D,H,W] ROI')
    if roi.dtype != torch.bool or not roi.any():
        raise ValueError('ROI must be nonempty boolean mask')
    return logits[:, channel, roi].mean()


@torch.no_grad()
def explain_region(model, x, region, roi, channel=2, max_sites=None):
    """One input reference and measured intervention effects for each selected site.

    Disruption = score(clean) - score(clean with altered site state).
    Restoration = score(altered with clean site state) - score(altered).
    Results are *node/module output* interventions, not discovered edge circuits.
    """
    model.eval()
    if not numeric_trace_equivalence(model, x):
        raise RuntimeError('Trace changed unpatched computation')
    altered = local_reference(x, region)
    clean_logits, clean, _ = captured_forward(model, x)
    alt_logits, alt, _ = captured_forward(model, altered)
    score0 = float(fixed_target_score(clean_logits, channel, roi))
    score1 = float(fixed_target_score(alt_logits, channel, roi))
    input_effect = score0 - score1
    sites = list(allowed_modules(model))
    if max_sites is not None:
        if max_sites < 0:
            raise ValueError('max_sites must be nonnegative')
        sites = sites[:max_sites]
    items = []
    for name in sites:
        disrupted, _, _ = captured_forward(model, x, patch=(name, alt[name]))
        restored, _, _ = captured_forward(model, altered, patch=(name, clean[name]))
        d = score0 - float(fixed_target_score(disrupted, channel, roi))
        r = float(fixed_target_score(restored, channel, roi)) - score1
        items.append({'module': name, 'kind': 'module_output_intervention',
                      'disruption': d, 'restoration': r,
                      'approximate_effect_match': abs(input_effect - r),
                      'evidence_status': 'intervened_not_cross_validated'})
    items.sort(key=lambda a: a['approximate_effect_match'])
    return {'method': 'SCEPT', 'stage': 'alpha_module_intervention',
            'reference': 'channel_volume_mean_SYNTHETIC_ONLY',
            'target_channel': channel, 'region': asdict(region),
            'input_effect': input_effect, 'clean_score': score0,
            'altered_score': score1, 'modules': items,
            'mechanistic_claim': 'node-level effects only; no verified edge circuit',
            'independent_validation': False}
