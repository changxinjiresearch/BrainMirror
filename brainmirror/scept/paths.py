"""Cross-layer, **path-specific** Transformer interventions (engineering v0.3).

A path is a sequence of DECLARED residual-branch outputs (edge gates) in
forward execution order. At each hop the prior branch's delta propagates only
through the residual identity stream, while all OFF-PATH branches are frozen to
their base-run values. The next branch is **recomputed** on the hybrid input:
  - MLP: normalised pre-MLP residual + incoming delta.
  - Attention: Q/K/V projections receive the hybrid normalised residual in
    the selected role(s), with other roles held at their base state.
Finally, ONLY the last gate's recomputed delta is patched into a normal base
forward; the remaining downstream model is allowed to run naturally.

This is exact for these *explicitly defined* interventions on the current
TransformerVIB3D's residual DAG, NOT recovery of a unique network circuit,
individual token-to-token Attention edges, all attention heads, or biology.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
from typing import Optional, Sequence

import torch

from .trace import captured_forward
from .scca import valid_declared_edges
from .explain import fixed_target_score

_EDGE_RE = re.compile(r'^blocks\.(\d+)\.(attn|mlp)_to_residual$')


@dataclass(frozen=True)
class PathHop:
    edge: str
    # For attention, choose which of the actual MHA Q/K/V inputs receives the
    # hybrid residual. 'qkv' is a joint input route, NOT one single atomic edge.
    role: str = 'qkv'

    def position(self):
        m = _EDGE_RE.fullmatch(self.edge)
        if m is None:
            raise ValueError(f'Not an explicit Transformer residual gate: {self.edge}')
        return int(m.group(1)), m.group(2)


def validate_path(model, hops: Sequence[PathHop]):
    if len(hops) < 2:
        raise ValueError('Need at least a source and a distinct downstream gate')
    from .model import TransformerVIB3D
    if not isinstance(model, TransformerVIB3D):
        raise TypeError("Explicit residual route adapter currently supports only TransformerVIB3D")
    edges = valid_declared_edges(model)
    if hops[0].role != "qkv":
        raise ValueError("Source gate has no Q/K/V input role")
    positions = []
    for hop in hops:
        if not isinstance(hop, PathHop) or hop.edge not in edges:
            raise ValueError('Every hop must be an explicitly declared edge')
        layer, branch = hop.position()
        if branch == 'attn' and hop.role not in ('q', 'k', 'v', 'qkv'):
            raise ValueError('Attention role must be q, k, v or qkv')
        if branch == 'mlp' and hop.role != 'qkv':
            raise ValueError('MLP branch does not have a Q/K/V input role')
        positions.append((layer, 0 if branch == 'attn' else 1))
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError('Hops must be strictly ordered along actual forward execution')
    # Repeated paths/branches are not allowed (and cannot pass order check).
    return tuple(hops)


@contextmanager
def _capture_residual_inputs(model, record):
    """Observe actual input of norm1 and norm2 in every Transformer block."""
    handles = []
    try:
        for i, block in enumerate(model.blocks):
            for branch, module in (('attn', block.norm1), ('mlp', block.norm2)):
                name = f'blocks.{i}.{branch}'
                def capture(_module, args, name=name):
                    if len(args) != 1 or not torch.is_tensor(args[0]):
                        raise RuntimeError('Unexpected norm input; path adapter invalid')
                    record[name] = args[0].detach().clone()
                handles.append(module.register_forward_pre_hook(capture))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture_path_baseline(model, x):
    """Return baseline logits, declared-gate traces and pre-branch residuals."""
    model.eval()
    residuals = {}
    with _capture_residual_inputs(model, residuals):
        logits, gates, _ = captured_forward(model, x)
    return logits, gates, residuals


@torch.no_grad()
def route_delta(model, hops, incoming, base_gates, base_residuals,
                token_mask: Optional[torch.Tensor] = None):
    """Compute changed output of FINAL hop; all intermediate non-path branches frozen.

    incoming: source-branch delta already calculated between two baselines.
    token_mask: optional [B,N] Boolean mask on the SOURCE gate delta only.
    Note: attention subsequently mixes token information, so destination
    changes are not constrained to the original token mask.
    """
    hops = validate_path(model, hops)
    source = hops[0].edge
    if incoming.shape != base_gates[source].shape:
        raise ValueError('Incoming source delta shape mismatch')
    if incoming.device != base_gates[source].device or incoming.dtype != base_gates[source].dtype:
        raise ValueError('Incoming source delta device/dtype mismatch')
    delta = incoming.detach().clone()
    if token_mask is not None:
        if (not torch.is_tensor(token_mask) or token_mask.dtype != torch.bool or
                token_mask.shape != delta.shape[:2] or token_mask.device != delta.device):
            raise ValueError('Invalid source token mask')
        delta = torch.where(token_mask.unsqueeze(-1), delta, torch.zeros_like(delta))
    if not torch.any(delta):
        return torch.zeros_like(base_gates[hops[-1].edge])
    for hop in hops[1:]:
        if not torch.any(delta):
            return torch.zeros_like(base_gates[hops[-1].edge])
        layer, branch = hop.position()
        block = model.blocks[layer]
        base = base_residuals[f'blocks.{layer}.{branch}']
        if delta.shape != base.shape:
            raise ValueError('Residual route dimensions differ: merging unsupported')
        if branch == 'mlp':
            # Only path delta propagates along residual additions. The base
            # Attention branch (if any) remains fixed in this hop.
            new_branch = block.mlp(block.norm2(base + delta))
        else:
            ref = block.norm1(base)
            changed = block.norm1(base + delta)
            q = changed if hop.role in ('q', 'qkv') else ref
            k = changed if hop.role in ('k', 'qkv') else ref
            v = changed if hop.role in ('v', 'qkv') else ref
            new_branch, _ = block.attn(q, k, v, need_weights=False)
        # Replace prior route delta with the variation through THIS selected
        # branch; the parallel residual bypass is not carried to the next hop.
        delta = new_branch - base_gates[hop.edge]
        if not torch.isfinite(delta).all():
            raise ArithmeticError('Nonfinite route delta')
    return delta


@torch.no_grad()
def intervene_path(model, x, hops, base_gates, base_residuals,
                   source_delta, token_mask=None):
    """Execute a controlled route-specific patch into the final edge only."""
    hops = validate_path(model, hops)
    final_delta = route_delta(model, hops, source_delta, base_gates,
                              base_residuals, token_mask)
    last = hops[-1].edge
    patched = base_gates[last] + final_delta
    output, _, _ = captured_forward(model, x, patch=(last, patched))
    return output, final_delta


@torch.no_grad()
def evaluate_path(model, clean_x, reference_x, hops, roi, channel=2,
                  token_mask=None, sparsity_penalty=0.04,
                  off_target_penalty=0.02, disruption_penalty=0.15):
    """Measured path-specific directional effects, *not* all-path total effects.

    In restoration, inject (clean-ref) SOURCE delta into a frozen-reference
    route, patch only final edge into reference forward. Disruption reverses
    clean/ref and the delta. The input effect is the total clean/ref change,
    hence route-specific effects may be much smaller or opposite in sign.
    """
    hops = validate_path(model, hops)
    if clean_x.ndim != 5 or clean_x.shape[1] != 4:
        raise ValueError('Expected 3D 4-channel model inputs')
    if clean_x.shape != reference_x.shape or clean_x.dtype != reference_x.dtype:
        raise ValueError('Inputs must share shape and dtype')
    if clean_x.device != reference_x.device:
        raise ValueError('Inputs must share device')
    if (roi.dtype != torch.bool or roi.shape != clean_x.shape[-3:] or
            not roi.any() or not (~roi).any() or roi.device != clean_x.device):
        raise ValueError('Invalid nonempty nonfull ROI on input device')
    if channel not in (0,1,2):
        raise ValueError('Unknown output channel')
    if not all(math.isfinite(p) and p >= 0 for p in
               (sparsity_penalty, off_target_penalty, disruption_penalty)):
        raise ValueError('Penalties must be finite and nonnegative')
    clean_logit, clean_gates, clean_res = capture_path_baseline(model, clean_x)
    ref_logit, ref_gates, ref_res = capture_path_baseline(model, reference_x)
    source = hops[0].edge
    delta = clean_gates[source] - ref_gates[source]
    restored, restore_delta = intervene_path(model, reference_x, hops, ref_gates,
                                             ref_res, delta, token_mask)
    disrupted, disrupt_delta = intervene_path(model, clean_x, hops, clean_gates,
                                               clean_res, -delta, token_mask)
    score_clean = float(fixed_target_score(clean_logit,channel,roi))
    score_ref = float(fixed_target_score(ref_logit,channel,roi))
    d_input = score_clean - score_ref
    d_restore = float(fixed_target_score(restored,channel,roi)) - score_ref
    d_disrupt = score_clean - float(fixed_target_score(disrupted,channel,roi))
    off = float((restored[:,channel,~roi]-ref_logit[:,channel,~roi]).abs().mean())
    if not all(math.isfinite(v) for v in (d_input,d_restore,d_disrupt,off)):
        raise ArithmeticError('Nonfinite path effect')
    denominator = max(abs(d_input),1e-8)
    objective = (((d_input-d_restore)/denominator)**2 +
                 disruption_penalty*((d_input-d_disrupt)/denominator)**2 +
                 sparsity_penalty*len(hops) + off_target_penalty*off/denominator)
    return {
        'path':[{'edge':h.edge,'input_role':h.role if h.position()[1]=='attn' else 'mlp'} for h in hops],
        'source_token_mask_applied':token_mask is not None,
        'input_effect':d_input, 'restoration':d_restore, 'disruption':d_disrupt,
        'fractional_recovery':d_restore/d_input if d_input else None,
        'off_target_change':off, 'objective':objective,
        'direction_aligned':bool(d_input*d_restore>0 and d_input*d_disrupt>=0),
        'route_delta_norm':float(restore_delta.norm()),
        'reverse_route_delta_norm':float(disrupt_delta.norm()),
        'evidence_status':'controlled_path_intervention_same_case',
        'semantics':'frozen_offpath_residual_path_final_edge_patch',
        'independent_patient_validation':False,
    }


@dataclass(frozen=True)
class PathSearchConfig:
    channel: int = 2
    max_candidates: int = 6
    max_hops: int = 3
    max_tested_paths: int = 24
    sparsity_penalty: float = 0.04
    off_target_penalty: float = 0.02
    disruption_penalty: float = 0.15
    min_input_effect: float = 1e-6
    attention_roles: tuple = ('q','k','v')

    def validate(self):
        if (self.channel not in (0,1,2) or self.max_candidates < 2 or
            self.max_hops < 2 or self.max_hops > 5 or self.max_tested_paths < 1 or
            not set(self.attention_roles).issubset({'q','k','v','qkv'}) or
            not self.attention_roles or min(self.sparsity_penalty,
            self.off_target_penalty,self.disruption_penalty) < 0 or
            not all(math.isfinite(x) for x in (self.min_input_effect,
                self.sparsity_penalty,self.off_target_penalty,self.disruption_penalty)) or
            self.min_input_effect <= 0):
            raise ValueError('Invalid path search configuration')


def _candidate_routes(sorted_candidates, max_hops, roles):
    """Deterministic proposed ordered paths, NOT confirmed causal circuits."""
    from itertools import combinations, product
    ordered = sorted(sorted_candidates, key=lambda s: (
        int(_EDGE_RE.fullmatch(s).group(1)),
        0 if '.attn_' in s else 1))
    for length in range(2, min(len(ordered), max_hops)+1):
        for names in combinations(ordered, length):
            role_options = [roles if '.attn_' in name else ('qkv',)
                            for name in names[1:]]
            for selected in product(*role_options):
                yield tuple([PathHop(names[0])] + [PathHop(name, role)
                      for name, role in zip(names[1:], selected)])


def explain_scca_paths(model, x, region, roi, config=None,
                       reference_strategy='channel_mean', validation_strategy='local_blur',
                       source_token_mask=None):
    """Budgeted cross-layer *route* search with reserved same-case reference.

    Output contains every tested route and its measured effects, not a claim of
    discovering a unique complete attention/neuron causal graph. Ranking uses
    ONLY the discovery reference; the other reference is frozen for evaluation.
    """
    from .scca import reference_volume, gradient_screen
    from .trace import numeric_trace_equivalence
    config = config or PathSearchConfig()
    config.validate()
    region.validate(x.shape[-3:])
    if (x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 4 or
       roi.dtype != torch.bool or roi.shape != x.shape[-3:] or
       roi.device != x.device or not roi.any() or not (~roi).any()):
        raise ValueError('Prototype expects one 4-channel MRI-shaped tensor with valid ROI')
    if reference_strategy == validation_strategy:
        raise ValueError('Independent reference strategy must differ from discovery')
    model.eval()
    meta = valid_declared_edges(model)
    candidates = [name for name in meta if _EDGE_RE.fullmatch(name)]
    if not candidates or not numeric_trace_equivalence(model,x):
        raise ValueError('Model not supported or tracing changed numerical output')
    ref = reference_volume(x,region,reference_strategy)
    clean_logits, clean_trace, _ = captured_forward(model,x)
    ref_logits, _, _ = captured_forward(model,ref)
    d_input = float(fixed_target_score(clean_logits,config.channel,roi) -
                    fixed_target_score(ref_logits,config.channel,roi))
    base = dict(method='SCCA',version='cross_layer_path_alpha_v0.3',
                interpretation='isolated frozen-offpath residual route; not complete causal circuit',
                source_region=region.__dict__.copy(),target_channel=config.channel,
                selection_reference=reference_strategy,
                validation_reference=validation_strategy,
                input_effect=d_input,independent_patient_validation=False,
                selection_budget=config.max_tested_paths)
    if abs(d_input) < config.min_input_effect:
        return {**base,'status':'unresolved_low_input_effect','selected':None,
                'tested_paths':[],'validation':{'status':'not_run'}}
    screen = gradient_screen(model,ref,clean_trace,candidates,roi,config.channel)
    selected_edges=sorted(screen,key=lambda n:(-abs(screen[n]),n))[:config.max_candidates]
    proposed=list(_candidate_routes(selected_edges,config.max_hops,config.attention_roles))
    # Priority balances both endpoints; gradient is ONLY a proposal heuristic.
    def priority(hops):
        return (-min(abs(screen[hops[0].edge]),abs(screen[hops[-1].edge])),
                len(hops),tuple((h.edge,h.role) for h in hops))
    proposed.sort(key=priority)
    proposed=proposed[:config.max_tested_paths]
    measured=[]
    for path in proposed:
        result=evaluate_path(model,x,ref,path,roi,config.channel,
                       token_mask=source_token_mask,
                       sparsity_penalty=config.sparsity_penalty,
                       off_target_penalty=config.off_target_penalty,
                       disruption_penalty=config.disruption_penalty)
        measured.append(result)
    if not measured:
        return {**base,'status':'unresolved_no_valid_path','selected':None,
                'tested_paths':[],'validation':{'status':'not_run'},
                'gradient_proposal_scores':screen}
    measured.sort(key=lambda r:(r['objective'],len(r['path']),
               tuple((x['edge'],x['input_role']) for x in r['path'])))
    best=measured[0]
    fixed_hops=tuple(PathHop(h['edge'],
                   h['input_role'] if h['input_role']!='mlp' else 'qkv')
                   for h in best['path'])
    holdout=reference_volume(x,region,validation_strategy)
    holdout_result=evaluate_path(model,x,holdout,fixed_hops,roi,config.channel,
                        token_mask=source_token_mask,
                        sparsity_penalty=config.sparsity_penalty,
                        off_target_penalty=config.off_target_penalty,
                        disruption_penalty=config.disruption_penalty)
    holdout_status=('unresolved_low_validation_effect'
                    if abs(holdout_result['input_effect']) < config.min_input_effect
                    else 'evaluated_same_case_reserved_reference')
    return {**base,'status':('candidate_selected_not_independently_validated'
                if best['direction_aligned'] else 'unresolved_conflicting_directions'),
            'gradient_proposal_scores':screen,
            'proposed_path_count':len(proposed),
            'tested_paths':measured,'selected':best,
            'validation':{'status':holdout_status,'strategy':validation_strategy,
                'independent_patient_validation':False,
                'same_case_different_reference':True,'fixed_path':holdout_result}}
