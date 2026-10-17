"""Exact module-level traces and *node-level* activation patching.

A patch is a model computation intervention, not a demonstrated path or
human neurobiological causal effect. No cross-model hidden-state swaps.
"""
from contextlib import contextmanager
import torch


def allowed_modules(model):
    # Explicit, versioned patch sites; actual module forward outputs are used.
    return {name: module for name, module in model.named_modules()
            if (name.endswith('.attn') or name.endswith('.mlp') or
                name.endswith('_to_residual') or name == 'vib')}


def _primary(output):
    # MHA returns (attention output, head weight matrix).
    return output[0] if isinstance(output, tuple) else output


def _replace(output, tensor):
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    return tensor


@contextmanager
def instrument(model, *, patch=None, records=None):
    """Capture module outputs; optionally inject one or multiple exact states.

    patch accepts (site, tensor), (site, tensor, boolean_token_mask), or
    {site: tensor}. `*_to_residual` sites correspond to declared single-consumer
    branch edges in TransformerVIB3D; ordinary sites are node interventions.
    """
    sites = allowed_modules(model)
    if patch is None:
        patches = {}
    elif isinstance(patch, dict):
        patches = patch
    elif isinstance(patch, tuple) and len(patch) in (2, 3):
        patches = {patch[0]: patch[1:]}
    else:
        raise TypeError('Patch must be a mapping or (site, tensor[, mask])')
    unknown = set(patches).difference(sites)
    if unknown:
        raise KeyError(f'Unknown patch sites: {sorted(unknown)}')
    handles = []
    try:
        for name, module in sites.items():
            def callback(_module, _input, output, name=name):
                value = _primary(output)
                if records is not None:
                    records[name] = value.detach().clone()
                    if name.endswith('.attn') and isinstance(output, tuple):
                        records[name + '.weights'] = output[1].detach().clone()
                if name not in patches:
                    return None
                specification = patches[name]
                if isinstance(specification, tuple):
                    if len(specification) not in (1, 2):
                        raise ValueError('Bad patch tuple')
                    replacement = specification[0]
                    mask = specification[1] if len(specification) == 2 else None
                else:
                    replacement, mask = specification, None
                if (not torch.is_tensor(replacement) or
                    replacement.shape != value.shape or
                    replacement.device != value.device or
                    replacement.dtype != value.dtype):
                    raise ValueError(f'Patch tensor mismatch at {name}')
                if mask is not None:
                    if (name == 'vib' or value.ndim != 3 or
                        not torch.is_tensor(mask) or mask.dtype != torch.bool or
                        mask.shape != value.shape[:2] or mask.device != value.device):
                        raise ValueError(f'Invalid token mask at {name}')
                    changed = torch.where(mask.unsqueeze(-1), replacement.detach(), value)
                else:
                    changed = replacement.detach().clone()
                return _replace(output, changed)
            handles.append(module.register_forward_hook(callback))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def captured_forward(model, x, patch=None):
    model.eval()
    records = {}
    with instrument(model, patch=patch, records=records):
        logits, details = model(x, sample=False, return_details=True)
    return logits.detach(), records, details


def numeric_trace_equivalence(model, x, atol=1e-6):
    model.eval()
    with torch.no_grad():
        plain = model(x, sample=False)
        traced, _, _ = captured_forward(model, x)
    return torch.allclose(plain, traced, atol=atol, rtol=atol)
