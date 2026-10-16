"""Exact module-level traces and *node-level* activation patching.

A patch is a model computation intervention, not a demonstrated path or
human neurobiological causal effect. No cross-model hidden-state swaps.
"""
from contextlib import contextmanager
import torch


def allowed_modules(model):
    # Explicit, versioned patch sites; actual module forward outputs are used.
    return {name: module for name, module in model.named_modules()
            if (name.endswith('.attn') or name.endswith('.mlp') or name == 'vib')}


def _primary(output):
    # MHA returns (attention output, head weight matrix).
    return output[0] if isinstance(output, tuple) else output


def _replace(output, tensor):
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    return tensor


@contextmanager
def instrument(model, *, patch=None, records=None):
    """Capture module outputs and optionally replace ONE selected module output.

    For SpatialVIB, replace only z, preserving KL/mu/logvar as diagnostics.
    Every hook is removed even on exception; inputs and weights not mutated.
    """
    sites = allowed_modules(model)
    if patch is not None and patch[0] not in sites:
        raise KeyError(f'Unknown patch site: {patch[0]}')
    handles = []
    try:
        for name, module in sites.items():
            def callback(_module, _input, output, name=name):
                value = _primary(output)
                if name == 'vib':
                    value = output[0]
                if records is not None:
                    records[name] = value.detach().clone()
                    if name.endswith('.attn') and isinstance(output, tuple):
                        records[name + '.weights'] = output[1].detach().clone()
                if patch is not None and patch[0] == name:
                    replacement = patch[1]
                    if replacement.shape != value.shape or replacement.device != value.device:
                        raise ValueError(f'Patch mismatch at {name}')
                    return _replace(output, replacement.detach().clone())
                return None
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
