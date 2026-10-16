"""Exact Spatial VIB token-to-decoder interventions on original single-scale model.

A latent intervention changes the chosen spatial z token/channel AFTER the
VIB sampler. Output is recomputed through the real ConvTranspose3d decoder;
the independent linear decoder oracle tests the identity of the change.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .model import TransformerVIB3D
from .trace import captured_forward


@dataclass(frozen=True)
class LatentSite:
    token: int
    channel: int | None = None


@torch.no_grad()
def intervene_spatial_vib(model, clean_x, reference_x, site):
    if not isinstance(model, TransformerVIB3D):
        raise TypeError('Expected original TransformerVIB3D')
    if (not isinstance(site, LatentSite) or clean_x.shape != reference_x.shape or
            clean_x.ndim != 5 or clean_x.shape[0] != 1 or clean_x.shape[1] != 4 or
            clean_x.dtype != reference_x.dtype or clean_x.device != reference_x.device):
        raise ValueError('Invalid latent site or paired MRI-shaped inputs')
    model.eval()
    clean_logits, _, clean = captured_forward(model, clean_x)
    ref_logits, _, ref = captured_forward(model, reference_x)
    z_clean, z_ref = clean['latent'], ref['latent']
    if not 0 <= site.token < z_ref.shape[1]:
        raise ValueError('Latent spatial token out of range')
    if site.channel is not None and not 0 <= site.channel < z_ref.shape[-1]:
        raise ValueError('Latent channel out of range')
    patch = z_ref.clone()
    if site.channel is None:
        patch[:, site.token] = z_clean[:, site.token]
    else:
        patch[:, site.token, site.channel] = z_clean[:, site.token, site.channel]
    output, _, _ = captured_forward(model, reference_x, patch=('vib', patch))
    dz = patch - z_ref
    sparse_grid = dz.transpose(1, 2).reshape(clean_x.shape[0], -1, *ref['grid'])
    # Bias cancellation makes this an independent oracle of the REAL decoder.
    expected_delta = F.conv_transpose3d(sparse_grid, model.decoder.weight,
                                        bias=None, stride=model.patch)
    actual_delta = output - ref_logits
    if not torch.allclose(actual_delta, expected_delta, atol=2e-6, rtol=2e-5):
        raise AssertionError('Spatial VIB->decoder numerical oracle failed')
    return {'clean_logits': clean_logits, 'reference_logits': ref_logits,
            'intervened_logits': output, 'latent_delta': dz,
            'expected_decoder_delta': expected_delta,
            'semantics': 'post_vib_latent_token_splice_real_decoder',
            'independent_patient_validation': False}
