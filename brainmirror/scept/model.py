"""Small inspectable 3D Transformer + *spatial* variational bottleneck.

Engineering integration, NOT the final SwinUNETR architecture or validated MRI model.
Tokens are explicit (D/p,H/p,W/p); three independent output logits are WT/TC/ET.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        h = self.norm1(x)
        delta, _weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        x = x + delta
        return x + self.mlp(self.norm2(x))


class SpatialVIB(nn.Module):
    def __init__(self, dim=32, latent_dim=16):
        super().__init__()
        self.mu = nn.Linear(dim, latent_dim)
        self.logvar = nn.Linear(dim, latent_dim)

    def forward(self, tokens, sample=False):
        mu = self.mu(tokens)
        lv = self.logvar(tokens).clamp(-12.0, 8.0)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu) if sample else mu
        kl = 0.5 * (mu.square() + lv.exp() - 1.0 - lv).mean()
        return z, kl, mu, lv


class TransformerVIB3D(nn.Module):
    """Minimal 3D token-based segmentation adapter; max_tokens caps quadratic attention."""
    def __init__(self, channels=4, dim=32, heads=4, depth=2, patch=4,
                 latent_dim=16, max_tokens=512):
        super().__init__()
        if channels != 4 or dim % heads or min(depth, patch, latent_dim, max_tokens) < 1:
            raise ValueError('Need 4 input modalities, dim divisible by heads, positive sizes')
        self.patch = patch
        self.max_tokens = max_tokens
        self.embed = nn.Conv3d(channels, dim, patch, stride=patch)
        self.position = nn.Linear(3, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.vib = SpatialVIB(dim, latent_dim)
        self.decoder = nn.ConvTranspose3d(latent_dim, 3, patch, stride=patch)

    def tokenize(self, x):
        if x.ndim != 5 or x.shape[1] != 4 or any(s % self.patch for s in x.shape[-3:]):
            raise ValueError('Expected [B,4,D,H,W], all spatial sizes divisible by patch')
        grid = tuple(s // self.patch for s in x.shape[-3:])
        count = grid[0] * grid[1] * grid[2]
        if count > self.max_tokens:
            raise ValueError(f'{count} tokens exceed max_tokens={self.max_tokens}; use 3D crops')
        token = self.embed(x).flatten(2).transpose(1, 2)
        coords = torch.stack(torch.meshgrid(
            *[torch.linspace(-1, 1, g, device=x.device, dtype=x.dtype) for g in grid],
            indexing='ij'), dim=-1).reshape(-1, 3)
        return token, coords, grid

    def encode(self, x, mask=None, mask_token=None):
        token, coords, grid = self.tokenize(x)
        if mask is not None:
            if mask_token is None or mask.shape != token.shape[:2]:
                raise ValueError('mask must be [B,N] and mask_token supplied')
            token = torch.where(mask.unsqueeze(-1), mask_token.to(token.dtype), token)
        token = token + self.position(coords).unsqueeze(0)
        for block in self.blocks:
            token = block(token)
        return self.norm(token), grid

    def forward(self, x, sample=False, return_details=False):
        tokens, grid = self.encode(x)
        z, kl, mu, logvar = self.vib(tokens, sample=sample)
        out = self.decoder(z.transpose(1, 2).reshape(x.shape[0], -1, *grid))
        if return_details:
            return out, {'kl': kl, 'mu': mu, 'logvar': logvar,
                         'latent': z, 'grid': grid}
        return out

    def encoder_parameters(self):
        for module in (self.embed, self.position, self.blocks, self.norm):
            yield from module.parameters()


def brats_dice_bce(logits, target):
    """WT/TC/ET overlapping labels, target [B,3,D,H,W]; no clinical claim."""
    if logits.shape != target.shape:
        raise ValueError('target shape must match three output channels')
    prob = logits.sigmoid()
    dims = (0, 2, 3, 4)
    dice = 1 - (2 * (prob * target).sum(dims) + 1) / (
        prob.sum(dims) + target.sum(dims) + 1)
    return dice.mean() + F.binary_cross_entropy_with_logits(logits, target)
