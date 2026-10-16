"""Lightweight 3D I-JEPA-*style* masked latent prediction pretrainer.

No claim to replicate official I-JEPA; no external pretrained weights are used.
Only the student encoder is transferred to downstream TransformerVIB3D.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class IJEPAPretrainer(nn.Module):
    def __init__(self, student, mask_ratio=0.5, momentum=0.996):
        super().__init__()
        if not 0 < mask_ratio < 1 or not 0 <= momentum < 1:
            raise ValueError('Invalid mask ratio or momentum')
        self.student = student
        self.teacher = deepcopy(student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Linear(student.norm.normalized_shape[0],
                                   student.norm.normalized_shape[0])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, student.norm.normalized_shape[0]))
        self.mask_ratio, self.momentum = mask_ratio, momentum

    def forward(self, x, mask=None):
        b = x.shape[0]
        if mask is None:
            with torch.no_grad():
                tokens, _, _ = self.student.tokenize(x)
            n = tokens.shape[1]
            k = max(1, min(n - 1, round(n * self.mask_ratio)))
            mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
            for i in range(b):
                mask[i, torch.randperm(n, device=x.device)[:k]] = True
        if mask.dtype != torch.bool or mask.ndim != 2 or not mask.any() or mask.all():
            raise ValueError('mask must be boolean [B,N] with visible and target tokens')
        visible, _ = self.student.encode(x, mask=mask, mask_token=self.mask_token)
        with torch.no_grad():
            target, _ = self.teacher.encode(x)
        if mask.shape != visible.shape[:2]:
            raise ValueError('mask shape does not match token count')
        pred = self.predictor(visible)
        return F.smooth_l1_loss(pred[mask], target[mask])

    @torch.no_grad()
    def update_teacher(self):
        # Called AFTER optimizer.step(); EMA teacher is never backpropagated.
        src = dict(self.student.named_parameters())
        for name, p in self.teacher.named_parameters():
            if name.startswith(('embed.', 'position.', 'blocks.', 'norm.')):
                p.lerp_(src[name], 1.0 - self.momentum)
        src_buf = dict(self.student.named_buffers())
        for name, b in self.teacher.named_buffers():
            if name in src_buf:
                b.copy_(src_buf[name])

    def pretrain_step(self, x, optimizer):
        self.student.train()
        self.teacher.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = self(x)
        loss.backward()
        optimizer.step()
        self.update_teacher()
        return float(loss.detach())


def transfer_pretrained_encoder(source, destination):
    """Copy ONLY pretrained encoder (embed/position/blocks/norm), not VIB/decoder."""
    for name in ('embed', 'position', 'blocks', 'norm'):
        getattr(destination, name).load_state_dict(getattr(source, name).state_dict())
    return destination
