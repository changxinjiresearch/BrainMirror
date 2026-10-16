"""Spatial VIB inserted into the *actual* MONAI SwinUNETR bottleneck.

The disabled adapter is mathematically equivalent to upstream forward.
When enabled, the bottleneck changes but the decoder and all skip connections
remain MONAI's original modules; skips bypass VIB by design.
"""
from __future__ import annotations
import torch
from torch import nn
from .monai_adapter import require_monai


class MonaiSpatialVIB(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.mu_delta=nn.Conv3d(channels,channels,1)
        self.logvar=nn.Conv3d(channels,channels,1)
        nn.init.zeros_(self.mu_delta.weight)
        nn.init.zeros_(self.mu_delta.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias,-4.)

    def forward(self,x,*,sample=False,noise=None):
        mu=x+self.mu_delta(x)
        logvar=self.logvar(x).clamp(-12,8)
        if sample:
            if noise is None:
                noise=torch.randn_like(mu)
            if noise.shape!=mu.shape:
                raise ValueError('VIB paired-noise shape mismatch')
            z=mu+torch.exp(.5*logvar)*noise
        else:
            z=mu
        kl=0.5*(mu.square()+logvar.exp()-1-logvar).mean()
        return z,{'mu':mu,'logvar':logvar,'kl':kl}


class MonaiSwinVIBBridge(nn.Module):
    def __init__(self, backbone:nn.Module,*,feature_size:int,enabled=True):
        super().__init__()
        SwinUNETR,_,_=require_monai()
        if not isinstance(backbone,SwinUNETR):
            raise TypeError('Requires a real MONAI SwinUNETR')
        self.backbone=backbone
        self.vib=MonaiSpatialVIB(16*feature_size)
        self.enabled=enabled
        self.last_evidence=None

    def forward(self,x,*,sample=False,noise=None):
        m=self.backbone
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            m._check_input_size(x.shape[2:])
        hs=m.swinViT(x,m.normalize)
        enc0=m.encoder1(x)
        enc1=m.encoder2(hs[0]);enc2=m.encoder3(hs[1]);enc3=m.encoder4(hs[2])
        bottleneck=hs[4]
        if self.enabled:
            bottleneck,self.last_evidence=self.vib(bottleneck,sample=sample,noise=noise)
        else:
            self.last_evidence=None
        dec4=m.encoder10(bottleneck)
        dec3=m.decoder5(dec4,hs[3])
        dec2=m.decoder4(dec3,enc3)
        dec1=m.decoder3(dec2,enc2)
        dec0=m.decoder2(dec1,enc1)
        return m.out(m.decoder1(dec0,enc0))
