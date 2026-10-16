"""Inspectable windowed/shifted-window 3D Transformer-VIB research adapter.

Executable Swin *style* implementation (not MONAI SwinUNETR), strictly
window-divisible grid at each scale. A trace stores actual per-window Q,K,V,
post-mask attention weights, global token IDs and residual gates. No fabricated
point-cloud graph or claim of clinical validation.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import SpatialVIB


def grid_index(z, y, x, grid):
    d,h,w=grid
    if not (0<=z<d and 0<=y<h and 0<=x<w):
        raise ValueError('out-of-grid token coordinates')
    return z*h*w+y*w+x


def unflatten(index, grid):
    d,h,w=grid
    if not isinstance(index,int) or index<0 or index>=d*h*w:
        raise ValueError('token out of bounds')
    return index//(h*w), (index//w)%h, index%w


def _windows(t, window):
    # [B,D,H,W,C] -> [B, n_windows, W^3, C]
    b,d,h,w,c=t.shape
    if any(s%window for s in (d,h,w)):
        raise ValueError('3D grid must be divisible by window size')
    return t.reshape(b,d//window,window,h//window,window,w//window,window,c)\
            .permute(0,1,3,5,2,4,6,7).reshape(b,-1,window**3,c)


def _restore_windows(t, grid, window):
    b,_,_,c=t.shape
    d,h,w=grid
    return t.reshape(b,d//window,h//window,w//window,window,window,window,c)\
            .permute(0,1,4,2,5,3,6,7).reshape(b,d,h,w,c)


def _region_labels(grid, window, shift, device):
    """Match shifted-window boundary mask semantics (not original window IDs).

    Labels split coordinates into 3 boundary segments per dimension. The
    *rolled feature* is grouped into consecutive shifted windows. Thus
    artificial last/first cyclic neighbours cannot exchange messages.
    """
    if shift==0:
        return torch.zeros(grid,device=device,dtype=torch.long)
    per=[]
    for size in grid:
        t=torch.zeros(size,device=device,dtype=torch.long)
        t[size-window:size-shift]=1
        t[size-shift:]=2
        per.append(t)
    return per[0][:,None,None]*9+per[1][None,:,None]*3+per[2][None,None,:]


class WindowAttention3D(nn.Module):
    def __init__(self, dim=16, heads=2, window=2, shift=0):
        super().__init__()
        if dim<1 or heads<1 or dim%heads or window<1 or not 0<=shift<window:
            raise ValueError('invalid window-attention settings')
        self.dim,self.heads,self.window,self.shift=dim,heads,window,shift
        self.qkv=nn.Linear(dim,3*dim)
        self.proj=nn.Linear(dim,dim)

    def geometry(self, grid, device):
        w=self.window
        if any(s%w for s in grid):
            raise ValueError('all scale dimensions must be window divisible')
        # Swin disables shift when an entire axis is a single window.
        shift=self.shift if all(s>w for s in grid) else 0
        ids=torch.arange(math.prod(grid),device=device).reshape(*grid)
        if shift:
            ids=torch.roll(ids,(-shift,)*3,(0,1,2))
        loc=_windows(ids[None,...,None],w)[0,:,:,0].long()
        labels=_region_labels(grid,w,shift,device)
        mask_labels=_windows(labels[None,...,None],w)[0,:,:,0]
        allowed=mask_labels[:,:,None].eq(mask_labels[:,None,:])
        # shifted windows contain real neighbours across ordinary window boundaries.
        return loc,allowed,shift

    def forward(self, x, grid, record=False):
        if x.ndim!=3 or x.shape[-1]!=self.dim or x.shape[1]!=math.prod(grid):
            raise ValueError('expected [B,prod(grid),dim]')
        b,n,c=x.shape
        loc,allowed,shift=self.geometry(grid,x.device)
        t=x.reshape(b,*grid,c)
        if shift:
            t=torch.roll(t,(-shift,)*3,(1,2,3))
        t=_windows(t,self.window)
        nw,p=self._numbers(grid)
        dh=c//self.heads
        qkv=self.qkv(t).reshape(b,nw,p,3,self.heads,dh).permute(3,0,1,4,2,5)
        q,k,v=qkv.unbind(0)
        raw_logits=(q@k.transpose(-1,-2))/math.sqrt(dh)
        logits=raw_logits.masked_fill(~allowed[None,:,None,:,:],torch.finfo(raw_logits.dtype).min)
        weights=logits.softmax(-1)
        head_output=weights@v
        concat=head_output.permute(0,1,3,2,4).reshape(b,nw,p,c)
        proj=self.proj(concat)
        restored=_restore_windows(proj,grid,self.window)
        if shift:
            restored=torch.roll(restored,(shift,)*3,(1,2,3))
        output=restored.reshape(b,n,c)
        if not record:
            return output
        return output, {'q':q,'k':k,'v':v,'raw_logits':raw_logits,
                        'logits':logits,'weights':weights,'head_output':head_output,
                        'window_output':proj,'output':output,'token_ids':loc,
                        'allowed':allowed,'effective_shift':shift,'grid':grid,
                        'head_width':dh}

    def _numbers(self,grid):
        return math.prod(s//self.window for s in grid),self.window**3


class SwinBlock3D(nn.Module):
    def __init__(self,dim=16,heads=2,window=2,shift=0):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=WindowAttention3D(dim,heads,window,shift)
        self.attn_to_residual=nn.Identity()
        self.norm2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.mlp_to_residual=nn.Identity()

    def forward(self,x,grid,record=False):
        if record:
            out,attention=self.attn(self.norm1(x),grid,record=True)
        else:
            out=self.attn(self.norm1(x),grid)
        gate=self.attn_to_residual(out)
        after_attn=x+gate
        mgate=self.mlp_to_residual(self.mlp(self.norm2(after_attn)))
        y=after_attn+mgate
        if record:
            return y,{'input':x,'attention':attention,'gate':gate,
                      'after_attn':after_attn,'mlp_gate':mgate,'output':y}
        return y


class MultiLayerSwinVIB3D(nn.Module):
    """Two-stage, >=2 blocks/stage shifted-window research model.

    Does not load MONAI SwinUNETR weights; not trained on medical scans.
    Fine skip bypasses Spatial VIB and MUST be frozen during route-only tests.
    """
    def __init__(self,dim=12,heads=3,patch=2,window=2,
                 fine_depth=2,coarse_depth=2,latent_dim=8,max_tokens=512):
        super().__init__()
        if (dim<1 or dim%heads or patch<1 or window<1 or
                fine_depth<2 or coarse_depth<2 or latent_dim<1 or max_tokens<1):
            raise ValueError('invalid multiblock swin configuration')
        self.dim,self.patch,self.window,self.max_tokens=dim,patch,window,max_tokens
        self.embed=nn.Conv3d(4,dim,patch,stride=patch)
        self.fine_pos=nn.Linear(3,dim)
        self.fine_blocks=nn.ModuleList([SwinBlock3D(dim,heads,window,0 if i%2==0 else window//2)
                                        for i in range(fine_depth)])
        self.fine_norm=nn.LayerNorm(dim)
        self.patch_merge=nn.Conv3d(dim,dim,2,stride=2)
        self.coarse_pos=nn.Linear(3,dim)
        self.coarse_blocks=nn.ModuleList([SwinBlock3D(dim,heads,window,0 if i%2==0 else window//2)
                                          for i in range(coarse_depth)])
        self.coarse_norm=nn.LayerNorm(dim)
        self.vib=SpatialVIB(dim,latent_dim)
        self.up=nn.ConvTranspose3d(latent_dim,latent_dim,2,stride=2)
        self.fuse=nn.Conv3d(latent_dim+dim,dim,1)
        self.decoder=nn.ConvTranspose3d(dim,3,patch,stride=patch)

    def grids(self,x):
        if x.ndim!=5 or x.shape[1]!=4 or not torch.is_floating_point(x):
            raise ValueError('MRI tensor must be [B,4,D,H,W] floating')
        if any(s%(2*self.patch*self.window) for s in x.shape[-3:]):
            raise ValueError('volume dimensions must be divisible by 2*patch*window')
        fine=tuple(s//self.patch for s in x.shape[-3:]);coarse=tuple(s//2 for s in fine)
        if math.prod(fine)>self.max_tokens:
            raise ValueError('fine token count over configured maximum')
        return fine,coarse

    @staticmethod
    def _positions(grid,device,dtype):
        coords=[torch.linspace(-1,1,s,device=device,dtype=dtype) for s in grid]
        return torch.stack(torch.meshgrid(*coords,indexing='ij'),-1).reshape(-1,3)

    def decode(self,coarse_norm_tokens,fine_skip,coarse_grid,sample=False,record=False):
        b=coarse_norm_tokens.shape[0]
        z,kl,mu,lv=self.vib(coarse_norm_tokens,sample=sample)
        up=self.up(z.transpose(1,2).reshape(b,-1,*coarse_grid))
        fused=self.fuse(torch.cat([up,fine_skip],dim=1))
        output=self.decoder(fused)
        if record:
            return output,{'z':z,'mu':mu,'logvar':lv,'kl':kl,'fused':fused}
        return output

    def forward(self,x,sample=False,record=False):
        fine,coarse=self.grids(x);b=x.shape[0]
        t=self.embed(x).flatten(2).transpose(1,2)
        t=t+self.fine_pos(self._positions(fine,x.device,x.dtype))[None]
        fine_initial=t
        f=[]
        for block in self.fine_blocks:
            if record:
                t,rec=block(t,fine,record=True);f.append(rec)
            else:t=block(t,fine)
        fine_pre_norm=t
        skip=self.fine_norm(t).transpose(1,2).reshape(b,self.dim,*fine)
        merged=self.patch_merge(skip)
        ct=merged.flatten(2).transpose(1,2)
        ct=ct+self.coarse_pos(self._positions(coarse,x.device,x.dtype))[None]
        coarse_initial=ct
        c=[]
        for block in self.coarse_blocks:
            if record:
                ct,rec=block(ct,coarse,record=True);c.append(rec)
            else:ct=block(ct,coarse)
        coarse_pre_norm=ct
        ct=self.coarse_norm(ct)
        if record:
            logits,dec=self.decode(ct,skip,coarse,sample=sample,record=True)
            return logits,{'fine_grid':fine,'coarse_grid':coarse,
                'fine_initial':fine_initial,'fine_steps':f,'fine_pre_norm':fine_pre_norm,
                'fine_skip':skip,'merged':merged,'coarse_initial':coarse_initial,
                'coarse_steps':c,'coarse_pre_norm':coarse_pre_norm,
                'coarse_normed':ct,'decoder':dec}
        return self.decode(ct,skip,coarse,sample=sample)

@torch.no_grad()
def execution_topology(model, x, *, max_edges=250000):
    """Export ACTUAL executed topology (window connectivity + 2x patch merging).

    Attention-edge eligibility is NOT an observed causal effect. Token indices
    are taken from the same geometry() function the forward executes.
    """
    if not isinstance(model,MultiLayerSwinVIB3D):
        raise TypeError('expected MultiLayerSwinVIB3D')
    fine,coarse=model.grids(x)
    stages={}
    edge_count=0
    for stage,grid,blocks in [('fine',fine,model.fine_blocks),
                              ('coarse',coarse,model.coarse_blocks)]:
        rows=[]
        for i,block in enumerate(blocks):
            ids,allowed,shift=block.attn.geometry(grid,x.device)
            edges=[]
            for window_idx in range(ids.shape[0]):
                for query_local in range(ids.shape[1]):
                    for key_local in range(ids.shape[1]):
                        if bool(allowed[window_idx,query_local,key_local]):
                            edges.append([int(ids[window_idx,query_local]),
                                          int(ids[window_idx,key_local]),int(window_idx)])
            edge_count+=len(edges)*block.attn.heads
            if edge_count>max_edges:
                raise ValueError('topology exceeds explicit edge budget')
            rows.append({'layer':i,'window':block.attn.window,
                         'effective_shift':shift,'heads':block.attn.heads,
                         'window_token_ids':ids.cpu().tolist(),
                         'per_head_query_key_edges':edges,
                         'tensor_ops':['LayerNorm','QKVLinear','MaskedSoftmax',
                                       'AVMatmul','OutputProjection','ResidualAdd',
                                       'LayerNorm','MLP','ResidualAdd']})
        stages[stage]={'grid':list(grid),'blocks':rows}
    merges=[]
    for z in range(fine[0]):
        for y in range(fine[1]):
            for xx in range(fine[2]):
                source=grid_index(z,y,xx,fine)
                target=grid_index(z//2,y//2,xx//2,coarse)
                merges.append([source,target])
    return {'model_type':'executed_3D_Swin_style_two_scale_NOT_MONAI_SwinUNETR',
            'stages':stages,'patch_merge_receptive_links':merges,
            'skip_route':'fine_norm->decoder_fusion_bypasses_coarse_spatial_VIB',
            'downstream':['coarse_norm','spatial_VIB_mu_or_sample',
                          'latent_transposed_convolution','fine_skip_fusion',
                          'decoder_transposed_convolution'],
            'attention_connectivity_not_causality':True,
            'expanded_head_attention_edge_count':edge_count}
