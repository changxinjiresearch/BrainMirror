"""Two-scale 3D Transformer-VIB plus precisely declared merge-mediated intervention.

This is a *compact two-scale prototype*, not Swin/shifted-window attention and
not a trained medical segmentation model. Spatial mapping is explicit:
fine-grid (d,h,w) -> coarse-grid (d//2,h//2,w//2). The decoder has a
frozen-reference fine skip route that BYPASSES the coarse Spatial VIB.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .model import TransformerBlock, SpatialVIB


def _coords(grid, device, dtype):
    return torch.stack(torch.meshgrid(*[
        torch.linspace(-1, 1, size, dtype=dtype, device=device) for size in grid],
        indexing='ij'), dim=-1).reshape(-1, 3)


def _flat_to_zyx(index, grid):
    d, h, w = grid
    if not isinstance(index, int) or not 0 <= index < d * h * w:
        raise ValueError('Token index out of spatial grid')
    return (index // (h*w), (index // w) % h, index % w)


class MultiScaleTransformerVIB3D(nn.Module):
    """Full attention within each of TWO 3D scales, with controlled token caps."""
    def __init__(self, dim=16, heads=4, patch=2, fine_depth=1, coarse_depth=1,
                 latent_dim=8, max_tokens=512):
        super().__init__()
        if (dim < 1 or dim % heads != 0 or heads < 1 or patch < 1 or
                latent_dim < 1 or fine_depth < 1 or coarse_depth < 1 or
                max_tokens < 1):
            raise ValueError('Invalid two-scale model configuration')
        self.dim, self.patch, self.max_tokens = dim, patch, max_tokens
        self.embed = nn.Conv3d(4, dim, patch, stride=patch)
        self.fine_pos = nn.Linear(3,dim)
        self.fine_blocks = nn.ModuleList([TransformerBlock(dim,heads) for _ in range(fine_depth)])
        self.fine_norm = nn.LayerNorm(dim)
        self.patch_merge = nn.Conv3d(dim,dim,kernel_size=2,stride=2)
        self.coarse_pos = nn.Linear(3,dim)
        self.coarse_blocks = nn.ModuleList([TransformerBlock(dim,heads) for _ in range(coarse_depth)])
        self.coarse_norm = nn.LayerNorm(dim)
        self.vib = SpatialVIB(dim,latent_dim)
        self.upsample_latent = nn.ConvTranspose3d(latent_dim,latent_dim,2,stride=2)
        self.skip_fusion = nn.Conv3d(dim+latent_dim,dim,1)
        self.decoder = nn.ConvTranspose3d(dim,3,patch,stride=patch)

    def grids(self, x):
        if (x.ndim != 5 or x.shape[1] != 4 or
                any(s % (2*self.patch) for s in x.shape[-3:])):
            raise ValueError('Expected 4 channels and each dimension divisible by 2*patch')
        fine = tuple(s//self.patch for s in x.shape[-3:])
        coarse = tuple(s//2 for s in fine)
        if fine[0]*fine[1]*fine[2] > self.max_tokens:
            raise ValueError('Fine token count exceeds max_tokens')
        return fine,coarse

    def fine_to_coarse(self, token_index, fine_grid):
        z,y,x = _flat_to_zyx(token_index,fine_grid)
        coarse = tuple(dim//2 for dim in fine_grid)
        if any(g%2 for g in fine_grid):
            raise ValueError('Fine grid must be even for exact patch merging')
        coarse_zyx = (z//2,y//2,x//2)
        d,h,w = coarse_zyx
        return d*coarse[1]*coarse[2]+h*coarse[2]+w

    def forward(self,x,sample=False,return_details=False):
        fine,coarse = self.grids(x)
        b=x.shape[0]
        t=self.embed(x).flatten(2).transpose(1,2)
        t=t+self.fine_pos(_coords(fine,x.device,x.dtype)).unsqueeze(0)
        for block in self.fine_blocks:
            t=block(t)
        t=self.fine_norm(t)
        fine_map=t.transpose(1,2).reshape(b,self.dim,*fine)
        # This skip remains an independent route to the decoder.
        skip=fine_map
        merged=self.patch_merge(fine_map)
        ct=merged.flatten(2).transpose(1,2)
        ct=ct+self.coarse_pos(_coords(coarse,x.device,x.dtype)).unsqueeze(0)
        for block in self.coarse_blocks:
            ct=block(ct)
        ct=self.coarse_norm(ct)
        z,kl,mu,lv=self.vib(ct,sample=sample)
        latent_grid=z.transpose(1,2).reshape(b,-1,*coarse)
        up=self.upsample_latent(latent_grid)
        fused=self.skip_fusion(torch.cat([up,skip],dim=1))
        logits=self.decoder(fused)
        if return_details:
            return logits, {'fine_grid':fine,'coarse_grid':coarse,
                    'mu':mu,'logvar':lv,'latent':z,'kl':kl,
                    'fine_skip':skip,'fused':fused}
        return logits


@dataclass(frozen=True)
class CrossScaleSite:
    fine_token: int
    fine_channel: int | None = None


@torch.no_grad()
def intervene_merge_path(model, clean_x, reference_x, site):
    """Source fine-grid token -> designated 2x merge cell -> coarse network ->
    Spatial VIB -> upsample -> decoder. Unselected fine skip is reference.

    Reference fine token is replaced with the clean token at the *input* of
    patch_merge for the synthetic merge calculation. Only the resulting merged
    cell is patched into a FRESH reference forward. The reference fine skip
    is thus frozen. All coarse downstream computation is live, NOT path frozen.
    """
    if not isinstance(model,MultiScaleTransformerVIB3D) or not isinstance(site,CrossScaleSite):
        raise TypeError('Expected two-scale adapter and CrossScaleSite')
    if (clean_x.shape!=reference_x.shape or clean_x.ndim!=5 or
            clean_x.shape[0]!=1 or clean_x.shape[1]!=4 or
            clean_x.dtype!=reference_x.dtype or clean_x.device!=reference_x.device):
        raise ValueError('Inputs must be matched [1,4,D,H,W]')
    model.eval()
    fine,coarse=model.grids(reference_x)
    loc=_flat_to_zyx(site.fine_token,fine)
    if site.fine_channel is not None and not 0<=site.fine_channel<model.dim:
        raise ValueError('Fine feature channel out of range')
    record={}
    handles=[]
    def fine_capture(tag):
        def f(_module,args):record[tag]=args[0].detach().clone()
        return f
    def merge_capture(_module,args,out):
        record['merged_base']=out.detach().clone()
    def coarse_out(_module,args,out):
        record['coarse_out']=out.detach().clone()
    try:
        handles.append(model.patch_merge.register_forward_pre_hook(fine_capture('ref_fine')))
        handles.append(model.patch_merge.register_forward_hook(merge_capture))
        base_logits, base_info=model(reference_x,return_details=True)
        # Record full clean fine features but nothing after the merge is used.
        for h in handles:h.remove()
        handles=[]
        handles.append(model.patch_merge.register_forward_pre_hook(fine_capture('clean_fine')))
        clean_logits,_=model(clean_x,return_details=True)
        for h in handles:h.remove()
        handles=[]
        ref_fine=record['ref_fine'];clean_fine=record['clean_fine']
        altered=ref_fine.clone()
        if site.fine_channel is None:
            altered[:,:,loc[0],loc[1],loc[2]]=clean_fine[:,:,loc[0],loc[1],loc[2]]
        else:
            c=site.fine_channel
            altered[:,c,loc[0],loc[1],loc[2]]=clean_fine[:,c,loc[0],loc[1],loc[2]]
        new_merge=model.patch_merge(altered)
        ref_merge=model.patch_merge(ref_fine)
        diff=new_merge-ref_merge
        target=(loc[0]//2,loc[1]//2,loc[2]//2)
        allowed=torch.zeros_like(diff)
        allowed[:,:,target[0],target[1],target[2]]=diff[:,:,target[0],target[1],target[2]]
        if not torch.allclose(diff,allowed,atol=1e-7,rtol=1e-6):
            raise AssertionError('Patch merge unexpectedly changed distant coarse tokens')
        merged_patch=ref_merge+allowed
        def patch_merged(_module,_input,_output):
            return merged_patch
        handles.append(model.patch_merge.register_forward_hook(patch_merged))
        intervened, patched_info=model(reference_x,return_details=True)
    finally:
        for h in handles:h.remove()
    if not torch.allclose(patched_info['fine_skip'],base_info['fine_skip'],atol=0,rtol=0):
        raise AssertionError('Offpath fine skip changed')
    cidx=model.fine_to_coarse(site.fine_token,fine)
    return {'reference_logits':base_logits,'clean_logits':clean_logits,
            'intervened_logits':intervened,'merge_delta':allowed,
            'latent_mu_delta':patched_info['mu']-base_info['mu'],
            'coarse_token':cidx,'coarse_zyx':target,'fine_zyx':loc,
            'fine_skip_unchanged':True,
            'semantics':'single_fine_token_patch_merge_to_coarse_full_downstream_frozen_fine_skip',
            'independent_patient_validation':False}


@torch.no_grad()
def intervene_cross_scale_atomic(model, clean_x, reference_x, fine_site, attention_site):
    """Fine token -> 2x merge -> ONE coarse head/Q/K weighted message -> VIB -> decoder.

    This is supported only with one coarse block so that downstream offpath
    *coarse* residual layers cannot silently introduce additional changes.
    Direct merge->coarse residual bypass and all fine skips are held reference.
    QK `logit` mode allows Softmax row recoupling; weighted_value mode isolates
    only one post-softmax message. No claim about natural causal uniqueness.
    """
    from .attention_atomic import AtomicAttentionSite, attention_parts, _delta_one_head_query
    if (not isinstance(model,MultiScaleTransformerVIB3D) or
            not isinstance(fine_site,CrossScaleSite) or
            not isinstance(attention_site,AtomicAttentionSite) or
            attention_site.layer!=0 or len(model.coarse_blocks)!=1):
        raise ValueError('v0.4 supports one coarse block and layer=0 only')
    # Validate all fine-token, channel and paired-input conditions here via the
    # same source->merge intervention oracle that handles offpath skip checks.
    merge_route=intervene_merge_path(model,clean_x,reference_x,fine_site)
    _,coarse=model.grids(reference_x)
    n=coarse[0]*coarse[1]*coarse[2]
    if (not 0<=attention_site.head<model.coarse_blocks[0].attn.num_heads or
            not 0<=attention_site.query<n or not 0<=attention_site.key<n or
            attention_site.mode not in ('weighted_value','logit')):
        raise ValueError('Invalid coarse attention site')
    ref_record={}
    changed_record={}
    handles=[]
    block=model.coarse_blocks[0]
    def capture_residual(record):
        def f(_module,args):record['residual']=args[0].detach().clone()
        return f
    def capture_gate(record):
        def f(_module,args,out):record['branch']=out.detach().clone()
        return f
    try:
        handles.append(block.norm1.register_forward_pre_hook(capture_residual(ref_record)))
        handles.append(block.attn_to_residual.register_forward_hook(capture_gate(ref_record)))
        base_logits,ref_details=model(reference_x,return_details=True)
        for h in handles:h.remove()
        handles=[]
        # Replace the actual 2x patch_merge output at only its specified cell.
        def merge_delta(_module,_inputs,output):return output+merge_route['merge_delta']
        handles.append(model.patch_merge.register_forward_hook(merge_delta))
        handles.append(block.norm1.register_forward_pre_hook(capture_residual(changed_record)))
        merged_logits,_=model(reference_x,return_details=True)
        for h in handles:h.remove()
        handles=[]
        before=attention_parts(block,ref_record['residual'])
        after=attention_parts(block,changed_record['residual'])
        if not torch.allclose(before['output'],ref_record['branch'],atol=2e-6,rtol=2e-5):
            raise AssertionError('Coarse attention decomposition not faithful')
        atom_delta=_delta_one_head_query(after,before,attention_site,block.attn.out_proj.weight)
        def atom_patch(_module,_inputs,output):
            value=output.clone()
            value[:,attention_site.query,:]+=atom_delta
            return value
        handles.append(block.attn_to_residual.register_forward_hook(atom_patch))
        isolated_logits,isolated_details=model(reference_x,return_details=True)
    finally:
        for h in handles:h.remove()
    return {'reference_logits':base_logits,'merge_route_logits':merged_logits,
            'isolated_logits':isolated_logits,'atomic_branch_delta':atom_delta,
            'latent_mu_delta':isolated_details['mu']-ref_details['mu'],
            'fine_token':fine_site.fine_token,'merge_coarse_token':merge_route['coarse_token'],
            'coarse_head':attention_site.head,'coarse_query':attention_site.query,
            'coarse_key':attention_site.key,'mode':attention_site.mode,
            'semantics':'fine_to_merge_to_one_coarse_head_message_to_vib_decoder_reference_skips',
            'independent_patient_validation':False}
