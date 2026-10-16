"""Version-gated, non-mutating MONAI SwinUNETR/Swin WindowAttention adapter.

Targets MONAI 1.5.1 as tested in CI. No MONAI import at module import time.
The tensor-atomic intervention replaces ONE weighted value message or ONE QK
logit in the *actual* pretrained WindowAttention, via a temporary forward hook.
This is a local computational intervention, not a unique causal path claim.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import re
from itertools import product
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

SUPPORTED_MONAI = ('1.5.1',)


def require_monai():
    try:
        import monai
        from monai.networks.nets import SwinUNETR
        from monai.networks.nets.swin_unetr import WindowAttention, SwinTransformerBlock
    except ImportError as exc:
        raise RuntimeError('MONAI 1.5.1 required: pip install monai==1.5.1') from exc
    if monai.__version__.split('+')[0] not in SUPPORTED_MONAI:
        raise RuntimeError(f'MONAI version {monai.__version__} is not validated; use 1.5.1')
    return SwinUNETR, WindowAttention, SwinTransformerBlock


def build_monai_model(*, in_channels=4, out_channels=3, feature_size=12,
                      depths=(1, 1, 1, 1), num_heads=(3, 3, 6, 12),
                      window_size=2, norm_name='batch'):
    """Tiny synthetic test config; NOT a pretrained clinical model."""
    SwinUNETR, _, _ = require_monai()
    return SwinUNETR(in_channels=in_channels, out_channels=out_channels,
                     feature_size=feature_size, depths=depths, num_heads=num_heads,
                     window_size=window_size, norm_name=norm_name, spatial_dims=3,
                     use_checkpoint=False)


def is_window_attention(module):
    return (hasattr(module, 'relative_position_bias_table')
            and hasattr(module, 'relative_position_index')
            and hasattr(module, 'qkv') and hasattr(module, 'proj')
            and hasattr(module, 'num_heads') and hasattr(module, 'scale'))


def _validate_eval(module):
    if module.training:
        raise ValueError('SCEPT atomic replay requires module.eval() (no dropout)')
    if module.attn_drop.training or module.proj_drop.training:
        raise ValueError('Attention dropout must be disabled')


def decompose_monai_attention(module: nn.Module, windows: torch.Tensor, mask=None):
    """Reproduce MONAI 1.5.1 WindowAttention.forward without parameter changes.

    Original inputs: windows [B*nW, tokens, channels], shift mask [nW,N,N].
    Returns actual Q/K/V, masked scores, normalized weights and output.
    """
    if not is_window_attention(module):
        raise TypeError('Not a compatible MONAI WindowAttention module')
    _validate_eval(module)
    b, n, c = windows.shape
    if c != module.dim or c % module.num_heads:
        raise ValueError('Window channel dimensions are inconsistent')
    qkv = module.qkv(windows).reshape(b, n, 3, module.num_heads, c // module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    logits = (q * module.scale) @ k.transpose(-2, -1)
    rpi = module.relative_position_index[:n, :n]
    bias = module.relative_position_bias_table[rpi.reshape(-1)].reshape(n,n,-1).permute(2,0,1).contiguous()
    logits = logits + bias.unsqueeze(0)
    if mask is not None:
        if mask.ndim != 3 or tuple(mask.shape[-2:]) != (n,n):
            raise ValueError('Shifted-window mask shape mismatch')
        nw = mask.shape[0]
        if b % nw:
            raise ValueError('Window count not divisible by number of masks')
        logits = (logits.reshape(b // nw, nw, module.num_heads, n, n)
                  + mask.unsqueeze(0).unsqueeze(2)).reshape(b, module.num_heads, n, n)
    weights = torch.softmax(logits, dim=-1)
    projected = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b,n,c)
    result = module.proj_drop(module.proj(projected))
    return {'q':q, 'k':k, 'v':v, 'logits':logits, 'weights':weights, 'output':result}


@dataclass(frozen=True)
class AttentionAtom:
    module: str
    window: int
    head: int
    query: int
    key: int
    kind: str = 'value_message'  # or 'qk_logit'


def patched_attention_output(module, clean_windows, reference_windows, mask, atom: AttentionAtom):
    """Change one site only, recompute MONAI output projection exactly.

    value_message: substitutes a single (weight_qk * V_k) vector.
    qk_logit: substitutes one QK logit, then softmax renormalizes whole row.
    Relative bias and shifted masks are included in both conditions.
    """
    if clean_windows.shape != reference_windows.shape:
        raise ValueError('Clean and reference attention window shapes differ')
    clean=decompose_monai_attention(module, clean_windows, mask)
    ref=decompose_monai_attention(module, reference_windows, mask)
    b,h,n=clean['weights'].shape[:3]
    if not (0 <= atom.window < b and 0 <= atom.head < h and 0 <= atom.query < n and 0 <= atom.key < n):
        raise IndexError('Attention atom coordinates outside actual tensor bounds')
    w,hidx,q,k=atom.window,atom.head,atom.query,atom.key
    if mask is not None:
        m=mask[w % mask.shape[0],q,k]
        if float(m.detach()) < -50:
            raise ValueError('Shifted-window forbidden connection; refusing intervention')
    weights = clean['weights'].clone()
    values = clean['v']
    messages = weights.to(values.dtype) @ values
    messages = messages.clone()
    if atom.kind == 'value_message':
        # One message replaced; all other weighted value contributions held clean.
        messages[w,hidx,q] = (messages[w,hidx,q] + weights[w,hidx,q,k].to(values.dtype)
                               * (ref['v'][w,hidx,k] - values[w,hidx,k]))
    elif atom.kind == 'qk_logit':
        # Single score changed; softmax necessarily couples every key in row.
        logits=clean['logits'].clone()
        logits[w,hidx,q,k] = ref['logits'][w,hidx,q,k]
        altered_weights = torch.softmax(logits[w,hidx,q],dim=-1).to(values.dtype)
        messages[w,hidx,q] = altered_weights @ values[w,hidx]
    else:
        raise ValueError('Only value_message or qk_logit atomic interventions supported')
    b,heads,n,d=messages.shape
    combined=messages.transpose(1,2).reshape(b,n,heads*d)
    return module.proj_drop(module.proj(combined))


@dataclass(frozen=True)
class WindowGeometry:
    """Exact MONAI 3D window/shift/padding coordinate reconstruction."""
    input_shape: tuple  # D,H,W before padding
    window_size: tuple
    shift_size: tuple

    @property
    def padded(self):
        return tuple(((d+w-1)//w)*w for d,w in zip(self.input_shape,self.window_size))

    @property
    def window_counts(self):
        return tuple(d//w for d,w in zip(self.padded,self.window_size))

    @property
    def total_windows(self):
        a,b,c=self.window_counts
        return a*b*c

    def position(self,window_index:int,token_index:int):
        """Return unshifted original-grid (d,h,w), or None if padded."""
        wc=self.window_counts;ws=self.window_size
        n=ws[0]*ws[1]*ws[2]
        if not 0<=window_index<self.total_windows or not 0<=token_index<n:
            raise IndexError('window/token index out of bounds')
        i,j,k=(window_index//(wc[1]*wc[2]),(window_index//wc[2])%wc[1],window_index%wc[2])
        p,q,r=(token_index//(ws[1]*ws[2]),(token_index//ws[2])%ws[1],token_index%ws[2])
        loc=tuple((a*w+v+s)%z for a,w,v,s,z in zip((i,j,k),ws,(p,q,r),self.shift_size,self.padded))
        return loc if all(x<d for x,d in zip(loc,self.input_shape)) else None


def window_geometry(block, spatial_shape):
    shape=tuple(int(d) for d in spatial_shape)
    if len(shape)!=3:
        raise ValueError('Only 3D Swin geometry supported')
    windows=tuple(min(int(w),d) for w,d in zip(block.window_size,shape))
    shifts=tuple(0 if d<=int(w) else int(s) for d,w,s in zip(shape,block.window_size,block.shift_size))
    return WindowGeometry(shape,windows,shifts)


class MonaiSwinObserver:
    """Capture real named-module topology and optional window inputs.

    Hooks are removed even on exceptions. Capture stores detached copies only;
    use on a small ROI/model or request a small number of sites to bound VRAM.
    """
    def __init__(self, model):
        self.model=model
        self.sites={name:m for name,m in model.named_modules() if is_window_attention(m)}
        if not self.sites:
            raise ValueError('Model has no compatible MONAI WindowAttention modules')
        self.events=[]
        self.windows={}
        self.geometry={}

    def inventory(self):
        result=[]
        for name,mod in self.model.named_modules():
            if name in self.sites:
                result.append({'name':name,'type':'window_attention','heads':mod.num_heads,'window_size':list(mod.window_size)})
            elif (name.startswith('backbone.') or name.startswith('swinViT.')
                  or re.match(r'^(encoder|decoder)\d+',name)) and name.count('.')<=3:
                result.append({'name':name,'type':type(mod).__name__})
        return result

    @contextmanager
    def recording(self, selected=None, *, capture_tensors=True):
        names=set(self.sites if selected is None else selected)
        if not names.issubset(self.sites):
            raise KeyError(f'Unknown attention sites: {names-set(self.sites)}')
        handles=[]
        self.events=[]
        self.windows={}
        self.geometry={}
        try:
            for name in sorted(names):
                mod=self.sites[name]
                parent_name=name[:-len('.attn')]
                parents=dict(self.model.named_modules())
                if parent_name in parents and hasattr(parents[parent_name],'shift_size'):
                    def block_before(module,args,kwargs,name=name):
                        x=args[0]
                        if x.ndim!=5:
                            raise ValueError('Expected MONAI block tensor [B,D,H,W,C]')
                        self.geometry[name]=window_geometry(module,x.shape[1:4])
                    handles.append(parents[parent_name].register_forward_pre_hook(block_before,with_kwargs=True))
                def before(module,args,kwargs,name=name):
                    x=args[0]
                    mask=kwargs.get('mask',args[1] if len(args)>1 else None)
                    self.events.append({'name':name,'input_shape':list(x.shape),
                                        'mask_shape':None if mask is None else list(mask.shape)})
                    if capture_tensors:
                        self.windows[name]={'input':x.detach().clone(),
                                            'mask':mask.detach().clone() if mask is not None else None}
                handles.append(mod.register_forward_pre_hook(before,with_kwargs=True))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def atom_patch(self, atom: AttentionAtom, reference_windows: torch.Tensor):
        if atom.module not in self.sites:
            raise KeyError(f'Unknown MONAI WindowAttention: {atom.module}')
        mod=self.sites[atom.module]
        calls=[]
        def patch(module,args,kwargs,original):
            if calls:
                raise RuntimeError('Attention module invoked twice; ambiguous patch target')
            calls.append(1)
            mask=kwargs.get('mask',args[1] if len(args)>1 else None)
            return patched_attention_output(module,args[0],reference_windows,mask,atom)
        handle=mod.register_forward_hook(patch,with_kwargs=True)
        try:
            yield calls
        finally:
            handle.remove()


def target_mean_logit(logits:torch.Tensor, class_index:int, roi=None):
    if logits.ndim != 5 or not 0 <= class_index < logits.shape[1]:
        raise ValueError('Expected [B,classes,D,H,W] logits and valid class')
    vals=logits[:,class_index]
    if roi is None:
        return vals.mean()
    if roi.shape != vals.shape[-3:]:
        raise ValueError('ROI shape must match output 3D grid')
    mask=roi.to(device=vals.device,dtype=torch.bool)
    if not mask.any():
        raise ValueError('Empty ROI')
    return vals[:,mask].mean()


def atomic_bidirectional_effect(model, original, reference, atom:AttentionAtom, *, class_index:int=0, roi=None):
    """Disrupt clean with reference message; restore reference with clean message.

    Both normal and patched runs use the REAL complete SwinUNETR decoder and skips.
    This is NOT full multi-layer isolation: other branches are free to respond.
    """
    if model.training:
        raise ValueError('model.eval() required')
    if original.shape != reference.shape:
        raise ValueError('Clean/reference MRI tensor shape mismatch')
    observer=MonaiSwinObserver(model)
    if atom.module not in observer.sites:
        raise KeyError(atom.module)
    with torch.no_grad():
        with observer.recording(selected=[atom.module]):
            raw_clean=model(original)
        clean_input=observer.windows[atom.module]['input']
        clean_mask=observer.windows[atom.module]['mask']
        with observer.recording(selected=[atom.module]):
            raw_ref=model(reference)
        ref_input=observer.windows[atom.module]['input']
        ref_mask=observer.windows[atom.module]['mask']
        if clean_mask is None:
            if ref_mask is not None:
                raise ValueError('Clean/reference attention mask disagreement')
        elif ref_mask is None or not torch.equal(clean_mask,ref_mask):
            raise ValueError('Clean/reference attention masks are different')
        with observer.atom_patch(atom,ref_input) as calls:
            disrupted=model(original)
        if len(calls)!=1:
            raise RuntimeError('Required patch site was never executed')
        with observer.atom_patch(atom,clean_input) as calls:
            restored=model(reference)
        if len(calls)!=1:
            raise RuntimeError('Required restore patch site was never executed')
    s=lambda x:float(target_mean_logit(x,class_index,roi))
    s_clean,s_ref,s_disrupt,s_restore=map(s,(raw_clean,raw_ref,disrupted,restored))
    return {'atom':asdict(atom),'score_clean':s_clean,'score_reference':s_ref,
            'spatial_effect':s_clean-s_ref,'disruption_effect':s_clean-s_disrupt,
            'restoration_effect':s_restore-s_ref,'score_disrupted':s_disrupt,
            'score_restored':s_restore,'evidence_status':'intervened_not_crossvalidated'}


class MonaiDecoderTrace:
    """Read named encoder/decoder/skip-input tensors from REAL MONAI modules."""
    def __init__(self, model):
        self.model=model
        self.activations={}
        self.input_args={}
        self.events=[]

    @contextmanager
    def recording(self,*,capture_tensors=False):
        handles=[]
        self.activations={};self.input_args={};self.events=[]
        try:
            for name,mod in self.model.named_modules():
                local_name=name.removeprefix('backbone.')
                if not (re.fullmatch(r'(encoder\d+|decoder\d+|out)',local_name)
                        or re.fullmatch(r'swinViT\.layers[1-4]\.0',local_name)):
                    continue
                def pre(module,args,kwargs,name=name):
                    self.input_args[name]=tuple(a.detach().clone() for a in args if torch.is_tensor(a)) if capture_tensors else None
                def post(module,args,kwargs,output,name=name):
                    if torch.is_tensor(output):
                        self.events.append({'site':name,'output_shape':list(output.shape)})
                        if capture_tensors:
                            self.activations[name]=output.detach().clone()
                handles.append(mod.register_forward_pre_hook(pre,with_kwargs=True))
                handles.append(mod.register_forward_hook(post,with_kwargs=True))
            yield self
        finally:
            for h in handles:h.remove()


@contextmanager
def substitute_decoder_skip(model, decoder_site:str, reference_skip:torch.Tensor):
    """Replace the second decoder argument (the original MONAI skip tensor).

    This is a skip-connection *node/input* intervention; do not call it a
    path-isolated or a validated causal circuit.
    """
    modules=dict(model.named_modules())
    if not re.fullmatch(r'(backbone\.)?decoder[1-5]',decoder_site) or decoder_site not in modules:
        raise KeyError('Unknown or unsupported decoder site')
    used=[]
    def replace(module,args,kwargs):
        if len(args)!=2 or not torch.is_tensor(args[1]) or reference_skip.shape!=args[1].shape:
            raise ValueError('Decoder skip intervention requires equal-shape second tensor argument')
        if used:
            raise RuntimeError('Decoder site invoked multiple times')
        used.append(True)
        return (args[0],reference_skip),kwargs
    handle=modules[decoder_site].register_forward_pre_hook(replace,with_kwargs=True)
    try:
        yield used
    finally:
        handle.remove()
