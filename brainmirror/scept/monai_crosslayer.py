"""SCEPT v0.8 — bounded automatic *controlled-cascade* search on MONAI 1.5.1.

A controlled cascade is a precisely specified INTERVENTION, not a unique causal
circuit.  A source atomic message is introduced on the reference MRI run.  At
subsequent selected blocks, ONLY a chosen attention message is re-computed from
its hybrid input and all other attention messages and MLP outputs are held at
reference values.  The incoming residual identity delta is *removed* at each
selected block output: this rules out bypass of the designated intermediate atom.
Between selected blocks only the residual identity signal and genuine MONAI
PatchMerging operations carry the delta.  Decoder skip inputs are held at their
reference values.  No model parameter or public MONAI implementation is changed.

Version-gated; eval only; batch size 1; no autograd/checkpointing; no patient
claims. Clinical validity and generic graph-wide causal completeness NOT tested.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import math
import re
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .monai_adapter import (
    AttentionAtom, MonaiSwinObserver, decompose_monai_attention,
    patched_attention_output, target_mean_logit,
)

_BLOCK_PATTERN = re.compile(r'^(?:backbone\.)?swinViT\.layers([1-4])\.0\.blocks\.(\d+)\.attn$')
_SKIP_PATTERN = re.compile(r'^(?:backbone\.)?decoder[1-5]$')


def _block_for_atom(atom: AttentionAtom) -> str:
    if _BLOCK_PATTERN.fullmatch(atom.module) is None:
        raise ValueError('Not a supported MONAI SwinUNETR attention module path')
    return atom.module[:-5]


def _ordered_sites(observer: MonaiSwinObserver):
    result=[]
    for name in observer.sites:
        match=_BLOCK_PATTERN.fullmatch(name)
        if match is not None:
            result.append(((int(match.group(1)), int(match.group(2))),name))
    result.sort()
    return result


def _check_inputs(model, clean, reference):
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('MONAI path search requires complete model.eval()')
    if clean.shape!=reference.shape or clean.ndim!=5 or clean.shape[0]!=1:
        raise ValueError('Only matched [1,C,D,H,W] volumes supported')
    if clean.device!=reference.device or clean.dtype!=reference.dtype:
        raise ValueError('Input device/dtype mismatch')
    if not torch.isfinite(clean).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite MRI input')
    if torch.is_grad_enabled():
        raise ValueError('Use torch.no_grad() for deterministic SCEPT tracing')


def _validate_path(path, observer):
    if not path:
        raise ValueError('Path must contain at least one atom')
    keys=[]
    for atom in path:
        if not isinstance(atom,AttentionAtom) or atom.module not in observer.sites:
            raise ValueError('Unknown or invalid path atom')
        match=_BLOCK_PATTERN.fullmatch(atom.module)
        if match is None: raise ValueError('Only official SwinUNETR block sites supported')
        keys.append((int(match.group(1)),int(match.group(2))))
    if keys!=sorted(set(keys)):
        raise ValueError('A path must follow distinct MONAI forward blocks')


def _register(module, fn, *, pre=False):
    if pre:
        return module.register_forward_pre_hook(fn,with_kwargs=True)
    return module.register_forward_hook(fn,with_kwargs=True)


@dataclass
class MonaiReference:
    original_score: float
    reference_score: float
    original_logits: torch.Tensor
    reference_logits: torch.Tensor
    ref_windows: dict
    clean_windows: dict
    masks: dict
    geometries: dict
    block_inputs: dict
    block_outputs: dict
    mlp_outputs: dict
    decoder_skips: dict
    model_id: int


def _clone(x):
    return x.detach().clone()


def capture_monai_reference(model, clean, reference, *, class_index=0, roi=None):
    """Capture one clean and one reference forward. No parameter modification."""
    _check_inputs(model,clean,reference)
    observer=MonaiSwinObserver(model)
    ordered=_ordered_sites(observer)
    if len(ordered)<2:
        raise ValueError('Need at least two MONAI attention blocks')
    modules=dict(model.named_modules())
    inputs,outputs,mlps,skips={},{},{},{}
    with ExitStack() as stack:
        for _,site in ordered:
            name=site[:-5]
            block=modules[name]
            def capture_in(module,args,kwargs,name=name):
                inputs[name]=_clone(args[0])
            def capture_out(module,args,kwargs,out,name=name):
                outputs[name]=_clone(out)
            def capture_mlp(module,args,kwargs,out,name=name):
                mlps[name]=_clone(out)
            stack.callback(_register(block,capture_in,pre=True).remove)
            stack.callback(_register(block,capture_out).remove)
            stack.callback(_register(block.mlp,capture_mlp).remove)
        for name,module in modules.items():
            if _SKIP_PATTERN.fullmatch(name):
                def skip_in(module,args,kwargs,name=name):
                    if len(args)!=2 or not torch.is_tensor(args[1]):
                        raise RuntimeError('Unexpected MONAI decoder API')
                    skips[name]=_clone(args[1])
                stack.callback(_register(module,skip_in,pre=True).remove)
        with observer.recording():
            ref_logits=model(reference)
    ref_windows={n:_clone(r['input']) for n,r in observer.windows.items()}
    masks={n:None if r['mask'] is None else _clone(r['mask']) for n,r in observer.windows.items()}
    geometries=dict(observer.geometry)
    with observer.recording():
        clean_logits=model(clean)
    clean_windows={n:_clone(r['input']) for n,r in observer.windows.items()}
    if set(ref_windows)!=set(clean_windows):
        raise ValueError('Input-dependent MONAI module topology is unsupported')
    for name,r in observer.windows.items():
        m=masks[name];other=r['mask']
        if (m is None)!=(other is None) or (m is not None and not torch.equal(m,other)):
            raise ValueError('Shifted-window masks changed across references')
    if len(outputs)!=len(ordered) or len(inputs)!=len(ordered) or len(mlps)!=len(ordered):
        raise RuntimeError('MONAI block/MLP capture is incomplete')
    if not skips:
        raise RuntimeError('No MONAI decoder skips captured')
    return MonaiReference(
        float(target_mean_logit(clean_logits,class_index,roi)),
        float(target_mean_logit(ref_logits,class_index,roi)),
        _clone(clean_logits),_clone(ref_logits),ref_windows,clean_windows,
        masks,geometries,inputs,outputs,mlps,skips,id(model),
    )


def _ensure_snapshot(model,snapshot):
    if snapshot.model_id!=id(model):
        raise ValueError('Snapshot belongs to a different model')
    if model.training: raise ValueError('model.eval() required')


def run_isolated_monai_chain(model, reference, snapshot:MonaiReference, path,
                             *, class_index=0, roi=None, cut_at:Optional[int]=None,
                             capture_windows=False):
    """Execute a serial atomic intervention through official MONAI modules.

    Cut_at: zero-base selected hop to use reference atom instead of hybrid
    atom; this must erase the upstream residual-only bypass at that hop.
    Returns logits, score, per-hop realized attention output delta and optional
    pre-attention hybrid windows for next-hop discovery.
    """
    _ensure_snapshot(model,snapshot)
    observer=MonaiSwinObserver(model)
    _validate_path(path,observer)
    if cut_at is not None and not 0<=cut_at<len(path):
        raise ValueError('cut_at is outside selected path')
    selected={_block_for_atom(atom):i for i,atom in enumerate(path)}
    modules=dict(model.named_modules())
    current_inputs={}
    per_hop={}
    hybrid_windows={}
    used=set()
    with ExitStack() as stack:
        for _,site in _ordered_sites(observer):
            bname=site[:-5]
            block=modules[bname]
            def block_pre(module,args,kwargs,name=bname):
                current_inputs[name]=args[0]
            stack.callback(_register(block,block_pre,pre=True).remove)

            # Freeze entire original MLP branch. Unselected blocks carry only
            # their incoming identity delta; selected blocks cancel it.
            def mlp_hook(module,args,kwargs,out,name=bname):
                base=snapshot.mlp_outputs[name]
                if base.shape!=out.shape or base.dtype!=out.dtype:
                    raise ValueError('MLP state disagreement')
                return base
            stack.callback(_register(block.mlp,mlp_hook).remove)

            def block_hook(module,args,kwargs,out,name=bname):
                prev=current_inputs[name]
                base_in=snapshot.block_inputs[name]
                base_out=snapshot.block_outputs[name]
                if prev.shape!=base_in.shape or out.shape!=base_out.shape:
                    raise ValueError('Block reference topology disagreement')
                delta=prev-base_in
                if name in selected:
                    # Remove identity carry-through: downstream changes now
                    # depend on that selected attention atom, not a bypass.
                    return out-delta
                # Off-path attention & MLP changes cannot propagate.
                return base_out+delta
            stack.callback(_register(block,block_hook).remove)

            if bname in selected:
                idx=selected[bname]
                atom=path[idx]
                def patch_attn(module,args,kwargs,out,atom=atom,idx=idx,name=site):
                    if idx in used:
                        raise RuntimeError('MONAI attention invoked twice')
                    used.add(idx)
                    x=args[0]
                    mask=kwargs.get('mask', args[1] if len(args)>1 else None)
                    ref=snapshot.ref_windows[name]
                    if ref.shape!=x.shape or (mask is None)!=(snapshot.masks[name] is None):
                        raise ValueError('Window topology disagreement')
                    if mask is not None and not torch.equal(mask,snapshot.masks[name]):
                        raise ValueError('Window mask changed')
                    # Source seed from *clean* condition; downstream from
                    # actual hybrid input, never from the full clean condition.
                    candidate=(snapshot.clean_windows[name] if idx==0 else x)
                    if cut_at is not None and idx==cut_at:
                        candidate=ref
                    result=patched_attention_output(module,ref,candidate,mask,atom)
                    # Patch at output of MONAI WindowAttention, not replacement
                    # of whole block or semantic token guesses.
                    delta=result-decompose_monai_attention(module,ref,mask)['output']
                    per_hop[idx]={'module':name,'atom':asdict(atom),
                                  'attention_delta_l2':float(torch.linalg.vector_norm(delta)),
                                  'incoming_window_delta_l2':float(torch.linalg.vector_norm(x-ref))}
                    return result
                stack.callback(_register(modules[site],patch_attn).remove)

            if capture_windows:
                def window_pre(module,args,kwargs,name=site):
                    hybrid_windows[name]=_clone(args[0])
                stack.callback(_register(modules[site],window_pre,pre=True).remove)

        for name,module in modules.items():
            if _SKIP_PATTERN.fullmatch(name):
                if name not in snapshot.decoder_skips:
                    raise ValueError('Missing decoder reference skip')
                def freeze_skip(module,args,kwargs,name=name):
                    ref_skip=snapshot.decoder_skips[name]
                    if len(args)!=2 or not torch.is_tensor(args[1]) or args[1].shape!=ref_skip.shape:
                        raise ValueError('Decoder skip topology mismatch')
                    return (args[0],ref_skip),kwargs
                stack.callback(_register(module,freeze_skip,pre=True).remove)
        with torch.no_grad():
            logits=model(reference)
    if used!=set(range(len(path))):
        raise RuntimeError(f'Path steps not all executed: {sorted(used)}')
    value=float(target_mean_logit(logits,class_index,roi))
    return {'score':value,'logits':logits.detach().clone(),
            'effect_from_reference':value-snapshot.reference_score,
            'hops':[per_hop[i] for i in range(len(path))],
            'hybrid_windows':hybrid_windows}


def _rank_changed_atoms(observer,snapshot,source_windows,site,roi,*,kinds,limit,
                        max_queries=96,max_keys_per_head=12):
    """Bounded changed-message screening, NOT exhaustive causal discovery.

    Deterministic ROI scan (max_queries per site). Only the strongest
    max_keys_per_head reference-vs-hybrid deltas per head are enumerated.
    These hard bounds keep real 3D Swin windows tractable.
    """
    geom=snapshot.geometries[site]
    ref=snapshot.ref_windows[site]
    target=source_windows
    if target.shape!=ref.shape: raise ValueError('Screening windows shape mismatch')
    if ref.shape[0]!=geom.total_windows:
        raise ValueError('Batch/window layout unsupported')
    parts=decompose_monai_attention(observer.sites[site],ref,snapshot.masks[site])
    candidate=decompose_monai_attention(observer.sites[site],target,snapshot.masks[site])
    qweight=parts['weights'].detach();values=parts['v'].detach()
    delta_v=(candidate['v']-values).detach()
    logit_delta=(candidate['logits']-parts['logits']).detach()
    pooled=F.adaptive_max_pool3d(roi.to(device=ref.device,dtype=ref.dtype)[None,None],geom.input_shape)[0,0]>0
    b,h,n,d=values.shape
    scores=[]
    queries_scanned=0
    for w in range(b):
        wi=w%geom.total_windows
        for q in range(n):
            qpos=geom.position(wi,q)
            if qpos is None or not bool(pooled[qpos]): continue
            if queries_scanned>=max_queries:break
            queries_scanned+=1
            keys=[key for key in range(n)
                  if geom.position(wi,key) is not None
                  and (snapshot.masks[site] is None or float(snapshot.masks[site][wi,q,key])>=-50)]
            if not keys:continue
            key_idx=torch.tensor(keys,device=ref.device,dtype=torch.long)
            for head in range(h):
                for kind in kinds:
                    if kind=='value_message':
                        candidates=qweight[w,head,q,key_idx].abs()*torch.linalg.vector_norm(delta_v[w,head,key_idx],dim=-1)
                    elif kind=='qk_logit':
                        candidates=logit_delta[w,head,q,key_idx].abs()
                    else:raise ValueError('Invalid atomic mode')
                    best_count=min(len(keys),max_keys_per_head)
                    values_top,positions=torch.topk(candidates,best_count,largest=True,sorted=True)
                    for value,pos in zip(values_top,positions):
                        score=float(value)
                        if score>0 and math.isfinite(score):
                            key=keys[int(pos)]
                            scores.append((score,AttentionAtom(site,w,head,q,key,kind)))
        if queries_scanned>=max_queries:break
    scores.sort(key=lambda t:(-t[0],t[1].module,t[1].window,t[1].head,t[1].query,t[1].key,t[1].kind))
    return scores[:limit],len(scores)


def discover_monai_crosslayer(model, clean, reference, roi, *,
                              class_index=0, max_hops=3, max_sites=5,
                              per_step=2, beam_width=2, max_exact=8,
                              kinds=('value_message','qk_logit'),
                              heldout_reference=None, effect_tol=1e-7):
    """Automatic ROI->real-window atoms->multi-block path candidate search.

    One primary reference is used for screening and selection. Optional second
    reference is evaluated ONLY after primary selection. All candidates and
    tested output effects are recorded; abstention if no effect/contradiction.
    """
    if roi.ndim!=3 or tuple(roi.shape)!=tuple(clean.shape[-3:]) or not bool(roi.any()):
        raise ValueError('ROI must be a nonempty mask on the input 3D grid')
    if not (2<=max_hops<=5 and max_sites>=2 and per_step>=1 and beam_width>=1 and max_exact>=1):
        raise ValueError('Invalid bounded search configuration')
    if set(kinds)-{'value_message','qk_logit'} or not kinds:
        raise ValueError('Unknown intervention kind')
    if not math.isfinite(effect_tol) or effect_tol<=0:raise ValueError('Invalid evidence tolerance')
    with torch.no_grad():
        snap=capture_monai_reference(model,clean,reference,class_index=class_index,roi=roi)
        obs=MonaiSwinObserver(model)
        ordered=[s for _,s in _ordered_sites(obs)][:max_sites]
        if len(ordered)<2: raise RuntimeError('Cannot discover cross-layer path')
        sites=ordered[:max_hops]
        ranked,screen_count=_rank_changed_atoms(obs,snap,snap.clean_windows[sites[0]],sites[0],roi,
                                                  kinds=kinds,limit=per_step)
        budget=0
        all_checks=[]
        beams=[]
        for score,atom in ranked:
            if budget>=max_exact:break
            candidate=[atom]
            result=run_isolated_monai_chain(model,reference,snap,candidate,
                                            class_index=class_index,roi=roi,capture_windows=True)
            budget+=1
            all_checks.append({'path':[asdict(a) for a in candidate],
                               'heuristic':score,'effect':result['effect_from_reference'],
                               'hops':result['hops'],'kind':'source_probe'})
            beams.append((abs(result['effect_from_reference']),candidate,result))
        if not beams:
            return {'status':'unresolved','reason':'no_changed_roi_atoms','tested':[],
                    'screened':screen_count,'forward_budget_used':budget,
                    'score_clean':snap.original_score,'score_reference':snap.reference_score}
        for site in sites[1:]:
            expanded=[]
            for _,path,prev in beams:
                choices,count=_rank_changed_atoms(obs,snap,prev['hybrid_windows'][site],site,roi,
                                                  kinds=kinds,limit=per_step)
                screen_count+=count
                for score,atom in choices:
                    if budget>=max_exact:break
                    full=path+[atom]
                    res=run_isolated_monai_chain(model,reference,snap,full,
                                                 class_index=class_index,roi=roi,capture_windows=True)
                    budget+=1
                    all_checks.append({'path':[asdict(a) for a in full],
                                       'heuristic':score,'effect':res['effect_from_reference'],
                                       'hops':res['hops'],'kind':'candidate'})
                    expanded.append((abs(res['effect_from_reference']),full,res))
                if budget>=max_exact:break
            if not expanded:break
            expanded.sort(key=lambda t:-t[0])
            beams=expanded[:beam_width]
        terminal=[x for x in beams if len(x[1])>=2]
        terminal.sort(key=lambda t:-t[0])
        best=terminal[0] if terminal else None
        heldout=None
        cutoff=None
        primary_status='unresolved'
        reason='no_valid_multiblock_path'
        if best is not None:
            effect=best[2]['effect_from_reference']
            # Essential negative control: substitute reference at last selected
            # hop; prior identity-stream delta must no longer leak downstream.
            cut=run_isolated_monai_chain(model,reference,snap,best[1],class_index=class_index,
                                         roi=roi,cut_at=len(best[1])-1)
            cutoff={'effect':cut['effect_from_reference'],
                    'max_logit_delta':float((cut['logits']-snap.reference_logits).abs().max())}
            if cutoff['max_logit_delta']>3e-5:
                reason='identity_bypass_failed'
            elif abs(effect)<=effect_tol:
                reason='insufficient_effect'
            else:
                primary_status='intervened'
                reason='primary_reference_only'
        if best is not None and heldout_reference is not None and primary_status=='intervened':
            if heldout_reference.shape!=clean.shape:
                raise ValueError('Heldout reference shape mismatch')
            hold=capture_monai_reference(model,clean,heldout_reference,class_index=class_index,roi=roi)
            valid=run_isolated_monai_chain(model,heldout_reference,hold,best[1],class_index=class_index,roi=roi)
            h_effect=valid['effect_from_reference']
            consistency=(abs(h_effect)>effect_tol and h_effect*best[2]['effect_from_reference']>0)
            heldout={'score_clean':hold.original_score,'score_reference':hold.reference_score,
                     'effect':h_effect,'direction_consistent':consistency}
            primary_status='cross_reference_supported' if consistency else 'conflicting'
            reason='same_direction_nonzero' if consistency else 'reference_disagreement'
        return {'status':primary_status,'reason':reason,'model':'monai_swinunetr_1.5.1',
                'scope':'controlled_residual_atom_cascade_not_unique_circuit',
                'score_clean':snap.original_score,'score_reference':snap.reference_score,
                'spatial_effect':snap.original_score-snap.reference_score,
                'screened':screen_count,'forward_budget_used':budget,
                'screening_scope':'bounded_roi_scan_96_queries_12_keys_per_head',
                'candidate_paths':all_checks,
                'selected_path':None if best is None else [asdict(a) for a in best[1]],
                'selected_effect':None if best is None else best[2]['effect_from_reference'],
                'negative_control':cutoff,'heldout':heldout}
