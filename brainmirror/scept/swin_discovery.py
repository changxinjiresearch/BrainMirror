"""SCCA v0.6 bounded, ROI-conditioned, multi-block 3D Swin-style path search.

Precision: every selected attention message is executed from real Q/K/V and
projected through the actual per-head out-projection; all *unselected* attention
and MLP branches are frozen to the baseline state. A source delta is injected
into one changed patch-embedding token. Each selected head/query/key step
replaces the *incoming* delta instead of allowing an unselected identity bypass.
The declared path traverses fine blocks -> true patch_merge -> coarse blocks ->
spatial VIB (deterministic mean) -> decoder with FROZEN fine skip.

This is an intervention-conditioned path, NOT a complete natural causal path,
not an exhaustive model graph, and not a MONAI/SwinUNETR integration.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import torch
from torch.nn import functional as F
from .swin3d import MultiLayerSwinVIB3D, unflatten
from .auto_discovery import reference_from_region


@dataclass(frozen=True)
class WindowAtom:
    stage: str
    layer: int
    window: int
    head: int
    query: int  # GLOBAL token index in the corresponding scale
    key: int    # GLOBAL token index
    mode: str   # weighted_value or logit


@dataclass(frozen=True)
class SwinDiscoveryConfig:
    primary_reference: str = 'blur'
    reserved_reference: str = 'outside_mean'
    beam_width: int = 3
    max_source_tokens: int = 3
    per_state_proposals: int = 8
    max_exact_paths: int = 8
    validate_top: int = 3
    min_input_effect: float = 1e-10
    min_source_amplitude: float = 1e-11
    min_proxy: float = 1e-14
    min_supported_fraction: float = 0.01
    off_target_weight: float = 0.1
    modes: tuple[str,...] = ('weighted_value','logit')

    def validate(self):
        if not (0 < self.beam_width <= 32 and 0 < self.max_source_tokens <= 64
                and 0 < self.per_state_proposals <= 256 and 0 < self.max_exact_paths <= 64
                and 0 < self.validate_top <= self.max_exact_paths
                and self.min_input_effect > 0 and self.min_source_amplitude >= 0
                and self.min_proxy >= 0 and 0 <= self.min_supported_fraction <= 1
                and self.off_target_weight >= 0):
            raise ValueError('invalid bounded discovery configuration')
        if (self.primary_reference == self.reserved_reference or
            set(self.modes)-{'weighted_value','logit'} or not self.modes):
            raise ValueError('distinct references and valid nonempty atomic modes required')


def _score(logits, channel, target):
    return logits[:,channel,:, :, :][:,target].mean()


def _validate(model, x, source, target, channel):
    if not isinstance(model,MultiLayerSwinVIB3D):
        raise TypeError('v0.6 requires the explicit two-scale multi-block Swin-style model')
    if not torch.is_tensor(x) or x.ndim!=5 or x.shape[0]!=1 or x.shape[1]!=4:
        raise ValueError('only single-channel-batch 4-modality 3D MRI-like input supported')
    if not torch.is_floating_point(x) or not bool(torch.isfinite(x).all()):
        raise ValueError('input must be finite floating tensor')
    model.grids(x)
    if x.device!=next(model.parameters()).device or x.dtype!=next(model.parameters()).dtype:
        raise ValueError('model/input dtype or device mismatch')
    if model.training or any(isinstance(m,torch.nn.Dropout) and m.training for m in model.modules()):
        raise ValueError('model must be in eval() for controlled interventions')
    if channel not in (0,1,2):
        raise ValueError('target channel must be WT/TC/ET (0/1/2)')
    for mask in (source,target):
        if not (torch.is_tensor(mask) and mask.device==x.device
                and mask.dtype==torch.bool and tuple(mask.shape)==tuple(x.shape[-3:])
                and bool(mask.any()) and bool((~mask).any())):
            raise ValueError('source and target must be valid nontrivial 3D boolean masks')


def _trace(model, x, target=None, channel=2, need_grad=False):
    with torch.enable_grad() if need_grad else torch.no_grad():
        logits,rec=model(x,record=True,sample=False)
        gradients=None
        if need_grad:
            gates=[item['gate'] for item in rec['fine_steps']+rec['coarse_steps']]
            grads=torch.autograd.grad(_score(logits,channel,target),
                [rec['fine_initial']]+gates,allow_unused=False)
            gradients={'source':grads[0].detach(),
                       'fine':[g.detach() for g in grads[1:1+len(rec['fine_steps'])]],
                       'coarse':[g.detach() for g in grads[1+len(rec['fine_steps']):]]}
    return logits.detach(),_detach(rec),gradients


def _detach(v):
    if torch.is_tensor(v):return v.detach()
    if isinstance(v,dict):return {k:_detach(x) for k,x in v.items()}
    if isinstance(v,list):return [_detach(x) for x in v]
    return v


def _location(attn,atom):
    ids=attn['token_ids']
    if not (isinstance(atom,WindowAtom) and 0<=atom.window<ids.shape[0] and
            0<=atom.head<attn['q'].shape[2]):
        raise ValueError('invalid atom window/head')
    qpos=(ids[atom.window]==atom.query).nonzero().flatten()
    kpos=(ids[atom.window]==atom.key).nonzero().flatten()
    if len(qpos)!=1 or len(kpos)!=1:
        raise ValueError('query/key not co-located in selected shifted window')
    q=int(qpos[0]); k=int(kpos[0]);
    if not bool(attn['allowed'][atom.window,q,k]):
        raise ValueError('masked or cyclic-wrap token interaction is not executable')
    return q,k


def _atom_delta(base,hybrid,atom,projection):
    """Post-projection attention branch delta ONLY for one global query token."""
    q,k=_location(base,atom);wi=atom.window;h=atom.head
    if atom.mode=='weighted_value':
        src=(hybrid['weights'][:,wi,h,q,k,None]*hybrid['v'][:,wi,h,k]
             -base['weights'][:,wi,h,q,k,None]*base['v'][:,wi,h,k])
    elif atom.mode=='logit':
        # Recompute the FULL query row using exactly one modified logit.
        old=base['logits'][:,wi,h,q,:]
        replaced=old.clone()
        replaced[:,k]=hybrid['logits'][:,wi,h,q,k]
        updated=F.softmax(replaced,dim=-1)
        src=((updated-base['weights'][:,wi,h,q,:]).unsqueeze(-1)*
             base['v'][:,wi,h,:,:]).sum(-2)
    else:
        raise ValueError('invalid atom mode')
    dh=base['head_width']
    return src@projection[:,h*dh:(h+1)*dh].T


def _candidate_vectors(base,hybrid,projection,gate_gradient,modes,min_proxy,limit,stage,layer):
    """Vectorized first-order screening; no screening score is causal evidence."""
    if not torch.equal(base['token_ids'],hybrid['token_ids']):
        raise AssertionError('window token correspondence changed')
    # [W,H,Q,K,head_width], B=1
    bb=base['weights'][0];ab=hybrid['weights'][0]
    bv=base['v'][0];av=hybrid['v'][0]
    vectors={}
    if 'weighted_value' in modes:
        vectors['weighted_value']=ab[...,None]*av[:,:,None,:,:]-bb[...,None]*bv[:,:,None,:,:]
    if 'logit' in modes:
        diff=(hybrid['raw_logits'][0]-base['raw_logits'][0]).clamp(-30,30)
        frac=(bb*torch.expm1(diff))
        coeff=frac/(1+frac).clamp_min(1e-10)
        vectors['logit']=coeff[...,None]*(bv[:,:,None,:,:]-base['head_output'][0][:,:,:,None,:])
    ww,h,q,k,dh=bb.shape[0],bb.shape[1],bb.shape[2],bb.shape[3],base['head_width']
    weights=torch.stack([projection[:,i*dh:(i+1)*dh].T for i in range(h)],dim=0)
    local_grad=gate_gradient[0][base['token_ids']]  # [W,Q,dim]
    ranking=[]
    for mode,deltas in vectors.items():
        vec_out=torch.einsum('whqkd,hdc->whqkc',deltas,weights)
        estimates=(vec_out*local_grad[:,None,:,None,:]).sum(-1)
        scores=estimates.abs()+vec_out.norm(dim=-1)*1e-8
        scores=scores.masked_fill(~base['allowed'][:,None,:,:],-1.)
        topk=min(limit,scores.numel())
        scores_flat=scores.reshape(-1)
        values,idxs=torch.topk(scores_flat,k=topk)
        for score,index in zip(values.tolist(),idxs.tolist()):
            if score<=min_proxy:continue
            widx=index//(h*q*k); rest=index%(h*q*k)
            hid=rest//(q*k); rem=rest%(q*k)
            qi=rem//k; ki=rem%k
            row_ids=base['token_ids'][widx]
            atom=WindowAtom(stage,layer,int(widx),int(hid),int(row_ids[qi]),int(row_ids[ki]),mode)
            vec=vec_out[widx,hid,qi,ki].reshape(1,-1).detach()
            ranking.append((float(score),float(estimates[widx,hid,qi,ki].detach()),atom,vec))
    ranking.sort(key=lambda a:(-a[0],a[2].stage,a[2].layer,a[2].window,
                                a[2].head,a[2].query,a[2].key,a[2].mode))
    return ranking[:limit]


def _inject_one_source(clean,base,source_index):
    delta=torch.zeros_like(base['fine_initial'])
    delta[:,source_index,:]=clean['fine_initial'][:,source_index,:]-base['fine_initial'][:,source_index,:]
    return delta


def _hybrid_parts(model,step,delta,block,grid):
    _,rec=block.attn(block.norm1(step['input']+delta),grid,record=True)
    if not torch.equal(rec['token_ids'],step['attention']['token_ids']):
        raise AssertionError('window ID mismatch during patching')
    return rec


def _next_delta(step,base_attn,hybrid,atom,block):
    if atom.stage not in ('fine','coarse'):
        raise ValueError('unknown stage')
    d=_atom_delta(base_attn,hybrid,atom,block.attn.proj.weight)
    output=torch.zeros_like(step['input'])
    output[:,atom.query,:]=d
    return output


def _bridge(model,ref,delta):
    """Precisely propagate one selected fine route via merge, never via fine skip."""
    b=ref['fine_initial'].shape[0];fine=ref['fine_grid']
    altered_pre=ref['fine_pre_norm']+delta
    altered=model.fine_norm(altered_pre).transpose(1,2).reshape(b,model.dim,*fine)
    newmerge=model.patch_merge(altered)
    diff=newmerge-ref['merged']
    nz=(diff.abs().sum(dim=1)>1e-11)
    if int(nz.sum())>1:
        raise AssertionError('isolated source must not affect multiple merge cells')
    return diff.flatten(2).transpose(1,2)


@torch.no_grad()
def execute_route(model,base,donor,source_index,fine_atoms,coarse_atoms):
    """Run a SELECTED path, freezing every off-path residual/MLP/skip component.

    This only recomputes selected attention messages in each block, the true
    nonlinear norm, 2x patch merging, deterministic VIB and decoder.
    """
    if len(fine_atoms)!=len(model.fine_blocks) or len(coarse_atoms)!=len(model.coarse_blocks):
        raise ValueError('one declared attention atom required per stage block')
    if not 0<=source_index<base['fine_initial'].shape[1]:
        raise ValueError('invalid source token')
    delta=_inject_one_source(donor,base,source_index)
    for i,(block,step,atom) in enumerate(zip(model.fine_blocks,base['fine_steps'],fine_atoms)):
        if atom.stage!='fine' or atom.layer!=i:
            raise ValueError('wrong fine stage order')
        h=_hybrid_parts(model,step,delta,block,base['fine_grid'])
        delta=_next_delta(step,step['attention'],h,atom,block)
    delta=_bridge(model,base,delta)
    for i,(block,step,atom) in enumerate(zip(model.coarse_blocks,base['coarse_steps'],coarse_atoms)):
        if atom.stage!='coarse' or atom.layer!=i:
            raise ValueError('wrong coarse stage order')
        h=_hybrid_parts(model,step,delta,block,base['coarse_grid'])
        delta=_next_delta(step,step['attention'],h,atom,block)
    z=model.coarse_norm(base['coarse_pre_norm']+delta)
    # The fine skip is the base-case skip, NOT the altered donor skip.
    return model.decode(z,base['fine_skip'],base['coarse_grid'],sample=False)


def _beam_step(model,beams,step,block,grid,grad,stage,layer,cfg):
    ranking=[];screened=0
    for beam in beams:
        hybrid=_hybrid_parts(model,step,beam['delta'],block,grid)
        base=step['attention']
        proposals=_candidate_vectors(base,hybrid,block.attn.proj.weight,grad,cfg.modes,
                                     cfg.min_proxy,cfg.per_state_proposals,stage,layer)
        screened+=int(base['allowed'].sum())*block.attn.heads*len(cfg.modes)
        for score,signed,atom,_ in proposals:
            delta=_next_delta(step,base,hybrid,atom,block)
            if delta.abs().max()<=0:continue
            path=beam['fine'] if stage=='fine' else beam['coarse']
            candidate={'source':beam['source'],
                       'fine':(path+[atom]) if stage=='fine' else beam['fine'],
                       'coarse':(path+[atom]) if stage=='coarse' else beam['coarse'],
                       'delta':delta,'last_proxy':score,'signed_proxy':signed}
            ranking.append(candidate)
    ranking.sort(key=lambda e:(-e['last_proxy'],e['source'],
                        tuple((x.stage,x.layer,x.window,x.head,x.query,x.key,x.mode)
                              for x in e['fine']+e['coarse'])))
    return ranking[:cfg.beam_width],screened


def _score_path(model,clean,ref,source,fine_atoms,coarse_atoms,target,channel):
    restored=execute_route(model,ref,clean,source,fine_atoms,coarse_atoms)
    disrupted=execute_route(model,clean,ref,source,fine_atoms,coarse_atoms)
    return {'restored_logits':restored,'disrupted_logits':disrupted,
            'restore_score':float(_score(restored,channel,target)),
            'disrupt_score':float(_score(disrupted,channel,target))}


def _summary_path(row):
    return {'source_fine_token':row['source'],'fine_atoms':[asdict(a) for a in row['fine']],
            'coarse_atoms':[asdict(a) for a in row['coarse']]}


@torch.no_grad()
def validate_fixed_route(model,x,reference_x,source,fine_atoms,coarse_atoms,target,channel):
    cl,clean,_=_trace(model,x)
    rl,ref,_=_trace(model,reference_x)
    z=_score_path(model,clean,ref,source,fine_atoms,coarse_atoms,target,channel)
    c=float(_score(cl,channel,target));r=float(_score(rl,channel,target))
    d=c-r
    return {'input_effect':d,'restoration':z['restore_score']-r,
            'disruption':c-z['disrupt_score'],
            'off_target_mean_abs':float((z['restored_logits'][:,channel][:,~target]-
                                           rl[:,channel][:,~target]).abs().mean())}


def discover_scca_swin(model,x,selected_mri_region,target_roi=None,channel=2,config=None):
    """Automatically choose source patch + Head/Q/K in EVERY windowed Swin block.

    Attribution-based beam screening selects paths under a strict compute cap;
    double-direction exact patching ranks finalists. Another reference
    strategy is RESERVED until after candidate selection. Same-patient
    sensitivity checks are not independent patient evidence.
    """
    cfg=config or SwinDiscoveryConfig();cfg.validate()
    target=selected_mri_region if target_roi is None else target_roi
    _validate(model,x,selected_mri_region,target,channel)
    primary=reference_from_region(x,selected_mri_region,cfg.primary_reference)
    reserved=reference_from_region(x,selected_mri_region,cfg.reserved_reference)
    clean_logits,clean,_=_trace(model,x)
    ref_logits,ref,grads=_trace(model,primary,target,channel,need_grad=True)
    clean_score=float(_score(clean_logits,channel,target));ref_score=float(_score(ref_logits,channel,target))
    effect=clean_score-ref_score
    output={'method':'SCCA','version':'0.6-swin-multiblock',
            'model':'executed_3d_window_shifted_window_two_scale_model_NOT_MONAI_SWINUNETR',
            'status':'unresolved', 'execution':'deterministic_VIB_mu',
            'source_voxels':int(selected_mri_region.sum()),
            'target_voxels':int(target.sum()),'target_channel':channel,
            'main_input_effect':effect,'reference_primary':cfg.primary_reference,
            'reference_reserved':cfg.reserved_reference,
            'candidate_screening':'per-layer linearized output-gradient, NOT causal effect',
            'path_semantics':'selected_attention_atom_at_each_block_merge_spatialVIB_decoder_frozen_fine_skip',
            'clinical_reference_validity':'NOT_ASSESSED_synthetic_diagnostic_references',
            'independent_patient_validation':False,'config':{**asdict(cfg),'modes':list(cfg.modes)},
            'source_candidates':[],'screened_atomic_count':0,'exact_tested_count':0,
            'candidates':[],'reserved_validation':[],'best':None}
    if abs(effect)<cfg.min_input_effect:
        output['status']='unresolved_low_primary_effect';return output
    delta=clean['fine_initial']-ref['fine_initial']
    amplitudes=delta.square().sum(-1).sqrt()[0]
    screen=((delta*grads['source']).sum(-1).abs()[0]+1e-8*amplitudes)
    source=[j for j in range(len(screen)) if float(amplitudes[j])>cfg.min_source_amplitude]
    source.sort(key=lambda j:(-float(screen[j]),j))
    source=source[:cfg.max_source_tokens]
    output['source_candidates']=[{'token':j,'zyx':list(unflatten(j,ref['fine_grid'])),
                'proxy':float(screen[j])} for j in source]
    if not source:
        output['status']='unresolved_no_changed_embedding';return output
    beams=[{'source':i,'fine':[],'coarse':[],'delta':_inject_one_source(clean,ref,i),
           'last_proxy':float(screen[i]),'signed_proxy':float(screen[i])} for i in source]
    counted=0
    with torch.no_grad():
        for i,block in enumerate(model.fine_blocks):
            beams,num=_beam_step(model,beams,ref['fine_steps'][i],block,ref['fine_grid'],
                                    grads['fine'][i],'fine',i,cfg)
            counted+=num
            if not beams:break
        if beams:
            for beam in beams:
                beam['delta']=_bridge(model,ref,beam['delta'])
            for i,block in enumerate(model.coarse_blocks):
                beams,num=_beam_step(model,beams,ref['coarse_steps'][i],block,ref['coarse_grid'],
                                        grads['coarse'][i],'coarse',i,cfg)
                counted+=num
                if not beams:break
    output['screened_atomic_count']=counted
    if not beams:
        output['status']='unresolved_no_eligible_path';return output
    tested=[]
    for beam in beams[:cfg.max_exact_paths]:
        score=_score_path(model,clean,ref,beam['source'],beam['fine'],beam['coarse'],target,channel)
        restoration=score['restore_score']-ref_score
        disruption=clean_score-score['disrupt_score']
        off=float((score['restored_logits'][:,channel][:,~target]-
                     ref_logits[:,channel][:,~target]).abs().mean())
        denom=max(abs(effect),cfg.min_input_effect)
        objective=(abs(effect-restoration)+abs(effect-disruption)+cfg.off_target_weight*off)/denom
        tested.append({**_summary_path(beam),'proxy_abs':beam['last_proxy'],
                       'restoration':restoration,'disruption':disruption,
                       'off_target_mean_abs':off,'objective':objective,
                       'direction_aligned':bool(effect*restoration>0 and effect*disruption>0),
                       'input_effect':effect})
    tested.sort(key=lambda e:(e['objective'],-e['proxy_abs'],e['source_fine_token']))
    output['candidates']=tested;output['exact_tested_count']=len(tested);output['best']=tested[0]
    reserved_logits,reserved_trace,_=_trace(model,reserved)
    d2=clean_score-float(_score(reserved_logits,channel,target))
    output['reserved_input_effect']=d2
    if abs(d2)<cfg.min_input_effect:
        output['status']='unresolved_low_reserved_effect';return output
    validation=[]
    for cand in tested[:cfg.validate_top]:
        fine_atoms=[WindowAtom(**v) for v in cand['fine_atoms']]
        coarse_atoms=[WindowAtom(**v) for v in cand['coarse_atoms']]
        ev=_score_path(model,clean,reserved_trace,cand['source_fine_token'],
                        fine_atoms,coarse_atoms,target,channel)
        restore=ev['restore_score']-float(_score(reserved_logits,channel,target))
        disrupt=clean_score-ev['disrupt_score']
        ratio=min(abs(restore),abs(disrupt))/max(abs(d2),cfg.min_input_effect)
        passes=bool(cand['direction_aligned'] and d2*restore>0 and d2*disrupt>0
                    and ratio>=cfg.min_supported_fraction)
        validation.append({'source_fine_token':cand['source_fine_token'],
                  'fine_atoms':cand['fine_atoms'],'coarse_atoms':cand['coarse_atoms'],
                  'restoration':restore,'disruption':disrupt,
                  'relative_min_effect':ratio,'cross_reference_direction_supported':passes})
    output['reserved_validation']=validation
    output['status']=('same_case_cross_reference_supported' if any(
        v['cross_reference_direction_supported'] for v in validation)
        else 'unresolved_or_conflicting_references')
    return output
