"""SCEPT v0.4: cross-layer atomic attention message -> Spatial VIB -> decoder.

Off-path residual branches are frozen to REFERENCE activations in the route
calculation. A selected head+query+key is one post-softmax weighted-value
message, or one coupled pre-softmax score. Only the selected message output
is propagated to the next hop. The final hop is the FINAL MLP residual branch;
from there the *unmodified real* layer norm, VIB and decoder execute.

This is an exact intervention on the DECLARED residual computation graph, NOT
an identification of a unique natural causal route or medical causal claim.
"""
from dataclasses import dataclass
import torch
from .model import TransformerVIB3D
from .paths import PathHop, validate_path, capture_path_baseline
from .attention_atomic import AtomicAttentionSite, attention_parts, _delta_one_head_query
from .trace import captured_forward


@dataclass(frozen=True)
class AtomicPathHop:
    """An individual head/query/key intervention at a declared MHA residual gate."""
    layer: int
    head: int
    query: int
    key: int
    mode: str = 'weighted_value'

    @property
    def edge(self):
        return f'blocks.{self.layer}.attn_to_residual'


def _validate_atomic_chain(model,hops,token_count):
    if not isinstance(model,TransformerVIB3D):
        raise TypeError('Only TransformerVIB3D is supported by atomic chain v0.4')
    if len(hops)<3 or not isinstance(hops[0],PathHop) or not isinstance(hops[-1],PathHop):
        raise ValueError('Expected source gate -> atomic Attention -> final MLP gate')
    if not any(isinstance(h,AtomicPathHop) for h in hops[1:-1]):
        raise ValueError('At least one head/query/key intervention is required')
    if hops[-1].edge!=f'blocks.{len(model.blocks)-1}.mlp_to_residual':
        raise ValueError('For a VIB/decoder-closed route, final gate must be last MLP')
    ordinary=[]
    from .attention_atomic import validate_site
    for h in hops:
        if isinstance(h,PathHop):ordinary.append(h)
        elif isinstance(h,AtomicPathHop):
            validate_site(model,AtomicAttentionSite(h.layer,h.head,h.query,h.key,h.mode),token_count)
            ordinary.append(PathHop(h.edge,'qkv'))
        else:raise ValueError('Undeclared path hop')
    validate_path(model,ordinary)


@torch.no_grad()
def intervene_atomic_chain(model,clean_x,reference_x,hops,source_mask=None,
                           latent_token=None,latent_channel=None):
    """Return isolated chain+optional VIB/decoder token patch effects.

    `source_mask` limits SOURCE branch tokens. `latent_token` selects one z
    token only AFTER the entire frozen-offpath chain has been evaluated;
    `latent_channel` optionally restricts z to one dimension. Each patch
    recomputes the actual downstream model, not a simulated heatmap.
    """
    if (clean_x.shape != reference_x.shape or clean_x.ndim!=5 or
            clean_x.shape[0]!=1 or clean_x.shape[1]!=4 or
            clean_x.dtype!=reference_x.dtype or clean_x.device!=reference_x.device):
        raise ValueError('Expected matched [1,4,D,H,W] inputs')
    model.eval()
    clean_logits,clean_gates,_=capture_path_baseline(model,clean_x)
    ref_logits,ref_gates,ref_res=capture_path_baseline(model,reference_x)
    source_edge=hops[0].edge if hops else ''
    if source_edge not in ref_gates:raise ValueError('Source gate not present')
    _validate_atomic_chain(model,hops,ref_gates[source_edge].shape[1])
    delta=clean_gates[source_edge]-ref_gates[source_edge]
    if source_mask is not None:
        if (not isinstance(source_mask,torch.Tensor) or source_mask.dtype!=torch.bool or
                source_mask.shape!=delta.shape[:2] or source_mask.device!=delta.device):
            raise ValueError('Source token mask mismatch')
        delta=delta*source_mask.unsqueeze(-1)
    step_records=[]
    for hop in hops[1:]:
        if isinstance(hop,AtomicPathHop):
            block=model.blocks[hop.layer]
            base_input=ref_res[f'blocks.{hop.layer}.attn']
            if base_input.shape!=delta.shape:raise ValueError('Residual dimension mismatch')
            baseline=attention_parts(block,base_input)
            hybrid=attention_parts(block,base_input+delta)
            site=AtomicAttentionSite(hop.layer,hop.head,hop.query,hop.key,hop.mode)
            msg=_delta_one_head_query(hybrid,baseline,site,block.attn.out_proj.weight)
            next_delta=torch.zeros_like(base_input)
            next_delta[:,hop.query,:]=msg
        else:
            layer,branch=hop.position()
            block=model.blocks[layer]
            base_input=ref_res[f'blocks.{layer}.{branch}']
            if base_input.shape!=delta.shape:raise ValueError('Residual dimension mismatch')
            if branch=='mlp':
                next_delta=(block.mlp(block.norm2(base_input+delta))-
                            ref_gates[hop.edge])
            else:
                qbase=block.norm1(base_input); qhybrid=block.norm1(base_input+delta)
                q=qhybrid if hop.role in ('q','qkv') else qbase
                k=qhybrid if hop.role in ('k','qkv') else qbase
                v=qhybrid if hop.role in ('v','qkv') else qbase
                output,_=block.attn(q,k,v,need_weights=False)
                next_delta=output-ref_gates[hop.edge]
        if not torch.isfinite(next_delta).all():raise ArithmeticError('Nonfinite route delta')
        delta=next_delta
        step_records.append({'edge':hop.edge if isinstance(hop,PathHop) else hop.edge,
                            'route_delta_norm':float(delta.norm()),
                            'site_type':'atomic_attention_message' if isinstance(hop,AtomicPathHop)
                                     else 'declared_residual_gate'})
    last=hops[-1].edge
    patched=ref_gates[last]+delta
    chain_logits,_,chain_details=captured_forward(model,reference_x,patch=(last,patched))
    _,_,ref_details=captured_forward(model,reference_x)
    z_delta=chain_details['latent']-ref_details['latent']
    latent_logits=None
    if latent_token is not None:
        if not isinstance(latent_token,int) or not 0<=latent_token<z_delta.shape[1]:
            raise ValueError('Invalid latent token')
        if latent_channel is not None and (not isinstance(latent_channel,int) or
                not 0<=latent_channel<z_delta.shape[2]):
            raise ValueError('Invalid latent channel')
        z_patch=ref_details['latent'].clone()
        if latent_channel is None:
            z_patch[:,latent_token,:]=chain_details['latent'][:,latent_token,:]
        else:
            z_patch[:,latent_token,latent_channel]=chain_details['latent'][:,latent_token,latent_channel]
        latent_logits,_,_=captured_forward(model,reference_x,patch=('vib',z_patch))
    return {'clean_logits':clean_logits,'reference_logits':ref_logits,
        'isolated_chain_logits':chain_logits,'latent_mediated_logits':latent_logits,
        'route_delta':delta,'latent_delta':z_delta,'steps':step_records,
        'semantics':'frozen_offpath_atomic_message_chain_then_spatial_vib_decoder',
        'independent_patient_validation':False}


@torch.no_grad()
def search_atomic_chains(model,clean_x,reference_x,source_edge,
                         middle_layer,queries,keys,heads,roi,channel=2,
                         max_tested=16,latent_token=None,source_mask=None,
                         min_input_effect=1e-6):
    """Bounded exhaustive enumeration over EXPLICITLY requested atomic sites.

    Discovery data are the supplied same-case pair. No independent reference,
    held-out patient, clinical or method-superiority validation is performed.
    Retains every tested candidate and signed intervention effect.
    """
    from itertools import product
    from .explain import fixed_target_score
    if (not isinstance(model,TransformerVIB3D) or
            not isinstance(max_tested,int) or max_tested<1 or
            channel not in (0,1,2) or middle_layer<0 or
            middle_layer>=len(model.blocks)-1):
        raise ValueError('Invalid atomic discovery settings')
    if (not torch.is_tensor(roi) or roi.dtype!=torch.bool or
            roi.shape!=clean_x.shape[-3:] or roi.device!=clean_x.device or
            not roi.any() or not (~roi).any()):
        raise ValueError('Fixed ROI must be nontrivial and reside on model device')
    source=PathHop(source_edge)
    final=PathHop(f'blocks.{len(model.blocks)-1}.mlp_to_residual')
    combos=list(product(tuple(sorted(set(heads))),tuple(sorted(set(queries))),
                        tuple(sorted(set(keys)))))
    if not combos:raise ValueError('No candidate atoms specified')
    # Hard budget: the enumerator never silently evaluates all O(H*N*N).
    combos=combos[:max_tested]
    tested=[]
    result0=None
    for h,q,k in combos:
        hop=AtomicPathHop(middle_layer,h,q,k)
        r=intervene_atomic_chain(model,clean_x,reference_x,[source,hop,final],
               source_mask=source_mask,latent_token=latent_token)
        d=float(fixed_target_score(r['clean_logits'],channel,roi)-
                fixed_target_score(r['reference_logits'],channel,roi))
        effect=float(fixed_target_score(r['isolated_chain_logits'],channel,roi)-
                     fixed_target_score(r['reference_logits'],channel,roi))
        if result0 is None:result0=d
        denom=max(abs(d),min_input_effect)
        tested.append({'head':h,'query':q,'key':k,
                       'input_effect':d,'chain_restoration':effect,
                       'signed_fraction':effect/d if abs(d)>=min_input_effect else None,
                       'absolute_relative_error':abs(d-effect)/denom,
                       'direction_aligned':d*effect>0,
                       'route_delta_norm':float(r['route_delta'].norm())})
    tested.sort(key=lambda v:(v['absolute_relative_error'],v['head'],v['query'],v['key']))
    status=('unresolved_low_input_effect' if abs(result0)<min_input_effect else
            'same_case_candidates_not_independently_validated')
    return {'version':'scca-atomic-v0.4','status':status,
            'selection_reference':'provided_reference_case',
            'source_edge':source_edge,'intermediate_layer':middle_layer,
            'last_mlp_gate':final.edge,'channel':channel,
            'tested_count':len(tested),'budget':max_tested,
            'candidates':tested,'best':tested[0] if abs(result0)>=min_input_effect else None,
            'independent_patient_validation':False,
            'semantics':'bounded_same_case_frozen_offpath_atomic_chain_discovery'}
