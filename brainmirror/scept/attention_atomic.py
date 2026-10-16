"""SCEPT v0.4: exact *defined* attention-head / token interventions.

A token-to-token ``weighted_value`` message is A[b,h,q,k] * V[b,h,k].
Changing just this post-softmax message is a well-defined *surgical*
intervention. It does NOT independently change the query/key computation or
claim that an attention weight measures causal importance.

A ``logit`` intervention swaps one pre-softmax QK score, renormalising the
whole query row; that operation may change contributions from other tokens.
These meanings are intentionally distinct and recorded in each output.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F

from .model import TransformerVIB3D
from .trace import captured_forward
from .paths import capture_path_baseline


@dataclass(frozen=True)
class AtomicAttentionSite:
    layer: int
    head: int
    query: int
    key: int
    mode: str = 'weighted_value'  # or 'logit'


def _check_block(block):
    attn = block.attn
    if (attn.dropout != 0 or not attn.batch_first or
            attn.bias_k is not None or attn.bias_v is not None or
            attn.add_zero_attn or attn.kdim != attn.embed_dim or
            attn.vdim != attn.embed_dim or not attn._qkv_same_embed_dim):
        raise ValueError('Unsupported attention variant: explicit no-dropout self-attention required')
    return attn


def attention_parts(block, pre_norm_residual):
    """Return algebraically inspectable attention intermediates, in batch/head axes.

    Reconstructs the same computation as the current nn.MultiheadAttention,
    dropout=0, self-attention, no masks. We compare with PyTorch in tests.
    """
    attn = _check_block(block)
    if pre_norm_residual.ndim != 3 or pre_norm_residual.shape[-1] != attn.embed_dim:
        raise ValueError('Expected [B,N,D] residual tensor')
    h = block.norm1(pre_norm_residual)
    q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    b, n, d = q.shape
    heads, dh = attn.num_heads, d // attn.num_heads
    def split(t):
        return t.reshape(b, n, heads, dh).permute(0, 2, 1, 3)
    q, k, v = split(q), split(k), split(v)
    logits = q @ k.transpose(-1, -2) / math.sqrt(dh)
    weights = torch.softmax(logits, dim=-1)
    raw = weights @ v
    concatenated = raw.permute(0, 2, 1, 3).reshape(b, n, d)
    output = F.linear(concatenated, attn.out_proj.weight, attn.out_proj.bias)
    return {'q': q, 'k': k, 'v': v, 'logits': logits,
            'weights': weights, 'head_output': raw, 'output': output,
            'head_width': dh}


def validate_site(model, site, n_tokens):
    if not isinstance(model, TransformerVIB3D):
        raise TypeError('Atomic v0.4 adapter supports single-scale TransformerVIB3D')
    if not isinstance(site, AtomicAttentionSite) or site.mode not in ('weighted_value', 'logit'):
        raise ValueError('Invalid atomic attention site or intervention mode')
    if (not 0 <= site.layer < len(model.blocks) or
        not 0 <= site.head < model.blocks[site.layer].attn.num_heads or
        not 0 <= site.query < n_tokens or not 0 <= site.key < n_tokens):
        raise ValueError('Head or token outside model topology')
    _check_block(model.blocks[site.layer])


def _delta_one_head_query(clean, base, site, out_proj_weight):
    """Calculate residual-branch output delta at selected query only."""
    h, q, k = site.head, site.query, site.key
    if site.mode == 'weighted_value':
        changed_message = clean['weights'][:, h, q, k, None] * clean['v'][:, h, k]
        base_message = base['weights'][:, h, q, k, None] * base['v'][:, h, k]
        head_delta = changed_message - base_message
    else:
        # Exactly one QK logit changes, but a softmax row is coupled.
        changed_row = base['logits'][:, h, q, :].clone()
        changed_row[:, k] = clean['logits'][:, h, q, k]
        new_weights = torch.softmax(changed_row, dim=-1)
        head_delta = ((new_weights - base['weights'][:, h, q, :]).unsqueeze(-1)
                      * base['v'][:, h]).sum(dim=1)
    dh = base['head_width']
    # out_proj: [D_out, D_in]; only the selected head is changed.
    kernel = out_proj_weight[:, h * dh:(h + 1) * dh]
    return head_delta @ kernel.T


@torch.no_grad()
def intervene_atomic_attention(model, clean_x, reference_x, site):
    """Patch exactly one head/query message (or QK score) into reference forward.

    ALL remaining attention heads, queries, residual branches and skip states
    are evaluated on the reference input. Only the selected output token's
    branch tensor receives the surgically computed delta.
    """
    if (clean_x.shape != reference_x.shape or clean_x.ndim != 5 or
            clean_x.shape[0] != 1 or clean_x.shape[1] != 4 or
            clean_x.dtype != reference_x.dtype or clean_x.device != reference_x.device):
        raise ValueError('Paired [1,4,D,H,W] inputs with matching dtype/device required')
    model.eval()
    clean_logits, clean_gates, clean_res = capture_path_baseline(model, clean_x)
    ref_logits, ref_gates, ref_res = capture_path_baseline(model, reference_x)
    n = ref_res['blocks.0.attn'].shape[1]
    validate_site(model, site, n)
    block = model.blocks[site.layer]
    base = attention_parts(block, ref_res[f'blocks.{site.layer}.attn'])
    changed = attention_parts(block, clean_res[f'blocks.{site.layer}.attn'])
    edge = f'blocks.{site.layer}.attn_to_residual'
    if not torch.allclose(base['output'], ref_gates[edge], atol=2e-6, rtol=2e-5):
        raise AssertionError('Manual attention decomposition diverged from model')
    delta = _delta_one_head_query(changed, base, site, block.attn.out_proj.weight)
    patch = ref_gates[edge].clone()
    patch[:, site.query, :] += delta
    output, _, _ = captured_forward(model, reference_x, patch=(edge, patch))
    return {
        'original_logits': clean_logits,
        'reference_logits': ref_logits,
        'intervened_logits': output,
        'branch_delta': delta,
        'patched_branch': patch,
        'reference_branch': ref_gates[edge],
        'site': site,
        'semantics': ('post_softmax_weighted_value_message_swap' if site.mode == 'weighted_value'
                      else 'single_qk_logit_swap_softmax_row_renormalized'),
        'independent_patient_validation': False,
    }
