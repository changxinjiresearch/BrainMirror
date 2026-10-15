"""P0 executable numerical trace: real PyTorch calculations on synthetic fixtures.

NOT a healthy-brain experiment. No semantic interpretation is inferred.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import torch

from .model import BrainAE, incoming_edges
from .mri import phantom, local_expand


def tensor_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def weights_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha(tensor).encode("ascii"))
    return digest.hexdigest()


def trace_pair(
    model: BrainAE,
    original: np.ndarray,
    roi: np.ndarray,
    strength_voxels: float = 1.0,
    *,
    layer: str = "conv3",
    channel: int | None = None,
    z: int = 2,
    y: int = 2,
    x: int = 2,
    source_tier: str = "synthetic_debug",
) -> dict:
    """Compute input intervention + genuine weighted edges + single-unit ablation.

    Current P0 only authorizes synthetic_debug: real MRI science needs a
    separate approved protocol, data provenance, QC and normalization.
    """
    if source_tier != "synthetic_debug":
        raise ValueError("P0 trace restricted to synthetic_debug until P1 protocol approval")
    if original.shape != (32, 32, 32) or roi.shape != (32, 32, 32):
        raise ValueError("P0 expects aligned 32^3 arrays")
    if not np.isfinite(original).all():
        raise ValueError("Non-finite input")
    edited, edited_mask, warp_qc = local_expand(original, roi, strength_voxels)
    base_tensor = torch.from_numpy(original.astype(np.float32))[None, None]
    edit_tensor = torch.from_numpy(edited)[None, None]
    model.eval()
    with torch.no_grad():
        baseline_recon, baseline_acts = model(base_tensor, capture=True)
        edited_recon, edited_acts = model(edit_tensor, capture=True)
        if channel is None:
            # Pick a measurable unit by value but do NOT assign any semantics.
            pre = baseline_acts[layer][0, :, z, y, x]
            channel = int(torch.argmax(pre).item())
        before = incoming_edges(model, baseline_acts, layer, channel, z, y, x)
        after = incoming_edges(model, edited_acts, layer, channel, z, y, x)
        ablated, _ = model(
            edit_tensor,
            capture=True,
            ablate={"layer": layer, "channel": channel, "z": z, "y": y, "x": x},
        )
    model_hash = weights_sha(model)
    return {
        "schema_version": "brainmirror.p0.trace.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tier": "synthetic_debug",
        "evidence_level": "L1_REAL_NUMERIC_SYNTHETIC_INPUT",
        "scientific_validation": "NOT_STARTED",
        "model_sha256": model_hash,
        "input_sha256": tensor_sha(base_tensor),
        "edited_input_sha256": tensor_sha(edit_tensor),
        "roi_sha256": hashlib.sha256(np.asarray(roi, bool).tobytes()).hexdigest(),
        "target_unit": {
            "id": f"{model_hash}:{layer}:c{channel}:z{z}:y{y}:x{x}",
            "layer": layer,
            "channel": channel,
            "z": z,
            "y": y,
            "x": x,
        },
        "intervention": {
            "kind": "synthetic_local_warp",
            "strength_voxels": strength_voxels,
            "qc": warp_qc,
            "roi_after_voxel_count": int(edited_mask.sum()),
            "anatomical_plausibility_validated": False,
        },
        "neuron": {
            "before": {
                "preactivation": before["preactivation"],
                "postactivation": before["activation"],
            },
            "after": {
                "preactivation": after["preactivation"],
                "postactivation": after["activation"],
            },
            "delta_postactivation": after["activation"] - before["activation"],
            "incoming_edge_type": "actual_conv3d_kernel_coefficient",
            "incoming_edge_count": len(after["edges"]),
            "incoming_edge_sample": after["edges"][:8],
            "bias": after["bias"],
            "recomputed_weighted_sum": after["verified_sum"],
            "max_absolute_arithmetic_error": max(
                abs(before["verified_sum"] - before["preactivation"]),
                abs(after["verified_sum"] - after["preactivation"]),
            ),
        },
        "model_output": {
            "mean_absolute_reconstruction_difference": float(
                torch.mean(torch.abs(edited_recon - baseline_recon))
            ),
            "mean_absolute_ablation_effect_on_edited_reconstruction": float(
                torch.mean(torch.abs(ablated - edited_recon))
            ),
        },
        "warnings": [
            "Synthetic ellipsoid input, NOT a real healthy brain MRI.",
            "Pretrained anatomical understanding NOT established.",
            "P0 warp is NOT anatomically validated.",
            "Measured ablation effect is local to autoencoder reconstruction.",
            "Weights are random unless a separately verified training checkpoint is loaded.",
        ],
    }


def smoke_experiment(seed: int = 17, strength: float = 1.25) -> dict:
    """Deterministic local debug: one true optimization step + real trace."""
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    model = BrainAE()
    phantom_volume, mask = phantom(seed)
    batch = torch.from_numpy(phantom_volume)[None, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    model.train()
    output = model(batch)
    loss = ((output - batch) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    record = trace_pair(model, phantom_volume, mask, strength_voxels=strength)
    record["debug_training"] = {
        "source": "one synthetic phantom",
        "optimizer": "Adam(lr=0.002)",
        "steps": 1,
        "loss_after_forward_before_update": float(loss.detach()),
        "trained_on_real_mri": False,
    }
    return record
