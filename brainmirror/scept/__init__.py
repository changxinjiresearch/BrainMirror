"""SCEPT prototype: observable Transformer-VIB and intervention-based explanations."""
from .model import TransformerVIB3D, SpatialVIB, brats_dice_bce
from .ijepa import IJEPAPretrainer, transfer_pretrained_encoder
from .trace import captured_forward, numeric_trace_equivalence, allowed_modules
from .explain import Region, local_reference, explain_region, fixed_target_score

__all__ = ['TransformerVIB3D', 'SpatialVIB', 'brats_dice_bce', 'IJEPAPretrainer',
           'transfer_pretrained_encoder', 'captured_forward',
           'numeric_trace_equivalence', 'allowed_modules', 'Region',
           'local_reference', 'explain_region', 'fixed_target_score']

from .scca import (SCCAConfig, explain_scca, valid_declared_edges, reference_volume,
                   evaluate_edge_set, gradient_screen)

from .paths import (PathHop, PathSearchConfig, capture_path_baseline,
                    route_delta, intervene_path, evaluate_path, explain_scca_paths)
