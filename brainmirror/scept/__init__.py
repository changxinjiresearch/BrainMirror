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

# v0.4 experimental, auditable units. Explicit reference-conditioned
# interventions; no clinical claims or unique circuit identification.
from .attention_atomic import (AtomicAttentionSite, attention_parts,
                               intervene_atomic_attention)
from .latent_bridge import LatentSite, intervene_spatial_vib
from .atomic_chain import (AtomicPathHop, intervene_atomic_chain,
                           search_atomic_chains)
from .multiscale import (MultiScaleTransformerVIB3D, CrossScaleSite,
                         intervene_merge_path, intervene_cross_scale_atomic)

# v0.5: ROI-conditioned automatic fine/coarse mediator and atomic head/Q/K
# discovery. Diagnostic references only; NO held-out patients or clinical claims.
from .auto_discovery import (AutoDiscoveryConfig, discover_scca_multiscale,
                             reference_from_region)

# v0.6: executable shifted-window Swin-style stages and bounded multi-block
# ROI-conditioned path discovery. NOT a drop-in MONAI SwinUNETR adapter.
from .swin3d import (WindowAttention3D, SwinBlock3D, MultiLayerSwinVIB3D,
                     execution_topology)
from .swin_discovery import (WindowAtom, SwinDiscoveryConfig,
                             discover_scca_swin, execute_route, validate_fixed_route)
