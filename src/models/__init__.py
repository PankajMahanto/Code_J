"""
Model layer — the three publishable novelties + the combined model.

• sgpe_encoder    : Novelty 1  (SGP-E)
• emgdcr_routing  : Novelty 3  (EMGD-CR)   ← must be built BEFORE encoder
• scad_decoder    : Novelty 2  (SCAD)
• ednftm          : full model that wires all three
• ablation_modules: plain baselines for the ablation study
"""
from .sgpe_encoder    import SGPEncoder
from .emgdcr_routing  import EMGDCapsuleRouting
from .scad_decoder    import SCADecoder
from .ednftm          import EDNeuFTMv2
from .ablation_modules import (
    VanillaMLPEncoder,
    VanillaDynamicRouting,
    VanillaSoftmaxDecoder,
)

__all__ = [
    "SGPEncoder",
    "EMGDCapsuleRouting",
    "SCADecoder",
    "EDNeuFTMv2",
    "VanillaMLPEncoder",
    "VanillaDynamicRouting",
    "VanillaSoftmaxDecoder",
]
