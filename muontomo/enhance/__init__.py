"""Modular ML/CV enhancement suite for the 3D reconstruction surface.

Every technique sharpens the reconstruction's ceiling-layer slice while being
gated by the model-free beam verification (muontomo.beams): a technique that
moves or invents beams is rejected. All techniques share ONE data/calibration
front-end (context.EnhanceContext) and register through base.REGISTRY, so adding
a technique never touches the core pipeline.

    python -m muontomo.enhance --run runs/production --method guided|dip|pnp|all
"""

from .base import REGISTRY, Enhancer  # noqa: F401
from .context import EnhanceContext, load_context  # noqa: F401

# Importing the technique modules registers them in REGISTRY. Heavy deps (torch,
# skimage) are imported lazily inside enhance() so this import stays cheap and
# the core test suite never pulls them in transitively.
from . import guided  # noqa: F401,E402
from . import dip  # noqa: F401,E402
from . import pnp  # noqa: F401,E402
from . import artifacts  # noqa: F401,E402  (registers "clean")
