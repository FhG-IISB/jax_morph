"""
jax_morph: JAX/Flax translation of the MORPH PDE foundation model.

A 1-to-1 translation of the MORPH model architecture from PyTorch to JAX/Flax,
maintaining exact weight compatibility for pretrained checkpoint conversion.

Reference:
    Rautela et al., "MORPH: PDE Foundation Models with Arbitrary Data Modality" (2025)
    https://arxiv.org/abs/2509.21670
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_morph")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from jax_morph.model import ViT3DRegression
from jax_morph.convert_weights import (
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
)
from jax_morph.configs import (
    MORPH_CONFIGS,
    CHECKPOINT_NAMES,
    HF_REPO_ID,
    morph_Ti,
    morph_S,
    morph_M,
    morph_L,
)

__all__ = [
    "__version__",
    "ViT3DRegression",
    "load_pytorch_state_dict",
    "convert_pytorch_to_jax_params",
    "MORPH_CONFIGS",
    "CHECKPOINT_NAMES",
    "HF_REPO_ID",
    "morph_Ti",
    "morph_S",
    "morph_M",
    "morph_L",
]
