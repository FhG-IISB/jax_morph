"""
MORPH model configurations and convenience constructors.

Each config maps to the official MORPH model variants from
``config/argument_parser.py`` in the MORPH repository.

Format: ``{conv_filter, dim, depth, heads, mlp_dim, max_ar, model_size}``.
"""

from typing import Any, Dict

#: Model configurations keyed by variant name.
MORPH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Ti": dict(conv_filter=8, dim=256,  depth=4,  heads=4,  mlp_dim=1024, max_ar=1,  model_size="Ti"),
    "S":  dict(conv_filter=8, dim=512,  depth=4,  heads=8,  mlp_dim=2048, max_ar=1,  model_size="S"),
    "M":  dict(conv_filter=8, dim=768,  depth=8,  heads=12, mlp_dim=3072, max_ar=1,  model_size="M"),
    "L":  dict(conv_filter=8, dim=1024, depth=16, heads=16, mlp_dim=4096, max_ar=16, model_size="L"),
}

#: HuggingFace checkpoint filenames.
CHECKPOINT_NAMES: Dict[str, str] = {
    "Ti": "morph-Ti-FM-max_ar1_ep225.pth",
    "S":  "morph-S-FM-max_ar1_ep225.pth",
    "M":  "morph-M-FM-max_ar1_ep290_latestbatch.pth",
    "L":  "morph-L-FM-max_ar16_ep189_latestbatch.pth",
}

#: HuggingFace repo ID for downloading pretrained checkpoints.
HF_REPO_ID = "mahindrautela/MORPH"


def _make_model(variant: str, **overrides):
    """
    Create a ``ViT3DRegression`` instance for a given variant.

    Args:
        variant: One of ``'Ti'``, ``'S'``, ``'M'``, ``'L'``.
        **overrides: Any ``ViT3DRegression`` attribute to override
            (e.g. ``dropout=0.0``, ``max_patches=512``).

    Returns:
        A ``ViT3DRegression`` module (un-initialised; call ``model.init(...)``).
    """
    from jax_morph.model import ViT3DRegression  # deferred to avoid circular import

    if variant not in MORPH_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(MORPH_CONFIGS)}"
        )

    cfg = MORPH_CONFIGS[variant]
    kwargs = dict(
        patch_size=8,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        heads_xa=32,
        mlp_dim=cfg["mlp_dim"],
        max_components=3,
        conv_filter=cfg["conv_filter"],
        max_ar=cfg["max_ar"],
        max_patches=4096,
        max_fields=3,
        dropout=0.0,
        emb_dropout=0.0,
        model_size=cfg["model_size"],
    )
    kwargs.update(overrides)
    return ViT3DRegression(**kwargs)


def morph_Ti(**overrides):
    """Create a MORPH-Ti model (9.9M parameters)."""
    return _make_model("Ti", **overrides)


def morph_S(**overrides):
    """Create a MORPH-S model (32.8M parameters)."""
    return _make_model("S", **overrides)


def morph_M(**overrides):
    """Create a MORPH-M model (125.6M parameters)."""
    return _make_model("M", **overrides)


def morph_L(**overrides):
    """Create a MORPH-L model (483.3M parameters)."""
    return _make_model("L", **overrides)
