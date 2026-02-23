"""
patchify_3d: JAX translation of src.utils.patchify_3d.custom_patchify_3d

Reshapes a 5D tensor (B, D, H, W, C) into patches of shape
(B, num_patches, patch_vol * C), matching the PyTorch implementation exactly.

Note: This is a pure function, not an nn.Module, since it has no learnable params.
"""

import jax.numpy as jnp
from typing import Tuple, Union


def custom_patchify_3d(
    x: jnp.ndarray,
    patch_size: Union[int, Tuple[int, int, int]],
) -> jnp.ndarray:
    """
    Flexible 3D patch extraction (channels-last).

    Args:
        x: (B, D, H, W, C) — channels-last input.
        patch_size: int or (pD, pH, pW).

    Returns:
        (B, num_patches, C * pD * pH * pW) — flattened patches.
    """
    B, D, H, W, C = x.shape

    if isinstance(patch_size, (tuple, list)):
        pz, py, px = patch_size
    else:
        pz = py = px = patch_size

    # Adjust for small axes
    pz = pz if D >= pz else D
    py = py if H >= py else H
    px = px if W >= px else W

    nz, ny, nx = D // pz, H // py, W // px
    assert (
        D % pz == 0 and H % py == 0 and W % px == 0
    ), f"Dimensions {(D, H, W)} must be divisible by patches {(pz, py, px)}"

    # (B, D, H, W, C) -> (B, nz, pz, ny, py, nx, px, C)
    x = x.reshape(B, nz, pz, ny, py, nx, px, C)

    # -> (B, nz, ny, nx, C, pz, py, px)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)

    # -> (B, num_patches, C * pz * py * px)
    x = x.reshape(B, nz * ny * nx, C * pz * py * px)

    return x
