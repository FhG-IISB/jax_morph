"""
Positional encodings: JAX/Flax translations of

- src.utils.positional_encoding_spatiotemporal_li_slice.PositionalEncoding_SLin_TSlice
- src.utils.positional_encoding_spatiotemporal_bilinear.PositionalEncoding_STBilinear

Both use a learned embedding table of shape (1, max_ar, max_patches, dim)
and interpolate to the actual (t, n_patches) at forward time.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


def _interpolate_linear_1d(x: jnp.ndarray, out_size: int) -> jnp.ndarray:
    """
    1D linear interpolation matching PyTorch's
    ``F.interpolate(x, size=out_size, mode='linear', align_corners=False)``.

    Args:
        x: (..., in_size) — input array.
        out_size: Target size for the last dimension.

    Returns:
        (..., out_size) — interpolated array.
    """
    in_size = x.shape[-1]
    if in_size == out_size:
        return x

    # PyTorch align_corners=False coordinate mapping
    out_idx = jnp.arange(out_size, dtype=jnp.float32)
    src = (out_idx + 0.5) * in_size / out_size - 0.5

    src_low = jnp.floor(src).astype(jnp.int32)
    src_high = src_low + 1

    src_low = jnp.clip(src_low, 0, in_size - 1)
    src_high = jnp.clip(src_high, 0, in_size - 1)

    w = src - jnp.floor(src)

    return x[..., src_low] * (1 - w) + x[..., src_high] * w


def _interpolate_bilinear_2d(
    x: jnp.ndarray,
    out_h: int,
    out_w: int,
    antialias: bool = False,
) -> jnp.ndarray:
    """
    2D bilinear interpolation matching PyTorch's
    ``F.interpolate(x, size=(out_h, out_w), mode='bilinear',
                    align_corners=False, antialias=antialias)``.

    Uses separable 1-D passes (width then height). When ``antialias=True``
    and a dimension is being down-sampled, the triangle (linear) kernel is
    widened by ``1/scale`` so that every input pixel contributes.

    Args:
        x: (..., in_h, in_w) — input array.
        out_h: Target height.
        out_w: Target width.
        antialias: Whether to apply antialiased filtering when downsampling.

    Returns:
        (..., out_h, out_w) — interpolated array.
    """
    import numpy as _np  # for static int computation

    in_h = x.shape[-2]
    in_w = x.shape[-1]

    def _interp_1d(arr, in_size, out_size):
        """1-D antialias-aware linear interpolation along the last axis."""
        if in_size == out_size:
            return arr

        scale = out_size / in_size
        out_idx = jnp.arange(out_size, dtype=jnp.float32)
        center = (out_idx + 0.5) * in_size / out_size - 0.5  # (out_size,)

        if antialias and scale < 1.0:
            kernel_width = 1.0 / scale
        else:
            kernel_width = 1.0

        support = int(_np.ceil(kernel_width))
        offsets = jnp.arange(-support, support + 1, dtype=jnp.float32)

        # (out_size, n_taps)
        src_idx = jnp.floor(center)[:, None] + offsets[None, :]
        dist = jnp.abs(center[:, None] - src_idx) / kernel_width
        weights = jnp.maximum(1.0 - dist, 0.0)

        # Zero out weights for out-of-bounds positions, then renormalize
        valid = (src_idx >= 0) & (src_idx < in_size)
        weights = weights * valid
        weights = weights / weights.sum(axis=-1, keepdims=True)

        src_idx = jnp.clip(src_idx.astype(jnp.int32), 0, in_size - 1)
        gathered = arr[..., src_idx]  # (..., out_size, n_taps)
        return (gathered * weights).sum(axis=-1)  # (..., out_size)

    # Separable: width first, then height (via transpose)
    x = _interp_1d(x, in_w, out_w)  # (..., in_h, out_w)
    x = x.swapaxes(-2, -1)  # (..., out_w, in_h)
    x = _interp_1d(x, in_h, out_h)  # (..., out_w, out_h)
    x = x.swapaxes(-2, -1)  # (..., out_h, out_w)
    return x


class PositionalEncodingSLinTSlice(nn.Module):
    """
    Learned positional embedding with time-slice + spatial linear interpolation.

    Used for Ti/S/M variants (max_ar <= 1 or model_size != 'L').

    Mirrors PyTorch ``PositionalEncoding_SLin_TSlice(max_ar, max_patches, dim)``.

    Attributes:
        max_ar:      Maximum auto-regressive timesteps.
        max_patches: Maximum number of spatial patches.
        dim:         Embedding dimension.
        emb_dropout: Dropout rate on positional encoding.
    """

    max_ar: int = 5
    max_patches: int = 512
    dim: int = 256
    emb_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, t, n_patches, dim) — input tensor (used for shape only).
            deterministic: If True, disable dropout.

        Returns:
            (B, t, n_patches, dim) — positional encoding.
        """
        B, t, n, D = x.shape

        # Learned table: (1, max_ar, max_patches, dim)
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, self.max_ar, self.max_patches, self.dim),
        )

        # Slice time: (1, t, max_patches, dim)
        pe = pos_embedding[:, :t, :, :]

        # Reshape for 1D interpolation over patch axis
        # (1, dim*t, max_patches)
        pe = pe.transpose(0, 3, 1, 2).reshape(1, D * t, -1)

        # Linear interpolate to n_patches
        # Match PyTorch F.interpolate(pe, size=n, mode='linear', align_corners=False)
        # Shape: (1, D*t, max_patches) -> (1, D*t, n)
        pe = _interpolate_linear_1d(pe, n)

        # Reshape back: (1, t, n, dim)
        pe = pe.reshape(1, D, t, n).transpose(0, 2, 3, 1)

        # Broadcast to batch
        pe = jnp.broadcast_to(pe, (B, t, n, D))

        # Dropout
        if not deterministic and self.emb_dropout > 0.0:
            pe = nn.Dropout(rate=self.emb_dropout)(pe, deterministic=False)

        return pe


class PositionalEncodingSTBilinear(nn.Module):
    """
    Learned positional embedding with bilinear interpolation over (time, patches).

    Used for the L variant (model_size == 'L' and max_ar > 1).

    Mirrors PyTorch ``PositionalEncoding_STBilinear(max_ar, max_patches, dim)``.

    Attributes:
        max_ar:      Maximum auto-regressive timesteps.
        max_patches: Maximum number of spatial patches.
        dim:         Embedding dimension.
        emb_dropout: Dropout rate on positional encoding.
    """

    max_ar: int = 16
    max_patches: int = 4096
    dim: int = 1024
    emb_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, t, n_patches, dim) — input tensor (used for shape only).
            deterministic: If True, disable dropout.

        Returns:
            (B, t, n_patches, dim) — positional encoding.
        """
        B, t, n, D = x.shape

        # Learned table: (1, max_ar, max_patches, dim)
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, self.max_ar, self.max_patches, self.dim),
        )

        # Reshape for bilinear interpolation: (1, dim, max_ar, max_patches)
        pe = pos_embedding.transpose(0, 3, 1, 2)

        # Bilinear interpolate (time, patches) from (max_ar, max_patches) -> (t, n)
        # Match PyTorch F.interpolate(pe, size=(t,n), mode='bilinear',
        #                             align_corners=False, antialias=True)
        pe = _interpolate_bilinear_2d(pe, t, n, antialias=True)

        # Back to (1, t, n, dim)
        pe = pe.transpose(0, 2, 3, 1)

        # Broadcast to batch
        pe = jnp.broadcast_to(pe, (B, t, n, D))

        # Dropout
        if not deterministic and self.emb_dropout > 0.0:
            pe = nn.Dropout(rate=self.emb_dropout)(pe, deterministic=False)

        return pe
