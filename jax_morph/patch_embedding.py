"""
HybridPatchEmbedding3D: JAX/Flax translation of
    src.utils.embedding_conv_patch_xatt_project.HybridPatchEmbedding3D

Pipeline: Conv features -> Patchify -> Zero-pad -> Linear project -> Field cross-attention.

Takes (B, t, F, C, D, H, W) input and produces (B, t, n_patches, embed_dim).

Internally works channels-last for JAX conv efficiency:
    (B*t*F, D, H, W, C) through ConvOperator, then reshapes.
"""

from typing import Tuple, Union

import jax.numpy as jnp
import flax.linen as nn

from jax_morph.conv_operator import ConvOperator
from jax_morph.patchify import custom_patchify_3d
from jax_morph.cross_attention import FieldCrossAttention


class HybridPatchEmbedding3D(nn.Module):
    """
    Hybrid patch embedding with conv features and field cross-attention.

    Mirrors PyTorch ``HybridPatchEmbedding3D(patch_size, max_components,
    conv_filter, embed_dim, heads_xa)``.

    Attributes:
        patch_size:      (pD, pH, pW) patch dimensions.
        max_components:  Maximum number of input components (channels).
        conv_filter:     Number of conv output feature maps.
        embed_dim:       Transformer embedding dimension.
        heads_xa:        Number of heads for field cross-attention.
    """

    patch_size: Tuple[int, int, int] = (8, 8, 8)
    max_components: int = 3
    conv_filter: int = 8
    embed_dim: int = 256
    heads_xa: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, t, F, C, D, H, W) — input volume.
            deterministic: If True, disable dropout.

        Returns:
            (B, t, n_patches, embed_dim).
        """
        B, t, F, C, D, H, W = x.shape
        pD, pH, pW = self.patch_size
        max_patch_vol = pW**3
        max_features = max_patch_vol * self.conv_filter

        # 1) Merge batch, time, fields for conv: (B*t*F, C, D, H, W)
        #    Convert to channels-last: (B*t*F, D, H, W, C)
        x = x.reshape(B * t * F, C, D, H, W)
        x = x.transpose(0, 2, 3, 4, 1)  # -> (B*t*F, D, H, W, C)

        # Apply conv features
        x = ConvOperator(
            max_in_ch=self.max_components,
            conv_filter=self.conv_filter,
            hidden_dim=8,
            name="conv_features",
        )(
            x
        )  # -> (B*t*F, D, H, W, conv_filter)

        # 2) Patchify: (B*t*F, n_patches, conv_filter * patch_vol)
        x = custom_patchify_3d(x, self.patch_size)
        n_patches = x.shape[1]
        features = x.shape[2]

        # Reshape to separate F: (B*t, F, n_patches, features)
        x = x.reshape(B * t, F, n_patches, features)

        # Bring patches to front: (B*t, n_patches, F, features)
        x = x.transpose(0, 2, 1, 3)

        # 3a) Reshape and zero-pad features: (B*t*n, F, features)
        x = x.reshape(-1, F, features)

        if features < max_features:
            pad_amt = max_features - features
            pad = jnp.zeros((x.shape[0], x.shape[1], pad_amt), dtype=x.dtype)
            x = jnp.concatenate([x, pad], axis=-1)

        # 3b) Project to embed_dim: (B*t*n, F, embed_dim)
        x = nn.Dense(self.embed_dim, name="projection")(x)

        # 4) Field cross-attention: (B*t*n, embed_dim)
        x = FieldCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.heads_xa,
            dropout=0.0,
            name="field_attn",
        )(x, deterministic=deterministic)

        # 5) Restore dims: (B, t, n_patches, embed_dim)
        x = x.reshape(B * t, n_patches, self.embed_dim)
        x = x.reshape(B, t, n_patches, self.embed_dim)

        return x
