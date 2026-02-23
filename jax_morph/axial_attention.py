"""
AxialAttention3DSpaceTime: JAX/Flax translation of
    src.utils.axial_attention_3dspacetime_2_lora.AxialAttention3DSpaceTime

Applies multi-head attention independently along each axis (time, depth,
height, width), then sums the results. This factored attention avoids the
O(n^4) cost of full 3D+T attention.

When t > 1, time-axis attention is also computed; otherwise only spatial
axes are used.
"""

from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn

from jax_morph.attention import LoRAMHA


class AxialAttention3DSpaceTime(nn.Module):
    """
    Factored axial attention over time + 3 spatial axes.

    Mirrors PyTorch ``AxialAttention3DSpaceTime(dim, heads, dropout, ...)``.

    Attributes:
        dim:     Embedding dimension.
        heads:   Number of attention heads.
        dropout: Attention dropout rate.
        rank:    LoRA rank (0 = no LoRA).
        alpha:   LoRA alpha.
        lora_p:  LoRA dropout.
    """

    dim: int
    heads: int
    dropout: float = 0.0
    rank: int = 0
    alpha: int = None
    lora_p: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        grid_size: Tuple[int, int, int],
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: (B, t, N, features) where N = D*H*W.
            grid_size: (D_patches, H_patches, W_patches).
            deterministic: If True, disable dropout.

        Returns:
            (B, t, N, features)
        """
        B, t, N, features = x.shape
        D, H, W = grid_size

        # Reconstruct 3D grid: (B, t, D, H, W, features)
        x = x.reshape(B, t, D, H, W, features)

        # -- 1) Time-axis attention (always instantiate for weight compatibility) --
        # (B, D, H, W, t, features) -> (B*D*H*W, t, features)
        xt = x.transpose(0, 2, 3, 4, 1, 5).reshape(B * D * H * W, t, features)
        xt = LoRAMHA(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=self.dropout,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="attn_t",
        )(xt, xt, xt, deterministic=deterministic)
        # -> (B, t, D, H, W, features)
        xt = xt.reshape(B, D, H, W, t, features).transpose(0, 4, 1, 2, 3, 5)
        # Only add time residual when t > 1 (matches PyTorch behavior)
        if t > 1:
            x = x + xt

        # -- 2) Depth-axis attention --
        # (B, t, H, W, D, features) -> (B*t*H*W, D, features)
        xd = x.transpose(0, 1, 3, 4, 2, 5).reshape(B * t * H * W, D, features)
        xd = LoRAMHA(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=self.dropout,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="attn_d",
        )(xd, xd, xd, deterministic=deterministic)
        # -> (B, t, D, H, W, features)
        xd = xd.reshape(B, t, H, W, D, features).transpose(0, 1, 4, 2, 3, 5)

        # -- 3) Height-axis attention --
        # (B, t, D, W, H, features) -> (B*t*D*W, H, features)
        xh = x.transpose(0, 1, 2, 4, 3, 5).reshape(B * t * D * W, H, features)
        xh = LoRAMHA(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=self.dropout,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="attn_h",
        )(xh, xh, xh, deterministic=deterministic)
        # -> (B, t, D, H, W, features)
        xh = xh.reshape(B, t, D, W, H, features).transpose(0, 1, 2, 4, 3, 5)

        # -- 4) Width-axis attention --
        # (B*t*D*H, W, features)
        xw = x.reshape(B * t * D * H, W, features)
        xw = LoRAMHA(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=self.dropout,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="attn_w",
        )(xw, xw, xw, deterministic=deterministic)
        # -> (B, t, D, H, W, features)
        xw = xw.reshape(B, t, D, H, W, features)

        # -- 5) Sum and flatten back --
        x_comb = x + xd + xh + xw  # (B, t, D, H, W, features)
        out = x_comb.reshape(B, t, D * H * W, features)
        return out
