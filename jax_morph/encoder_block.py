"""
EncoderBlock: JAX/Flax translation of
    src.utils.transformer_encoder_axialattention_3dspacetime_lora.EncoderBlock

Pre-norm transformer block:
    x = x + AxialAttention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

MLP uses LoRALinear layers (dormant when rank=0).
"""

from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn

from jax_morph.axial_attention import AxialAttention3DSpaceTime
from jax_morph.attention import LoRALinear


class EncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block with axial attention.

    Mirrors PyTorch ``EncoderBlock(dim, heads, mlp_dim, dropout, ...)``.

    Attributes:
        dim:      Embedding dimension.
        heads:    Number of attention heads.
        mlp_dim:  Hidden dimension of the MLP.
        dropout:  Dropout rate.
        lora_r_attn: LoRA rank for attention projections.
        lora_r_mlp:  LoRA rank for MLP projections.
        lora_alpha:  LoRA alpha.
        lora_p:      LoRA dropout.
    """

    dim: int
    heads: int
    mlp_dim: int
    dropout: float = 0.0
    lora_r_attn: int = 0
    lora_r_mlp: int = 0
    lora_alpha: int = None
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
            x: (B, t, N, dim) — N = D*H*W.
            grid_size: (D_patches, H_patches, W_patches).
            deterministic: If True, disable dropout.

        Returns:
            (B, t, N, dim).
        """
        # -- Axial attention block --
        residual = x
        x = nn.LayerNorm(epsilon=1e-5, name="norm1")(x)
        x = AxialAttention3DSpaceTime(
            dim=self.dim,
            heads=self.heads,
            dropout=self.dropout,
            rank=self.lora_r_attn,
            alpha=self.lora_alpha,
            lora_p=self.lora_p,
            name="axial_attn",
        )(x, grid_size, deterministic=deterministic)
        x = residual + x

        # -- MLP block --
        residual = x
        x = nn.LayerNorm(epsilon=1e-5, name="norm2")(x)

        x = LoRALinear(
            self.mlp_dim,
            use_bias=True,
            rank=self.lora_r_mlp,
            alpha=self.lora_alpha,
            lora_p=self.lora_p,
            name="mlp_0",
        )(x, deterministic=deterministic)
        x = nn.gelu(x, approximate=False)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=False)

        x = LoRALinear(
            self.dim,
            use_bias=True,
            rank=self.lora_r_mlp,
            alpha=self.lora_alpha,
            lora_p=self.lora_p,
            name="mlp_1",
        )(x, deterministic=deterministic)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=False)

        x = residual + x
        return x
