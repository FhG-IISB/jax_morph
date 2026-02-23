"""
ViT3DRegression: JAX/Flax translation of
    src.utils.vit_conv_xatt_axialatt2.ViT3DRegression

Top-level MORPH model that composes:
    Patch Embedding -> Positional Encoding -> Transformer Blocks -> Decoder

Supports both training (deterministic=False) and inference (deterministic=True).

Model variants (filters, dim, depth, heads, mlp_dim):
    Ti: (8,  256,   4,  4, 1024)  — max_ar=1
    S:  (8,  512,   8,  4, 2048)  — max_ar=1
    M:  (8,  768,  12,  8, 3072)  — max_ar=1
    L:  (8, 1024,  16, 16, 4096)  — max_ar=16, uses bilinear pos encoding

Reference:
    Rautela et al., "MORPH: PDE Foundation Models with Arbitrary Data Modality" (2025)
"""

from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn

from jax_morph.patch_embedding import HybridPatchEmbedding3D
from jax_morph.positional_encoding import (
    PositionalEncodingSLinTSlice,
    PositionalEncodingSTBilinear,
)
from jax_morph.encoder_block import EncoderBlock
from jax_morph.decoder import SimpleDecoder


class ViT3DRegression(nn.Module):
    """
    MORPH Vision Transformer for 3D PDE regression.

    Input:  (B, t, F, C, D, H, W)
    Output: (B, F, C, D, H, W)  — prediction at the last timestep.

    Mirrors PyTorch ``ViT3DRegression(patch_size, dim, depth, heads, ...)``.

    Attributes:
        patch_size:      Patch size (int or tuple).
        dim:             Transformer embedding dimension.
        depth:           Number of transformer blocks.
        heads:           Number of attention heads per block.
        heads_xa:        Number of heads for field cross-attention.
        mlp_dim:         MLP hidden dimension.
        max_components:  Maximum input components (channels).
        conv_filter:     Conv feature extractor output channels.
        max_ar:          Maximum auto-regressive timesteps.
        max_patches:     Maximum number of patches.
        max_fields:      Maximum number of fields.
        dropout:         Dropout rate.
        emb_dropout:     Embedding dropout rate.
        lora_r_attn:     LoRA rank for attention (0 = dormant).
        lora_r_mlp:      LoRA rank for MLP (0 = dormant).
        lora_alpha:      LoRA alpha.
        lora_p:          LoRA dropout.
        model_size:      Model variant ('Ti', 'S', 'M', 'L').
    """

    patch_size: int = 8
    dim: int = 256
    depth: int = 4
    heads: int = 4
    heads_xa: int = 32
    mlp_dim: int = 1024
    max_components: int = 3
    conv_filter: int = 8
    max_ar: int = 1
    max_patches: int = 4096
    max_fields: int = 3
    dropout: float = 0.1
    emb_dropout: float = 0.1
    lora_r_attn: int = 0
    lora_r_mlp: int = 0
    lora_alpha: int = None
    lora_p: float = 0.0
    model_size: str = "Ti"

    def _get_patch_info(self, volume: Tuple[int, int, int]):
        D, H, W = volume
        pD, pH, pW = self._patch_tuple()
        pD = 1 if D == 1 else pD
        pH = 1 if H == 1 else pH
        pW = 1 if W == 1 else pW
        patch_sizes = (pD, pH, pW)
        assert (
            D % pD == 0 and H % pH == 0 and W % pW == 0
        ), "Each axis must be divisible by its patch_size"
        n_patches = (D // pD, H // pH, W // pW)
        return patch_sizes, n_patches

    def _patch_tuple(self):
        if isinstance(self.patch_size, (tuple, list)):
            return tuple(self.patch_size)
        return (self.patch_size, self.patch_size, self.patch_size)

    @nn.compact
    def __call__(
        self, vol: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.

        Args:
            vol: (B, t, F, C, D, H, W) — input volume.
            deterministic: If True, disable all dropout.

        Returns:
            Tuple of:
                enc: (B, t, n, dim)   — patch embedding output.
                z:   (B, t, n, dim)   — transformer output.
                x_last: (B, F, C, D, H, W) — decoded output at last timestep.
        """
        B, t, F, C, D, H, W = vol.shape
        pD, pH, pW = self._patch_tuple()
        max_patch_vol = pW**3
        max_decoder_out_ch = self.max_fields * self.max_components * max_patch_vol

        # 1) Patch embedding: (B, t, n, dim)
        x = HybridPatchEmbedding3D(
            patch_size=(pD, pH, pW),
            max_components=self.max_components,
            conv_filter=self.conv_filter,
            embed_dim=self.dim,
            heads_xa=self.heads_xa,
            name="patch_embedding",
        )(vol, deterministic=deterministic)
        enc = x

        # 2) Positional encoding
        if self.model_size == "L" and self.max_ar > 1:
            pe = PositionalEncodingSTBilinear(
                max_ar=self.max_ar,
                max_patches=self.max_patches,
                dim=self.dim,
                emb_dropout=self.emb_dropout,
                name="pos_encoding",
            )(x, deterministic=deterministic)
        else:
            pe = PositionalEncodingSLinTSlice(
                max_ar=self.max_ar,
                max_patches=self.max_patches,
                dim=self.dim,
                emb_dropout=self.emb_dropout,
                name="pos_encoding",
            )(x, deterministic=deterministic)

        x = x + pe
        if not deterministic and self.emb_dropout > 0.0:
            x = nn.Dropout(rate=self.emb_dropout)(x, deterministic=False)

        # 3) Transformer blocks
        (pD_actual, pH_actual, pW_actual), (D_p, H_p, W_p) = self._get_patch_info(
            (D, H, W)
        )
        grid_size = (D_p, H_p, W_p)
        patch_vol = pD_actual * pH_actual * pW_actual

        for i in range(self.depth):
            x = EncoderBlock(
                dim=self.dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                lora_r_attn=self.lora_r_attn,
                lora_r_mlp=self.lora_r_mlp,
                lora_alpha=self.lora_alpha,
                lora_p=self.lora_p,
                name=f"transformer_blocks_{i}",
            )(x, grid_size, deterministic=deterministic)
        z = x

        # 4) Decode: (B, t, n, out_ch)
        x = SimpleDecoder(
            dim=self.dim,
            max_out_ch=max_decoder_out_ch,
            name="decoder",
        )(x, F, C, patch_vol)

        # 5) Reshape to output volume
        b, t_out, n, cpd = x.shape
        assert cpd == F * C * patch_vol

        # Take last timestep
        x_last = x[:, -1, :, :]  # (b, n, F*C*patch_vol)

        # (b, n, F, C, pD, pH, pW)
        x_last = x_last.reshape(b, n, F, C, pD_actual, pH_actual, pW_actual)

        # (b, D_p, H_p, W_p, F, C, pD, pH, pW)
        x_last = x_last.reshape(b, D_p, H_p, W_p, F, C, pD_actual, pH_actual, pW_actual)

        # Reorder to (B, F, C, D_p, pD, H_p, pH, W_p, pW)
        x_last = x_last.transpose(0, 4, 5, 1, 6, 2, 7, 3, 8)

        # Final reshape: (B, F, C, D, H, W)
        x_last = x_last.reshape(b, F, C, D, H, W)

        return enc, z, x_last
