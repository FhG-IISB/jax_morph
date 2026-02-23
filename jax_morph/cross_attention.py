"""
FieldCrossAttention: JAX/Flax translation of src.utils.crossattention_fields.FieldCrossAttention

Collapses multiple fields into a single token per patch via a learned query
and cross-attention mechanism. Uses nn.MultiHeadDotProductAttention internally
to match the PyTorch nn.MultiheadAttention weight layout.

Weight mapping from PyTorch nn.MultiheadAttention:
- in_proj_weight (3*E, E) -> split into q_proj.kernel, k_proj.kernel, v_proj.kernel
- in_proj_bias (3*E,) -> split into q_proj.bias, k_proj.bias, v_proj.bias
- out_proj.weight (E, E) -> out_proj.kernel (transposed)
- out_proj.bias (E,) -> out_proj.bias
"""

import jax.numpy as jnp
import flax.linen as nn


class FieldCrossAttention(nn.Module):
    """
    Collapses F fields into 1 token via cross-attention with a learned query.

    Mirrors PyTorch ``FieldCrossAttention(patch_dim, heads, dropout)``.

    Attributes:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout:   Attention dropout rate.
    """

    embed_dim: int
    num_heads: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (Bn, F, embed_dim) — F field tokens per patch.
            deterministic: If True, disable dropout.

        Returns:
            (Bn, embed_dim) — single fused token per patch.
        """
        Bn, F, E = x.shape

        # Learned query: (1, 1, E), broadcast to (Bn, 1, E)
        q = self.param("q", nn.initializers.normal(stddev=1.0), (1, 1, E))
        q = jnp.broadcast_to(q, (Bn, 1, E))

        # Cross-attention: query attends to field tokens
        # We implement MHA manually to match PyTorch nn.MultiheadAttention
        # weight layout exactly (in_proj_weight, in_proj_bias, out_proj)
        head_dim = E // self.num_heads

        # Q projection (from learned query)
        q_proj = nn.DenseGeneral(features=(self.num_heads, head_dim), name="q_proj")(
            q
        )  # (Bn, 1, heads, head_dim)

        # K, V projections (from field tokens)
        k_proj = nn.DenseGeneral(features=(self.num_heads, head_dim), name="k_proj")(
            x
        )  # (Bn, F, heads, head_dim)

        v_proj = nn.DenseGeneral(features=(self.num_heads, head_dim), name="v_proj")(
            x
        )  # (Bn, F, heads, head_dim)

        # Scaled dot-product attention
        # (Bn, heads, 1, head_dim) @ (Bn, heads, head_dim, F) -> (Bn, heads, 1, F)
        scale = head_dim**-0.5
        q_proj = q_proj.transpose(0, 2, 1, 3)  # (Bn, heads, 1, head_dim)
        k_proj = k_proj.transpose(0, 2, 1, 3)  # (Bn, heads, F, head_dim)
        v_proj = v_proj.transpose(0, 2, 1, 3)  # (Bn, heads, F, head_dim)

        attn_weights = jnp.matmul(q_proj, k_proj.transpose(0, 1, 3, 2)) * scale
        attn_weights = nn.softmax(attn_weights, axis=-1)

        if not deterministic and self.dropout > 0.0:
            attn_weights = nn.Dropout(rate=self.dropout)(
                attn_weights, deterministic=False
            )

        # (Bn, heads, 1, F) @ (Bn, heads, F, head_dim) -> (Bn, heads, 1, head_dim)
        out = jnp.matmul(attn_weights, v_proj)

        # Merge heads: (Bn, 1, E)
        out = out.transpose(0, 2, 1, 3).reshape(Bn, 1, E)

        # Output projection
        out = nn.Dense(E, name="out_proj")(out)

        # Squeeze sequence dim: (Bn, E)
        return out.squeeze(1)
