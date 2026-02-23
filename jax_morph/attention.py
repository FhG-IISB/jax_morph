"""
ScaledDotProductAttention & LoRALinear & LoRAMHA: JAX/Flax translations of

- src.utils.sdpa.ScaledDotProductAttention
- src.utils.lora_linear.LoRALinear
- src.utils.lora_mha.LoRAMHA

For pretrained foundation model weights, LoRA rank=0, so the LoRA path
is dormant and only the base Linear weights are used. LoRA support is
included for fine-tuning compatibility.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn


def scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
) -> jnp.ndarray:
    """
    Scaled dot-product attention.

    Args:
        q, k, v: (B, h, L, d)
        dropout_rate: Attention dropout rate.
        deterministic: If True, disable dropout.

    Returns:
        (B, h, L, d)
    """
    B, H, L, d = q.shape
    if L == 1:
        return v  # attention over one token is identity

    scale = 1.0 / math.sqrt(d)
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Stable softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = nn.softmax(scores, axis=-1)

    if not deterministic and dropout_rate > 0.0:
        attn = nn.Dropout(rate=dropout_rate)(attn, deterministic=False)

    return jnp.matmul(attn, v)


class LoRALinear(nn.Module):
    """
    Linear layer with optional LoRA adapters.

    y = x @ W^T + (alpha/r) * (x @ A^T) @ B^T + bias

    If rank=0, behaves like a plain Dense layer (no LoRA params).

    Mirrors PyTorch ``LoRALinear(in_features, out_features, bias, rank, alpha, p)``.

    Attributes:
        features:  Output dimension.
        use_bias:  Whether to include a bias term.
        rank:      LoRA rank (0 = no LoRA).
        alpha:     LoRA scaling alpha (defaults to 2*rank if None).
        lora_p:    LoRA dropout probability.
    """

    features: int
    use_bias: bool = True
    rank: int = 0
    alpha: int = None
    lora_p: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Base linear
        y = nn.Dense(self.features, use_bias=self.use_bias, name="base")(x)

        # LoRA path (only if rank > 0)
        if self.rank > 0:
            alpha = self.alpha if self.alpha is not None else 2 * self.rank
            scaling = alpha / self.rank
            in_features = x.shape[-1]

            A = self.param(
                "A",
                nn.initializers.kaiming_uniform(),
                (in_features, self.rank),
            )
            B = self.param(
                "B",
                nn.initializers.zeros,
                (self.rank, self.features),
            )

            lora_in = x
            if not deterministic and self.lora_p > 0.0:
                lora_in = nn.Dropout(rate=self.lora_p)(lora_in, deterministic=False)

            # (x @ A) @ B
            upd = jnp.dot(jnp.dot(lora_in, A), B)
            y = y + scaling * upd

        return y


class LoRAMHA(nn.Module):
    """
    Multi-head attention with optional LoRA on Q/K/V/O projections.

    Mirrors PyTorch ``LoRAMHA(embed_dim, num_heads, dropout, rank, alpha, p)``.

    Attributes:
        embed_dim:   Embedding dimension.
        num_heads:   Number of attention heads.
        dropout:     Attention dropout rate.
        rank:        LoRA rank for all projections (0 = no LoRA).
        alpha:       LoRA alpha.
        lora_p:      LoRA dropout.
    """

    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    rank: int = 0
    alpha: int = None
    lora_p: float = 0.0

    @nn.compact
    def __call__(
        self,
        q_in: jnp.ndarray,
        k_in: jnp.ndarray,
        v_in: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            q_in, k_in, v_in: (B, L, C) — query, key, value inputs.
            deterministic: If True, disable dropout.

        Returns:
            (B, L, C) — attention output.
        """
        B, L, C = q_in.shape
        head_dim = self.embed_dim // self.num_heads

        # Project Q, K, V
        q = LoRALinear(
            self.embed_dim,
            use_bias=True,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="q",
        )(q_in, deterministic=deterministic)

        k = LoRALinear(
            self.embed_dim,
            use_bias=True,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="k",
        )(k_in, deterministic=deterministic)

        v = LoRALinear(
            self.embed_dim,
            use_bias=True,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="v",
        )(v_in, deterministic=deterministic)

        # Reshape to multi-head: (B, L, heads, head_dim) -> (B, heads, L, head_dim)
        q = q.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # SDPA
        y = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )

        # Merge heads
        y = y.transpose(0, 2, 1, 3).reshape(B, L, C)

        # Output projection
        y = LoRALinear(
            self.embed_dim,
            use_bias=True,
            rank=self.rank,
            alpha=self.alpha,
            lora_p=self.lora_p,
            name="o",
        )(y, deterministic=deterministic)

        return y
