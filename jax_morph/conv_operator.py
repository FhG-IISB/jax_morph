"""
ConvOperator: JAX/Flax translation of src.utils.convolutional_operator.ConvOperator

A convolutional feature extractor that pads input channels to max_in_ch,
projects down with a 1x1x1 conv, then applies a doubling stack of 3x3x3 convs
with LeakyReLU activations.

Weight-compatible with the PyTorch version when loaded via convert_weights.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class ConvOperator(nn.Module):
    """
    Conv feature extractor: pad channels -> 1x1 proj -> doubling 3x3 stack.

    Mirrors PyTorch ``ConvOperator(max_in_ch, conv_filter, hidden_dim=8)``.

    Attributes:
        max_in_ch:   Maximum number of input channels (e.g. 3).
        conv_filter: Final number of output feature maps.
        hidden_dim:  Intermediate channel count after 1x1 conv (default 8).
    """

    max_in_ch: int = 3
    conv_filter: int = 8
    hidden_dim: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, D, H, W, in_ch) — channels-last input.

        Returns:
            (B, D, H, W, conv_filter) — channels-last output.
        """
        in_ch = x.shape[-1]

        # Pad to max_in_ch if needed
        if in_ch < self.max_in_ch:
            pad_sz = self.max_in_ch - in_ch
            pad = jnp.zeros((*x.shape[:-1], pad_sz), dtype=x.dtype)
            x = jnp.concatenate([x, pad], axis=-1)

        # 1x1x1 projection: max_in_ch -> hidden_dim
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(1, 1, 1),
            use_bias=False,
            name="input_proj",
        )(x)

        # Doubling 3x3x3 conv stack
        prev = self.hidden_dim
        i = 0
        while prev < self.conv_filter:
            nxt = min(prev * 2, self.conv_filter)
            x = nn.Conv(
                features=nxt,
                kernel_size=(3, 3, 3),
                padding="SAME",
                use_bias=False,
                name=f"conv_stack_{i}",
            )(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            prev = nxt
            i += 1

        # Final conv
        x = nn.Conv(
            features=self.conv_filter,
            kernel_size=(3, 3, 3),
            padding="SAME",
            use_bias=False,
            name=f"conv_stack_{i}",
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        return x
