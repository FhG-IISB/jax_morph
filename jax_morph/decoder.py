"""
SimpleDecoder: JAX/Flax translation of src.utils.simple_decoder.SimpleDecoder

LayerNorm + Linear head that maps from embed_dim to (fields * components * patch_vol).
Output is sliced to the actual out_ch when it's less than max_out_ch.
"""

import jax.numpy as jnp
import flax.linen as nn


class SimpleDecoder(nn.Module):
    """
    Decoder: LayerNorm -> Linear -> optional slice.

    Mirrors PyTorch ``SimpleDecoder(dim, max_out_ch)``.

    Attributes:
        dim:         Input embedding dimension.
        max_out_ch:  Maximum output channels (fields * max_components * max_patch_vol).
    """

    dim: int
    max_out_ch: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        fields: int,
        components: int,
        patch_vol: int,
    ) -> jnp.ndarray:
        """
        Args:
            x: (B, t, n, dim)
            fields: Number of output fields.
            components: Number of components.
            patch_vol: Volume of one patch (pD * pH * pW).

        Returns:
            (B, t, n, out_ch) where out_ch = fields * components * patch_vol.
        """
        out_ch = fields * patch_vol * components

        x = nn.LayerNorm(epsilon=1e-5, name="norm")(x)
        x = nn.Dense(self.max_out_ch, name="linear")(x)

        if out_ch < self.max_out_ch:
            x = x[..., :out_ch]

        return x
