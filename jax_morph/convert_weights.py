"""
Weight conversion utility: PyTorch -> JAX/Flax parameter mapping.

Maps every key in the PyTorch ``ViT3DRegression`` state_dict to the
corresponding leaf in the Flax parameter tree, applying transpositions
where needed.

Key mapping rules:
- ``nn.Conv3d.weight`` (out, in, D, H, W) -> ``nn.Conv.kernel`` (D, H, W, in, out)
- ``nn.Linear.weight`` (out, in) -> ``nn.Dense.kernel`` (in, out)
- ``nn.Linear.bias`` -> ``nn.Dense.bias``
- ``nn.LayerNorm.weight`` -> ``nn.LayerNorm.scale``
- ``nn.LayerNorm.bias`` -> ``nn.LayerNorm.bias``
- ``nn.MultiheadAttention.in_proj_weight`` (3*E, E) -> split Q/K/V kernels
- ``nn.MultiheadAttention.in_proj_bias`` (3*E,) -> split Q/K/V biases
- ``nn.MultiheadAttention.out_proj.weight/bias`` -> ``out_proj.kernel/bias``
- ``nn.Parameter`` -> ``param`` (as-is)
"""

from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to numpy, handling GPU tensors."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def load_pytorch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a PyTorch MORPH checkpoint and extract the model state dict.

    Handles the MORPH checkpoint format: ``{'model_state_dict': state_dict}``.
    Also strips ``'module.'`` prefix from DataParallel-wrapped checkpoints.
    """
    assert torch is not None, "PyTorch is required to load checkpoints"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # Strip 'module.' prefix from DataParallel
    if sd and next(iter(sd)).startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


def _conv3d_weight_to_jax(w: np.ndarray) -> np.ndarray:
    """Transpose Conv3d weight: (out, in, D, H, W) -> (D, H, W, in, out)."""
    return np.transpose(w, (2, 3, 4, 1, 0))


def _linear_weight_to_jax(w: np.ndarray) -> np.ndarray:
    """Transpose Linear weight: (out, in) -> (in, out)."""
    return w.T


def convert_pytorch_to_jax_params(
    pytorch_state_dict: Dict[str, Any],
    jax_params: Dict[str, Any],
    heads_xa: int = 32,
) -> Dict[str, Any]:
    """
    Convert a PyTorch ViT3DRegression state_dict to a Flax parameter dict.

    Takes the initialized JAX params tree as template to ensure the structure
    matches, then overwrites leaf values with converted PyTorch weights.

    Args:
        pytorch_state_dict: PyTorch model state_dict.
        jax_params: Initialized JAX parameter tree (from model.init()).
        heads_xa: Number of cross-attention heads (needed for weight reshape).

    Returns:
        ``{'params': nested_dict}`` with converted weights.
    """
    import jax.numpy as jnp
    from flax.core import freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict

    sd = {k: torch_to_numpy(v) for k, v in pytorch_state_dict.items()}

    # Build JAX param dict from PyTorch state_dict
    params = {}

    def _set(path: str, value: np.ndarray):
        keys = tuple(path.split("/"))
        params[keys] = jnp.array(value)

    # ── Patch Embedding ──
    _convert_patch_embedding(sd, _set, heads_xa=heads_xa)

    # ── Positional Encoding ──
    _convert_pos_encoding(sd, _set)

    # ── Transformer Blocks ──
    _convert_transformer_blocks(sd, _set)

    # ── Decoder ──
    _convert_decoder(sd, _set)

    # Merge into the JAX params tree
    jax_flat = flatten_dict(unfreeze(jax_params["params"]))

    converted_count = 0
    missing_in_pt = []
    for jax_key in jax_flat:
        if jax_key in params:
            pt_val = params[jax_key]
            jax_val = jax_flat[jax_key]
            if pt_val.shape != jax_val.shape:
                raise ValueError(
                    f"Shape mismatch at {'/'.join(jax_key)}: "
                    f"PyTorch={pt_val.shape}, JAX={jax_val.shape}"
                )
            jax_flat[jax_key] = pt_val
            converted_count += 1
        else:
            missing_in_pt.append("/".join(jax_key))

    if missing_in_pt:
        # Filter out LoRA params which are expected to be missing
        non_lora_missing = [
            k
            for k in missing_in_pt
            if "A" not in k.split("/")[-1] and "B" not in k.split("/")[-1]
        ]
        if non_lora_missing:
            print(f"WARNING: {len(non_lora_missing)} JAX params not found in PyTorch:")
            for k in non_lora_missing[:20]:
                print(f"  {k}")

    # Check for unconverted PyTorch keys
    pt_keys_used = set()
    _collect_used_pt_keys(sd, pt_keys_used)
    unused = set(sd.keys()) - pt_keys_used
    if unused:
        print(f"WARNING: {len(unused)} PyTorch keys not mapped:")
        for k in sorted(unused)[:20]:
            print(f"  {k}: {sd[k].shape}")

    print(f"Converted {converted_count}/{len(jax_flat)} JAX parameters")

    result = {"params": unflatten_dict(jax_flat)}
    return freeze(result)


def _collect_used_pt_keys(sd, used_set):
    """Track which PyTorch keys we expect to consume."""
    for k in sd:
        used_set.add(k)


def _convert_patch_embedding(sd, _set, heads_xa=32):
    """Convert patch_embedding.* keys."""
    prefix = "patch_embedding"

    # ConvOperator: input_proj + conv_stack
    _convert_conv_operator(
        sd, _set, f"{prefix}.conv_features", f"patch_embedding/conv_features"
    )

    # Linear projection
    key = f"{prefix}.projection.weight"
    if key in sd:
        _set("patch_embedding/projection/kernel", _linear_weight_to_jax(sd[key]))
    key = f"{prefix}.projection.bias"
    if key in sd:
        _set("patch_embedding/projection/bias", sd[key])

    # Field cross-attention
    _convert_field_cross_attention(
        sd,
        _set,
        f"{prefix}.field_attn",
        "patch_embedding/field_attn",
        heads_xa=heads_xa,
    )


def _convert_conv_operator(sd, _set, pt_prefix, jax_prefix):
    """Convert ConvOperator weights."""
    # input_proj (1x1x1 Conv3d, no bias)
    key = f"{pt_prefix}.input_proj.weight"
    if key in sd:
        _set(f"{jax_prefix}/input_proj/kernel", _conv3d_weight_to_jax(sd[key]))

    # conv_stack: Sequential of [Conv3d, LeakyReLU, Conv3d, LeakyReLU, ...]
    # PyTorch uses conv_stack.0.weight, conv_stack.2.weight, etc. (skipping activations)
    # JAX uses conv_stack_0, conv_stack_1, etc. (sequential conv indices)
    jax_conv_idx = 0
    pt_idx = 0
    while True:
        key = f"{pt_prefix}.conv_stack.{pt_idx}.weight"
        if key not in sd:
            break
        _set(
            f"{jax_prefix}/conv_stack_{jax_conv_idx}/kernel",
            _conv3d_weight_to_jax(sd[key]),
        )
        jax_conv_idx += 1
        pt_idx += 2  # Skip LeakyReLU (every other module in Sequential)


def _convert_field_cross_attention(sd, _set, pt_prefix, jax_prefix, heads_xa=32):
    """Convert FieldCrossAttention (nn.MultiheadAttention + learned query)."""
    # Learned query parameter
    key = f"{pt_prefix}.q"
    if key in sd:
        _set(f"{jax_prefix}/q", sd[key])

    # nn.MultiheadAttention in_proj_weight: (3*E, E) -> split into Q, K, V
    # Then reshape for DenseGeneral(features=(num_heads, head_dim)):
    #   kernel shape = (embed_dim, num_heads, head_dim)
    key = f"{pt_prefix}.attn.in_proj_weight"
    if key in sd:
        w = sd[key]  # (3*E, E)
        E = w.shape[0] // 3
        head_dim = E // heads_xa
        wq, wk, wv = w[:E], w[E : 2 * E], w[2 * E :]
        # PyTorch: F.linear(x, w, b) = x @ w.T + b
        # Flax DenseGeneral kernel: (E, heads, head_dim)
        _set(f"{jax_prefix}/q_proj/kernel", wq.T.reshape(E, heads_xa, head_dim))
        _set(f"{jax_prefix}/k_proj/kernel", wk.T.reshape(E, heads_xa, head_dim))
        _set(f"{jax_prefix}/v_proj/kernel", wv.T.reshape(E, heads_xa, head_dim))

    key = f"{pt_prefix}.attn.in_proj_bias"
    if key in sd:
        b = sd[key]  # (3*E,)
        E = b.shape[0] // 3
        head_dim = E // heads_xa
        bq, bk, bv = b[:E], b[E : 2 * E], b[2 * E :]
        _set(f"{jax_prefix}/q_proj/bias", bq.reshape(heads_xa, head_dim))
        _set(f"{jax_prefix}/k_proj/bias", bk.reshape(heads_xa, head_dim))
        _set(f"{jax_prefix}/v_proj/bias", bv.reshape(heads_xa, head_dim))

    # out_proj (regular Dense)
    key = f"{pt_prefix}.attn.out_proj.weight"
    if key in sd:
        _set(f"{jax_prefix}/out_proj/kernel", _linear_weight_to_jax(sd[key]))
    key = f"{pt_prefix}.attn.out_proj.bias"
    if key in sd:
        _set(f"{jax_prefix}/out_proj/bias", sd[key])


def _convert_pos_encoding(sd, _set):
    """Convert positional encoding weights."""
    key = "pos_encoding.pos_embedding"
    if key in sd:
        _set("pos_encoding/pos_embedding", sd[key])


def _convert_transformer_blocks(sd, _set):
    """Convert transformer_blocks.{i}.* keys."""
    i = 0
    while True:
        pt_prefix = f"transformer_blocks.{i}"
        jax_prefix = f"transformer_blocks_{i}"

        # Check if this block exists
        if f"{pt_prefix}.norm1.weight" not in sd:
            break

        # norm1
        _set(f"{jax_prefix}/norm1/scale", sd[f"{pt_prefix}.norm1.weight"])
        _set(f"{jax_prefix}/norm1/bias", sd[f"{pt_prefix}.norm1.bias"])

        # norm2
        _set(f"{jax_prefix}/norm2/scale", sd[f"{pt_prefix}.norm2.weight"])
        _set(f"{jax_prefix}/norm2/bias", sd[f"{pt_prefix}.norm2.bias"])

        # Axial attention: attn_t, attn_d, attn_h, attn_w
        for axis in ["t", "d", "h", "w"]:
            _convert_lora_mha(
                sd,
                _set,
                f"{pt_prefix}.axial_attn.attn_{axis}",
                f"{jax_prefix}/axial_attn/attn_{axis}",
            )

        # MLP: two LoRALinear layers
        # PyTorch: mlp.0 (LoRALinear), mlp.1 (GELU), mlp.2 (Dropout),
        #          mlp.3 (LoRALinear), mlp.4 (Dropout)
        _convert_lora_linear(sd, _set, f"{pt_prefix}.mlp.0", f"{jax_prefix}/mlp_0")
        _convert_lora_linear(sd, _set, f"{pt_prefix}.mlp.3", f"{jax_prefix}/mlp_1")

        i += 1


def _convert_lora_mha(sd, _set, pt_prefix, jax_prefix):
    """Convert LoRAMHA weights (Q/K/V/O projections)."""
    for proj in ["q", "k", "v", "o"]:
        _convert_lora_linear(sd, _set, f"{pt_prefix}.{proj}", f"{jax_prefix}/{proj}")


def _convert_lora_linear(sd, _set, pt_prefix, jax_prefix):
    """Convert LoRALinear weights (base linear + optional LoRA A/B)."""
    # Base linear
    key = f"{pt_prefix}.base.weight"
    if key in sd:
        _set(f"{jax_prefix}/base/kernel", _linear_weight_to_jax(sd[key]))
    key = f"{pt_prefix}.base.bias"
    if key in sd:
        _set(f"{jax_prefix}/base/bias", sd[key])

    # LoRA params (may not exist if rank=0)
    key = f"{pt_prefix}.A"
    if key in sd:
        # PyTorch A: (rank, in_features) -> JAX: (in_features, rank)
        _set(f"{jax_prefix}/A", sd[key].T)
    key = f"{pt_prefix}.B"
    if key in sd:
        # PyTorch B: (out_features, rank) -> JAX: (rank, out_features)
        _set(f"{jax_prefix}/B", sd[key].T)


def _convert_decoder(sd, _set):
    """Convert decoder weights."""
    prefix = "decoder"

    # LayerNorm
    key = f"{prefix}.norm.weight"
    if key in sd:
        _set("decoder/norm/scale", sd[key])
    key = f"{prefix}.norm.bias"
    if key in sd:
        _set("decoder/norm/bias", sd[key])

    # Linear
    key = f"{prefix}.linear.weight"
    if key in sd:
        _set("decoder/linear/kernel", _linear_weight_to_jax(sd[key]))
    key = f"{prefix}.linear.bias"
    if key in sd:
        _set("decoder/linear/bias", sd[key])
