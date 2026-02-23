"""
Convert pretrained MORPH PyTorch weights to JAX/Flax msgpack format.

Usage:
    python scripts/convert.py --input morph-Ti-FM-max_ar1_ep225.pth --output morph-Ti.msgpack --model-size Ti

Loads the PyTorch checkpoint, initializes the JAX model, maps all parameters,
validates the mapping, and saves as msgpack.
"""

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes, from_bytes

from jax_morph import ViT3DRegression, load_pytorch_state_dict, convert_pytorch_to_jax_params
from jax_morph.configs import MORPH_CONFIGS as MORPH_MODELS


def flatten_params(d, prefix=""):
    """Flatten nested dict to list of (path, array) tuples."""
    from flax.core import FrozenDict
    if isinstance(d, FrozenDict):
        d = dict(d)
    result = []
    for k, v in sorted(d.items()):
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (dict, FrozenDict)):
            result.extend(flatten_params(v, path))
        else:
            result.append((path, v))
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert MORPH PyTorch weights to JAX msgpack")
    parser.add_argument("--input", "-i", required=True, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output", "-o", default=None, help="Output msgpack path (default: <input>.msgpack)")
    parser.add_argument("--model-size", "-m", choices=list(MORPH_MODELS.keys()), default="Ti",
                        help="Model variant (Ti, S, M, L)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + ".msgpack"

    cfg = MORPH_MODELS[args.model_size]

    # ── Load PyTorch checkpoint ──
    print(f"Loading PyTorch checkpoint: {args.input}")
    sd = load_pytorch_state_dict(args.input)
    print(f"  PyTorch state_dict: {len(sd)} parameters")

    # ── Initialize JAX model ──
    print(f"Initializing JAX model ({args.model_size})...")
    model = ViT3DRegression(
        patch_size=8,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        heads_xa=32,
        mlp_dim=cfg["mlp_dim"],
        max_components=3,
        conv_filter=cfg["conv_filter"],
        max_ar=cfg["max_ar"],
        max_patches=4096,
        max_fields=3,
        dropout=0.0,
        emb_dropout=0.0,
        model_size=cfg["model_size"],
    )

    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 1, 1, 1, 8, 8, 8))
    jax_params = model.init(rng, dummy, deterministic=True)
    print(f"  JAX model initialized")

    # ── Convert weights ──
    print("Converting PyTorch weights to JAX...")
    converted = convert_pytorch_to_jax_params(sd, jax_params, heads_xa=32)

    # ── Summary ──
    jax_flat = flatten_params(converted["params"])
    n_jax = len(jax_flat)
    total_elements = sum(int(np.prod(v.shape)) for _, v in jax_flat)
    print(f"\n  JAX params: {n_jax} leaf arrays")
    print(f"  Total parameters: {total_elements:,}")

    print("\n  Parameter groups:")
    groups = {}
    for path, arr in jax_flat:
        group = path.split(".")[0]
        if group not in groups:
            groups[group] = {"count": 0, "elements": 0}
        groups[group]["count"] += 1
        groups[group]["elements"] += int(np.prod(arr.shape))
    for g in sorted(groups):
        print(f"    {g}: {groups[g]['count']} arrays, {groups[g]['elements']:,} params")

    # ── Serialize ──
    print(f"\nSaving to: {args.output}")
    serialized = to_bytes(converted)
    with open(args.output, "wb") as f:
        f.write(serialized)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # ── Verify roundtrip ──
    print("Verifying saved file...")
    with open(args.output, "rb") as f:
        loaded_bytes = f.read()
    loaded = from_bytes(converted, loaded_bytes)

    loaded_flat = flatten_params(loaded["params"])
    assert len(loaded_flat) == n_jax, f"Mismatch: saved {n_jax}, loaded {len(loaded_flat)}"

    max_diff = 0.0
    for (orig_path, orig_arr), (load_path, load_arr) in zip(jax_flat, loaded_flat):
        assert orig_path == load_path
        diff = float(np.max(np.abs(np.array(orig_arr) - np.array(load_arr))))
        max_diff = max(max_diff, diff)

    print(f"  Verification: {n_jax} params loaded, max roundtrip diff = {max_diff:.2e}")
    print("\nDone!")


if __name__ == "__main__":
    main()
