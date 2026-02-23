"""
Pretrained weight equivalence test: verify that the JAX model with converted
weights produces the same output as the PyTorch model with original weights.

Downloads a checkpoint from HuggingFace if not present locally, runs both
models on the same random input, and compares outputs.

Usage:
    python scripts/compare.py --model-size Ti
    python scripts/compare.py --model-size Ti --checkpoint /path/to/morph-Ti.pth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

# ── MORPH repo path (for importing PyTorch model) ──
MORPH_ROOT = os.environ.get("MORPH_ROOT", os.path.expanduser("~/MORPH"))
sys.path.insert(0, MORPH_ROOT)

from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression as ViT3DRegression_PT
from jax_morph import ViT3DRegression as ViT3DRegression_JAX
from jax_morph import load_pytorch_state_dict, convert_pytorch_to_jax_params
from jax_morph.configs import MORPH_CONFIGS as MORPH_MODELS, CHECKPOINT_NAMES


def get_checkpoint(model_size, checkpoint_path=None):
    """Get checkpoint path, downloading from HuggingFace if needed."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path

    # Check local models directory
    local_path = os.path.join(MORPH_ROOT, "models", "FM", CHECKPOINT_NAMES[model_size])
    if os.path.exists(local_path):
        return local_path

    # Download from HuggingFace
    print(f"Downloading {CHECKPOINT_NAMES[model_size]} from HuggingFace...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="mahindrautela/MORPH",
        filename=CHECKPOINT_NAMES[model_size],
        subfolder="models/FM",
        repo_type="model",
        resume_download=True,
    )
    return path


def create_pytorch_model(cfg):
    """Create and return PyTorch model."""
    model = ViT3DRegression_PT(
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
    return model


def create_jax_model(cfg):
    """Create and return JAX model."""
    model = ViT3DRegression_JAX(
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
    return model


def compare_outputs(pt_out, jax_out, name="output"):
    """Compare PyTorch and JAX outputs."""
    pt_np = pt_out.detach().cpu().numpy()
    jax_np = np.array(jax_out)

    abs_diff = np.abs(pt_np - jax_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    rel_diff = abs_diff / (np.abs(pt_np) + 1e-8)
    max_rel = rel_diff.max()

    print(f"  {name}:")
    print(f"    Shape: PT={pt_np.shape}, JAX={jax_np.shape}")
    print(f"    Max abs diff:  {max_diff:.6e}")
    print(f"    Mean abs diff: {mean_diff:.6e}")
    print(f"    Max rel diff:  {max_rel:.6e}")
    print(f"    PT  range: [{pt_np.min():.4f}, {pt_np.max():.4f}]")
    print(f"    JAX range: [{jax_np.min():.4f}, {jax_np.max():.4f}]")

    return max_diff


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX MORPH outputs")
    parser.add_argument("--model-size", "-m", choices=list(MORPH_MODELS.keys()), default="Ti")
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to .pth checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fields", type=int, default=1)
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--spatial", type=int, default=16, help="Spatial size (D=H=W)")
    args = parser.parse_args()

    cfg = MORPH_MODELS[args.model_size]
    print(f"=== MORPH {args.model_size} Equivalence Test ===")
    print(f"Config: {cfg}")

    # ── Get checkpoint ──
    ckpt_path = get_checkpoint(args.model_size, args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")

    # ── Create random input ──
    np.random.seed(args.seed)
    S = args.spatial
    vol_np = np.random.randn(args.batch, 1, args.fields, args.components, S, S, S).astype(np.float32)
    vol_pt = torch.from_numpy(vol_np)
    vol_jax = jnp.array(vol_np)
    print(f"Input shape: {vol_np.shape}")

    # ── PyTorch model ──
    print("\n--- PyTorch ---")
    pt_model = create_pytorch_model(cfg)
    sd = load_pytorch_state_dict(ckpt_path)
    pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    pt_params = sum(p.numel() for p in pt_model.parameters())
    print(f"  Parameters: {pt_params:,}")

    t0 = time.time()
    with torch.no_grad():
        enc_pt, z_pt, pred_pt = pt_model(vol_pt)
    pt_time = time.time() - t0
    print(f"  Forward time: {pt_time:.3f}s")
    print(f"  enc shape: {enc_pt.shape}")
    print(f"  z shape: {z_pt.shape}")
    print(f"  pred shape: {pred_pt.shape}")

    # ── JAX model ──
    print("\n--- JAX ---")
    jax_model = create_jax_model(cfg)

    rng = jax.random.PRNGKey(0)
    jax_params = jax_model.init(rng, vol_jax, deterministic=True)
    jax_params = convert_pytorch_to_jax_params(sd, jax_params, heads_xa=32)

    jax_flat = jax.tree.leaves(jax_params)
    jax_n_params = sum(x.size for x in jax_flat)
    print(f"  Parameters: {jax_n_params:,}")

    # Warmup
    _ = jax_model.apply(jax_params, vol_jax, deterministic=True)

    t0 = time.time()
    enc_jax, z_jax, pred_jax = jax_model.apply(jax_params, vol_jax, deterministic=True)
    jax_time = time.time() - t0
    print(f"  Forward time: {jax_time:.3f}s")
    print(f"  enc shape: {enc_jax.shape}")
    print(f"  z shape: {z_jax.shape}")
    print(f"  pred shape: {pred_jax.shape}")

    # ── Compare ──
    print("\n--- Comparison ---")
    d1 = compare_outputs(enc_pt, enc_jax, "Encoder output")
    d2 = compare_outputs(z_pt, z_jax, "Transformer output")
    d3 = compare_outputs(pred_pt, pred_jax, "Prediction")

    threshold = 1e-3
    max_d = max(d1, d2, d3)
    if max_d < threshold:
        print(f"\n✓ PASS: Max absolute difference {max_d:.2e} < {threshold:.0e}")
    else:
        print(f"\n✗ FAIL: Max absolute difference {max_d:.2e} >= {threshold:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
