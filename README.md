# jax_morph

> **Note:** This package is designed to be used with [jNO](https://github.com/FhG-IISB/jNO).

> **Warning:** This is a research-level repository. It may contain bugs and is subject to continuous change without notice.

JAX/Flax translation of the **MORPH** PDE foundation model, maintaining exact 1-to-1 weight compatibility with the original PyTorch implementation for pretrained checkpoint conversion.

## Overview

MORPH is a family of PDE foundation models from Los Alamos National Laboratory, trained on multi-physics simulation data. This repository provides a pure JAX/Flax reimplementation of the full model architecture (`ViT3DRegression`).

### Architecture

```
Input (B, t, F, C, D, H, W)
        │
        ▼
┌───────────────────────┐
│  HybridPatchEmbedding │  ConvOperator → Patchify → Dense → FieldCrossAttention
│  (patch_embedding)    │  Conv3D feature extraction + learned cross-attention
└──────────┬────────────┘
           │  (B, t, n_patches, dim)
           ▼
┌───────────────────────┐
│  Positional Encoding  │  Learned embedding table, interpolated to (t, n_patches)
│  (pos_encoding)       │  Ti/S/M: 1D linear | L: 2D bilinear with antialias
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  N × EncoderBlock     │  Each block:
│  (transformer_blocks) │    1. LayerNorm → AxialAttention (t, d, h, w axes)
│                       │    2. LayerNorm → MLP (with LoRA support)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  SimpleDecoder        │  LayerNorm → Dense → Unpatchify
│  (decoder)            │
└──────────┬────────────┘
           │
           ▼
Output (B, F, C, D, H, W)
```

### Model Variants

| Variant | Dim | Depth | Heads | MLP Dim | max_ar | Parameters |
|---------|-----|-------|-------|---------|--------|------------|
| Ti      | 256 |     4 |     4 |   1,024 |      1 |       9.9M |
| S       | 512 |     4 |     8 |   2,048 |      1 |      32.8M |
| M       | 768 |     8 |    12 |   3,072 |      1 |     125.6M |
| L       | 1024|    16 |    16 |   4,096 |     16 |     483.3M |

All variants use `conv_filter=8`, `patch_size=8`, `heads_xa=32`, `max_patches=4096`.

## Reference

Rautela et al., "MORPH: PDE Foundation Models with Arbitrary Data Modality" (2025)
- Paper: <https://arxiv.org/abs/2509.21670>
- Weights: <https://huggingface.co/mahindrautela/MORPH>
- Code: <https://github.com/lanl/MORPH>

## Installation

```bash
# Using uv (recommended)
uv venv
uv pip install -e .

# For GPU support (CUDA 12)
uv pip install -e ".[gpu]"

# For weight conversion from PyTorch
uv pip install -e ".[convert]"
```

## Weight Conversion

Convert a pretrained PyTorch checkpoint to JAX msgpack format:

```bash
python scripts/convert.py --input morph-Ti.pth --output morph-Ti.msgpack --model-size Ti
```

The script:
1. Loads the PyTorch checkpoint (`model_state_dict` format)
2. Initialises the JAX model and maps all parameters
3. Validates shapes and reports parameter counts
4. Saves as msgpack with roundtrip verification

### Weight Mapping Rules

| PyTorch | Flax | Transformation |
|---------|------|----------------|
| `nn.Conv3d.weight` (O,I,D,H,W) | `nn.Conv.kernel` (D,H,W,I,O) | Transpose (2,3,4,1,0) |
| `nn.Linear.weight` (O,I) | `nn.Dense.kernel` (I,O) | Transpose |
| `nn.Linear.bias` | `nn.Dense.bias` | As-is |
| `nn.LayerNorm.weight` | `nn.LayerNorm.scale` | As-is |
| `nn.LayerNorm.bias` | `nn.LayerNorm.bias` | As-is |
| `nn.MultiheadAttention.in_proj_weight` (3E,E) | Q/K/V `kernel` (E,H,d) | Split + transpose + reshape |
| `nn.MultiheadAttention.in_proj_bias` (3E,) | Q/K/V `bias` (H,d) | Split + reshape |
| `nn.MultiheadAttention.out_proj.*` | `out_proj.kernel/bias` | Transpose |
| `nn.Parameter` | `param` | As-is |
| `LoRALinear.A` (R,I) | `A` (I,R) | Transpose |
| `LoRALinear.B` (O,R) | `B` (R,O) | Transpose |

## Usage

### Using Convenience Constructors

```python
import jax
import jax.numpy as jnp
from jax_morph import morph_Ti, load_pytorch_state_dict, convert_pytorch_to_jax_params

# Create model
model = morph_Ti()

# Initialise
rng = jax.random.PRNGKey(0)
dummy = jnp.zeros((1, 1, 1, 1, 8, 8, 8))
params = model.init(rng, dummy, deterministic=True)

# Convert and load PyTorch weights
pt_sd = load_pytorch_state_dict("morph-Ti-FM-max_ar1_ep225.pth")
params = convert_pytorch_to_jax_params(pt_sd, params)

# Run inference
enc, z, pred = model.apply(params, dummy, deterministic=True)
```

### Loading Converted Weights

```python
import jax.numpy as jnp
from flax.serialization import from_bytes
from jax_morph import morph_Ti

model = morph_Ti()

# Load pre-converted msgpack weights
with open("morph-Ti.msgpack", "rb") as f:
    params = from_bytes(target=None, encoded_bytes=f.read())

# Inference: x is (B, t, F, C, D, H, W)
enc, z, pred = model.apply(params, x, deterministic=True)
```

### Using Individual Components

```python
from jax_morph.patch_embedding import HybridPatchEmbedding3D
from jax_morph.encoder_block import EncoderBlock
from jax_morph.axial_attention import AxialAttention3DSpaceTime
from jax_morph.cross_attention import FieldCrossAttention
from jax_morph.decoder import SimpleDecoder
from jax_morph.positional_encoding import PositionalEncodingSLinTSlice
```

### Explicit Model Construction

```python
from jax_morph import ViT3DRegression

model = ViT3DRegression(
    patch_size=8, dim=1024, depth=16, heads=16, heads_xa=32,
    mlp_dim=4096, max_components=3, conv_filter=8,
    max_ar=16, max_patches=4096, max_fields=3,
    dropout=0.0, emb_dropout=0.0, model_size="L",
)
```

## Equivalence Testing

Verify that the JAX model with converted weights produces the same output as the PyTorch model:

```bash
# Requires cloned MORPH repo (https://github.com/lanl/MORPH)
export MORPH_ROOT=/path/to/MORPH
python scripts/compare.py --model-size Ti
```

The script downloads the checkpoint from HuggingFace, runs both models on identical random input, and compares outputs.

### Results

| Variant | Max Abs Diff | Mean Abs Diff | Parameters Matched | Status |
|---------|-------------|---------------|-------------------|--------|
| Ti      |    4.96e-05 |      5.90e-06 |          9,932,920 | PASS   |
| S       |    1.22e-04 |      2.06e-05 |         32,849,512 | PASS   |
| M       |    9.77e-04 |      1.00e-04 |        125,574,248 | PASS   |
| L       |    9.16e-05 |      1.15e-05 |        483,293,400 | PASS   |

All models pass with max absolute difference < 1e-3.

## Project Structure

```
jax_morph/
├── jax_morph/                  # Core library
│   ├── __init__.py             # Public API exports
│   ├── configs.py              # Model configs + convenience constructors
│   ├── model.py                # ViT3DRegression (top-level model)
│   ├── patch_embedding.py      # HybridPatchEmbedding3D
│   ├── conv_operator.py        # ConvOperator (3D conv feature extractor)
│   ├── patchify.py             # 3D patchification utility
│   ├── cross_attention.py      # FieldCrossAttention (multi-field aggregation)
│   ├── positional_encoding.py  # Learned positional encodings (linear / bilinear)
│   ├── encoder_block.py        # EncoderBlock (LayerNorm + attention + MLP)
│   ├── axial_attention.py      # AxialAttention3DSpaceTime (t, d, h, w axes)
│   ├── attention.py            # SDPA, LoRALinear, LoRAMHA
│   ├── decoder.py              # SimpleDecoder (LayerNorm + Dense + unpatchify)
│   └── convert_weights.py      # PyTorch → Flax parameter mapping
├── scripts/
│   ├── convert.py              # CLI: convert .pth → .msgpack
│   └── compare.py              # CLI: PyTorch vs JAX equivalence test
├── pyproject.toml
├── LICENSE
├── .gitignore
└── README.md
```

## Module Details

### Model (`model.py`)

**`ViT3DRegression`** — Top-level model composing patch embedding, positional encoding, N transformer blocks, and a decoder. Handles input reshaping `(B, t, F, C, D, H, W)` → patch tokens → output prediction. Returns `(encoder_output, transformer_output, prediction)`.

### Patch Embedding (`patch_embedding.py`)

**`HybridPatchEmbedding3D`** — Conv3D feature extraction via `ConvOperator`, 3D patchification, linear projection, and learned field cross-attention via `FieldCrossAttention`. Aggregates multiple input fields into a single token sequence.

### Conv Operator (`conv_operator.py`)

**`ConvOperator`** — 1×1×1 input projection followed by a stack of 3×3×3 convolutions with channel doubling and LeakyReLU activations. Uses channels-last layout.

### Cross Attention (`cross_attention.py`)

**`FieldCrossAttention`** — Learned query tokens attend to patch tokens from multiple fields via multi-head attention. Uses `DenseGeneral` for Q/K/V projections matching PyTorch's `nn.MultiheadAttention`.

### Positional Encoding (`positional_encoding.py`)

Two variants of learned positional embeddings:

- **`PositionalEncodingSLinTSlice`** — Time-slicing + 1D linear interpolation over patches. Used for Ti/S/M variants (max_ar ≤ 1).
- **`PositionalEncodingSTBilinear`** — 2D bilinear interpolation with antialiasing over (time, patches). Used for L variant (max_ar > 1).

Custom interpolation functions replicate PyTorch's `F.interpolate(align_corners=False, antialias=True)` coordinate mapping exactly.

### Encoder Block (`encoder_block.py`)

**`EncoderBlock`** — Pre-norm transformer block: LayerNorm → AxialAttention → residual → LayerNorm → MLP → residual. Uses exact GELU (`approximate=False`) and LayerNorm `epsilon=1e-5` to match PyTorch defaults.

### Axial Attention (`axial_attention.py`)

**`AxialAttention3DSpaceTime`** — Sequential attention over four axes: time (t), depth (d), height (h), width (w). Each axis uses `LoRAMHA` (multi-head attention with optional LoRA adapters).

### Attention (`attention.py`)

- **`scaled_dot_product_attention`** — Manual SDPA (compatible with any JAX version).
- **`LoRALinear`** — Dense layer with optional low-rank adaptation (`A`, `B` matrices).
- **`LoRAMHA`** — Multi-head attention with separate LoRA-enabled Q/K/V/O projections.

### Decoder (`decoder.py`)

**`SimpleDecoder`** — LayerNorm → Dense projection → reshape back to volumetric output. Applies to only the last timestep.

### Convert Weights (`convert_weights.py`)

- **`load_pytorch_state_dict`** — Loads MORPH `.pth` checkpoints (handles `model_state_dict` wrapper and `module.` prefix from DataParallel).
- **`convert_pytorch_to_jax_params`** — Full parameter conversion with shape validation.

## Implementation Notes

### Key Differences from PyTorch

1. **Channels-last convolutions**: JAX Conv3D uses `(D, H, W, C_in, C_out)` kernel layout vs PyTorch's `(C_out, C_in, D, H, W)`.

2. **LayerNorm epsilon**: Flax defaults to `1e-6`; PyTorch defaults to `1e-5`. We explicitly set `epsilon=1e-5` everywhere.

3. **GELU activation**: Flax's `nn.gelu` defaults to the approximate version; PyTorch uses exact. We use `nn.gelu(x, approximate=False)`.

4. **Positional encoding interpolation**: `jax.image.resize` uses different coordinate conventions than PyTorch's `F.interpolate(align_corners=False)`. Custom interpolation functions replicate the half-pixel coordinate mapping `(i + 0.5) * in_size / out_size - 0.5`.

5. **Antialias bilinear interpolation**: For the L model, PyTorch's `antialias=True` zeros out-of-bounds kernel weights before renormalisation. Our implementation matches this exactly via separable 1D passes with a widened triangle kernel.

6. **LoRA dormant parameters**: When `lora_rank=0`, LoRA A/B matrices are initialised but never contribute to the output. This matches PyTorch's dormant-LoRA pattern for future fine-tuning.

7. **attn_t always instantiated**: Even when `t=1`, the temporal attention module is created (but its residual is not added) to ensure consistent parameter trees for weight loading.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
