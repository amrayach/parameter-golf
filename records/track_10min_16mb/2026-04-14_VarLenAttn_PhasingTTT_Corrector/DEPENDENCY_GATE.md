# Dependency Gate — PR #1610 Reproduction

Verified from pinned SHA `ca1919539dc6e328ea890cb03ad3ca1c5a84da55` on 2026-04-14.

## Critical Runtime Dependencies

| Package | Version | Source | Used in train_gpt.py |
|---------|---------|--------|---------------------|
| torch | 2.9.1+cu128 | PyPI + CUDA index | Core framework, DDP, compile |
| flash_attn_interface (FA3) | latest | Separate install | `flash_attn_varlen_func`, `flash_attn_with_kvcache` |
| triton | bundled with torch | pip torch dep | Fused MLP kernel (`@triton.jit`) |
| brotli | any | PyPI | Artifact compression in `serialize()` |
| sentencepiece | any | PyPI | `build_sentencepiece_luts()` |
| numpy | any | PyPI | Standard numerics |
| lzma | stdlib | Python 3.x | Code wrapper compression |

## Non-Critical / Transitive Dependencies

| Package | Notes |
|---------|-------|
| kernels | Listed in requirements.txt but NOT imported in train_gpt.py |
| tqdm | Not imported in train_gpt.py |
| huggingface-hub | Not imported in train_gpt.py (data likely pre-staged) |
| datasets | Not imported in train_gpt.py |
| tiktoken | Not imported in train_gpt.py |
| zstandard | Not imported in train_gpt.py (brotli used instead) |
| python-minifier | Used during serialize() for code minification |
| typing-extensions==4.15.0 | Pinned; likely torch dep |
| setuptools | Build dep |

## FA3 Install Path

FA3 is NOT available via standard pip. It must be installed from source or a prebuilt wheel:

```bash
# From #1610 README:
# FA3 must be installed separately; see README.md
# Typical RunPod/NGC approach:
pip install flash-attn --no-build-isolation
# or from wheel matching torch 2.9.1 + CUDA 12.8
```

The exact FA3 version is not pinned. The import path is `flash_attn_interface`, which is FA3's API (FA2 uses `flash_attn`).

## Verification Commands (RunPod 8xH100)

```bash
# 1. Install deps
uv pip install torch==2.9.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
# FA3 installed separately per above

# 2. Smoke test: import all deps
python3 -c "
import torch, numpy, brotli, sentencepiece, triton, lzma
from flash_attn_interface import flash_attn_varlen_func
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'triton {triton.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
"

# 3. Model shape test (no training, no data):
python3 -c "
import sys; sys.path.insert(0, '.')
# Instantiate model with #1610 defaults and verify shapes
# (This requires the full train_gpt.py to be importable)
"
```

## RunPod Launch Commands (from #1610 README)

```bash
# Full training + eval (one seed):
SEED=0 ARTIFACT_DIR="runs/varlen0" \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Eval-only on existing checkpoint:
SEED=0 EVAL_ONLY_PATH="runs/varlen0/final_model.pt" \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: RunPod uses `torchrun --standalone`, NOT Slurm `srun`. The Pegasus
launcher rules do not apply to RunPod.

## Gate Status

- [x] requirements.txt fetched from pinned SHA
- [x] Critical imports identified (6 packages)
- [x] Non-critical / transitive deps identified (8 packages)
- [x] FA3 install path documented
- [x] Launch commands documented from README
- [x] Smoke test commands prepared
- [ ] Smoke test executed on RunPod (Session B)
