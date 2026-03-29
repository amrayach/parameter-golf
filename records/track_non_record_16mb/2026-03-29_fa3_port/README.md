# Session 05 Phase 1: FA3 Port

**Status**: READY TO TEST
**Parent**: `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor`
**Delta**: Replace SDPA attention with direct FA3 (`flash_attn_interface`)

## Changes vs Anchor

| Change | Anchor (SDPA) | FA3 Port |
|--------|--------------|----------|
| Import | `F.scaled_dot_product_attention` | `flash_attn_interface.flash_attn_func` |
| Tensor layout | `(B, H, T, D)` for q/k | `(B, T, H, D)` for all |
| Rotary cache | `(1, 1, T, rd/2)` | `(1, T, 1, rd/2)` |
| q_gain broadcast | `[None, :, None, None]` | `[None, None, :, None]` |
| Post-attention | `.transpose(1, 2).contiguous()` | removed (already B,T,H,D) |
| GQA | `enable_gqa=True` flag | automatic (Hkv < H broadcast) |
| SDPA backend flags | `enable_flash_sdp(True)` etc. | removed |

## Container

NGC `25.02` (`PyTorch 2.11.0+cu130`, CUDA 13.0) + installed FA3 wheel:

```bash
pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
```

## Microbenchmark Context

Isolated attention kernel (B=16, T=2048, H=8, Hkv=4, D=64):
- SDPA flash (25.02): 1.889 ms/iter
- FA3 direct (25.02): 0.165 ms/iter (11.44x faster)

Full training impact TBD.

## Run Commands

Smoke (1xH100):
```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:10:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    pip install --no-cache-dir sentencepiece zstandard \
      "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -1 &&
    python records/track_non_record_16mb/2026-03-29_fa3_port/train_gpt.py
  '
```

Full 8xH100 (after smoke passes):
```bash
srun -p H100 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    pip install --no-cache-dir sentencepiece zstandard \
      "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -1 &&
    python records/track_non_record_16mb/2026-03-29_fa3_port/train_gpt.py
  '
```

## Results

Pending.
