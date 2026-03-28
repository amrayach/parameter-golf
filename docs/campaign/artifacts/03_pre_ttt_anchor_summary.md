# 03 Pre-TTT Anchor Summary

Date: 2026-03-28
Status: Complete

## Anchor Definition

The Session 03 anchor is a clean 2026-03-21-style pre-TTT stack ported onto the repo-root `train_gpt.py` skeleton. It uses the SDPA attention backend (not flash_attn_3) and hardcodes all architecture/training constants.

### Feature Set

| Feature | Status |
|---------|--------|
| 11L 512d 8H/4KV U-Net skip | Ported |
| 3x relu^2 MLP | Ported |
| SmearGate | Ported |
| BigramHash (2048 buckets, 128d) | Ported |
| XSA on last 4 layers | Ported (adapted for SDPA layout) |
| Partial RoPE (16/64) | Ported (with NTK scaling, root cache layout) |
| Layerwise LN scale | Ported |
| EMA (decay=0.997) | Ported |
| Muon WD=0.04, Adam WD=0.04 | Ported |
| Mixed int6+zstd export | Ported |
| Stride-64 sliding eval | Ported |
| Orthogonal init + proj scaling | Ported |

### Excluded Features

Late QAT, SWA, MTP, VE, DTG, GPTQ-lite, flash_attn_interface, TTT, warmdown 3500.

### Key Adaptation

SDPA uses (B, H, T, D) tensor layout vs donor's flash_attn_3 (B, T, H, D). XSA adapted by transposing SDPA output before self-value subtraction. Rotary cache uses root's `[None, None, :, :]` shape.

## Parameter Count

26,829,913 (matches donor exactly).

## Target

`val_bpb` in 1.123-1.128 range on 8xH100 in 600s.

## Measured Results

| Metric | Value |
|--------|-------|
| GPU / node | 8x NVIDIA H100 80GB HBM3 (SXM5) / serv-3342 |
| Container | nvcr.io_nvidia_pytorch_26.03-py3.sqsh |
| Data path | /fscratch (low-latency) |
| Steps completed | 6,564 / 9,000 |
| Step average | 91.37 ms |
| Pre-quant EMA val_bpb | 1.14472403 |
| Post-quant roundtrip val_bpb | 1.15247273 |
| **Sliding s64 val_bpb** | **1.12904446** |
| Artifact size (int6+zstd) | 15,692,752 bytes |
| Code size | 58,572 bytes |
| Total submission size | 15,751,324 bytes |
| Peak memory | 21,274 MiB allocated / 22,070 MiB reserved |
| Compressor used | zstd |

## Comparison with Donor

| Metric | This run | Donor (2026-03-21) | Delta |
|--------|----------|-------------------|-------|
| Sliding s64 val_bpb | 1.1290 | 1.1248 | +0.0042 |
| Steps | 6,564 | 7,051 | -487 |
| Step average | 91.37 ms | ~85 ms | +6.4 ms |
| Artifact | 15,751,324 | 15,612,308 | +139,016 |

## Bottleneck Analysis

This run validates the anchor. The remaining donor gap is small enough that the next step should be a narrow delta sweep, not a redesign.

What the run says clearly:

- The Session 03 port is real: final sliding `val_bpb=1.12904446` is far better than the root `8xH100` baseline and close to the donor.
- Throughput is one plausible bottleneck: `91.37 ms` vs donor `~85 ms` leaves a `487` step deficit in the same wallclock.
- Export fidelity still matters: pre-quant EMA `1.14472403` to int6 roundtrip `1.15247273` is a `+0.00774870` gap, so not all remaining headroom should be assigned to throughput alone.

Conclusion:

- Treat backend/perf and export/model changes as separate measurements.
- Do not assume the entire `+0.0042` donor gap is explained by step count alone.

## Next Recommended Delta

**Session 04: targeted delta sweep, one change per run**

Recommended order:

1. **GPTQ-lite clip search** as the first export-side delta. The current roundtrip gap is large enough to justify a clean post-training quantization refinement run.
2. **LeakyReLU²** as the first cheap model-side delta. It is simple, public, and easy to attribute.
3. **One small schedule or token-path tweak** (`warmdown`/EMA threshold or one bigram/smear change), only after the first two are measured.

If throughput becomes the dominant concern, benchmark backend/perf parity as its own control. Do not stack backend, export, and model changes in the same Session 04 run.
