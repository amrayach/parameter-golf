# Session 04 Delta 1: GPTQ-lite Clip Search

Date: 2026-03-28
Track: non_record_16mb
Status: Pending run
Base: Session 03 pre-TTT anchor (`val_bpb=1.12904446` sliding s64)

## What this is

A single isolated change on top of the Session 03 anchor: replace fixed row-max int6 quantization with GPTQ-lite percentile clip search. Training loop is identical to the anchor. Only the post-training quantization function changes.

## Delta from Session 03 anchor

| Component | Anchor | Delta 1 |
|-----------|--------|---------|
| `quantize_int6_per_row` | Fixed row-max scale | 5-percentile MSE clip search |
| Training loop | Unchanged | Unchanged |
| Model architecture | Unchanged | Unchanged |
| Export pipeline | `mixed_int6+zstd` | `mixed_int6+zstd` (same, different quantizer) |

### GPTQ-lite clip search

For each 2D weight matrix, tries 5 candidate clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0). For each candidate, computes per-row scales from that percentile of the row's absolute values, quantizes the entire matrix, reconstructs it, and measures the matrix-wide MSE. The candidate with the lowest MSE wins. 1D tensors fall back to simple max-based clipping.

Source: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` lines 885-904.

### Known subtlety: clip range

The anchor clips to `[-32, 31]` (full signed 6-bit). The GPTQ-lite version clips to `[-31, 31]` (symmetric). This matches the proven 2026-03-22 donor exactly.

## Architecture

Identical to Session 03 anchor. See `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/README.md`.

| Parameter | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 |
| MLP multiplier | 3.0 |
| Sequence length | 2048 |
| Batch tokens | 786,432 |

## Launch command

```bash
salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00 --mem=200G

bash scripts/pegasus_optimized_launcher.sh \
  delta1_gptq_lite_8xh100 \
  records/track_non_record_16mb/2026-03-28_delta1_gptq_lite/train_gpt.py
```

## Acceptance checks

| Check | Expected |
|-------|----------|
| Training steps | ~6564 (identical to anchor) |
| Step average | ~91.37 ms (identical to anchor) |
| Pre-quant EMA val_bpb | ~1.14472403 (identical to anchor) |
| Roundtrip gap | < 0.00774870 (anchor gap: 1.15247273 − 1.14472403) |
| Artifact size | < 16,000,000 bytes |

## Success metric

- **Primary**: roundtrip gap (int6 roundtrip − pre-quant EMA) is smaller than 0.00774870
- **Headline**: sliding s64 val_bpb < 1.12904446

## Run results

_Pending — fill after run completes._
