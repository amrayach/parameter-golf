# Faithful PR #1394 SP8192 RunPod Strict Proof

**val_bpb: 1.0852** (3-seed mean sliding s64) | **15.99 MB** max artifact | **8xH100 SXM**, **588.0s** max train time

Submitted artifact: **seed 1337** with `val_loss = 2.80194051`, `val_bpb = 1.08471849`, `bytes_total = 15,986,188`.

## Summary

This folder packages the faithful PR `#1394` SP8192 reproduction run on RunPod `8xH100 SXM`.
It uses the current local `train_gpt.py` packaging path, which counts code bytes via the minified
self-extracting wrapper instead of the larger human-readable source file.

## Strict Multi-Seed Results

| Seed | Steps | train_time | Pre-quant EMA BPB | Roundtrip exact BPB | Sliding s64 BPB | nats | model_bytes | bytes_total |
|------|------:|-----------:|------------------:|--------------------:|----------------:|-----:|------------:|------------:|
| 42 | 5098 | 588045ms | 1.08896745 | 1.10263175 | 1.08576707 | 2.80464912 | 15,969,716 | 15,987,537 |
| 1337 | 5067 | 588044ms | 1.08993204 | 1.10134414 | 1.08471849 | 2.80194051 | 15,968,367 | 15,986,188 |
| 2025 | 5091 | 588038ms | 1.09017390 | 1.10167670 | 1.08513825 | 2.80302480 | 15,968,705 | 15,986,526 |
| **Mean** | | | **1.08969113** | **1.10188420** | **1.08520794** | **2.80320481** | | |
| **Std** | | | | | **0.00052775** | **0.00136325** | | |

## BPB Reference Points

Merged official `#1019`: `1.11473509` BPB
Open clean PR `#1394` mean: `1.08563` BPB
Mean delta vs merged `#1019`: `-0.02952715` BPB
Mean delta vs open `#1394` reference: `-0.00042206` BPB

Because this bundle uses `SP8192`, its token-level `val_loss` nats are tokenizer-dependent.
Cross-line comparison should therefore use **BPB**, not `val_loss` nats against the older SP1024 merged line.

## Artifact Byte Accounting

- `compressed_model_bytes` comes directly from the run log.
- `code_bytes_counted` is recomputed from `train_gpt.py` using the same lzma+base85 wrapper counted by the current export path.
- `bytes_total` in this folder is therefore `compressed_model_bytes + code_bytes_counted`.

## Environment Notes

- hardware: RunPod `8xH100 SXM`
- tokenizer/data: `fineweb10B_sp8192` + `fineweb_8192_bpe.model`
- packaging note: the original 2026-04-07 seed-1337 smoke run was launched before the local code-byte fix,
  so its logged `Total submission size` may exceed the recomputed `bytes_total` used here.

## How to Run

Install Python dependencies first:

```bash
pip install -r requirements.txt
```

Then run from the repo root:

```bash
PYTHONUNBUFFERED=1 \
DATA_DIR=./data \
SEED=1337 \
RUN_ID=submission \
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`: faithful PR `#1394` repro snapshot with current packaging fix
- `train_seed*.log`: one log per claimed seed
- `submission.json`: leaderboard metadata for the best seed in this bundle
