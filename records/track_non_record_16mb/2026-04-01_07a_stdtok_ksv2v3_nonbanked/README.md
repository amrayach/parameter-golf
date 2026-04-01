# Session 07a: Standard-Tokenizer KSV2/KSV3 Non-Banked Pivot

## Goal

Land the clean local non-banked pivot toward the PR `#1130` / `#1212` family without inheriting the `#1089` banking thesis.

## Included

- 12-layer local skip-connected stack
- residual lambdas
- split early/late LR groups
- PR-style coprime multi-shard loader
- MiLe loss with wallclock triangle schedule
- cache/backout path at `CACHE_LAYER=7`
- bigger bigram (`5120`) and VE (`VE_DIM=128`, `VE_LAYERS=5,9,10`)
- compile-stable precomputed RoPE tables
- eager eval by default
- optional window attention
- optional explicit mixed-seq training via `SEQ_LENS_PER_GPU` + `SEQS_PER_GPU`

## Explicitly Not Included

- Parameter Banking
- Turbo-Muon
- EngramLite
- GPTQ
- TTT
- automatic FA3 dependency for the default path

## Defaults

- `TRAIN_SEQ_LEN=2048`
- `EVAL_SEQ_LEN=6144`
- `EVAL_STRIDE=128`
- `WINDOW_SIZE=-1` by default, so FA3 is not required for the first landing
- mixed-seq is opt-in and currently requires explicit `SEQS_PER_GPU`

## Launch Notes

- Default 8x launch is single-seq and window-off.
- To enable the public-family windowed path, set:

```bash
WINDOW_SIZE=512
WINDOW_ATTN_LAYERS=2,4,6,8,10
```

- To enable mixed-seq, set both:

```bash
SEQ_LENS_PER_GPU=2048,2048,2048,2048,2048,6144,6144,6144
SEQS_PER_GPU=36,36,36,36,36,10,10,10
```
