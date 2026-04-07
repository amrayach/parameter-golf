# Local Prep: PR #1413 parallel-residual and n-gram variant

This folder is a local, non-submission prep variant built from upstream `pr1413` and
the corrected causal n-gram sidecars from `pr1437`. It is designed for the planned
A/B/C/D/E RunPod batch.

What is added on top of faithful `#1413`:

- `PARALLEL_RESIDUAL_START` hook, default `-1` (disabled)
- `SKIP_TRAINING` hook, default `0`
- causal token-only n-gram tilt support, default disabled
- sidecars `ngram_tilt.py` and `fused_expert_kernel.cpp`

Intended runs:

- `B`: `PARALLEL_RESIDUAL_START=7`
- `D`: `PARALLEL_RESIDUAL_START=7 LOOP_START=3`
- `E`: reuse the `D` checkpoint with `SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1`

Important defaults for corrected token-only n-gram tilt:

- `NGRAM_BASE_BETA=2.0`
- `NGRAM_AGREE_BONUS=0.1`
- `NGRAM_WITHIN_THRESHOLD=0.25`
- `NGRAM_WITHIN_BETA=0.0`
- `NGRAM_WORD_THRESHOLD=0.8`
- `NGRAM_WORD_BETA=0.0`
- `NGRAM_OPEN_TABLE_BITS=26`
- `NGRAM_ORDER_STRIDE=2`

Materialization summary:

- local record folder: `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep`
- wrapped code bytes: `17390`
- decoded source sha256: `5d38d696f1d57c92882c72678f0cc52653ecf8ce1947b62965df80cfea3d2beb`
- sidecar sha256:
  - `ngram_tilt.py`: `065ced48efcd5ae633f4307d254a0d3e475641878a0dc580f8e677b6e56aa379`
  - `fused_expert_kernel.cpp`: `6b11646609508a84f7c2d9ddd9cdb4c133c2474ec83a50b78313d96664984056`
