# D Stack Ablation Study Design

Prepared for Section VII of PR `#1017`.

## Scope

This document designs a one-component-removed ablation study for the canonical `D` stack:

- archive run: `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/`
- local record folder: `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/`

Canonical `D` is the `#1413` stack plus the local hooks added by `scripts/prepare_pr1413_variants.py`, with the seed-0 run configured as:

- `PARALLEL_RESIDUAL_START=7`
- `LOOP_START=3`
- `LOOP_END=5`
- `QK_GAIN_INIT=5.0`
- `TTT_ENABLED=1`
- `TTT_LR=0.005`
- `TTT_EPOCHS=3`

Important scope boundary:

- `SKIP_TRAINING`
- `NGRAM_TILT_ENABLED`
- `NGRAM_*`

are `E`-only hooks. They live in the same prepared folder, but they are dormant in `D` and are not part of this ablation matrix.

## Existing Archive Coverage

The fetched A/B/C/D seed-0 archive already gives a 2x2 ablation for two stack levers:

| Existing comparison | What it isolates |
| --- | --- |
| `A` vs `B` | parallel residuals at the upstream loop window |
| `A` vs `C` | widened loop window (`3..5` vs `4..5`) without parallel residuals |
| `C` vs `D` | parallel residuals at the widened loop window |
| `B` vs `D` | widened loop window with parallel residuals enabled |

The archive logs also already contain both `quantized_sliding_window` and `legal_ttt_exact`, so the TTT contribution can be read without a dedicated retrain.

## Ablation Matrix

| Component | What it does | Control mechanism | Ablation config | Requires retraining? | Expected BPB direction | Interaction risks | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Score-first exact TTT | Runs chunked sliding-window evaluation, scoring each token before SGD adaptation on later chunk data. | `TTT_ENABLED`, `TTT_LR`, `TTT_EPOCHS`, `TTT_CHUNK_TOKENS`, `TTT_MOMENTUM`, `TTT_BATCH_SEQS`, `TTT_GRAD_CLIP` in the decoded `train_gpt.py`; active in `pr1413_combo_s0/run_meta.env`. | `TTT_ENABLED=0`. No retrain required; the current archive already logs the no-TTT metric as `quantized_sliding_window`. | NO | HURT | Future training ablations should always report both pre-TTT and post-TTT BPB, otherwise the TTT effect is confounded with the training change. | HIGH |
| Parallel residuals on layers `>= 7` | For later blocks, runs attention and MLP from the same `x_in` and adds both branches in parallel instead of feeding attention output into the MLP sequentially. | Local hook added by `scripts/prepare_pr1413_variants.py`; enabled by `PARALLEL_RESIDUAL_START=7`. | `PARALLEL_RESIDUAL_START=-1`. Already archive-backed by `A/B` and `C/D`. | YES | HURT | Interacts with looping because the effective execution order changes once looping is active. Use the existing `A/B/C/D` lattice, not only `D` vs `A`. | HIGH |
| Widened loop window `3..5` | Reuses blocks `3..5` three times once looping activates, increasing effective depth and changing the encoder/decoder split used for skip connections. | `NUM_LOOPS`, `LOOP_START`, `LOOP_END`, `ENABLE_LOOPING_AT`; `D` uses `NUM_LOOPS=2 LOOP_START=3 LOOP_END=5`. | For the D-specific ablation, revert to upstream `LOOP_START=4 LOOP_END=5`. For full recurrence removal, use `NUM_LOOPS=0`. The `3..5` vs `4..5` comparison is already archive-backed by `A/C` and `B/D`. | YES | HURT | Full loop removal is not the same as reverting only the widened window. Loop controls also change skip topology and the meaning of later-block hooks. | HIGH |
| XSA on all 11 layers | Applies the XSA head-space projection after causal attention in every block because `XSA_LAST_N=11` equals `NUM_LAYERS=11`. | `XSA_LAST_N=11` and `self.blocks[i].attn.use_xsa=True` for the last `n` blocks. | `XSA_LAST_N=0`. | YES | HURT | Lives inside the same attention block as GQA, partial RoPE, and QK gain. Change only one attention subcomponent at a time. | HIGH |
| Learned skip-path weights | Adds encoder-to-decoder skip tensors through `skip_weights` before each decoder block. | `self.skip_weights` is always created from the loop-derived encoder/decoder index lists; there is no env flag for full removal. | Code change: set `self.num_skip_weights = 0` or bypass the `if skip_idx < self.num_skip_weights and skips:` block in `forward_logits()`. | YES | HURT | Strongly entangled with looping because loop settings determine how many skip edges exist. Gate-off is a different ablation from skip-off. | HIGH |
| Skip gates | Gates each skip tensor with a learned sigmoid before mixing it into the decoder stream. | `SKIP_GATES_ENABLED=1` creates `self.skip_gates`; otherwise the skip path falls back to pure additive skips. | `SKIP_GATES_ENABLED=0`. | YES | UNKNOWN | This does not remove the skip path, only the gating. Interpret against the skip-weight ablation separately. | HIGH |
| Residual input mixing (`resid_mix`) | Learns a two-way mix between the current stream `x` and the original embedding stream `x0` at every block input. | `self.resid_mix` in `Block`; always active, no env flag. | Code change: replace `x_in = mix[0] * x + mix[1] * x0` with `x_in = x`. | YES | UNKNOWN | Affects every block, and its meaning changes once looping and skip connections are altered. | HIGH |
| Logit softcap | Applies `softcap * tanh(logits / softcap)` before the cross-entropy loss and before eval logits are scored. | `LOGIT_SOFTCAP=30.0`. | Set `LOGIT_SOFTCAP=1e9` to approximate no cap, or code-change the return path to emit `logits_proj` directly. | YES | UNKNOWN | This changes both training dynamics and eval-time logits; do not mix it with export-only ablations. | HIGH |
| Three-way optimizer split | Uses `Muon` for matrix weights, `AdamW` for token/scalar controls, and a separate head optimizer when embeddings are untied. | `Optimizers` class partitions params into `optimizer_tok`, `optimizer_muon`, `optimizer_scalar`, and optional `optimizer_head`. | Code change: collapse to one optimizer family, e.g. a single `AdamW(base_model.parameters(), ...)`, and remove the `CONTROL_TENSOR_NAME_PATTERNS` partitioning. | YES | HURT | Tied vs untied embeddings changes whether the head optimizer exists. Keep the tie setting fixed while testing optimizer structure. | HIGH |
| High QK gain init | Starts the learned per-head `q_gain` at `5.0` in the D archive rather than the source default `4.0`. | `QK_GAIN_INIT=5.0` in `run_meta.env`; implemented via learned `self.q_gain`. | `QK_GAIN_INIT=4.0` to revert to the source default. If the goal is full removal, that is a separate code-change ablation. | YES | UNKNOWN | Prior `07c1` evidence says larger QK gain can hurt elsewhere, so treat this as a high-value stack-specific check. | HIGH |
| Tied embeddings | Shares the input embedding matrix with the output projection and swaps the token optimizer LR to `TIED_EMBED_LR`. | `TIE_EMBEDDINGS=1`; when false, the model instantiates `lm_head` and uses `HEAD_LR`. | `TIE_EMBEDDINGS=0`. | YES | HURT | Untying adds parameters, activates the head optimizer, and changes the artifact-size tradeoff. This is not a pure architecture-only ablation under the 16 MB cap. | HIGH |
| LeakyReLU squared MLP | Uses `leaky_relu(0.5)^2` in the MLP instead of the older `relu^2` variant seen in earlier anchors. | Hard-coded in `MLP.forward()`. | Code change back to the earlier repo baseline: `x = F.relu(self.fc(x)); return self.proj(x.square())`. | YES | UNKNOWN | Prior local results suggest this effect may be near-flat, so it is informative but less likely to surprise than the high-priority stack hooks. | MEDIUM |
| RMSNorm everywhere | Uses RMSNorm for block pre-norms, final norm, embedding norm, and Q/K normalization rather than LayerNorm-style mean-centering. | `RMSNorm` class plus `F.rms_norm()` call sites; no env flag. | Code change: swap `RMSNorm` and the `F.rms_norm` call sites to LayerNorm equivalents. | YES | UNKNOWN | Not independently clean from `LN_SCALE`; a LayerNorm swap changes both the norm family and the meaning of the layerwise norm scaling hook. | LOW |
| Layerwise norm scaling | Multiplies normalized block inputs by `1 / sqrt(layer_idx + 1)` when `LN_SCALE=1`. | `LN_SCALE=1` controls `self.ln_scale_factor`. | `LN_SCALE=0`. | YES | UNKNOWN | Best interpreted with RMSNorm held fixed. Also interacts with looping because reused layers see the same physical index but more effective depth. | MEDIUM |
| Partial RoPE (`ROPE_DIMS=16`) | Applies rotary position encoding to only the first 16 dimensions of each head instead of the full head dimension. | `ROPE_DIMS=16`; the model rewires each block's `Rotary` helper when `ROPE_DIMS > 0`. | `ROPE_DIMS=0` to revert to full-head RoPE. | YES | UNKNOWN | Shares the attention block with XSA, GQA, and QK gain. Keep all three fixed while testing this row. | MEDIUM |
| Grouped-query attention | Uses `NUM_KV_HEADS=4` with `NUM_HEADS=8`, reducing K/V projection size versus full MHA. | `NUM_KV_HEADS=4`, `NUM_HEADS=8`. | `NUM_KV_HEADS=8` to revert to full MHA. | YES | UNKNOWN | Changes parameter count, K/V projection shapes, speed, and artifact bytes, so it is less apples-to-apples than most rows. | LOW |
| Byte-shuffled Brotli export | Byte-shuffles the serialized quantized payload before compression to improve Brotli density without changing the model itself. | `_byte_shuffle()` inside `_compress()`; active for both `brotli` and `lzma` export. | Code change: bypass `_byte_shuffle()` in `_compress()` and `_byte_unshuffle()` in `_decompress()`. | NO | FLAT (size-only) | This is a size-only ablation. Keep compressor choice fixed while testing it, or the result is uninterpretable. | HIGH |

## Recommended Execution Order

1. Reuse what already exists before spending new H100 time.
   - Read `quantized_sliding_window` vs `legal_ttt_exact` from the current `A/B/C/D` console logs to get the TTT contribution.
   - Reuse the existing `A/B/C/D` archive for the parallel-residual and widened-loop-window comparisons.
2. Run the no-retrain export check offline.
   - Byte-shuffle off on the preserved `D` checkpoint.
3. First-wave new retraining ablations.
   - `XSA_LAST_N=0`
   - `SKIP_GATES_ENABLED=0`
   - skip-path removal
   - `resid_mix` fixed to identity
   - optimizer split collapsed
   - `QK_GAIN_INIT=4.0`
   - `LOGIT_SOFTCAP=1e9`
4. Second-wave new retraining ablations.
   - `TIE_EMBEDDINGS=0`
   - LeakyReLU squared back to `relu^2`
   - `LN_SCALE=0`
   - `ROPE_DIMS=0`
5. Low-priority architecture sweeps.
   - LayerNorm swap
   - `NUM_KV_HEADS=8`

Rationale:

- The first-wave rows are either already suggested by the archive, cheap to specify, or likely to explain a meaningful fraction of D's gain.
- The second-wave rows are still valid one-change ablations, but they are less likely to overturn the ranking.
- The last two rows are the least clean under the competition cap and the most likely to need retuning after the ablation.

## Entangled Rows

These components cannot be interpreted as independent if changed together:

- Parallel residuals and loop-window changes.
  - Use the existing `A/B/C/D` 2x2 structure. `D` vs `A` alone is not a clean attribution.
- Skip weights and skip gates.
  - `SKIP_GATES_ENABLED=0` keeps additive skips alive. It is not the same experiment as removing the skip path.
- Looping and skip-path ablations.
  - Loop controls change `encoder_indices`, `decoder_indices`, and therefore the number and placement of skip edges.
- Tied embeddings and optimizer structure.
  - Untying embeddings adds `lm_head` and activates `optimizer_head`, so the tied-embedding ablation must keep the intended optimizer plan explicit.
- XSA, partial RoPE, GQA, and QK gain.
  - All four live inside the same attention block. Only one should move at a time.
- RMSNorm and `LN_SCALE`.
  - `LN_SCALE=0` isolates the scale-decay factor. A LayerNorm swap changes the entire norm family.
- Byte-shuffle and compressor choice.
  - Keep `COMPRESSOR` fixed while testing byte shuffle.

## Run Count Estimate

For the full matrix above:

- Already archive-backed, no new training run needed:
  - score-first exact TTT
  - parallel residuals
  - widened loop window `3..5`
- Offline, no retraining, no H100 needed:
  - byte-shuffle off
- New retraining runs still required:
  - 13

Two practical totals:

- Incremental Section VII batch, reusing the fetched `A/B/C/D` archive: `13` new training runs plus `1` offline export replay.
- Full fresh batch from scratch, if archive reuse is not allowed: `15` training runs plus `1` offline export replay.

Recommended seed policy:

- one seed for the full matrix first, to rank components by effect size
- three seeds only for the top 2 to 3 surprising rows after the single-seed pass

## Checkpoint Reuse Note

These rows can share the canonical `D` checkpoint:

- score-first exact TTT
  - no retraining; every `D`-style run should log both `quantized_sliding_window` and `legal_ttt_exact`
- byte-shuffled Brotli export
  - no retraining; rerun only the export path on the preserved `final_model.pt`

Everything else in the table changes the training-time architecture, optimizer, or initialization and should be treated as a retraining ablation.
