# Session 05: Throughput + Pre-TTT + TTT Audit

**Date**: 2026-03-29
**Status**: Complete (audit only — no implementation)
**Anchor reference**: Session 03 sliding s64 val_bpb `1.12904446`, step_avg `91.37 ms`, 6564 steps, artifact `15751324 bytes`

## Fixed Facts

| Metric | Anchor (S03) | 2026-03-22 Record | #1 Record (pre-TTT) | #1 Record (post-TTT) | Threshold |
|---|---|---|---|---|---|
| val_bpb (sliding s64) | 1.1290 | 1.1233 | 1.1218 | 1.1194 | ≤1.1178 |
| step_avg | 91.37 ms | ~84 ms | 83.4 ms | — | — |
| steps | 6564 | ~7100 | 7185 | — | — |
| artifact | 15.75 MB | 15.55 MB | 15.95 MB | — | ≤16 MB |

Gap from anchor to threshold: **0.0112 BPB**

## Key Reference: 2026-03-22 Record as Portability Bridge

The 2026-03-22 record (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`) uses the **same architecture class** as our anchor:
- CastedLinear modules (not Parameter Banking)
- DDP wrapping
- Standard sequential Muon (not Parallel Muon)

It additionally includes: FA3, VE128 (layers 9-10), Tight SWA (every 50 steps), warmdown 3500, Late QAT (threshold 0.15), and GPTQ-lite clip search.

**Attribution framing**: The 2026-03-22 README attributes only **-0.0013 BPB** to its incremental changes over PR #374 (GPTQ-lite: -0.0006, EMA: -0.0006, warmdown 3500: -0.0002, QAT@0.15: -0.0001). The remaining gap between our anchor (1.1290) and 2026-03-22 (1.1233) involves the full PR #374 feature set (FA3, VE128, SWA, etc.) and cannot be cleanly decomposed into per-feature deltas from cross-record comparison alone.

This record is used as a **compatibility and portability bridge** — it proves these features work together in our architecture class. It is the primary porting reference for first-wave changes. Per-feature gains on our specific anchor must still be measured via isolated deltas.

---

## Audit 1: Throughput (91.37ms → 83.4ms)

### Gap Decomposition (Hypothesized, Not Measured)

| Contributor | Hypothesis | In Anchor? | Confidence |
|---|---|---|---|
| **FA3 vs SDPA** | Largest single contributor | No | MEDIUM |
| **Parallel Muon** (async 3-phase overlap) | Secondary contributor | No | MEDIUM |
| **No DDP wrapper** (manual grad comm) | Minor contributor | No | LOW |
| **Batched NS5** (3D bmm on banks) | Minor contributor | No | LOW |

The 8ms gap is a cross-record observation. The per-contributor split is hypothesis, not measured decomposition.

### FA3 Assessment

**Recommendation**: FA3 is the first throughput target.

**Why**: It is architecturally independent of Parameter Banking and Parallel Muon. The 2026-03-22 record proves it works in our CastedLinear/DDP/standard-Muon architecture at ~84ms/step.

**Follow-up benchmark result** (`scripts/bench_fa3_vs_sdpa.py`, single H100, isolated attention kernel):

- NGC `26.03`: SDPA flash `1.967 ms/iter`
- NGC `25.02` + installed `flash_attn_3` wheel:
  - SDPA flash `1.889 ms/iter`
  - direct FA3 `0.165 ms/iter`
  - relative kernel speedup: `11.44x`

This benchmark is enough to justify an isolated FA3 training delta, but it is **not** an end-to-end training throughput measurement.

**Container stance after benchmark**:
- NGC `26.03` remains the standard stable container path for ordinary runs
- NGC `25.02` + installed FA3 wheel is the current explicit-FA3 experiment path

**Tensor layout change** (anchor `CausalSelfAttention.forward`, lines 616-634):

| Aspect | SDPA (current) | FA3 (target) |
|---|---|---|
| Q/K/V layout | B,H,T,D (`.transpose(1,2)`) | B,T,H,D (no transpose) |
| Attention call | `F.scaled_dot_product_attention(q, k, v_sdpa, is_causal=True, enable_gqa=...)` | `flash_attn_3_func(q, k, v, causal=True)` |
| Post-attention | `.transpose(1,2).contiguous()` | Direct reshape to (B,T,D) |
| RoPE cache shape | `(1, 1, T, rd/2)` | `(1, T, 1, rd/2)` |
| q_gain broadcast | `[None, :, None, None]` | `[None, None, :, None]` |

### Throughput Impact

Unknown until measured in the full anchor loop. The microbenchmark strongly favors direct FA3 at the kernel level, but the real training-step effect still depends on projection layers, RoPE, optimizer cost, DDP, and backward pass overhead. A smoke run must measure actual `step_avg` before stronger claims.

---

## Audit 2: Pre-TTT Stack Gap (1.1290 → 1.1218)

### First-Wave Features (Portable to CastedLinear Architecture)

All features below are present in the 2026-03-22 record and confirmed compatible with our architecture class.

#### FW-1: Flash Attention 3

- **Porting reference**: 2026-03-22 `train_gpt.py`, attention call site
- **Confidence in benefit**: HIGH (throughput), magnitude UNKNOWN
- **Effort**: 2-4 hours
- **Artifact impact**: None
- **Risk**: MEDIUM (experiment container and launcher path)
- **Dependencies**: None
- **Gate**: use the measured benchmark-backed FA3 experiment path; do not assume the kernel-only speedup transfers directly to training

#### FW-2: Value Embedding (VE128, Layers 9-10)

- **Porting reference**: 2026-03-22 `train_gpt.py`, `ValueEmbedding` class
- **What it does**: Reinjects token identity into attention values via a shared embedding (dim=128) projected to kv_dim, with learnable per-layer scale parameters
- **Confidence in benefit**: MEDIUM (present in all top records, but isolated contribution unknown)
- **Effort**: 2-3 hours
- **Artifact impact**: +~100KB (128×1024 embed + 128→kv_dim proj + 2 scales). Fits in 248KB headroom.
- **Risk**: LOW
- **Dependencies**: None

#### FW-3: Warmdown 3500

- **What it does**: Changes `warmdown_iters` from 3000 to 3500. Under wallclock-based warmdown, this starts the decay earlier.
- **Confidence in benefit**: LOW-MEDIUM (-0.0002 claimed in 2026-03-22 README)
- **Effort**: 0.5 hours (single constant)
- **Artifact impact**: Neutral
- **Risk**: VERY LOW
- **Dependencies**: None

#### FW-4: Tight SWA (Every 50 Steps)

- **Porting reference**: 2026-03-22 `train_gpt.py`, SWA collection logic
- **What it does**: Accumulates model state snapshots every 50 steps when lr_scale < 0.2 (during warmdown). Averages them for final evaluation alongside EMA.
- **Confidence in benefit**: MEDIUM (present in all top records)
- **Effort**: 1-2 hours
- **Artifact impact**: None (CPU-side state only)
- **Risk**: LOW
- **Dependencies**: Best with FW-3 (earlier warmdown start = more SWA snapshots)

#### FW-5: Late QAT (Threshold 0.15)

- **Porting reference**: 2026-03-22 `train_gpt.py`, `CastedLinear.forward` STE branch
- **What it does**: Enables straight-through-estimator fake quantization in CastedLinear when lr_scale drops below 0.15, training the network to be quantization-aware late in training.
- **Confidence in benefit**: LOW (-0.0001 claimed)
- **Effort**: 1-2 hours
- **Artifact impact**: Smaller roundtrip-to-prequant gap
- **Risk**: MEDIUM — torch.compile(fullgraph=True) may not handle the dynamic toggle of `_qat_enabled`. Needs testing.
- **Dependencies**: Test after FW-1

#### FW-6: LeakyReLU(0.5)² (Re-test with FA3)

- **Session 04 result**: -0.000003 BPB isolated (NEUTRAL), +0.72ms/step, -168KB artifact
- **Hypothesis**: Throughput coupling ate the gain. With FA3 headroom, the +0.72ms penalty should be absorbed.
- **Confidence in benefit**: LOW-MEDIUM (coupling hypothesis plausible but unverified)
- **Effort**: 0.5 hours
- **Artifact impact**: -168KB (measured, frees headroom)
- **Risk**: LOW
- **Dependencies**: After FW-1 (requires FA3 throughput headroom)

### Second-Wave Features (Hard, Deferred)

#### SW-1: Parameter Banking

- **What it does**: Replaces 66 separate CastedLinear modules with 4 contiguous 3D `nn.Parameter` banks. Enables batched Newton-Schulz.
- **Ablation claim**: ±0.0000 in isolation
- **Effort**: 8-12 hours (full architecture rewrite: init, forward threading, export unbanking)
- **Risk**: HIGH
- **Why deferred**: The 2026-03-22 record achieves 1.1233 without it. Only needed if Parallel Muon (SW-2) throughput gain is critical.

#### SW-2: Parallel Muon (Async 3-Phase Overlap)

- **What it does**: Async reduce-scatter for banks → Adam on non-bank params while RS in-flight → wait for RS + batched NS5 + async all-gather. Removes DDP.
- **Effort**: 6-8 hours
- **Risk**: HIGH (requires SW-1, removes DDP, manual grad communication)
- **Why deferred**: Coupled to Parameter Banking. The 2026-03-22 record runs standard Muon at ~84ms.

#### SW-3: Bigram 3072

- **Confusion**: #1 ablation tested 2048→3072 (-0.0009 claimed), but #1 submission uses 1536
- **Effort**: 1 hour, but artifact size constraint is binding (248KB headroom)
- **Why deferred**: Direction ambiguous. Current anchor (2048) matches the 2026-03-22 reference.

---

## Audit 3: TTT Correctness

### Protocol Summary

The #1 record implements score-first TTT in `eval_val_sliding_ttt()` (lines 1074-1229):

1. Validation tokens split into ~61 non-overlapping 32K-token chunks
2. For each chunk:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` — disables gradient tracking and prohibits in-place weight mutation. Model weights are frozen during scoring.
   - **TRAIN**: SGD(lr=0.002, momentum=0.9, cosine decay) on the already-scored chunk. 3 epochs, all blocks unfrozen, grad clip 1.0.
3. Chunk N is scored by model adapted on chunks 0..N-1
4. Last chunk is scored but never trained on

### Legality Assessment

The score-first protocol **appears compliant and matches public precedent** (PR #461 framework):

- `torch.inference_mode()` disables gradient computation and prohibits in-place weight mutation during scoring
- Each chunk is scored before any adaptation that could depend on it
- Causal attention within windows prevents intra-window leakage
- The last chunk is never trained on, maintaining a clean evaluation boundary

This assessment is based on code inspection and consistency with the published PR #461 protocol. It is not a formal verification or challenge organizer ruling.

### Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Chunk size | 32,768 tokens | #1 env var `TTT_CHUNK_TOKENS` |
| Optimizer | SGD + momentum(0.9) | #1 env var `TTT_MOMENTUM` |
| Learning rate | 0.002 (cosine decay across chunks) | #1 env var `TTT_LR` |
| Epochs per chunk | 3 | #1 env var `TTT_EPOCHS` |
| Frozen blocks | 0 (all adapt) | #1 env var `TTT_FREEZE_BLOCKS` |
| Gradient clip | 1.0 | #1 env var `TTT_GRAD_CLIP` |
| Batch seqs | 32 | #1 env var `TTT_BATCH_SEQS` |

### Eval-Time Budget

| Phase | Estimated Time |
|---|---|
| Training | 600s (≤10 min) |
| Standard eval (int6 roundtrip + sliding) | ~120s |
| TTT scoring (inference_mode, all chunks) | ~200s |
| TTT training (60 chunks × 3 epochs) | ~210s |
| **Total eval** | **~530s** (fits 10-min eval budget) |

### Portability

The TTT function is self-contained. Porting requires:
1. Copy `eval_val_sliding_ttt()` (lines 1074-1229)
2. Add hyperparameters to `Hyperparameters` dataclass
3. Add SGD optimizer creation for TTT phase
4. Ensure `forward_logits()` exists on anchor model (or add thin wrapper)
5. Call after standard sliding eval

Multi-GPU handling is built in (the function already does `dist.all_reduce` for gradients).

**Measured gain on #1 stack**: -0.0025 BPB (3-seed mean, std 0.0006). Gain on our stack is unknown until measured.

### Engineering Cost

Estimated 4-6 hours to integrate and validate:
- 2h: Port function and add config
- 1h: Add forward_logits wrapper and test
- 1-3h: Run on Pegasus 8xH100, validate results, debug if needed

---

## Recommendation

### Implementation Lanes

**Lane A: Isolated Audit Path** (strict attribution)
- Each feature tested as isolated delta against Session 03 anchor
- Slower but produces defensible per-feature evidence
- Recommended starting approach

**Lane B: Reproduction-Oriented Path** (leaderboard speed)
- Port full 2026-03-22 feature set as one bundle in a dedicated branch
- Faster path to competitive BPB, loses per-feature attribution
- Switch to this if Lane A progress is too slow or if leaderboard urgency increases

**Start with Lane A for FA3**, then reassess after 2-3 isolated deltas.

### Concrete Next Steps

| Step | Change | Gate |
|---|---|---|
| **0** | Implement FA3 isolated delta (Session 05 FW-1) | Use `25.02` + installed FA3 wheel path |
| **1** | Run short Pegasus smoke to measure actual `step_avg` and stability | FW-1 code compiles and launches |
| **2** | Full `600s` FA3 run | Smoke is healthy |
| **3** | VE128 isolated delta | FA3 is positive or neutral-but-faster |
| **4** | Stack winners, measure combined | Steps 0-3 measured |
| **5** | Warmdown 3500 + SWA + Late QAT | Sequential isolated deltas or stacked |
| **6** | LeakyReLU² re-test | After FA3 integrated (throughput-coupling test) |
| **7** | TTT integration | On stacked first-wave base with measured pre-TTT BPB |

### Scenario Analysis

| Scenario | Pre-TTT | Post-TTT | Notes |
|---|---|---|---|
| **Optimistic**: First-wave features stack well, TTT works | ~1.122 | ~1.119 | Requires multiple features to contribute |
| **Moderate**: FA3 helps throughput, 1-2 others help BPB | ~1.125 | ~1.122 | Still above threshold without Banking |
| **Pessimistic**: Only throughput gains, quality features ~neutral | ~1.127 | ~1.124 | Would need second-wave for threshold |
| **Best-case with second-wave** | ~1.120 | ~1.117 | Requires Banking + Parallel Muon |

These are scenarios, not forecasts. Each measured step narrows the range.

### Out of Scope

- MTP, Gated Attention, Value Residual, DTG, LAWA (all disabled in #1)
- GPTQ-lite clip search (proven to violate artifact cap — Session 04 Delta 1)
- Larger tokenizer / vocab changes
- Parameter Banking / Parallel Muon in first-wave

---

## Session 05 Decisions

1. **2026-03-22 record is the primary first-wave porting reference** for pre-TTT features. It shares our CastedLinear/DDP/standard-Muon architecture.
2. **2026-03-23 #1 record is the TTT reference** for score-first protocol porting.
3. **FA3 is the first implementation target** — highest expected throughput contribution, architecturally independent.
4. **Parameter Banking and Parallel Muon are second-wave** — not needed based on 2026-03-22 evidence.
5. **LeakyReLU² re-test is gated on FA3** — the throughput-coupling hypothesis must be tested, not assumed.
6. **Lane A (isolated deltas) is the default** until per-feature evidence accumulates or time pressure forces Lane B.
