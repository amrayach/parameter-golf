# Agent Sync

Date: 2026-03-31

## Current Objective

Session 05g (XSA-8 throughput recovery) is the active next candidate. 05c-plus remains the best measured branch.

Session 05c-plus is **measured and quality-positive** (sliding s64 `1.12558`, delta `-0.00347` vs anchor) but throughput regressed (`100.39 ms`, +9ms).
Session 05f is **measured and worse than 05c-plus** (`1.12661`, delta `+0.00103` vs 05c-plus) with no throughput recovery.
Session 05g reduces `xsa_last_n` from 11 to 8 on the 05c-plus base to test whether partial XSA scope reduction recovers throughput while retaining quality.

GPTQ is **permanently parked** — failed on both relu² anchor (7 ablations) and leaky_relu² 05c-plus (05e probe: 44/66 layers worse than naive).

## Challenge Reality

- Official leaderboard entry is **record-gated**, not top-5-open-entry.
- A record submission must beat the current official SOTA by at least `0.005` nats and show `p < 0.01`.
- Current official merged #1 is PR #1019 at `1.1147` BPB (3-seed mean `1.88218` nats).
- Record threshold: `<= 1.87718` nats.
- Current open frontier is lower:
  - PR #1089: `1.1086` BPB, 3-seed mean
  - PR #1060: `1.1122` BPB, 3-seed mean

## Current Mainline Plan

### Phase 1: Session 05c-plus training bundle (MEASURED — quality positive, throughput regressed)

Plan: `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
Code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

Four changes on the Session 03 anchor:
1. **XSA 4 → 11** — XSA on all layers (trivial constant)
2. **VE128 on layers 9-10** — shared ValueEmbedding (new module)
3. **Warmdown 3000 → 3500** — trivial constant
4. **LeakyReLU(0.5)²** — replaces relu in MLP (one line, aligns with PR #1019)

SWA is **not included** — dead code in both PR #1019 and #634 (collected but only EMA applied).

#### 8xH100 measured result (2026-03-31)

| Metric | 05c-plus | Anchor | Delta |
|--------|----------|--------|-------|
| sliding s64 val_bpb | **1.12557920** | 1.12904446 | **-0.00347** |
| pre_quant EMA exact | 1.14186715 | 1.14472403 | -0.00286 |
| int6 roundtrip exact | 1.14933197 | 1.15247273 | -0.00314 |
| step_avg_ms | **100.39** | 91.37 | **+9.02** |
| steps | 5977 | 6564 | -587 |
| bytes_total | 15,589,271 | 15,751,324 | -162,053 |

**Assessment**: Quality-positive (sliding s64 improved by 0.00347 despite 587 fewer steps). Throughput regressed materially (+9ms, exceeds the +5ms gate). Not a seed-validation branch yet. 05c-plus remains the best measured branch even after the 05f follow-up failed to improve it.

### Phase 3: Session 05f refinement (MEASURED — NEGATIVE vs 05c-plus)

Code: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py`

Three additional changes on the 05c-plus base:
1. **BigramHash vocab 2048 → 3072** — reduces hash collisions
2. **BigramHash dim 128 → 112** — partially offsets parameter increase
3. **Warmdown 3500 → 4000** — more gradual cooldown

#### 8xH100 measured result (2026-03-31)

| Metric | 05f | 05c-plus | Delta vs 05c-plus |
|--------|-----|----------|-------------------|
| sliding s64 val_bpb | **1.12660664** | 1.12557920 | **+0.00103** |
| pre_quant EMA exact | 1.14190308 | 1.14186715 | +0.00004 |
| int6 roundtrip exact | 1.15026661 | 1.14933197 | +0.00093 |
| step_avg_ms | 100.51 | 100.39 | +0.12 |
| steps | 5977 | 5977 | 0 |
| bytes_total | 15,630,854 | 15,589,271 | +41,583 |

**Assessment**: Negative follow-up. BigramHash 3072x112 + warmdown 4000 did not improve 05c-plus and did not recover throughput. 05c-plus remains the best measured branch in this line.

### Phase 4: Checkpoint diagnostics (DONE — conclusions below)

Diagnostic conclusions (from 05c-plus and 05f checkpoint analysis):
- Export / quantization is not the current bottleneck
- 05c-plus and 05f have nearly identical float→int6 damage profiles
- No strong late-block-only quantization hotspot
- mlp.proj zeroing is real but diffuse, not the next tactical lever
- 05f failed because of training-side changes, not quantization
- VE is alive enough to keep
- Bigram is not the next lever
- XSA scope is the first knob to relax for throughput recovery

### Phase 5: Session 05g — XSA-8 throughput recovery (NEXT)

Code: `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py`

Single change on the 05c-plus base:
- `xsa_last_n` 11 → 8 (XSA on layers 3-10, removed from layers 0-2)

Hypothesis: XSA scope increase (4→11) is the leading contributor to the +9ms throughput regression. Reducing to 8 should partially recover throughput while retaining most of the quality gain.

Next steps:
1. Sync to Pegasus: `git pull` on `/netscratch/$USER/parameter-golf`
2. Run 1xGPU smoke (see README in 05g folder)
3. If smoke passes, run 8xH100 full (see README in 05g folder)
4. Compare vs 05c-plus on sliding s64, step_avg_ms, and artifact size

### Phase 2: GPTQ probe on 05c-plus architecture (DONE — NEGATIVE)

Session 05e: `records/track_non_record_16mb/2026-03-31_05e_gptq_probe/`
- Pre-quant EMA exact replay: `3.95543154`
- Naive int6 roundtrip exact replay: `3.96902897`
- GPTQ int6 roundtrip exact replay: `3.96902897`
- Result: **worse_than_naive_rowmax = 44/66 (67%)** — kill threshold exceeded
- No same-checkpoint BPB gain over naive replay
- LeakyReLU(0.5)² + VE128 did NOT unblock GPTQ
- GPTQ parked permanently for this model family

### Parked

- Session 05b GPTQ on current anchor (7 ablations, all failed, code proven correct)
- Saved-container FA3 throughput path
- TTT
- Broad novelty probes

## Session 05b GPTQ: Conclusive Parking Summary

Seven ablations on the same Session 03 checkpoint:

| # | Variant | gptq_diag | Roundtrip gap | Outcome |
|---|---------|-----------|---------------|---------|
| 1 | Initial smoke (1xH100) | 66/66 | +0.212 | Bug found |
| 2 | Loop fix + percentile search | 66/66 | +0.335 | Still bad |
| 3 | actorder=False | 66/66 | +0.395 | Worse |
| 4 | block_size=full | 66/66 | +0.395 | No change |
| 5 | Hessian normalize+damp | 66/66 | +0.337 | Identical |
| 6 | PR #1019 verbatim transplant | 66/66 | +0.337 | **Byte-identical MSE** |
| 7 | AR self-gen calibration | crash | N/A | Non-PD Hessian |

Key conclusion: ablation #6 proves the GPTQ code is functionally correct. The failure is model-specific, not a code bug. PR #1019 uses `leaky_relu(0.5)` while our anchor uses `relu`.

### Session 05e GPTQ Probe: Falsification Result (2026-03-31)

Tested whether LeakyReLU(0.5)² + VE128 unblocks GPTQ. **It does not.**

- Architecture: 05c-plus (warmdown 3500, XSA 11, VE128, LeakyReLU(0.5)²)
- 66 layers GPTQ'd, 68 Hessians collected
- **worse_than_naive_rowmax = 44/66 (67%)** — exceeds 50% kill threshold
- Same-checkpoint replay was flat vs naive:
  - pre-quant exact: `3.95543154`
  - naive roundtrip exact: `3.96902897`
  - GPTQ roundtrip exact: `3.96902897`
- Hessian collection: 2979ms, GPTQ quantization: 31488ms
- Hardware: RTXA6000 serv-2108 (export-only comparison, GPU speed irrelevant)

GPTQ is now parked permanently for this model family. The activation function is not the root cause.

## Fixed Reference Results

- Session 03 anchor (`8xH100`, `serv-3342`)
  - sliding s64 `val_bpb=1.12904446`
  - pre-quant EMA `val_bpb=1.14472403`
  - int6 roundtrip `val_bpb=1.15247273`
  - steps `6564`
  - step_avg `91.37 ms`
  - artifact `15751324` bytes

## Canonical Files

- Shared mutable state: `docs/campaign/AGENT_SYNC.md`
- Append-only measured results: `docs/campaign/results_log.jsonl`
- Stable rules: `CLAUDE.md`
- Checkpoint diagnostic utility: `diagnose_weights.py`
- 05c-plus plan: `docs/superpowers/plans/2026-03-30-session-05c-plus.md`
- 05c-plus code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- 05f follow-up: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/`
- GPTQ experiment (parked): `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/`
- GPTQ probe on 05c-plus: `records/track_non_record_16mb/2026-03-31_05e_gptq_probe/`
- 05c-plus checkpoint diagnostics: `diagnostics/2026-03-31_05c_plus/`
- Codex memory:
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`

## Workspace

- Local repo: `/home/amay/Work/parameter-golf`
- Remote repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` by default.
