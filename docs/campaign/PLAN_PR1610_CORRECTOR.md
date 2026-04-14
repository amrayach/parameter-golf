# Plan: PR #1610 Reproduction + Posterior Corrector

Revision 3 (2026-04-14). Locked after 8-model convergent review.

## Context

- Arc: #1101 (1.129) -> #1307 (1.110) -> #1598 (1.081, frozen) -> this PR (target <=1.070)
- Clean legal frontier: PR #1610 at 1.0728 BPB
- Budget: $212 RunPod (~35 runs), 16 days remaining
- Base: use #1610's `train_gpt.py` directly (NOT patch D variant)

## Why #1610 Direct, Not D-Patch

Our D variant is a 558-line LZMA-minified file missing critical #1610 features:
BatchedTTTLoRA, phased TTT eval loop, DocumentPackingLoader, Triton fused MLP kernel,
`_build_cu_seqlens`, weight banking. Patching D to match #1610 would require
reconstructing 2800+ lines. Using #1610 directly and adding the corrector is safer.

PR #1610 body says "builds directly on #1530, training is unchanged" but the code
has completely different defaults: vocab 8192 vs 1024, 11 vs 9 layers, mlp_mult 4 vs 2,
seq_len 2048 vs 1024, int6+brotli vs int8+zlib. The body is misleading.

Verified at pinned SHAs:
- #1610: `ca1919539dc6e328ea890cb03ad3ca1c5a84da55`
- #1530: `7dca3ded46d2d6f537fda34777e5d0cad47f9232`

## Byte Budget (from #1610 logs)

Counted = compressed model + brotli-compressed code (per Issue #1017).

| Seed | Total counted bytes | Headroom (vs 16,000,000) |
|------|--------------------:|-------------------------:|
| 0    | 15,996,697          | 3,303                    |
| 1    | 15,995,985          | 4,015                    |
| 2    | 15,988,805          | 11,195                   |

Adding corrector code (~1-3 KB uncompressed) increases compressed code by ~200-800 bytes.
Must monitor `_compressed_code_size()` after every corrector change.

## Corrector Formulation

Full-vocab prefix-only posterior corrector with unigram base and sparse residual updates.
Math = implementation (one path, no drift):

```python
# For position t, context distribution q_t(v):
#   q_t(v) = (unigram_count[v] + ngram_bonus[v]) / Z_t
# where:
#   unigram_count[v] = count of token v in scored prefix [0, t) (Laplace +1)
#   ngram_bonus[v]   = sum of matching n-gram continuation counts for v
#   Z_t = sum over all v of (unigram_count[v] + ngram_bonus[v])
#
# Context logit bias: context_logits[v] = log(q_t(v))
# Final: p_t(v) = softmax(neural_logits[v] + alpha * context_logits[v])

log_u = torch.log(unigram_counts + 1)       # [V], Laplace-smoothed
log_Z = torch.logsumexp(log_u, dim=0)        # scalar
base_bias = alpha * (log_u - log_Z)           # [V], once per chunk

# Per position: sparse delta from n-gram hits only
delta = torch.zeros(V)                        # reused buffer
for (tok, count, total) in ngram_hits:
    delta[tok] = alpha * (log((count+1)/(total+V)) - (log_u[tok] - log_Z))
logits_f[:, pos, :] += base_bias + delta
delta.zero_()
```

Hard constraints:
1. NO dense [B, S, V] correction tensor
2. Prefix-scan (left-to-right) mandatory for update path
3. Score-before-update: query(t) reads [0,t), then score, then update(t, x_t)
4. Laplace base guarantees full-vocab support (Condition 2)

## Corrector + PhasingTTT Interaction

Reset corrector state after global SGD. The n-gram statistics were accumulated against
the pre-SGD model; keeping stale corrections adds noise.

## Execution Phases

### Phase 0: Pre-Implementation Gates (before any coding)

- **Gate 0.1**: Feature checklist diff (D vs #1610) -- already completed in plan
- **Gate 0.2**: Byte budget table -- already completed above
- **Gate 0.3**: 1-page legality spec answering: why valid under Condition 2, how
  score-before-update enforced, how chunking boundary handled, no realized-token dependence

### Phase 1: Zero-Cost Implementation (Apr 14-20)

**Phase 1A: Baseline Reproduction Runbook** (FIRST deliverable)

1. Dependency gate: verify exact deps from #1610 requirements.txt at pinned SHA
2. No-train startup smoke test: import all deps, instantiate model, verify shapes
3. Create branch `submission/pr1610-corrector` from main
4. Copy #1610's exact `train_gpt.py` as base
5. Warmup safety: verify no real val tokens used for compile warmup

**Phase 1B: Corrector Implementation** (AFTER 1A runbook is complete)

Staged approach:
- Step A: Single-order logistic-domain n-gram mixing with Laplace smoothing
- Step B: Multi-order (3 orders) with count mixing across orders
- Step C: Only if B shows signal, add confidence gating and order weighting

Performance: must use vectorized PyTorch or C++ extension, not pure Python dicts.
CPU microbenchmark gate: projected overhead must be < 50s on full eval.
GPU in-situ benchmark gate: projected total eval time must be < 580s.

Hash table memory: start with `table_bits=20` (36 MB total, negligible vs model).

Unit tests (`tests/test_corrector.py`):
1. `test_causality` -- position t uses only tokens 0..t-1
2. `test_full_vocab_normalization` -- softmax sums to 1.0 at every position
3. `test_score_before_update` -- hash state unchanged until after score
4. `test_no_realized_token_dependence` -- correction at t independent of x_t
5. `test_single_pass` -- same result regardless of chunk boundaries
6. `test_laplace_nonzero` -- every vocab entry gets nonzero context_logits
7. `test_reset_after_sgd` -- state cleared after reset()
8. `test_no_dense_bsv_tensor` -- no [B,S,V] allocation in production path

Package dry-run gate: assemble submission folder with stub, verify < 16,000,000 bytes.

### Phase 2: Compute Execution (Apr 21-27)

**Step 2.1: Baseline Reproduction (2 runs, ~$12)**

Use #1610's published seeds {0, 1, 2}.

- Run 0a: #1610 base, LoRA TTT only. Expected ~1.073 BPB.
  - Gate A: seed-0 within 0.003 of published 1.07258, eval < 600s, artifact < 16MB
  - Kill: BPB > 1.078 -> investigate. BPB > 1.085 -> abort.
- Run 0b: Same + PhasingTTT (requires Gate A pass). Expected ~1.072 BPB.
- Run 0c: Seeds 1, 2 with best config.
  - Gate B: 3-seed mean within 0.002 of published. ALL corrector compute blocked until Gate B passes.

**Step 2.2: Corrector Ablation (3-6 runs, ~$36)**

First test is eval-only on existing checkpoint (no fresh training).

| Run | Step | Alpha | Orders   | table_bits | Kill? |
|-----|------|------:|----------|------------|-------|
| 1a  | A    | 0.3   | [8]      | 20         | No    |
| 1b  | B    | 0.3   | [5,8,12] | 20         | No    |
| 1c  | B    | 0.1   | [5,8,12] | 20         | YES   |
| 1d  | B    | best  | best     | 22         | If signal |
| 1e  | B    | best  | best     | best       | If 1d improves |
| 1f  | C    | best  | best     | best       | Final |

Kill after 1a-1c: all three < 0.001 BPB gain -> KILL corrector, go to fallback.

**Step 2.3: Multi-Seed Validation (3-5 runs, ~$30)**

Seeds {0, 1, 2} then {1337, 42}. Need 3-seed mean <= record bar, p < 0.01 vs SOTA.

**Budget: $108 allocated + $104 unallocated**

### Phase 3: Polish and Submit (Apr 27-30)

- Minify corrector to < 2KB
- Write submission README (4-PR arc narrative)
- Submit record-track PR
- Buffer for reviewer questions

## Fallback Cascade

If corrector < 0.001 BPB after kill criteria:

1. **Level 1A (export-only)**: per-layer adaptive clip_sigmas, int7 embeddings. 1-2 requant runs (~$12). 1-day time-box.
2. **Level 1B (retrain-required)**: MATRIX_LR=0.026. 1 run (~$6). Only if 1A shows partial signal.
3. **Level 2**: Clean reproduction + corrector evidence + rate-distortion writeup. Frame as hypothesis test.
4. **Level 3**: Non-record writeup only with #1598 evidence package.

## Session Execution Split

| Session | Scope | Produces |
|---------|-------|----------|
| A | Sync docs, create branch, dependency gate | Branch, pinned source, runbook checklist |
| B | Baseline reproduction only | Gate A/B pass, seed results |
| C | Corrector skeleton + legality tests + benchmark | test_corrector.py, microbenchmark |
| D | First eval-only corrector trial | Ablation results, kill/proceed decision |
| E | Fallback / README / submission polish | Final PR |

## Critical Files

| File | Role |
|------|------|
| `/tmp/pr1610_train_gpt_pinned.py` | #1610 source at SHA ca191953 -- THE source base |
| `/tmp/pr1530_train_gpt_full.py` | #1530 source (cross-reference only) |
| `docs/campaign/AGENT_SYNC.md` | Current source of truth for campaign state |
| `docs/campaign/PLAN_PR1610_CORRECTOR.md` | This file -- execution plan |

## Verification Checklist (before claiming any step complete)

- [ ] Microbenchmark: corrector overhead < 50s projected
- [ ] Code correctness: all unit tests pass
- [ ] Legality: 1-page spec verified against Issue #1017
- [ ] Artifact size: wc -c < 16,000,000
- [ ] Eval time: < 600s (prefer < 590s)
- [ ] BPB sanity: no result below 0.8 (almost certainly broken)
- [ ] Reproducibility: 3+ seeds, stddev < 0.002
- [ ] Warmup safety: no real val tokens for compile warmup
