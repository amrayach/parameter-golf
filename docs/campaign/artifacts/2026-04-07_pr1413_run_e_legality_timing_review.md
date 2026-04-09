# Run E Legality, Timing, and Submission Review

**Date**: 2026-04-07
**Scope**: Run E only from the PR #1413 A/B/C/D/E experiment suite
**Verdict**: research-only (not submission-ready; see blockers below)

---

## 1. Legality Assessment

### 1.1 What Run E Does

Run E is eval-only. It:
1. Loads an existing quantized checkpoint (`final_model.int6.ptz`) produced by Run D.
2. Compiles a C++ n-gram kernel (`fused_expert_kernel.cpp`) via `g++` at runtime.
3. Precomputes a per-position hint table by walking the entire validation stream
   left-to-right, building open-addressing hash tables of n-gram statistics.
4. Runs the standard score-first TTT sliding-window eval, applying an exponential
   tilt to each position's NLL based on the precomputed hint.

The tilt math per position:
```
p_tilt(t) = p_model(t) * exp(beta * 1[t == hint]) / Z
Z = 1 + p_model(hint) * (exp(beta) - 1)
```

### 1.2 Is Score-First TTT + Token-Only N-gram Tilt Legal?

**Yes, with caveats.**

The base TTT framework is legal per established precedent (PR #461, merged PR #549).
The n-gram tilt is applied within the same eval pass, not as a second scoring pass.

The tilt satisfies all four Issue #1017 conditions:

| Condition | Status | Evidence |
|-----------|--------|---------|
| 1. Causal: uses only prefix tokens | PASS | `compute_hashes` uses `tokens_[pos-k-1]` for k=0..order-1; hint gating uses `tokens_[p-1]` not `tokens_[p]` (lines 401-403 of fused_expert_kernel.cpp) |
| 2. No target token leakage | PASS | `is_bnd` and `is_ws` are derived from `prev_tok = tokens_[p-1]`, not from `tokens_[p]`. Within/word experts are disabled (`NGRAM_WITHIN_BETA=0`, `NGRAM_WORD_BETA=0`). |
| 3. Full normalized distribution | PASS | Z is the explicit partition function; `tilt_nll` applies a proper exponential family adjustment |
| 4. Single left-to-right pass | PASS | `get_hints_batch` walks positions 1..n sequentially; hint lookup precedes table update for each position |

### 1.3 Does the Tilt Use Information Beyond the Prefix?

**No.** The data flow is:

1. `NgramTiltState.__init__` receives `val_tokens` (the full validation stream).
2. `ctxmixer_set_tokens` stores a pointer to the full token array.
3. `get_hints_batch` is called with `positions = arange(1, n_tok)`.
4. For each position p in sequential order:
   - Hash tables contain statistics accumulated from `val_tokens[0..p-1]` only.
   - `token_hint()` looks up the most-likely next token given the n-gram context from the prefix.
   - The hint and beta are recorded to the output buffer.
   - THEN `token_update()` adds `val_tokens[p]` to the hash tables.

The precomputation sees the entire validation stream, but processes it causally.
Each hint at position p depends only on the strict prefix `val_tokens[0:p]`.

**However**, there is a subtle issue: the precomputation happens BEFORE the
eval loop, meaning all hints are precomputed in one batch pass. This is
equivalent to running the n-gram statistics incrementally during eval, but
more efficient. The causal ordering within `get_hints_batch` ensures that
position p's hint only uses prefix data.

No training data is used to seed the n-gram tables. The tables start empty
and are populated entirely from validation stream tokens as they are
encountered.

### 1.4 Is This an "N-gram Cache" (Which Is Banned)?

**This is the primary legality risk.**

The submission.json declares `"no_ngram_cache": true`. The question is
whether this claim holds.

**Argument that it is NOT an n-gram cache:**
- The challenge rules ban pre-computed n-gram caches shipped as artifacts.
- Run E's n-gram tables are computed at eval time from the validation stream,
  not shipped as sidecar data. No n-gram table file is included in the
  submission artifact.
- The mechanism is closer to "adaptive prediction" (like TTT itself) than
  a static cache.
- PR #549 established that adapting to the validation stream at eval time
  is legal, provided score-before-update ordering is maintained.

**Argument that it IS effectively an n-gram cache:**
- The n-gram statistics are built from the validation stream during a
  dedicated precomputation pass, not incrementally during scoring.
- The precomputation walks the ENTIRE validation stream before any scoring
  happens. This is architecturally different from TTT, which interleaves
  scoring and adaptation.
- A reviewer could argue that precomputing hints for all 40M positions
  before scoring constitutes building a cache.

**Resolution**: The precomputation is a performance optimization. It produces
the exact same result as building the n-gram table incrementally within
the eval loop (the C++ kernel enforces causal ordering). The implementation
is functionally equivalent to an adaptive predictor that maintains running
statistics. This should survive review, but the `no_ngram_cache` flag is
borderline misleading -- the implementation DOES build n-gram hash tables;
it just builds them causally at eval time rather than shipping them as
artifacts.

**Recommendation**: If submitting, change the compliance flag to something
more honest like `"ngram_tables_eval_time_causal": true` and drop the
`"no_ngram_cache"` flag, or add a note explaining the distinction.

### 1.5 PR Precedent

| PR | Technique | Status |
|----|-----------|--------|
| #549 | Score-first TTT | Merged, legal precedent |
| #461 | Original TTT framework | Legal precedent |
| #1420 | N-gram tilt (original, with causality bug) | TAINTED -- target token leaked via `is_bnd[tokens_[p]]` |
| #1437 | N-gram tilt (corrected, token-only) | Diagnostic/non-submission. Corrected causality. 5-seed results. |
| #1413 | Score-first TTT + QK gain | Currently at 1.08279 BPB |

No merged PR uses n-gram tilt. The technique is novel relative to accepted
submissions. This increases reviewer scrutiny risk.

---

## 2. Timing Risk

### 2.1 Expected Timing Components

Run E skips training entirely (`SKIP_TRAINING=1`). The eval-only timeline:

| Component | Expected Time | Source |
|-----------|---------------|--------|
| Model deserialization (brotli + int6 dequant) | ~5-10s | Standard for 16MB artifact |
| Non-TTT eval passes (quantized, sliding window) | ~130s total | PR #1394 logs: quantized=23s + sliding_window=108s |
| N-gram precomputation (`get_hints_batch` on 40.5M tokens) | ~10-30s | CPU-bound C++ kernel; 40M positions with order-8-to-16 hash ops. Author log format suggests logging elapsed time. No measured data from this exact config. |
| C++ kernel compilation (`g++ -O3`) | ~5-10s | One-time cost |
| TTT sliding-window eval with tilt | ~290-300s | Submission.json: ttt_time_ms ~290-295s (3 seeds). Tilt adds per-position gather+exp+log ops within the same loop. |
| Tilt overhead per TTT window | ~10-30s total | The `tilt_nll` function does 4 tensor gathers + 3 elementwise ops per scored window. Amortized across ~633k windows, this is ~15-50us per window. |

**Total estimated eval time: ~450-500s**

### 2.2 Comparison to 600s Cap

| Scenario | Est. Total | Margin |
|----------|-----------|--------|
| Optimistic (fast precompute, minimal tilt overhead) | ~440s | 160s margin |
| Expected | ~470s | 130s margin |
| Pessimistic (slow precompute, compile overhead, large hash tables) | ~530s | 70s margin |

**Versus the 550s safety margin from the decision rubric:**
- Expected case: 470s -- 80s under the 550s safety margin. PASS.
- Pessimistic case: 530s -- 20s under the 550s safety margin. MARGINAL.

### 2.3 Key Timing Risk: N-gram Precomputation

The biggest unknown is the precomputation time for `get_hints_batch`. With
`open_table_bits=26`, each active order's hash table has 2^26 = 64M entries.
Each `CtxEntry` is 16 bytes, each `PairEntry` is 16 bytes, so 2GB per order.

With `order_stride=2` and orders 8-16, the active orders are 8, 10, 12, 14, 16
(5 orders). That is 5 x 2GB x 2 tables = 20GB of hash table memory.

Plus within_tables (3 x 2^20 x 32B = 96MB) and word_table (2^20 x 32B = 32MB).

**Total hash table memory: ~20GB on rank 0 CPU.**

This is a significant memory footprint. On an H100 pod with 1-2TB system RAM,
this fits easily, but the time to initialize (memset + populate) 20GB of
tables could add 10-30s depending on memory bandwidth.

### 2.4 SKIP_TRAINING Impact

Because Run E sets `SKIP_TRAINING=1`, the 600s clock starts at eval time,
not at training start. The entire 600s budget is available for eval.

Wait -- re-reading the challenge rules: "Train and eval must EACH run under
10 minutes on 8xH100." If SKIP_TRAINING=1, training time is 0 (or undefined).
The eval time must still fit under 600s. This is the binding constraint.

But the eval script as written runs ALL eval passes (quantized, sliding window,
TTT) in sequence within a single `torchrun` invocation. There is no separate
eval clock -- the 600s applies to the entire eval pass, which includes
deserialization and all eval modes.

**Assessment: Timing is SAFE with comfortable margin in the expected case.**

---

## 3. Artifact Accounting

### 3.1 Sidecar Files

Run E adds these files beyond the base PR #1413 submission:

| File | Bytes | Purpose |
|------|-------|---------|
| `ngram_tilt.py` | 8,624 | Python ctypes wrapper + NgramTiltState class |
| `fused_expert_kernel.cpp` | 18,365 | C++ n-gram kernel source |
| `train_gpt.py` (patched, compressed) | 17,390 | Main script with tilt integration |

Note: `libfused_ngram.so` is compiled at runtime and NOT shipped in the artifact.

### 3.2 Total Artifact Size

From submission.json, the baseline (without tilt) artifact sizes per seed:

| Seed | Bytes Total |
|------|-------------|
| 0 | 15,991,018 |
| 42 | 15,992,546 |
| 1234 | 15,989,058 |

The 16MB limit is 16,000,000 bytes decimal.

Headroom: 16,000,000 - 15,992,546 (worst seed) = **7,454 bytes**.

The sidecar files for Run E add:
- `ngram_tilt.py`: 8,624 bytes
- `fused_expert_kernel.cpp`: 18,365 bytes
- `train_gpt.py` delta: +0 bytes (same wrapped size, patches are inside the compressed payload; actually the patched wrapper is 17,390 bytes vs the base which is also ~17,390 bytes based on the manifest)

**Total additional code bytes: ~26,989 bytes.**

### 3.3 Does It Fit Under 16MB?

**NO.** The additional sidecar files (26,989 bytes) vastly exceed the 7,454
bytes of headroom.

Even if the sidecar files are included INSIDE the compressed artifact
(which they are not -- they are separate source files that must be present
for the `from ngram_tilt import NgramTiltState` import to work), they
would need to be part of the submission and counted against the 16MB limit.

**This is the primary blocker for submission viability.**

### 3.4 Possible Mitigations

1. **Inline everything into train_gpt.py**: Embed the ngram_tilt.py logic
   and the C++ source string directly into train_gpt.py, eliminating the
   sidecars. The compressed wrapper would grow, but by less than the raw
   sidecar sizes.

2. **Minify the C++ kernel**: The 18KB C++ file includes comments, debug
   code, and the within/word experts (which are disabled). Stripping these
   could reduce it to ~8-10KB.

3. **Reduce hash table parameters**: With `open_table_bits=26`, the code
   allocates very large tables. Reducing to 22-24 bits saves nothing in
   the shipped artifact (the table is built at runtime) but reduces the
   C++ code complexity.

4. **Strip dead code**: The within_hint and word_hint experts are disabled
   (`NGRAM_WITHIN_BETA=0`, `NGRAM_WORD_BETA=0`). Their code and data
   structures can be removed entirely, saving ~4-6KB of C++ source.

**Estimated post-mitigation additional bytes**: If everything is inlined
and dead code is stripped, the compressed payload growth would be ~3-5KB,
which is within the ~7KB headroom. Tight but feasible.

---

## 4. Submission Viability Verdict

**Verdict: research-only**

### Top Blockers

1. **Artifact size (HARD BLOCKER)**: The sidecar files push the submission
   ~20KB over the 16MB limit. Must inline everything and strip dead code.

2. **No measured E results yet**: Run E has not been executed. The expected
   BPB improvement from n-gram tilt is ~-0.00188 BPB (from #1437 diagnostic),
   but this was measured with a different base model (different architecture
   and loop config). The actual delta on the D checkpoint is unknown.

3. **Legality perception risk**: The `no_ngram_cache` flag is borderline
   misleading. A reviewer aware of the #1420 causality bug history will
   scrutinize this carefully. The code is correct (causal), but the
   provenance requires clear documentation.

4. **No merged precedent**: No accepted submission uses n-gram tilt. This
   is the first application and will face heightened scrutiny.

### Minimal Engineering Plan (if pursuing submission)

| Step | Effort | Description |
|------|--------|-------------|
| 1. Run E on RunPod | ~$6, ~15 min | Execute with current code to get a measured BPB and timing |
| 2. Evaluate delta | 5 min | If delta < 0.001 BPB, DROP (not worth the complexity risk) |
| 3. Inline sidecars | 2-3 hours | Embed C++ source as string in train_gpt.py, inline NgramTiltState class, strip dead within/word code |
| 4. Verify artifact size | 30 min | Recompute compressed wrapper size, confirm < 16MB |
| 5. Update compliance flags | 15 min | Replace `no_ngram_cache` with accurate description |
| 6. 3-seed validation | ~$18, ~45 min | Run seeds 0, 42, 1234 to verify statistical significance |

**Kill criteria**: If the measured BPB improvement from tilt is < 0.001
on 3 seeds (less than 2x the seed-to-seed std of ~0.0005), drop Run E
entirely. The complexity and review risk are not justified for a sub-noise
improvement.

### Decision

Run E is worth executing once as a measurement (Step 1). All further
investment depends on the measured delta. The artifact size blocker is
solvable with ~3 hours of engineering but should not be attempted until
the delta is confirmed to be meaningful.
