# Legality Specification: Full-Vocab Posterior Corrector

Gate 0.3 document. Answers the four required questions from the execution plan.

## Background

The corrector adds a context-dependent logit bias to the base model's output
logits before scoring. At position t, the bias is derived from token statistics
in the already-scored prefix [0, t). The final scored distribution is:

    p_t(v) = softmax(neural_logits_t(v) + alpha * context_logits_t(v))

where `context_logits_t(v) = log(q_t(v))` and q_t is a Laplace-smoothed unigram
distribution augmented by sparse n-gram hits.

## Question 1: Why is the corrector valid under Condition 2?

**Condition 2** (Issue #1017): "The model must output a proper probability
distribution over all tokens at every position — no tokens may be assigned
zero probability."

**Answer**: `softmax(neural_logits + alpha * context_logits)` produces a
full-vocab distribution for any real-valued `context_logits` vector, because:

1. `softmax` is defined over all V tokens and produces positive values for
   every entry (exp(x) > 0 for all finite x).
2. `context_logits[v] = log(q_t(v))` where `q_t(v) >= 1/Z_t` for all v,
   because every token has Laplace-smoothed count >= 1. So `context_logits[v]`
   is always a finite real number.
3. The sum of the output distribution is exactly 1.0 by definition of softmax.

This is **unconditionally valid** — it does not depend on the value of alpha,
the n-gram order, or the hash table state. Even if the hash table is empty
(all n-gram bonuses are zero), the Laplace unigram base guarantees full-vocab
support.

**Key invariant**: `unigram_count[v] >= 1` for all v in V (Laplace smoothing).
This means `q_t(v) > 0` for all v, which means `log(q_t(v))` is finite for
all v, which means `softmax(... + alpha * log(q_t(v)))` assigns nonzero
probability to all v.

## Question 2: How is score-before-update enforced?

**Score-before-update** (Issue #1017): "The score for token x_t must be
computed before x_t is used for any adaptation."

**Answer**: The corrector maintains this by construction, both at the LoRA TTT
level and at the n-gram hash table level.

### LoRA TTT level (existing #1610 code, lines 2510-2537)

The chunk loop in `eval_val_ttt_phased` enforces:
1. **SCORE** (line 2510): `per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)`
2. **ACCUMULATE** (line 2512): `_accumulate_bpb(per_tok_loss, ...)` — records NLL
3. **UPDATE** (line 2526-2537): `if needs_train: ... cur_opt.step()` — adapts LoRA

The update only happens AFTER scoring and accumulation. The `needs_train` flag
is False for the last chunk of each document, so the final scored tokens
never cause a LoRA update.

### N-gram hash table level (corrector code, to be added)

The corrector will follow the same pattern:
1. **QUERY** `corrector.get_context_logits(t)` — reads hash state built from [0, t)
2. **SCORE** — compute cross_entropy on `logits + alpha * context_logits`
3. **UPDATE** `corrector.update(t, x_t)` — adds x_t to hash tables

The update call is placed AFTER the score computation. This is enforced by
the sequential prefix-scan loop: position t+1 cannot be queried until
position t's update is complete.

### Global SGD phase boundary

After `train_val_ttt_global_sgd_distributed()` returns (line 2610-2612), the
base model weights have changed. The corrector state must be reset at this
point because:
- The n-gram statistics were accumulated against the pre-SGD model
- Keeping stale corrections would add noise against the post-SGD model
- The reset is an explicit `corrector.reset()` call after the SGD step

## Question 3: How is chunking prevented from creating boundary artifacts?

**Answer**: The corrector uses **global state** across chunks within a phase,
not per-chunk state.

Within a phase (pre-SGD or post-SGD), the hash table accumulates tokens from
ALL chunks and ALL document batches processed so far. When a new chunk or new
document batch starts, the hash table already contains the full history of
the current phase. There are no "chunk boundaries" or "batch boundaries"
in the corrector state — only a phase boundary.

The only reset points are:
1. **Phase transition**: After global SGD, the corrector is reset because the
   base model has changed. (Implemented: `eval_val_ttt_phased` lines ~2689–2691.)
2. **End of eval**: The corrector is discarded.

Within a phase, the corrector sees a monotonically growing prefix of scored
tokens, regardless of how the LoRA TTT chunks or document batches divide the data.

Note: The LoRA TTT resets per document-batch (`reusable_lora.reset()` at the
top of the batch loop), but the corrector does NOT reset per document-batch.
The corrector's unigram and n-gram tables span all document batches within a phase.

### Chunk-static bias approximation

The corrector bias is computed once per chunk (at chunk start) and applied
uniformly to all positions within that chunk. This is a deliberate approximation:

- **Why**: A per-position bias would require either (a) 32× more GPU forward
  passes, or (b) a dense [B, S, V] bias tensor. Both violate hard constraints
  from the plan (time budget / no dense [B,S,V]).
- **Legality**: The bias at any position within chunk c uses only tokens from
  chunks [0, c), i.e., strictly before the current chunk. Token x_t within
  chunk c does NOT influence the bias at position t (score-before-update holds
  at chunk granularity). This satisfies Condition 2 and the no-realized-token
  property.
- **Limitation**: Token t+1 within chunk c does not see x_t. Within-chunk
  n-gram context is not captured. This reduces corrector effectiveness compared
  to a true position-level prefix scan, but is the only feasible implementation
  given the [B,S,V] prohibition.

## Question 4: Does the correction depend on the realized token x_t?

**Answer**: No.

`get_context_logits(t)` reads only the hash table state, which contains
tokens [0, t). The token at position t (x_t) has not yet been added to
the hash table when the context logits are computed.

The correction at position t is a deterministic function of:
- The token sequence [0, t) — the scored prefix
- The n-gram order and table configuration — fixed hyperparameters
- The alpha mixing weight — a fixed hyperparameter

It does NOT depend on:
- x_t (the token being scored)
- The neural model's prediction for position t
- Any future tokens [t+1, ...)

This means the corrector satisfies the **prefix-only** property: the logit
bias at position t is fully determined before x_t is observed.

## Summary Table

| Property | Status | Mechanism |
|----------|--------|-----------|
| Full-vocab distribution (Condition 2) | Valid | Laplace smoothing guarantees q_t(v) > 0 for all v |
| Score-before-update | Enforced | Sequential: query -> score -> update |
| No chunk boundary artifacts | No artifacts | Global hash state within phase; reset only at SGD boundary |
| No realized-token dependence | Independent | get_context_logits(t) reads only [0, t) |
| Legal under Issue #1017 | Yes | All four conditions satisfied by construction |
