# N-gram Tilt `#1017` Compliance Audit

Date: 2026-04-09

Scope:

- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/ngram_tilt.py`
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp`
- `scripts/prepare_pr1413_variants.py`

## Verdict

The C++ kernel's score-before-update guarantee holds for the legality-critical edge cases reviewed here:

- first scored token in the validation stream
- chunk boundaries in score-first TTT
- sliding-window resets / overlapping-context restarts

I did not find an edge case where the hint for position `p` depends on `tokens[p]` or any later token.

## Evidence

| Edge case | Why the guarantee holds |
| --- | --- |
| First scored token | Python reserves position `0` as "no hint" and only precomputes positions `1..n_tok-1` ([ngram_tilt.py](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/ngram_tilt.py#L123), [ngram_tilt.py](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/ngram_tilt.py#L164), [ngram_tilt.py](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/ngram_tilt.py#L177)). In the kernel, lookup happens before any update for the current token ([fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L398), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L438), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L445)). |
| Sliding-window reset | The eval loop scores only the suffix of nonzero windows and maps those scored targets back to their global stream positions with `gp = arange(ws+s+1, ws+wlen+1)` ([scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L167), [scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L188), [scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L189)). That means the first scored token after a reset uses the last context token as `p-1`, which matches the strict-prefix rule enforced in the kernel via `prev_tok = tokens_[p-1]` ([fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L401), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L402), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L403)). |
| Chunk boundaries | Windows are assigned to chunks by their first scored position, then each chunk is fully scored before the model is adapted on that chunk ([scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L167), [scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L168), [scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L191), [scripts/prepare_pr1413_variants.py](/home/amay/Work/parameter-golf/scripts/prepare_pr1413_variants.py#L193)). The n-gram hints are read-only during scoring and already indexed by global target position, so chunk-local TTT updates do not retroactively change the hint for any token already scored. |
| Single-pass hint construction | The kernel walks positions in increasing order, emits the hint/beta pair, and only then updates all tables with the current token ([fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L382), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L415), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L438), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L445)). |

## Finding

`NGRAM_WITHIN_BETA=0.0` and `NGRAM_WORD_BETA=0.0` do not fully remove those experts from the decision path.

- `within_hint()` and `word_hint()` still emit candidates when their thresholds fire; only the returned beta is zero ([fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L233), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L234), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L271), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L272)).
- All nonnegative candidates enter the same `cands[]` array, and any matching hint receives `agree_bonus_` even if one of the matching experts had beta `0.0` ([fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L421), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L422), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L423), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L427), [fused_expert_kernel.cpp](/home/amay/Work/parameter-golf/records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/fused_expert_kernel.cpp#L430)).

Implication:

- This is not a score-before-update or target-leakage bug.
- It does mean the current "token-only" description is semantically overstated unless `agree_bonus` is also neutralized or zero-beta experts are gated out of `cands[]`.

## Residual Risk Outside This Audit

This audit clears the strict-prefix / score-before-update path. It does not resolve the separate reviewer-perception risk around the dedicated full-stream precompute pass and how that should be described relative to any "`no_ngram_cache`" claim.

## Deferred R1 Note

The `within_beta` / `word_beta` sweep remains a paid-session experiment only. Keep the current local defaults unchanged; log the sweep as a deferred R1 item rather than touching the knobs in local legality work.
