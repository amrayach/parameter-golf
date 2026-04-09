# Follow-Up Experiment Ladder: Post-PR-#1413 A/B/C/D/E Batch

Created: 2026-04-07
Branch: `07c1-base-runpod-strict-submission`
Prerequisite: A/B/C/D/E batch results are in hand and interpreted per `2026-04-07_pr1413_post_batch_decision_rubric.md`

---

## Context Summary

| Reference | BPB | Status |
|-----------|-----|--------|
| Our #1394 baseline (3-seed mean) | 1.08521 | Completed, locked |
| Upstream #1413 reference (3-seed mean) | 1.08279 | Reproducing now (Run A) |
| Our #1437 corrected (5-seed mean) | 1.08091 | Measured locally, causality-fixed |
| Open #1420 (claimed, pre-fix) | 1.08014 | Affected by same causality bug as our #1437 pre-fix |
| Open #1416 pre-quant TTT | 1.07948 | Untouched, higher review risk |

Key constraint: corrected #1437 (1.08091) only beats #1413 (1.08279) by ~0.00188 BPB. The three stacked levers (parallel residual, 3-layer loop, causal ngram tilt) produce small individual deltas. The largest single lever is the 3-layer loop at -0.00128 BPB.

---

## Tier 1: Low-Risk Immediate Follow-Up

### Experiment F: Multi-Seed Confirmation of Batch Winner + TTT Hyperparameter Sweep

**What**: After identifying the batch winner (best of A/B/C/D/E), run 2 additional seeds on it (42, 1337) on the same pod session. Then, on seed 0 of the winner, sweep one TTT hyperparameter: `TTT_LR` in {0.003, 0.007, 0.010} (current default is 0.005). This is eval-only -- reuse the existing quantized checkpoint and just re-run the TTT eval pass with different LR values. Each TTT eval takes ~290s so 3 sweeps fit in ~15 min.

**Rationale**: This is the highest-EV next step because it resolves two questions simultaneously at minimal cost:
1. Is the batch winner's advantage real (multi-seed confirmation) or within seed noise?
2. Is TTT_LR=0.005 optimal, or is there free BPB on the table from a slightly different learning rate?

The TTT LR sweep is essentially free once you have a checkpoint -- it is eval-only, reusing the same quantized model. The #1413 upstream chose LR=0.005 without documented ablation. Even a 0.0003 BPB improvement from a better LR would compound with every subsequent experiment.

**Expected upside**: Multi-seed confirmation: ~0 BPB (confirms existing signal). TTT LR sweep: 0.0002-0.0005 BPB improvement if the current LR is suboptimal. Combined: solidifies the base and may find a free improvement.

**Main risk**: TTT LR sweep finds that 0.005 is already optimal (most likely outcome -- it was presumably tuned by the upstream author). Pod session may not have enough time remaining if A/B/C/D/E took longer than expected.

**Needs offline prep?**: No. Everything runs from the existing batch infrastructure. The TTT LR sweep only requires overriding one env var (`TTT_LR`) on an existing checkpoint.

**Estimated cost**: ~$11 (2 seed runs at ~17 min each = 34 min + 3 TTT-only eval reruns at ~7 min each = 21 min, total ~55 min at $21.52/hr).

**Kill criterion**:
- Multi-seed: if the 3-seed mean of the winner is more than +0.0005 BPB worse than the upstream #1413 reference of 1.08279, the local payload has a reproduction issue. Stop and debug the wrapper before proceeding.
- TTT LR sweep: if no LR beats the default by more than 0.0002 BPB, lock 0.005 and move on. Do not sweep more than 3 values.
- Time budget: if the pod has been running >90 min total (including the A/B/C/D/E batch), skip the TTT LR sweep and just do the multi-seed runs.

---

## Tier 2: Medium-Risk Packaging Follow-Up

### Experiment G: Stack #1437 Techniques onto the Batch Winner (Parallel Residual + 3-Layer Loop + Causal Token-Only Tilt)

**What**: Build a new local record folder that starts from the batch winner's training config and adds all three #1437 levers together:
1. `PARALLEL_RESIDUAL_START=7` (if not already in the winner)
2. `LOOP_START=3, LOOP_END=5` (if not already in the winner)
3. Causal token-only n-gram tilt at eval time with `NGRAM_WITHIN_BETA=0, NGRAM_WORD_BETA=0` (the causality-corrected config)

If the batch winner is already Run D (which has both parallel residual and loop), then this experiment is simply adding n-gram tilt to the D checkpoint -- which is exactly what Run E already tests. In that case, redefine Experiment G as: run the full D+E config at 3 seeds to get a proper mean, then compare against the corrected #1437 5-seed mean of 1.08091.

If the batch winner is NOT D (e.g., A or C wins), then build a new variant that layers all three #1437 levers onto the winner's base config.

**Rationale**: The corrected #1437 achieves 1.08091, which is ~0.00188 BPB better than #1413's 1.08279. Our A/B/C/D/E batch tells us which individual levers are positive on our hardware/setup, so we can make an informed decision about which to include. The full #1437 stack is the most direct path to 1.080x territory.

The offline prep is straightforward because `scripts/prepare_pr1413_variants.py` already contains the parallel-residual hook, skip-training hook, and n-gram tilt integration. The stack wrapper in `records/.../ParallelResid7_TiltPrep/` already supports all three levers via env vars.

**Expected upside**: 0.0010-0.0020 BPB improvement over the batch winner, targeting ~1.0810-1.0820 BPB (based on corrected #1437 5-seed mean of 1.08091). This would put us in the same tier as the corrected #1420 and within striking distance of the clean open frontier.

**Main risk**:
1. The n-gram tilt adds very little once the causality bug is fixed. The corrected delta from tilt alone was only ~0.00182 BPB (1.08273 sliding pre-tilt to 1.08091 post-tilt). On our hardware/tokenizer, the benefit may be smaller.
2. Eval timing: the n-gram C++ kernel build + 5 GB host-RAM cache + tilt application adds ~30-40s to eval. If TTT eval is already using ~290s, total eval is ~330s which is safe but leaves less margin.
3. The combined stack may not be strictly additive. The interaction between parallel residuals and the loop layers could differ from what #1437 reports if our reproduction of the base differs from theirs.

**Needs offline prep?**: Partially. If the batch winner is D, no additional prep is needed -- the stack wrapper already exists. If the winner is A or C, a new variant folder needs to be built by running `prepare_pr1413_variants.py` with modified defaults (or manually setting env vars at launch). Estimated offline prep: ~30 min of scripting work.

**Estimated cost**: 3 seeds at ~17-20 min each (tilt adds ~3 min per run to eval) = ~57 min, approximately $20. If reusing the same pod session as Experiment F, add to the existing session.

**Kill criterion**:
- If the 3-seed mean with all three levers is worse than 1.08200, the full stack is not producing the expected benefit. Check whether tilt is actually contributing by comparing D (no tilt) vs D+tilt on the same seed.
- If eval time exceeds 550s on any seed, the tilt implementation is too slow for submission. Park tilt and submit the training-only winner.
- If artifact size exceeds 15,950,000 bytes with the stack wrapper, check code byte count -- the stack wrapper is 17,390 bytes vs 16,719 for base, leaving less model budget.
- Total RunPod spend on Experiments F+G should not exceed $40. If approaching that, take the best result in hand.

---

## Tier 3: High-Risk Frontier Follow-Up

### Experiment H: TTT Epochs/Batch Tuning + Aggressive LR Schedule for Closing the Gap to 1.080

**What**: Starting from the best checkpoint produced by Experiment G (or the batch winner if G fails), do a targeted TTT configuration search to squeeze maximum eval-time improvement. The specific axes to sweep:

1. **TTT_EPOCHS**: current default is 3. Try {4, 5}. More epochs means the model adapts more to the val distribution on each chunk. Higher risk of overfitting to individual chunks but precedent from #1416 (pre-quant TTT, 1.07948) suggests more adaptation is legal and beneficial.
2. **TTT_BATCH_SEQS**: current default is likely 1-2. Try {2, 4} to see if larger batches stabilize the SGD updates within the TTT eval window. This may speed up or slow down TTT depending on memory/compute tradeoffs.
3. **TTT_FREEZE_BLOCKS**: current default is 0 (no frozen blocks during TTT). Try {2, 4} -- freezing the first N blocks during TTT reduces the parameter count being adapted and may act as regularization, preventing the model from forgetting low-level features while adapting high-level ones to the val distribution.
4. **Cosine LR decay steepness**: the current TTT uses a cosine schedule over chunks. Try a more aggressive warmup (first 5% of chunks at 0.5x LR) to stabilize early adaptation.

This is NOT a full grid search. It is a sequential greedy search: fix the best value from axis 1, then sweep axis 2, then 3, then 4. Each sweep point is a single TTT eval pass (~290-350s) reusing the same checkpoint. Total: ~8-12 eval runs.

**Rationale**: The gap from the best achievable #1413+#1437 stack (~1.0809) to the uncorrected open frontier (~1.080) is ~0.001 BPB. This gap is small enough that TTT hyperparameter tuning could close it. PR #1416 achieves 1.07948 via pre-quant TTT (illegal under current rules), but its TTT mechanics reveal that the model has significant capacity to adapt at eval time -- our score-first TTT is leaving BPB on the table by using conservative hyperparameters.

The key insight: TTT is the only lever that improves BPB without changing training or model size. Every other improvement axis (more layers, wider model, better quantization) runs into the 600s / 16MB constraints. TTT hyperparameter tuning has no artifact-size cost and only trades eval-time budget for BPB.

**Expected upside**: 0.0005-0.0015 BPB improvement over the Tier 2 result. Best case: reaching ~1.0795-1.0805 BPB, which would be competitive with the corrected open frontier. Worst case: confirming that TTT_LR=0.005, epochs=3, freeze=0 is already near-optimal and the remaining gap requires architectural changes.

**Main risk**:
1. Eval timing: TTT_EPOCHS=5 instead of 3 adds ~67% to TTT time. If TTT currently takes ~290s, 5 epochs would take ~480s, leaving only ~120s for the rest of eval. This is dangerously close to the 600s cap.
2. Overfitting: more TTT epochs on small chunks can memorize chunk-specific patterns rather than learning generalizable val-distribution features. The BPB could improve on some seeds and worsen on others.
3. Interaction with tilt: if n-gram tilt is active, aggressive TTT may fight the tilt -- the model adapts its softmax to reduce the tilt's influence, neutralizing the benefit.
4. This is the most GPU-expensive experiment: 8-12 eval runs at ~7-12 min each = 56-144 min, potentially $20-50.

**Needs offline prep?**: Yes. Need to:
1. Add `TTT_BATCH_SEQS` env var plumbing to the stack wrapper if not already present.
2. Build a small sequential sweep script that iterates over the axes and logs results in a parseable format.
3. Pre-decide the sweep order and values to avoid on-pod deliberation.
Estimated offline prep: ~1-2 hours.

**Estimated cost**: $20-50 on RunPod depending on how many sweep points are needed. Could be done on the same pod session as F+G if time permits, or on a separate session.

**Kill criterion**:
- If TTT_EPOCHS=4 does not improve over epochs=3 by at least 0.0003 BPB, do not try epochs=5 (diminishing returns confirmed).
- If any single eval run exceeds 570s, that TTT configuration is not submittable. Drop it immediately.
- If after sweeping 2 axes the best result is still worse than 1.0805, the TTT tuning approach has diminishing returns. Accept the result and consider whether the remaining gap justifies moving to #1416 (pre-quant TTT, higher review risk).
- Hard budget: do not spend more than $50 total across Experiments F+G+H. Whatever result you have at that point is the campaign answer for this cycle.

---

## Decision Flow Summary

```
A/B/C/D/E batch completes
        |
        v
Identify winner by rubric (see post_batch_decision_rubric.md)
        |
        v
[Experiment F] Multi-seed the winner + TTT LR sweep (same pod, ~$11)
        |
        v
Lock the confirmed winner + best TTT LR as the new stable base
        |
        v
[Experiment G] Stack all #1437 levers onto the locked base (~$20)
        |
    +-----------+
    |           |
  works      fails / neutral
    |           |
    v           v
Lock G as    Submit the F result
new base     as-is; park tilt
    |
    v
[Experiment H] TTT hyperparameter sweep on G's checkpoint (~$20-50)
    |
    +----> If result < 1.0805: strong submission candidate
    +----> If result > 1.0815: accept and consider #1416 lane
    +----> If result > 1.0830: something is wrong; debug before proceeding
```

## Budget Envelope

| Experiment | Est. cost | Cumulative | Offline prep needed |
|-----------|----------|-----------|-------------------|
| F (multi-seed + TTT LR) | ~$11 | ~$11 | None |
| G (full #1437 stack) | ~$20 | ~$31 | 0-30 min depending on batch winner |
| H (TTT tuning sweep) | ~$20-50 | ~$51-81 | 1-2 hours |

Hard stop: $80 total across all three experiments. If budget is exhausted, submit the best measured result.

## What Comes After

If all three experiments complete and the best result is in the 1.080-1.082 range:
- The submission is competitive with the corrected open frontier tier
- Next campaign cycle should investigate #1416 (pre-quant TTT) mechanics to understand the remaining 0.002 BPB gap
- Consider whether originality requirements can be met by a novel combination of the known levers

If the best result is still above 1.082:
- The local reproduction has a systematic gap vs upstream
- Before investing more GPU time, diff the training dynamics (step count, loss curves) against upstream logs
- Consider whether the RunPod 8xH100 SXM environment differs from the upstream GCP 8xH100 SXM in a way that matters (NCCL backend, driver version, etc.)
