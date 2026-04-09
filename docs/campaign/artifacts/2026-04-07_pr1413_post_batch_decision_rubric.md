# Post-Batch Decision Rubric: PR #1413 A/B/C/D/E Suite

Created: 2026-04-07  
Branch: `07c1-base-runpod-strict-submission`  
Use after: batch completes on RunPod

---

## 1. Metrics Comparison Table

Extract the following from each run's `console.txt` and archived log file (`logs/<RUN_ID>.txt`):

| Metric | Where to find it | Units | Notes |
|--------|------------------|-------|-------|
| Sliding BPB (s64) | Final eval line in console.txt, `sliding_bpb=` or equivalent | BPB | Primary decision metric. Lower is better. |
| Train wallclock | `train_time=` or cumulative step timing in log | ms | Must be under 600,000 ms. Flag if over 580,000 ms (tight margin). |
| Eval wallclock | Time from end-of-training to final BPB line | ms | Must be under 600,000 ms. Run E is eval-only, so this is its total time. |
| Artifact bytes | `ls -l final_model.int6.ptz` in archive dir | bytes | Must be under 16,000,000. Flag if over 15,900,000 (tight margin). |
| Steps reached | Last `step=` in training log | int | Fewer steps at same BPB = more efficient. More steps at worse BPB = overhead problem. |
| Code bytes | `variant_manifest.json` wrapper_bytes or counted submission size | bytes | LocalBase = 16,719; Stack = 17,390. Both well under 16MB total. |
| Errors/warnings | grep for `CUDA error`, `OOM`, `nan`, `inf`, `NCCL timeout`, `Traceback` | flag | Any of these invalidates the run. |

**Template table to fill post-run:**

| Run | Label | sliding BPB | delta vs A | train ms | eval ms | artifact bytes | steps | errors |
|-----|-------|-------------|-----------|----------|---------|----------------|-------|--------|
| A | faithful #1413 ctrl | | 0.0 (ref) | | | | | |
| B | PARALLEL_RESIDUAL_START=7 | | | | | | | |
| C | LOOP_START=3, LOOP_END=5 | | | | | | | |
| D | B+C combined | | | | | | | |
| E | D checkpoint + ngram tilt eval | | | | | | | |

## 2. Decision Thresholds

All deltas are in BPB units. "vs A" means `(run X BPB) - (run A BPB)`.

| Verdict | Delta vs A | Rationale |
|---------|-----------|-----------|
| **Promising, invest more seeds** | < -0.0010 (i.e. at least 0.001 BPB improvement) | At this single-seed granularity, 0.001 BPB is roughly 1 sigma of seed noise based on the #1394 3-seed spread (1.08472 to 1.08577, range ~0.001). A full 0.001 win is worth confirming with 2 more seeds. |
| **Weakly promising, run 1 more seed** | -0.0010 to -0.0004 | Could be noise. One confirmatory seed is cheap (~$3.60). |
| **Neutral, deprioritize** | -0.0004 to +0.0004 | Within seed noise. Not worth more seeds. Move to #1437. |
| **Negative, drop** | > +0.0005 | Actively hurts. Kill the branch. |

**Reproduction gap check** (Run A vs upstream #1413 reference of 1.08279):

| Gap (A - 1.08279) | Interpretation |
|--------------------|---------------|
| < +0.0005 | Faithful reproduction. Proceed with ablation results as-is. |
| +0.0005 to +0.0015 | Minor gap, likely seed variance. Note it but trust the relative A/B/C/D ranking. |
| > +0.0015 | Reproduction problem. The local payload differs meaningfully from upstream. Debug the wrapper or env before trusting any ablation result. Do NOT run more seeds until resolved. |

**Run E special threshold** (eval-only ngram tilt on D checkpoint):

| E BPB vs D BPB | Interpretation |
|-----------------|---------------|
| < -0.0005 | Ngram tilt is a free eval-time win. Stack it onto whatever training winner emerges. |
| -0.0005 to +0.0002 | Negligible. Not worth the code complexity and eval-time cost. |
| > +0.0002 | Tilt hurts. Drop it. |
| E eval time > 550,000 ms | Even if BPB wins, it is dangerously close to the 600s eval cap. Needs optimization before submission. |

## 3. Decision Tree

**If D wins (combined parallel residual + loop adjustment is best):**
1. This is the expected best case -- both levers are additive.
2. Run 2 more seeds on D (seeds 42, 1337) in the same pod session.
3. If D 3-seed mean beats A 1-seed by >0.0008, lock D as the base for #1437 stacking.
4. Evaluate whether E (ngram tilt on D) adds marginal gain; if so, the submission candidate is D+E.
5. Next step: build #1437 stacked on D's training config.

**If C wins (loop adjustment alone is the driver):**
1. Parallel residual adds nothing or hurts. Drop PARALLEL_RESIDUAL_START=7.
2. Run 2 more seeds on C.
3. The simpler config (no stack wrapper, uses LocalBase folder) is preferable for maintenance.
4. Next step: stack #1437 techniques onto C's config, not D's.

**If B wins (parallel residual alone is the driver):**
1. Loop adjustment (LOOP_START=3) adds nothing or hurts. Drop it.
2. Run 2 more seeds on B.
3. Next step: try PARALLEL_RESIDUAL_START at 6 and 8 as a quick 2-run sweep to find the optimum layer.

**If A is best (modifications hurt):**
1. Both levers are noise or negative on this codebase. Do not invest more seeds in B/C/D.
2. Record the negative result in `results_log.jsonl`.
3. Immediate pivot: run faithful #1413 at 2 more seeds (42, 1337) to establish a solid 3-seed mean.
4. Then move directly to #1437 stacking on the unmodified #1413 base.
5. Consider whether the gap between A and upstream 1.08279 is worth debugging, or whether #1437 is the faster path to the frontier.

**If E shows gain but is slow (BPB improves but eval > 500s):**
1. The ngram tilt idea is valid but the current implementation is too heavy.
2. Profile where eval time goes: is it the ngram table build, the tilt application, or the sliding window itself?
3. Options: reduce NGRAM_OPEN_TABLE_BITS from 26 to 24 (4x smaller table), reduce NGRAM_ORDER_STRIDE, or only tilt on a subset of positions.
4. Do NOT submit with eval time >550s -- too risky under contest variance.

**If A doesn't reproduce #1413 reference (gap > 0.0015):**
1. Stop all ablation analysis. The relative B/C/D results are untrustworthy.
2. Diff the local `train_gpt.py` wrapper against the raw upstream #1413 source byte-for-byte.
3. Check whether the LZMA2 wrapper decode produces identical Python source (the `decoded_source_sha256` in `variant_manifest.json` should match).
4. Check env var defaults: LOOP_START=4, LOOP_END=5, PARALLEL_RESIDUAL_START=-1 must all match upstream defaults.
5. Check data: SP8192 tokenizer and shard checksums must match upstream.
6. Do not spend more than 30 minutes debugging. If unresolved, fall back to fetching #1413 source live on-pod with FETCH_PAYLOAD=1.

## 4. Follow-Up Seed Plan

**Scenario: Clear winner identified (delta vs A < -0.0008)**

| Item | Plan |
|------|------|
| Seeds to run | 2 additional (42, 1337) on the winner |
| Pod session | Same session if pod is still running. Each run takes ~10 min, so 2 runs = ~20 min = ~$7.20 additional cost. |
| Total cost for 3-seed confirmation | ~$3.60/run x 2 = ~$7.20 on top of the batch cost |
| Decision after 3 seeds | If 3-seed mean of winner < 1.08279 (upstream #1413), lock it and move to #1437. If 3-seed mean is 1.08279-1.08350, it is a valid #1413 reproduction with a minor local tweak. If >1.08400, something is wrong. |

**Scenario: Weakly promising (delta -0.0004 to -0.0010)**

| Item | Plan |
|------|------|
| Seeds to run | 1 additional (42) on the best variant |
| Pod session | Same session. ~$3.60 additional. |
| Decision after 2 seeds | If both seeds agree on direction, run a 3rd. If they disagree, drop the variant and move to #1437 on base #1413. |

**Scenario: A is best / all modifications neutral**

| Item | Plan |
|------|------|
| Seeds to run | 2 additional on A (42, 1337) |
| Pod session | Same session. ~$7.20 additional. |
| Purpose | Establish a solid 3-seed #1413 mean to compare against #1437 later. |
| Next move | Build #1437 stack immediately. |

**Scenario: Reproduction failure (A >> upstream reference)**

| Item | Plan |
|------|------|
| Seeds to run | 0 until root cause is found |
| Pod session | Keep the pod alive for debugging (max 30 min = ~$10.75 debug budget) |
| Fallback | Re-run A with FETCH_PAYLOAD=1 to bypass the local wrapper entirely |

## 5. Kill Criteria

| Branch / lever | Kill trigger | Action |
|----------------|-------------|--------|
| PARALLEL_RESIDUAL_START=7 (B) | 2-seed mean delta vs A > +0.0003 or 1-seed delta > +0.0008 | Drop permanently. Record negative result. |
| LOOP_START=3 (C) | Same as above | Drop permanently. |
| Combined D | If both B and C are individually negative, do not even run D multi-seed. | Skip and record. |
| Ngram tilt (E) | Eval wallclock > 550s regardless of BPB, OR BPB delta vs D > +0.0002 | Drop or park for optimization. Do not submit. |
| Entire #1413 local mod branch | 3-seed mean of best variant > 1.08350 (i.e., further from frontier than vanilla #1413) | Abandon local mods. Use vanilla #1413 as-is and move to #1437. |
| Pod session overall | Total pod time exceeding 90 minutes (~$32) without a locked 3-seed result | Terminate pod. Regroup offline. |
| GPU budget for this experiment suite | Total RunPod spend on #1413 ablations exceeding $50 | Hard stop. Whatever you have is the answer. Move to #1437. |

**Meta criterion (soft kill):** If the best achievable result from this entire A/B/C/D/E suite is still >1.08200 BPB (i.e., not closing the gap to the frontier meaningfully beyond vanilla #1413's 1.08279), do not invest more seeds in these levers. However, still finish interpreting the batch — the ablation results may reveal which individual component (parallel residual start layer, loop range, ngram tilt) is worth carrying forward into the #1437 stack, even if the #1413-only result isn't competitive on its own. The path to the frontier runs through #1437 and #1416 techniques, not through parallel residual or loop tuning on #1413 alone.
