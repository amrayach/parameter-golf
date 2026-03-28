# Session 06: Attribution-Graph Sidecar Probe (Reframed)

Preferred mode: Execution

Use `/research-engineer` if available. Keep the probe small, cheap, and falsifiable.

## Motivation

The original RFN framework (bachelor thesis) operates on MLPs only. Anthropic's Attribution Graphs paper (https://transformer-circuits.pub/2025/attribution-graphs/methods.html) provides a much better template for transformers, but their full method (cross-layer transcoders, learned sparse features) is too expensive for this campaign.

This session implements a **cheap hybrid**: use the attribution-graph philosophy (node-to-logit influence scoring) but with existing model components as nodes instead of learned CLT features. The goal is to determine if this ranking predicts quantization sensitivity better than simple weight magnitude.

## Goal

Test whether a stripped-down attribution-graph-style module ranking outperforms magnitude heuristics for predicting quantization/ablation sensitivity in the Parameter Golf GPT model.

## Read these first

- @train_gpt.py (the model architecture)
- @docs/campaign/artifacts/03_pre_ttt_anchor_summary.md (current anchor state)
- @docs/Abschlussarbeit_379315.pdf (pages 22-30, RFN extraction theory)
- @/home/amay/Work/BachExpGraph/src/explain.py (original explainer interface)

## Method

### Node Definition
Nodes = existing model components at the granularity where quantization decisions are made:
- Per attention head (Q, K, V, O projections)
- Per MLP sublayer (up projection, down projection)
- Per transformer block (coarse-grained)
- Embedding layer

### Edge Computation (Cheap Attribution)
For ~100 representative validation sequences:
1. Forward pass through the model, cache all intermediate activations
2. For each node (component), compute local Jacobian of logit loss w.r.t. node output
3. Under frozen attention patterns (detach attention weights from computation graph), compute indirect influence: how much does zeroing/scaling this node's output change the final logit distribution?
4. Aggregate influence scores across sequences → per-node importance ranking

### Baseline Comparators
1. **Weight magnitude:** L2 norm of each component's weight matrix
2. **Activation magnitude:** Mean L2 norm of component outputs across validation batch
3. **Gradient magnitude:** Mean gradient norm during final training steps

### Validation Experiment
For the top-N and bottom-N ranked components under each method:
1. Quantize bottom-N components to int5 (more aggressive), keep top-N at int8
2. Measure val_bpb degradation for each ranking method
3. The method whose ranking produces the least degradation under mixed-precision quantization wins

### Success Criterion
Attribution-graph ranking must produce measurably less val_bpb degradation than weight magnitude ranking under the same int5/int8 split. If the difference is < 0.0002 BPB or attribution is worse, the verdict is **stop**.

## Constraints

- Sidecar research probe only — does NOT block leaderboard push
- Total compute budget: ≤ 2 hours on 1xH100 (not 8x)
- No CLT training, no sparse feature learning, no replacement model
- Keep instrumentation code under 200 lines
- Compare against at least 2 baseline heuristics

## Deliverables

- `research/attribution_probe/probe.py` — minimal instrumentation code
- `research/attribution_probe/results.json` — ranking comparison data
- `docs/campaign/artifacts/06_attribution_graph_sidecar_probe.md` — analysis memo

## Definition of Done

- One explicit prediction task (mixed-precision sensitivity ranking)
- At least two baseline comparators (magnitude, gradient)
- Hard verdict: continue (attribution provides useful signal) or stop (does not outperform baselines)
- If continue: concrete recommendation for how to use the signal in the main campaign

## Commit message

`feat(campaign): add attribution-graph sidecar probe (reframed from RFN)`
