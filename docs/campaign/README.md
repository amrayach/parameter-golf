# Parameter Golf Campaign

This directory turns the campaign plan into operator-ready assets for a 7-10 day push on the challenge.

## Current Status

For live execution state, always defer to:
- `AGENTS.md`
- `docs/campaign/AGENT_SYNC.md`
- `CLAUDE.md`

As of 2026-03-31:
- the active mainline is compression-path feasibility on top of the measured 05c-plus fallback
- Session 05c-plus is the best measured branch in the current family
- Session 05f and Session 05g are measured negatives vs 05c-plus
- GPTQ is permanently parked for this model family after the 05e export-only probe on the 05c-plus architecture
- the next live execution step is rerunning the corrected compression probe on the saved 05c-plus / 05g artifacts
- for exact smoke-vs-launch gating, use `docs/campaign/AGENT_SYNC.md` rather than this summary

## Objective

Primary goal:
- Determine quickly whether there is a realistic path to a respectable leaderboard submission from this environment.

Secondary goal:
- If the answer is yes, reproduce a strong public pre-TTT stack, test a narrow set of improvements, and only then consider TTT or RFN-guided ideas.

## Current read of the field

As of 2026-03-30, the public leaderboard pattern is clear:
- The biggest gains came from stacking practical wins, not from speculative new theory.
- The strongest recurring levers are sliding-window eval, compression-aware training and export, deeper and wider models funded by compression, cheap architectural tweaks, and later TTT.
- The current official merged #1 is PR `#1019` at `1.1147` BPB.

Implication:
- The fastest serious route is a strong non-TTT anchor first.
- RFN is worth probing only as a sidecar until it proves signal on a tiny transformer experiment.

## Hardware stance

Primary execution base:
- Pegasus `H100` partition first.

Why:
- The Pegasus docs describe `H100` as `H100-SXM5`, 8 GPUs per node, NVSwitch-connected.
- That is the correct class for meaningful iteration, even if final challenge verification still belongs on Runpod or equivalent.

Important references:
- `@docs/Pegasus_Server_documentation.txt`
- `@docs/dfki-nlp-pegasus-bridle-8a5edab282632443.txt`

## Session order

These session docs are historical planning assets. They are useful for lineage and rationale, but they are not the current execution source of truth. Use `AGENT_SYNC.md` for the live plan.

1. `sessions/01_lineage_and_environment_audit.md`
Preferred mode: Planning
Goal: map the winning lineage, confirm repo mechanics, and lock the execution protocol.

2. `sessions/02_pegasus_baseline_ladder.md`
Preferred mode: Execution
Goal: verify the Pegasus environment and reproduce the baseline ladder cheaply.

3. `sessions/03_pre_ttt_anchor_port.md`
Preferred mode: Execution
Goal: port a strong pre-TTT stack into your own non-record folder and measure how close it lands.

4. `sessions/04_targeted_delta_sweep.md`
Preferred mode: Execution
Goal: test a narrow set of low-complexity deltas on top of the anchor.

5. `sessions/05_ttt_correctness_audit.md`
Preferred mode: Planning
Goal: decide whether the TTT path is correct, legal, and worth integrating.

6. `sessions/06_rfn_sidecar_probe.md`
Preferred mode: Execution
Goal: test whether RFN signals predict sensitivity better than simple baselines.

7. `sessions/07_go_no_go_review.md`
Preferred mode: Planning
Goal: decide whether to continue the challenge or stop and redirect effort.

## Gates

Gate 1:
- By session 2, baseline behavior on Pegasus must look credible in wallclock, artifact size, and post-quant `val_bpb`.

Gate 2:
- By session 3, the anchor stack should land near the public `1.123-1.128` band or reveal one concrete fixable bottleneck.

Gate 3:
- By session 4, at least one targeted delta must show believable upside.

RFN gate:
- Continue the RFN track only if its rankings beat or complement magnitude-based heuristics on a small controlled experiment.

## Workflow rules for all Claude sessions

- Explore first. Do not start by editing blindly.
- Keep competitive experiments in self-contained folders under `records/track_non_record_16mb/`.
- Always document what changed, what was measured, and what failed.
- End each execution session with:
  - an updated README or note,
  - reproducibility details,
  - a scoped git commit.

## Local context worth reusing

Challenge core:
- `@README.md`
- `@train_gpt.py`

Key public records:
- `@records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`
- `@records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `@records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `@records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

RFN thesis and code:
- `@docs/Abschlussarbeit_379315.pdf`
- `@/home/amay/Work/BachExpGraph/README.md`
- `@/home/amay/Work/BachExpGraph/src/main.py`
- `@/home/amay/Work/BachExpGraph/src/explain.py`
- `@/home/amay/Work/BachExpGraph/src/mlp.py`

## Templates

- `PROMPT_TEMPLATE.md`
- `PEGASUS_H100_RUNBOOK.md`
- `templates/RUN_MANIFEST_TEMPLATE.md`
- `templates/EXPERIMENT_SUMMARY_TEMPLATE.md`
