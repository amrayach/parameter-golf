# Session Handoff

Date: 2026-03-30

## Current Truths

- Session 03 pre-TTT anchor port is complete and remains the fixed reference.
- Sliding s64 val_bpb: `1.12904446` on `8xH100 SXM5`, `serv-3342`.
- Pre-quant EMA val_bpb: `1.14472403`.
- Int6 roundtrip val_bpb: `1.15247273`.
- Steps: `6564`, step_avg: `91.37 ms`.
- Artifact: `15751324` bytes (model `15692752` + code `58572`).
- Official leaderboard entry is record-gated. Must beat current merged #1.
- Current official merged #1 is PR `#1019` at `1.1147` BPB (3-seed mean `1.88218` nats).
- NGC 26.03 container + fscratch is the confirmed stable Pegasus path.
- Three agents coordinate on this repo: Claude Code, Codex, and Antigravity.

## What Matters Now

- **Session 05c-plus** is the active objective: a training-quality bundle on the Session 03 anchor.
- Code is implemented and pushed: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- Smoke test script is ready: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/smoke_test.sh`
- Next step: run 1xGPU Pegasus smoke, then 8xH100 full run (Pegasus or RunPod $25 credits).
- Do not work on GPTQ, FA3, TTT, or SWA until 05c-plus training results are measured.

## 05c-plus Bundle

Four changes on Session 03 anchor:
1. XSA 4 → 11 (all layers)
2. VE128 on layers 9-10 (shared ValueEmbedding)
3. Warmdown 3000 → 3500
4. LeakyReLU(0.5)² replacing ReLU²

Not included: SWA (dead code in PR #1019 and #634), GPTQ (parked), FA3 (ABI issue).

Target: sliding s64 val_bpb < 1.126 (anchor 1.129).

## Parked Work

### GPTQ (Session 05b) — parked after 7 ablations

Seven ablations on the same Session 03 checkpoint all failed. Ablation #6 (PR #1019 verbatim transplant) produced byte-identical MSE to the local code, proving the GPTQ code is correct. The failure is model-specific: relu creates sparse Hessians, leaky_relu does not. GPTQ may become viable after 05c-plus trains with LeakyReLU².

GPTQ replay on a 05c-plus checkpoint requires a merge step: the parked 05b script has the old architecture (no VE, relu²).

### FA3 — parked (ABI issue)

11.44x kernel speedup negated by pip torch downgrade. Parked until NGC-native path exists.

### TTT — parked

Parked pending stronger pre-TTT base.

## Source of Truth Files

- `AGENTS.md` — shared entry point
- `docs/campaign/AGENT_SYNC.md` — mutable objectives and results
- `CLAUDE.md` — standing rules
- `docs/codex-memory/decisions.md` — locked decisions
- `docs/codex-memory/project-state.md` — project state
- `docs/codex-memory/next-session.md` — next actions
- `docs/superpowers/plans/2026-03-30-session-05c-plus.md` — 05c-plus plan
