# Session Handoff

Date: 2026-03-31

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
- `docs/campaign/AGENT_SYNC.md` is the live source of truth if any summary file disagrees.

## What Matters Now

- **Session 05c-plus** is now the best measured branch, not the next run target.
- Best measured code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- Session 05f follow-up is measured negative: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py`
- Session 05g follow-up is measured negative: `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py`
- GPTQ is permanently parked for this model family after Session 05e on the 05c-plus architecture.
- Smoke harnesses exist:
  - `records/track_non_record_16mb/2026-03-30_training_bundle_plus/cpu_smoke.sh`
  - `records/track_non_record_16mb/2026-03-30_training_bundle_plus/smoke_test.sh`
- Current next step: finish compression-path feasibility, then choose one coherent larger fork.
- Do not work on GPTQ, FA3, TTT, or more local XSA/bigram micro-deltas.

## 05c-plus / 05f status

05c-plus measured on `8xH100`:
- sliding s64 `1.12557920`
- pre-quant EMA `1.14186715`
- int6 roundtrip `1.14933197`
- `step_avg=100.39 ms`
- quality-positive, throughput-regressed

05f measured on `8xH100`:
- sliding s64 `1.12660664`
- pre-quant EMA `1.14190308`
- int6 roundtrip `1.15026661`
- `step_avg=100.51 ms`
- negative vs 05c-plus, no throughput recovery

05g measured on `8xH100`:
- sliding s64 `1.12584234`
- pre-quant EMA `1.14203044`
- int6 roundtrip `1.14963535`
- `step_avg=98.67 ms`
- slight speed recovery, slight quality loss, and over cap on the old export path

Conclusion:
- keep 05c-plus as the best measured branch
- do not continue the 05f / 05g local tweak line

## Current Diagnostic Approaches

Artifacts:
- `diagnostics/2026-03-31_05c_plus/final_model.pt`
- `diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- `diagnostics/2026-03-31_05c_plus/train.log`
- `diagnostics/2026-03-31_05c_plus/diagnostics_float.txt`
- `diagnostics/2026-03-31_05c_plus/diagnostics_int6.txt`

Commands:
- `python scripts/diagnostics/diagnose_weights.py final_model.pt`
- `python scripts/diagnostics/diagnose_weights.py final_model.pt final_model.int6.ptz`
- `python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`

Use:
- single-checkpoint weight statistics
- float-vs-int6 quantization-damage proxy on the same checkpoint
- compression-path feasibility on saved artifacts
- correlation with measured 05c-plus / 05f / 05g logs before selecting the next branch

Limit:
- checkpoint-weight diagnostic only, not activation-level evidence

## Parked Work

### GPTQ (Session 05b + 05e) — permanently parked

Session 05b produced seven failed ablations on the Session 03 checkpoint. Ablation #6 (PR #1019 verbatim transplant) produced byte-identical MSE to the local code, proving the GPTQ code is correct. Session 05e then tested the leading rescue hypothesis directly on the 05c-plus architecture and falsified it:
- same-checkpoint naive replay: `3.96902897`
- same-checkpoint GPTQ replay: `3.96902897`
- `worse_than_naive_rowmax = 44/66`

Conclusion: GPTQ is permanently parked for this model family. Do not schedule more GPTQ work on the current branch.

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
- `scripts/diagnostics/diagnose_weights.py` — checkpoint diagnostics
- `scripts/diagnostics/compress_probe.py` — export-path feasibility probe
- `docs/superpowers/plans/2026-03-30-session-05c-plus.md` — 05c-plus plan
