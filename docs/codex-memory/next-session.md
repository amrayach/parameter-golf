# Next Session

## Phase

**Session 05g is the active next candidate. 05c-plus remains the best measured branch.**

GPTQ is permanently parked. Focus on naive int6 export only.

## Immediate next action

1. Sync 05g to Pegasus: `git pull` on `/netscratch/$USER/parameter-golf`
2. Run 1xGPU smoke (see `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/README.md`)
3. If smoke passes, run 8xH100 full
4. Compare vs 05c-plus: sliding s64 val_bpb, step_avg_ms, artifact size
5. If 05g recovers throughput while keeping quality, it becomes the new best branch

## What happened in Session 05c-plus / 05f (MEASURED)

8xH100 result:
- sliding s64 val_bpb: `1.12557920` (anchor delta: **-0.00347**, positive)
- pre_quant EMA: `1.14186715`
- int6 roundtrip: `1.14933197`
- step_avg: `100.39 ms` (+9.02ms vs anchor, **regressed**)
- steps: `5977` (587 fewer than anchor due to throughput)
- artifact: `15,589,271` bytes

Quality-positive but throughput regressed materially. Not a seed-validation branch.

05f 8xH100 follow-up:
- sliding s64 val_bpb: `1.12660664` (**worse** than 05c-plus by `+0.00103`)
- pre_quant EMA: `1.14190308`
- int6 roundtrip: `1.15026661`
- step_avg: `100.51 ms` (no throughput recovery)
- artifact: `15,630,854` bytes (+41,583 vs 05c-plus)

Conclusion: 05f is negative. Do not continue that line.

## Current diagnostic workflow

Artifacts:
- `diagnostics/2026-03-31_05c_plus/final_model.pt`
- `diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- `diagnostics/2026-03-31_05c_plus/train.log`
- `diagnostics/2026-03-31_05c_plus/diagnostics_float.txt`
- `diagnostics/2026-03-31_05c_plus/diagnostics_int6.txt`

Utility:
- `diagnose_weights.py`

Approaches:
- single-checkpoint weight statistics:
  - `python diagnose_weights.py final_model.pt`
- float-vs-int6 comparison on the same checkpoint:
  - `python diagnose_weights.py final_model.pt final_model.int6.ptz`
- interpret both reports together with the measured 05c-plus / 05f logs before choosing the next branch

Scope:
- useful for weight norms, outliers, sparsity, SmearGate / VE / Bigram scale inspection, and float-vs-int6 damage proxies
- not sufficient for activation-level claims by itself

## Files to read first

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/README.md`
4. `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py` (diff vs 05c-plus: xsa_last_n 11→8 only)
