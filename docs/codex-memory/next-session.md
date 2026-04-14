# Next Session

## Phase

**Updated 2026-04-14.** Strategy pivot: reproduce `#1610` directly + posterior corrector. Plan locked at `docs/campaign/PLAN_PR1610_CORRECTOR.md` (Revision 3).

## Immediate next action

Read only:
1. `AGENTS.md`
2. `docs/campaign/AGENT_SYNC.md`
3. `CLAUDE.md`
4. `docs/campaign/PLAN_PR1610_CORRECTOR.md`

**Task for next session (Session B): baseline reproduction only.**

1. **Verify pinned source exists.**
   - `/tmp/pr1610_train_gpt_pinned.py` should exist from prior session
   - If missing, re-fetch from GitHub at SHA `ca1919539dc6e328ea890cb03ad3ca1c5a84da55`

2. **Run dependency gate.**
   - Fetch `requirements.txt` from #1610 at pinned SHA
   - Verify: flash_attn_interface (FA3), triton, brotli, sentencepiece, torch version
   - Document any version mismatches

3. **Run baseline reproduction (Gate A then Gate B).**
   - Seed 0 first, verify within 0.003 of published 1.07258
   - Then seeds 1, 2 for 3-seed mean

4. **Do not start the corrector.**

Subsequent sessions:
- **Session C**: corrector skeleton + legality tests + benchmark
- **Session D**: first eval-only corrector trial
- **Session E**: fallback / README / submission polish

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

05g 8xH100 follow-up:
- sliding s64 val_bpb: `1.12584234` (**worse** than 05c-plus by `+0.00026`)
- pre_quant EMA: `1.14203044`
- int6 roundtrip: `1.14963535`
- step_avg: `98.67 ms` (modest recovery)
- artifact: `16,475,467` bytes (**over the cap** on the old export path)

Conclusion: 05f and 05g are both negative follow-ups. Do not continue the local tweak line.

## Current diagnostic workflow

Artifacts:
- `diagnostics/2026-03-31_05c_plus/final_model.pt`
- `diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- `diagnostics/2026-03-31_05c_plus/train.log`
- `diagnostics/2026-03-31_05c_plus/diagnostics_float.txt`
- `diagnostics/2026-03-31_05c_plus/diagnostics_int6.txt`

Utility:
- `scripts/diagnostics/diagnose_weights.py`
- `scripts/diagnostics/compress_probe.py`

Approaches:
- single-checkpoint weight statistics:
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt`
- float-vs-int6 comparison on the same checkpoint:
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt final_model.int6.ptz`
- export-path feasibility:
  - `python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- interpret these reports together with the measured 05c-plus / 05f / 05g logs before choosing the next larger fork

Scope:
- useful for weight norms, outliers, sparsity, SmearGate / VE / Bigram scale inspection, and float-vs-int6 damage proxies
- not sufficient for activation-level claims by itself

## Files to read first

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `scripts/diagnostics/compress_probe.py`
4. `diagnostics/README.md`
5. `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
