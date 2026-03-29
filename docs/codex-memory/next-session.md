# Next Session

## Phase

**Session 05b GPTQ implementation complete but has a CORRECTNESS BUG. Must debug before 8xH100 run.**

## Immediate next action

1. **Debug the GPTQ roundtrip quality regression** — 0.212 BPB gap vs anchor's 0.00775
   - Code: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
   - Plan: `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`
   - The GPTQ pipeline runs cleanly (66 layers, 0 fallbacks, 4.2s) but produces garbage weights
2. **Debug approach**: Add per-layer MSE comparison (GPTQ vs naive), try disabling actorder, verify Hinv_chol usage against PR #634/#1019 diffs
3. After fix: re-run 1xH100 smoke, verify roundtrip gap < 0.01
4. Then: full 8xH100 600s run
5. Then: Session 05c training bundle (XSA-all + VE128 + SWA + warmdown3500)

## Key files for debugging

- `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py` — lines 490-571 (`gptq_quantize_layer`)
- `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md` — full algorithm description
- PR #634 and #1019 on openai/parameter-golf — reference implementations (use `gh pr diff`)

## 1xH100 smoke test result (2026-03-29 20:43 UTC+2)

- Node: serv-3340, 906 steps, step_avg ~663ms
- Pre-quant EMA: val_bpb=1.4775 (normal for 1xH100 906 steps)
- Roundtrip: val_bpb=1.6896 (**gap: 0.212 BPB — catastrophically bad**)
- GPTQ stats: 66 GPTQ layers, 0 naive, 0 Cholesky fallbacks, 4236ms
- Artifact: 7,754,877 bytes (under cap)
- Job killed by time limit before sliding eval completed

## Session 05 audit summary

The audit identified the 2026-03-22 record as the primary porting reference (same CastedLinear/DDP architecture, has FA3+VE+SWA+warmdown3500+QAT). First-wave features are: FA3, VE128, warmdown 3500, SWA, Late QAT, LeakyReLU² re-test (gated on FA3). TTT appears compliant via score-first protocol. Follow-up benchmark: direct FA3 on `25.02` + wheel beat SDPA flash by `11.44x` in the isolated attention kernel benchmark. See full audit: `docs/campaign/artifacts/05_ttt_correctness_audit.md`.

## Prerequisites (all satisfied)

- Session 03 anchor verified: sliding s64 val_bpb `1.12904446`, 6564 steps, 91.37ms/step
- Remaining donor gap is small (`+0.00419944` on final sliding), so broad redesign is unnecessary
- NGC container + fscratch path confirmed on Pegasus
- Launcher lesson locked: use `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, NOT torchrun
- Saved FA3 Pegasus container built and import-verified
- `1xH100` FA3 smoke completed without stability issues
- `8xH100` FA3 saved-container run completed and regressed vs anchor
- int6+zstd roundtrip artifact: `15751324` bytes, headroom `248676` bytes

## Session 05 FW-1 closeout

`2026-03-29_fa3_port` vs Session 03 anchor:

- Sliding s64 val_bpb: `1.12958984` (worse by `+0.00054538`)
- Pre-quant EMA val_bpb: `1.14532979` (worse by `+0.00060576`)
- Roundtrip val_bpb: `1.15296145` (worse by `+0.00048872`)
- Artifact: `15529557` bytes (smaller by `221767` bytes)
- Step_avg: `92.67 ms` (`+1.30 ms` slower, `-90` steps)

Conclusion:
- The current saved-container FA3 runtime is a clean negative result.
- The likely issue is runtime-level regression from the pip-installed generic torch stack replacing the tuned NGC build.
- Do not rerun this runtime path as-is.

## Session 04 closeout

1. ~~Delta 1: GPTQ-lite percentile clip search~~ — **COMPLETE (FAILED)**
   - Sliding s64 val_bpb: `1.12941356` (worse than anchor by `+0.00036910`)
   - Artifact: `16219752` bytes — OVER the `16000000` byte cap
   - Conclusion: hurts zstd compressibility more than it helps quantization quality

2. ~~Delta 2: LeakyReLU^2~~ — **COMPLETE (NEUTRAL)**
   - Sliding s64 val_bpb: `1.12904123` (effectively identical, `-0.00000323`)
   - Pre-quant EMA val_bpb: `1.14438546` (slightly better, `-0.00033857`)
   - Roundtrip val_bpb: `1.15222198` (slightly better, `-0.00025075`)
   - Artifact: `15582968` bytes (168KB smaller)
   - Step_avg: `92.09 ms` (+0.72 ms slower, -53 steps)
   - Conclusion: not a standalone graduating delta. Keep as possible stack component.

3. Session 04 decision
   - Close the micro-delta sweep at `1 failed + 1 neutral`
   - Do not force a Delta 3 by default
   - Open Session 05 instead

## Measurement discipline

- Each delta is a separate run with one change
- Compare against Session 03 anchor as the fixed reference
- Record: GPU, steps, step_avg, sliding s64 val_bpb, pre-quant EMA val_bpb, int6 roundtrip val_bpb, artifact size
- Only combine deltas after each is measured in isolation

## Session 05 target

- strengthen the pre-TTT base relative to `1.12904446`
- understand and, if justified, integrate TTT on top of a stronger base
- identify the highest-value portable pieces of the local `1.1194` public stack

## Read order for the next fresh session

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `docs/campaign/artifacts/04_targeted_delta_sweep.md`
4. `docs/campaign/sessions/05_ttt_correctness_audit.md`
5. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

## Next investigation shape

```bash
ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh | sort -V

srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 --mem=64G --time=00:10:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  bash -c 'python -c "import torch; print(torch.__version__)"'
```
