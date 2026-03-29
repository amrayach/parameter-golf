# Project State

Date: 2026-03-29

## Objective

Primary:
- use the verified Pegasus `8xH100` path to advance from the Session 03 anchor into Session 05
- improve both the pre-TTT base and the TTT plan rather than keep extending Session 04 micro-deltas

Secondary:
- keep the Session 03 anchor as the new fixed reference
- preserve exact launch, logging, artifact, and evaluation discipline

Stretch:
- reach a clearly improved `8xH100` pre-TTT and post-TTT story that justifies a stronger compute request or leaderboard-adjacent claim

## Current campaign state

- campaign scaffolding exists under `docs/campaign/`
- shared handoff file is `docs/campaign/AGENT_SYNC.md`
- evidence summary is `docs/campaign/artifacts/2026-03-28_a100_evidence_summary.md`
- coordination entry points exist:
  - `AGENTS.md`
  - `CLAUDE.md`
- Session 03 anchor run is complete

## Verified hardware state

- Pegasus `A100-80GB` path works
- Pegasus `1xH100` path works
- Pegasus `8xH100` path works when launched with Slurm-native `srun`
- Pegasus `8xH100` path does **not** work reliably with `torchrun --standalone` on `serv-3342`
- NGC 26.03 container on Pegasus confirmed working with fscratch setup
- Saved Pegasus FA3 container exists at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- `1xH100` FA3 smoke is confirmed healthy
- Stock `25.02` + `--no-deps` FA3 import is not viable on Pegasus

## Locked baseline facts

- `1xA100` 600s baseline post-roundtrip exact: `val_bpb=1.37140771`
- `1xH100` 600s baseline post-roundtrip exact: `val_bpb=1.30594735`
- `8xH100` 600s baseline post-roundtrip exact: `val_bpb=1.23368511`
- `8xH100` baseline step average: `51.66 ms`
- `8xH100` baseline artifact size: `15871532` bytes

## Current measured anchors

- `8xH100` root baseline: `val_bpb=1.23368511` (step_avg `51.66 ms`, artifact `15871532` bytes)
- `8xH100` Session 03 anchor:
  - sliding s64 val_bpb: `1.12904446`
  - pre-quant EMA val_bpb: `1.14472403`
  - int6 roundtrip val_bpb: `1.15247273`
  - steps: `6564`, step_avg: `91.37 ms`
  - artifact: `15751324` bytes (model `15692752` + code `58572`)
  - GPU: `8xH100 SXM5`, `serv-3342`, NGC 26.03 container

## Launcher lesson

Use:
- Slurm-shaped allocation with `--ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6`
- Slurm-native `srun`
- env mapping inside the launch:
  - `LOCAL_RANK=$SLURM_LOCALID`
  - `RANK=$SLURM_PROCID`
  - `WORLD_SIZE=$SLURM_NTASKS`

Do not use:
- `torchrun --standalone` for Pegasus `8xH100`

## What has been demonstrated

- end-to-end training, evaluation, compression, and roundtrip validation
- controlled negative results (`LowerLR`, `Warmdown3600`)
- small A100 seed spread
- first challenge-shaped root baseline on real `8xH100`
- Session 03 pre-TTT anchor port: sliding s64 val_bpb `1.12904446` on `8xH100`
- int6+zstd roundtrip under the 16MB cap with `248676` bytes headroom
- small remaining donor gap with both throughput and export fidelity still worth isolated measurement
- NGC container + fscratch confirmed as optimized Pegasus path
- GPTQ-lite percentile clip search does not help at this scale (Session 04 Delta 1 negative result: worse BPB + artifact cap violation)
- LeakyReLU^2 activation is neutral (Session 04 Delta 2: sliding s64 val_bpb effectively identical at `1.12904123`, but slightly better quantization metrics and 168KB smaller artifact; slower step time cancels quality gain)
- The local public `1.1194` record is not “TTT only”: its pre-TTT base is already `1.1218` at `83.4 ms`, so stronger pre-TTT work and throughput matter before TTT can close the remaining gap
- Direct FA3 on Pegasus is now benchmark-backed as the first implementation target: in the isolated attention benchmark, `25.02` + wheel ran direct FA3 at `0.165 ms/iter` vs SDPA flash at `1.889 ms/iter` in the same container. This is kernel-only evidence and still needs full-training validation.
- The FA3 deployment path is now operationally locked: build the saved `25.02` FA3 container once, then reuse it for smoke and full jobs.
- The first `1xH100` FA3 smoke trained normally and stabilized near `640 ms/step` after warmup.
- The first full `8xH100` FA3 run on that saved-container path is a clean negative result: slower, fewer steps, and worse BPB than the SDPA anchor.

## Session 05b: Full Hessian GPTQ (2026-03-29)

- Implementation: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
- Plan: `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`
- Commit: `e00bc0a` pushed to origin/main
- Algorithm: post-training calibration (128 seqs), Cholesky error compensation, block_size=128, actorder, percdamp=0.01
- 4 new functions (~200 lines): `_make_hessian_hook`, `collect_hessians`, `gptq_quantize_layer`, `gptq_mixed_quantize_int6`
- Export path restructured: rank-0-only GPTQ, barrier, all ranks read file for eval
- **1xH100 smoke test: CORRECTNESS BUG** — roundtrip gap 0.212 BPB (27x worse than anchor's 0.00775)
  - 66 layers GPTQ'd, 0 Cholesky fallbacks, 4.2s quantization, 7.75MB artifact
  - Pipeline mechanics work, but quantized weights reconstruct poorly
  - Must debug before 8xH100 run

## What has not happened yet

- no correct Full Hessian GPTQ result yet (bug found in first smoke test)
- no vendor-tuned NGC FA3 runtime result yet
- no positive end-to-end FA3 result yet
- no top-tier leaderboard-adjacent result yet
- no measured VE128 delta yet

## Best next move

- **Debug the GPTQ roundtrip quality regression** — top priority
- Add per-layer MSE comparison, try disabling actorder, verify against PR diffs
- After fix: re-smoke on 1xH100, then full 8xH100 run
- Then Session 05c training bundle (XSA-all + VE128 + SWA + warmdown3500)
- Do not spend time on FA3 or TTT until GPTQ is fixed
