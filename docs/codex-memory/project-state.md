# Project State

Date: 2026-03-31

## Objective

Primary:
- smoke and run Session 05f (BigramHash 3072x112 + warmdown 4000) on `8xH100`
- base: 05c-plus, which measured quality-positive (sliding s64 `1.12558`, delta `-0.00347`) but throughput-regressed (`100.39 ms`)

Secondary:
- keep the Session 03 anchor as the fixed reference
- preserve exact launch, logging, artifact, and evaluation discipline

Completed:
- Session 05e GPTQ probe: **negative result** (44/66 layers worse than naive). GPTQ permanently parked.

## Current campaign state

- campaign scaffolding exists under `docs/campaign/`
- shared handoff file is `docs/campaign/AGENT_SYNC.md`
- evidence summary is `docs/campaign/artifacts/2026-03-28_a100_evidence_summary.md`
- coordination entry points exist:
  - `AGENTS.md`
  - `CLAUDE.md`
- Session 03 anchor run is complete
- Session 05b GPTQ implementation exists but is parked on the current anchor after 7 conclusive ablations
- Session 05c-plus code and smoke harness are implemented and pushed
- Session 05e same-checkpoint export-only replay completed and closed the GPTQ question for this model family

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
- `8xH100` Session 05c-plus (quality-positive, throughput regressed):
  - sliding s64 val_bpb: `1.12557920` (anchor delta: **-0.00347**)
  - pre-quant EMA val_bpb: `1.14186715`
  - int6 roundtrip val_bpb: `1.14933197`
  - steps: `5977`, step_avg: `100.39 ms` (+9.02ms vs anchor)
  - artifact: `15589271` bytes
  - GPU: `8xH100`, NGC 26.03 container

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
- Direct FA3 on Pegasus was benchmark-backed as a hypothesis, but the saved-container end-to-end path is now a measured negative result.
- The FA3 deployment path is operationally understood, but the current saved-container runtime is not a throughput candidate.
- The first `1xH100` GPTQ smoke successfully exercised Hessian collection, quantization, compression, reload, and eval.
- That same smoke also exposed a correctness failure in the current GPTQ quantizer: roundtrip exact `1.68963326` vs pre-quant exact `1.47753094`.
- The first replay-based Hessian repair also failed: `replay_ref_hfix` reached `2.15770170` from pre-quant `1.82064877`, with `gptq_diag` still reporting `66/66` layers worse than both naive baselines.
- The 05e architecture probe also failed to rescue GPTQ on the new stack:
  - pre-quant exact `3.95543154`
  - naive roundtrip exact `3.96902897`
  - GPTQ roundtrip exact `3.96902897`
  - `worse_than_naive_rowmax = 44/66`

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
  - The `1xH100` training metrics are not anchor-comparable because the smoke run uses a different `WORLD_SIZE` and therefore different `grad_accum_steps`
- **2026-03-29 code repair landed, rerun pending**
  - local PR diff found the key loop mismatch: `W_block[:, j + 1:]` vs PR `W_block[:, j:]`
  - the repaired code now matches the PR structure for:
    - within-block residual propagation
    - 5-percentile reconstruction search
    - symmetric `[-31, 31]` clamp
    - block-only `attn` / `mlp` Hessian targeting
  - export now writes `gptq_layer_diagnostics.json` with per-layer naive-vs-GPTQ MSE and worst-block summaries
  - this repo does not currently contain a saved checkpoint for same-checkpoint replay
  - this local shell does not have `torch`, so verification here only reached `py_compile`
- **2026-03-29 server replay still failed**
  - `gptq_diag: worse_than_legacy_rowmax=66 worse_than_percentile_naive=66`
  - roundtrip exact `2.15604597` vs pre-quant exact `1.82064982`
  - the remaining failure is systematic
  - export-only replay mode is now landed so the next ablations can use the saved `final_model.pt` directly
- **2026-03-30 smaller Hessian-path repair also failed**
  - commit on `main`: `9cea7e9`
  - `replay_ref_hfix`: `1.82064877 -> 2.15770170`, gap `+0.33705293`
  - `gptq_diag` remains `66/66` worse than both naive baselines
  - the smaller forward-hook + average+damp patch is not enough
  - next step should be a more faithful single-PR Hessian/quantization transplant

## What has not happened yet

- no vendor-tuned NGC FA3 runtime result yet
- no top-tier leaderboard-adjacent result yet
- no seed-validation run yet for 05c-plus (throughput regression makes it premature)
- no 05f (bigram3072 + warmdown4000) smoke or run yet

## Best next move

- **Smoke 05f on 1xGPU, then launch on 8xH100 if smoke passes**
- Code: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py`
- Base: 05c-plus (XSA-all + VE128 + warmdown 3500 + LeakyReLU(0.5)²)
- Additional changes: BigramHash 2048→3072, dim 128→112, warmdown 3500→4000
- 05c-plus measured quality-positive (sliding s64: 1.12558, delta -0.00347 vs anchor) but throughput regressed (+9ms)
- SWA excluded (dead code), GPTQ excluded (permanently parked)
