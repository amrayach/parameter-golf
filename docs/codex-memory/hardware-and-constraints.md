# Hardware And Constraints

## Challenge constraints

- train in under `10 minutes` on `8xH100 SXM`
- artifact under `16,000,000` bytes total
- final metric is `val_bpb`
- official leaderboard entry is record-gated: must beat the current official SOTA by at least `0.005` nats with enough logs for `p < 0.01`

## Pegasus verified state

- `H100` allocation works for `1xH100`
- `8xH100` single-node allocation works
- `8xH100` NCCL works under Slurm-native `srun`
- `torchrun --standalone` is currently not the correct launcher for Pegasus `8xH100`
- NGC 26.03 container confirmed working on Pegasus
- Saved FA3 container confirmed at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- Stock NGC `25.02` + `--no-deps` FA3 install fails with `undefined symbol: aoti_torch_abi_version`
- `1xH100` FA3 smoke trains stably at about `640 ms/step` after warmup
- `8xH100` FA3 on the saved `25.02` container is slower than the SDPA anchor (`92.67 ms` vs `91.37 ms`)
- Challenge-shaped `8xH100` or `8xH200` runs must include `--nodes=1`
- `/fscratch` confirmed as optimized data staging path (avoids `/netscratch` I/O bottlenecks)

## Current measured anchors

- `1xA100` root baseline: `1.37140771`
- `1xH100` root baseline: `1.30594735`
- `8xH100` root baseline: `1.23368511`
- `8xH100` Session 03 anchor (sliding s64): `1.12904446`
- `8xH100` Session 03 anchor (int6 roundtrip): `1.15247273`

## Artifact pressure

- `8xH100` root baseline artifact: `15871532` bytes
- `8xH100` Session 03 int6+zstd artifact: `15751324` bytes (model `15692752` + code `58572`)
- remaining headroom under cap: `248676` bytes
- size discipline is now important for any competition-phase change
- GPTQ-lite clip search INCREASES artifact size to `16219752` bytes — OVER the `16000000` cap by `219752` bytes
- GPTQ-lite is NOT a viable export path: it hurts zstd compressibility more than it helps quantization quality
- Anchor int6+zstd with fixed row-max remains the viable export path

## RunPod

- reserve for final validation only unless external credits are granted

## Current phase

- **Session 05c-plus is the current best measured branch**
- Session 05f and Session 05g are measured negatives vs 05c-plus
- Infrastructure uncertainty is no longer the blocker
- Current blocker is compression-path feasibility for the next larger fork
- GPTQ is parked (7 ablations, code correct, failure model-specific)
- FA3 is parked (ABI issue with NGC container)
- SWA is excluded (dead code in reference PRs)
- Checkpoint diagnostics are now part of the mainline workflow:
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt`
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt final_model.int6.ptz`
- Compression feasibility is now part of the mainline workflow:
  - `python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- RunPod $25 credits reserved for one decisive 8xH100 run after Pegasus diagnostics and branch selection are settled

## Practical rules

- Do not hide Pegasus job output with `| tail -1`; use unbuffered Python instead
- The saved-container FA3 runtime is a measured negative result; do not rerun as a throughput candidate
- Future FA3 work requires a vendor-tuned NGC runtime, not the current pip-replaced stack
- GPTQ-lite clip search confirmed as NOT helpful
- Current measured export candidate:
  - `custom-shuffle + brotli-10` on saved 05c-plus / 05g artifacts
  - byte-shuffle contributes only `~8-10 KB`; custom serialization + brotli is the dominant gain
- Attention microbenchmark (kernel-only, not full training throughput):
  - `26.03` SDPA flash: `1.967 ms/iter`
  - `25.02` SDPA flash: `1.889 ms/iter`
  - `25.02` direct FA3: `0.165 ms/iter`
