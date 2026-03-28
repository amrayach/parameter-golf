# Next Session

## Immediate next action

**Pegasus verification is PARTIAL.** Account access confirmed, but GPU allocation not yet tested (cluster saturated on 2026-03-27).

Read-only anchor analysis is now complete:
- `docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md`
- `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`

Fresh sessions should treat those two docs as the current source of truth before touching Session 03.

Current read-only conclusion:
- prefer a clean `2026-03-21`-style anchor for Session 03
- treat `2026-03-22` GPTQ-lite as a likely first post-anchor refinement, not part of the smallest credible anchor
- do not treat late QAT, DTG, VE, or tight SWA as anchor defaults without new validation
- treat the repo-root `train_gpt.py` as the cleaner donor skeleton, but not a near-anchor; Session 03 must port multiple feature clusters, not just tune env vars

Public-state note checked on 2026-03-27:
- merged `main` README still lists `2026-03-22` (`1.1228`) as the top merged non-TTT result
- PR `#693` and PR `#875` are stronger open non-TTT claims, but remain unmerged
- PR `#910` is expected-result-only, not measured proof
- PR `#893` is a two-pass n-gram branch, not the pre-TTT anchor path
- do not pivot Session 03 solely because of open PR claims

Locked Session 03 implementation order:
1. freeze anchor constants in the new Session 03 script
2. port SmearGate + BigramHash into the root token path
3. port partial RoPE + XSA + LN scale into attention/block/GPT wiring
4. add Muon/Adam weight decay and EMA
5. replace root int8+zlib export with mixed int6 + zstd roundtrip
6. add stride-64 sliding eval and final anchor logging
7. verify GPTQ-lite, VE, DTG, SWA, late QAT, MTP, and TTT all remain out of scope

Fresh Codex sessions can use these installed local skills if helpful:
- `research-engineer` for rigorous diff analysis and critique
- `gptq` for quantization-specific context
- `model-pruning` for compression-side comparisons
- `transformer-lens-interpretability` for future sidecar/mech-interp work

Remaining verification steps:
1. Retry `salloc -p H100 --nodes=1 --gpus=1 --time=00:10:00` during off-peak hours
2. Run `nvidia-smi -L` and `nvidia-smi topo -m` once allocated
3. Update `docs/campaign/artifacts/02a_pegasus_verification.md` with results
4. Only then proceed to baseline training from Session 02

If Pegasus remains saturated and the explicit goal shifts to compute-grant evidence rather than H100 parity, the fallback development package is already specified in `03b`:
1. `1` root baseline evidence run
2. `1` narrow clean-anchor smoke run
3. preferred hardware order: Pegasus `H200`, Pegasus `A100-80GB`, then remaining Runpod quick-start credit
4. capture GPU type, steps, wallclock, final `val_bpb`, artifact size, eval mode, and compile/export warnings

Those fallback runs are useful for a `Development grant` application, but they do not satisfy the H100 parity gate.

## Required artifact

Create:
- `docs/campaign/artifacts/02a_pegasus_verification.md`

It must capture:
- partition availability
- account access to H100-class partitions
- QoS and fairshare clues
- whether `--nodes=1 --gpus=8` allocation is possible
- actual `nvidia-smi -L` output on an allocated node
- actual `nvidia-smi topo -m` output on an allocated node

## Commands to run on Pegasus

```bash
sinfo -N -p H100,H100-RP,H100-SEE,H100-PCI -o "%P %N %G %t %c %m"
scontrol show partition H100
scontrol show partition H100-RP
scontrol show partition H100-SEE
sshare -u "$USER"
sacctmgr show assoc where user="$USER" format=Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxJobs

salloc -p H100 --nodes=1 --gpus=8 --time=00:05:00 --gpu-bind=none
hostname
nvidia-smi -L
nvidia-smi topo -m
exit
```

If `sacctmgr` is unavailable to your user, continue anyway.

If `H100` does not schedule, try:
- `H100-RP`
- `H100-SEE`

## Stop conditions

Stop and reassess before training if:
- your account cannot access H100-class partitions
- you cannot get 8 GPUs on one node
- QoS blocks short 8-GPU jobs
- allocated hardware is not the expected H100 SXM class

Stop and reassess before pivoting strategy if:
- an open PR becomes merged and clearly changes the non-TTT frontier
- you are tempted to skip the clean anchor and jump straight to GDN, two-pass n-gram rescoring, or other open-claim branches without first owning the current stack

## Allowed interim work while Pegasus is saturated

- read `docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md` before touching Session 03
- read `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md` before implementing Session 03 from the root script
- read `docs/codex-memory/session-handoff.md` in fresh sessions before resuming
- documentation updates under `docs/campaign/artifacts/`
- preparation for Session 03 without implementing or training anything
- if explicitly needed for a compute-grant application, prepare or run the two-run development evidence package from `03b`, but label it non-parity evidence

## First development evidence package after verification or if H100 remains blocked

Run 1:
- root baseline evidence run
- preferred hardware order: Pegasus H200, Pegasus A100-80GB, then remaining Runpod quick-start fallback

Run 2:
- narrow clean-anchor smoke port
- preferred hardware order: Pegasus H200, Pegasus A100-80GB, then remaining Runpod quick-start fallback

For both runs, capture:
- GPU type
- steps completed
- wallclock
- final `val_bpb`
- artifact size
- eval mode
- compile warnings
- export warnings
