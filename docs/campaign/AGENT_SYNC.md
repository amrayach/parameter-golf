# Agent Sync

Date: 2026-03-28

## Current Objective

Turn the completed Pegasus `A100-80GB` evidence runs into a short, defensible compute-grant evidence note.

This is not yet an H100-parity validation campaign.

## In Scope

- Summarize the completed `A100-80GB` smoke run
- Summarize the completed `600s` root baseline run
- Summarize the completed `600s` `LowerLR` comparison run
- Produce a short grant-ready evidence note with exact measured metrics

## Out Of Scope

- Claiming H100 parity
- Starting the Session 03 anchor port
- Treating RFN as the mainline strategy
- Spending RunPod budget except for final validation later
- Arbitrary trainer edits unrelated to hardware compatibility or the current baseline/comparison plan

## Current Hardware Stance

- Parity target: Pegasus `H100`
- Active development target: Pegasus `A100-80GB`
- Development fallback: other Pegasus GPUs only when A100/H100 are unavailable

## Status Snapshot

- Pegasus operator path: confirmed working
- A100 smoke run: complete
- A100 `600s` baseline run: complete
- A100 `600s` `LowerLR` comparison: complete
- A100 `600s` baseline seed-42 reproducibility run: complete
- A100 `600s` warmdown-only variant: complete
- Current best measured A100 result: root baseline (`val_bpb=1.37140771`)
- Baseline seed spread is small (`+0.00319322` BPB from seed `1337` to seed `42`)
- Immediate next deliverable: written summary, not another rerun

## Canonical Workspaces

- Local repo: `/home/amay/Work/parameter-golf`
- Remote repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` as the default remote sync path.
Use `rsync` only when local uncommitted changes need to be pushed quickly.

## Latest Measured Results

Date: 2026-03-27
Node: `serv-3333`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_baseline_smoke`

Measured outputs:

- Train setup: `1` shard, `200` iterations, `TRAIN_BATCH_TOKENS=65536`
- `amp_dtype: bf16`
- Step average at finish: `154.57 ms`
- Pre-roundtrip eval: `val_loss=3.6186`, `val_bpb=2.1432`
- Post-roundtrip exact eval: `val_loss=3.67022861`, `val_bpb=2.17371612`
- Post-roundtrip eval time: `250881 ms`
- Peak memory: `1548 MiB allocated`, `1566 MiB reserved`
- Total submission size `int8+zlib`: `7066088` bytes

Interpretation:

- Pegasus execution path is working end to end on available hardware.
- Artifact size is comfortably below the `16,000,000` byte cap.
- This run is development evidence only, not H100-equivalent validation.

Date: 2026-03-28
Node: `serv-3333`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_baseline_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `907` steps in `600119 ms`
- Pre-roundtrip eval: `val_loss=2.3117`, `val_bpb=1.3691`
- Post-roundtrip exact eval: `val_loss=2.31556447`, `val_bpb=1.37140771`
- Post-roundtrip eval time: `22204 ms`
- Peak memory: `10253 MiB allocated`, `10578 MiB reserved`
- Total submission size `int8+zlib`: `12046627` bytes

Date: 2026-03-28
Node: `serv-3333`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_lowerlr_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `908` steps in `600185 ms`
- Pre-roundtrip eval: `val_loss=2.3142`, `val_bpb=1.3706`
- Post-roundtrip exact eval: `val_loss=2.32600988`, `val_bpb=1.37759407`
- Post-roundtrip eval time: `22206 ms`
- Peak memory: `10253 MiB allocated`, `10530 MiB reserved`
- Total submission size `int8+zlib`: `10723611` bytes

Comparison:

- On this 1xA100 600s setup, `LowerLR` is worse than the root baseline by `+0.00618636` BPB post-roundtrip.
- `LowerLR` does reduce artifact size by `1323016` bytes, but size was not the limiting factor here.
- For grant evidence, the useful conclusion is that the operator path is reproducible and controlled comparisons are already discriminating between variants.

Date: 2026-03-28
Node: `serv-3338`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_baseline_seed42_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `900` steps in `600537 ms`
- Pre-roundtrip eval: `val_loss=2.3165`, `val_bpb=1.3720`
- Post-roundtrip exact eval: `val_loss=2.32095610`, `val_bpb=1.37460093`
- Post-roundtrip eval time: `22483 ms`
- Peak memory: `10253 MiB allocated`, `10578 MiB reserved`
- Total submission size `int8+zlib`: `12018778` bytes

Reproducibility read:

- Baseline seed `42` is worse than baseline seed `1337` by `+0.00319322` BPB post-roundtrip.
- Step time and memory are effectively unchanged across baseline seeds.
- This is strong enough to claim the baseline behavior is reproducible on Pegasus `A100-80GB`.

Date: 2026-03-28
Node: `serv-3338`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_warmdown3600_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap, `WARMDOWN_ITERS=3600`
- `amp_dtype: bf16`
- Stopped at `903` steps in `600661 ms`
- Pre-roundtrip eval: `val_loss=2.3568`, `val_bpb=1.3958`
- Post-roundtrip exact eval: `val_loss=2.38171775`, `val_bpb=1.41058741`
- Post-roundtrip eval time: `22360 ms`
- Peak memory: `10253 MiB allocated`, `10530 MiB reserved`
- Total submission size `int8+zlib`: `9951155` bytes

Schedule read:

- Warmdown-only is worse than the root baseline by `+0.03917970` BPB post-roundtrip.
- Warmdown-only is also worse than `LowerLR` by `+0.03299334` BPB post-roundtrip.
- It does reduce artifact size by `2095472` bytes versus the root baseline, but size was not the bottleneck.
- Current evidence says the root schedule should remain the A100 anchor.

## Next Actions

### 1. Extract comparable evidence lines

```bash
grep -E "amp_dtype:|step:.*val_loss:|stopping_early:|peak memory|Serialized model int8\\+zlib|Total submission size int8\\+zlib|final_int8_zlib_roundtrip" /netscratch/$USER/a100-*.log
```

### 2. Write grant-ready summary

The summary should include:

- the successful `A100-SXM4-80GB` smoke run
- the `600s` baseline result
- the `600s` `LowerLR` comparison result
- the `600s` baseline seed-42 reproducibility result
- the `600s` warmdown-only negative result
- the conclusion that baseline currently beats `LowerLR` on this setup
- the conclusion that baseline seed sensitivity appears small on this setup
- the conclusion that extending warmdown alone is harmful on this setup
- the fact that artifact sizes are already under the challenge cap

### 3. Optional next experiment only after the summary exists

If another controlled A100 run is needed, do not repeat `LowerLR`.
Pick a different single-change variant.

Candidate next variants after the summary:

- a pure artifact-size tradeoff variant
- an eval-side control, only if kept clearly separate from training-side comparisons

Warmdown-only has now been tested and should not be repeated unless coupled to another materially different change.

## Evidence Required From Each Run

- GPU model
- Exact command
- Train wallclock / `step_avg`
- Final post-roundtrip `val_bpb`
- Artifact size
- Peak memory
- Any compile or export warnings

## Decision Rule

The two-run A100 baseline/comparison pair now exists.
Do not broaden scope further until it is summarized into a short grant-ready evidence note.
