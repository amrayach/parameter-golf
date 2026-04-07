# Project State

Date: 2026-04-07

## Objective

Primary:

- preserve the completed RunPod `#1394` SP8192 baseline as the new stable base
- use the fetched local `#1394` archive and strict proof folder as the durable evidence base
- move the foreground branch hunt to faithful `#1413`

Secondary:

- keep `07c1` evidence background-only
- do not let the failed Pegasus TTT reruns consume foreground attention

## Current campaign state

- `05c-plus` is superseded
- `07c` and `07c1` validated the move away from the old compression line, but they are no longer the foreground branch
- the faithful `#1394` SP8192 RunPod baseline is now complete on `8xH100 SXM`
- packaging fix is validated on the real RunPod checkpoint:
  - counted code bytes: `17,821`
  - all three completed seeds are under the 16 MB cap
- the preserved archive has now been fetched locally:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256: `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- the extracted local recovery directory is:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07`
- preserved archive contents include:
  - all three seed logs
  - copied `train_gpt.py`
  - summary / manifest / checksums
  - exact `2025` checkpoint binaries
- exact `1337` and `42` checkpoint binaries are not preserved; only their logs and measured outputs remain
- the local strict proof folder has been built:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- the hardened RunPod data/deps prep helper exists and is ready:
  - `scripts/runpod_prepare_sp8192.sh`

## Current best measured results

### Best current foreground line

Completed `#1394` RunPod 3-seed baseline:

- `1337`
  - sliding s64: `1.08471849`
  - roundtrip exact: `1.10134414`
  - bytes total: `15,986,188`
- `42`
  - sliding s64: `1.08576707`
  - roundtrip exact: `1.10263175`
  - bytes total: `15,987,537`
- `2025`
  - sliding s64: `1.08513825`
  - roundtrip exact: `1.10167670`
  - bytes total: `15,986,526`

Aggregate:

- mean sliding s64: `1.08520794`
- mean roundtrip exact: `1.10188420`
- max bytes total: `15,987,537`

### Best `07c1` strict evidence

- `07c1_runpod_base_s2025_strict`
  - script: `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`
  - sliding s64: `1.10894203`
  - roundtrip exact: `1.11863096`
  - `val_loss = 1.87233216` nats
  - bytes total: `15,728,840`
  - train time: `598052ms`

## Competitive reality

Open frontier references as of 2026-04-07:

- merged official `#1019`: `1.11473509` BPB / `1.88217853` nats
- open clean `#1394`: `1.08563`
- open `#1412`: `1.08354`
- open `#1413`: `1.08279`
- open `#1420`: `1.08014`
- open `#1416`: `1.07948`
- open `#1437`: `1.07800`

Interpretation:

- our completed `#1394` baseline is competitive with the clean SP8192 tier
- it is about `0.00042` BPB better than the open `#1394` reference mean, on only 3 seeds instead of 5
- it is still behind `#1413`, `#1420`, `#1416`, and `#1437`
- the next real move is therefore `#1413`, not another `#1394` rerun and not more `07c1` polish

## Background `07c1` state

Locked `07c1` findings:

- strict RunPod base proof covers four seeds
- best strict seed: `1.10894203` BPB / `1.87233216` nats
- strict 4-seed significance vs merged `#1019` is still strong:
  - delta: `-0.00796789` nats
  - Welch `t = -6.8667`
  - one-sided `p = 0.001785`
- falsified levers:
  - `QK_GAIN=5.0`
  - `MLP_MULT=3.08`
- `MLP_MULT=3.5` remains over cap under current export
- TTT remains unresolved because the repaired Pegasus reruns have not yet produced a clean measured answer

Pegasus background jobs:

- `2740306` `07c1_ttt_s1337_fix`
- `2740307` `07c1_ttt_s2025_fix`
- latest state for both jobs on Pegasus: `FAILED`

## Verified hardware / workflow state

- RunPod `8xH100 SXM` is a valid foreground reproduction lane
- Pegasus remains the canonical validator, but is not the current foreground execution path
- for this migration pod shape:
  - `/workspace` persists across `Stop`
  - `/root` is not reliable across `Stop`
- the old `scripts/runpod_fetch_logs.sh` assumption of a clean direct SSH/`rsync` path is not reliable for the current gateway-only pod
- direct TCP SSH forwarding can still refuse connections even while `sshd` is listening inside the pod
- the working recovery path was a PTY-driven gateway shell with local base64 capture and checksum verification

## What has been demonstrated

- faithful `#1394` reproduction can be completed under cap on RunPod `8xH100 SXM`
- the packaging fix is sufficient to turn the real `1337` smoke result into a valid under-cap artifact
- all three foreground seeds stayed in-family:
  - `1337`: `1.08471849`
  - `42`: `1.08576707`
  - `2025`: `1.08513825`
- `07c1` remains a credible background evidence line, but is clearly behind the SP8192 frontier family

## What remains unresolved

- whether `#1413` reproduces cleanly on the same RunPod lane
- whether `#1437` should be climbed only after `#1413` or whether one later direct reproduction is worth it
- whether repaired `07c1` TTT is positive enough to matter strategically at all
- which originality lane to open once the `#1413` question is answered
- what to do only after the first faithful `#1413` seed-0 result is in hand

## Best next move

**Strategy pivot (2026-04-07):** Instead of launching only one faithful `#1413` seed, the A/B/C/D/E offline experiment suite is now prepared for one-shot batch execution.

Prepared artifacts (ready locally, pushed to git):

- `scripts/prepare_pr1413_variants.py` — materializes both local record folders from real local refs
- `scripts/runpod_1413_batch.sh` — sequential batch runner for A/B/C/D/E
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase/` — faithful #1413 base mirror (16,719 code bytes)
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/` — stack variant with parallel-residual and n-gram hooks (17,390 code bytes)

On the next paid RunPod `8xH100 SXM` pod:

1. `git pull --ff-only` to sync the prepared folders
2. `bash scripts/runpod_1413_batch.sh 0` to run A/B/C/D/E in sequence
3. Run E requires the D checkpoint — the batch runner enforces this automatically
4. Terminate/delete the old CPU-only recovery pod when convenient
5. Keep Pegasus `07c1` TTT jobs in the background only

Remaining offline blockers:

- proof of actual execution on a real pod is still needed (no pod run yet)
- artifact-cap margin for B/D/E (code + model) is unmeasured until a real stack run completes
