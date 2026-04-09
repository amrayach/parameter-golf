# Hardware And Constraints

## Challenge constraints

- train in under `10 minutes` on `8xH100 SXM`
- evaluate in under `10 minutes`
- artifact under `16,000,000` bytes total
- final metric is `val_bpb`
- official record claims are gated by the merged official baseline in **nats**, with enough logs for `p < 0.01`

## Current measured anchors

### Foreground SP8192 base

- faithful RunPod `#1394` baseline on `8xH100 SXM`
  - `1337`: sliding `1.08471849`, roundtrip `1.10134414`, bytes `15,986,188`
  - `42`: sliding `1.08576707`, roundtrip `1.10263175`, bytes `15,987,537`
  - `2025`: sliding `1.08513825`, roundtrip `1.10167670`, bytes `15,986,526`
  - mean sliding `1.08520794`

### Background `07c1` evidence

- strict RunPod `07c1` best seed:
  - `1.10894203` sliding BPB
  - `1.11863096` roundtrip BPB
  - `1.87233216` nats
  - `15,728,840` bytes

## Pegasus state

- Pegasus remains the canonical validator and the background TTT lane.
- There is not currently a clean immediate single-node `8xH100` block free for this foreground step.
- `torchrun --standalone` is not the right launcher on Pegasus multi-GPU; use Slurm-native `srun`.
- Latest checked `07c1` Pegasus TTT reruns:
  - `2740306`: `FAILED`
  - `2740307`: `FAILED`

## RunPod state

- RunPod is the foreground execution lane for SP8192 reproductions when credits are available.
- The `#1394` recovery archive is already fetched locally and verified.
- The old CPU-only recovery pod no longer holds unique artifacts and may be terminated/deleted.
- The next paid RunPod session should be a fresh `8xH100 SXM` pod for faithful `#1413`.
- The live pod-side launcher is:
  - `scripts/runpod_1413.sh`

## Persistence / recovery rules

- For the migration/recovery pod shape, `/workspace` was the persistence boundary; `/root` was not reliable across `Stop`.
- Direct TCP SSH forwarding can still refuse connections even when `sshd` is listening inside the pod.
- The successful `#1394` recovery path was a PTY-driven gateway shell with local base64 capture and checksum verification.
- Archive recovery is now complete; do not spend more paid time on it.

## Artifact discipline

- Packaging fix is validated on the real RunPod checkpoint:
  - counted code bytes `17,821`
- All three completed `#1394` foreground seeds are under the 16 MB cap.
- Exact `2025` binaries are preserved locally.
- Exact `1337` and `42` binaries are not preserved because they were already overwritten before the archive was made.

## Practical rules

- Use a fresh H100 pod for the next `#1413` step rather than trying to reuse or migrate the old CPU-only recovery pod.
- Start the next paid session with:
  - `git pull --ff-only`
  - `bash scripts/runpod_1413.sh 0`
- Keep `07c1` as background evidence only.
- Treat `05c-plus`, GPTQ, and the old width/compression branch family as archival, not active.
