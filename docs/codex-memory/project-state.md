# Project State

Date: 2026-04-08

## Objective

Primary:

- preserve the completed RunPod `#1394` SP8192 baseline as the new stable base
- use the fetched local `#1394` archive and strict proof folder as the durable evidence base
- move the foreground branch hunt to the measured `D` bundle and a corrected token-only `E` follow-up

Secondary:

- keep `07c1` evidence background-only
- do not let the failed Pegasus TTT reruns consume foreground attention

## Current campaign state

- `05c-plus` is superseded
- `07c` and `07c1` validated the move away from the old compression line, but they are no longer the foreground branch
- the faithful `#1394` SP8192 RunPod baseline is now complete on `8xH100 SXM`
- the `pr1413` RunPod batch archive is now fetched locally:
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205`
- the canonical `D` 5-seed bundle is the new clean foreground base:
  - seeds `0,42,1234,1337,2025`
  - mean TTT BPB `1.08128837`
  - sample stddev `0.00058943`
- seed `7` is preserved as an extra sixth seed:
  - TTT BPB `1.08167555`
- the old eval-only `E` seed-0 run is promising but not clean evidence:
  - `pr1413_ngram_eval_s0`
  - TTT BPB `1.08078425`
  - predates the public causal correction discussion around `#1420` / `#1437`
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

Completed local `D` RunPod bundle:

- `0`
  - sliding s64: `1.08261177`
  - score-first TTT: `1.08093485`
  - bytes total: `15,992,638`
- `42`
  - sliding s64: `1.08401205`
  - score-first TTT: `1.08113936`
  - bytes total: `15,990,501`
- `1234`
  - sliding s64: `1.08247918`
  - score-first TTT: `1.08091630`
  - bytes total: `15,990,023`
- `1337`
  - sliding s64: `1.08258969`
  - score-first TTT: `1.08112499`
  - bytes total: `15,989,185`
- `2025`
  - sliding s64: `1.08379404`
  - score-first TTT: `1.08232635`
  - bytes total: `15,989,883`

Aggregate:

- canonical 5-seed mean TTT BPB: `1.08128837`
- canonical 5-seed sample stddev: `0.00058943`
- all-6 mean TTT BPB: `1.08135290`
- max bytes total: `15,994,511`

Previous stable proof base remains the completed `#1394` RunPod 3-seed baseline:

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

Open frontier references as of 2026-04-08:

- merged official `#1019`: `1.11473509` BPB / `1.88217853` nats
- open clean `#1394`: `1.08563`
- open `#1412`: `1.08354`
- open `#1413`: `1.08279`
- open `#1420`: title now reports `1.08309`; the old `1.08014` number is no longer the clean anchor after the causal-fix discussion
- open `#1416`: author admitted the pre-quant validation TTT issue and said they are stripping TTT
- open `#1423`: comment thread points out direct validation fine-tuning before quantization
- open `#1437`: `1.08091`

Interpretation:

- local `D` is now ahead of `#1413`, `#1460`, and the currently titled `#1420` number
- `#1416` and `#1423` should not be treated as clean frontier anchors
- the clean public target is the corrected `#1437` number at `1.08091`
- the remaining gap from canonical `D` to `#1437` is only `0.00037837` BPB
- the next real move is therefore a corrected token-only `E` / `#1437`-style follow-up on top of `D`, not a faithful `#1413` restart

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
- local frontier review no longer has to be fully manual:
  - `scripts/sync_pr_frontier.py` now syncs `openai/parameter-golf` PR metadata through `gh`, stores a normalized JSON cache under `artifacts/gh_frontier/`, and writes a sortable CSV for score-first or campaign-relevance views

## What remains unresolved

- whether a corrected token-only `E`-style layer on top of `D` holds enough of the old seed-0 gain to beat `#1437`
- whether export-only GPTQ refinements (CDQuant / OWC) produce any measurable gain at all over plain `D`
- whether eval-only TTT refinements (optimizer / freeze policy) produce any measurable gain at all over plain `D`
- whether one training-split-only Fisher / `(g*w)^2` guided export policy produces anything after the checkpoint-reuse levers are exhausted
- whether the current local prep folders should be regenerated from the corrected public `#1437` code before the next pod session
- whether repaired `07c1` TTT is positive enough to matter strategically at all
- which originality lane to open only after the corrected `E` seed-0 and Fisher sidecar are both measured

## Best next move

**Strategy pivot (2026-04-08):** The old A/B/C/D/E batch is no longer the next launch plan as-written. The next launch plan is a corrected, token-only, reviewer-defensible `E`-style attack on top of the measured `D` base, followed by checkpoint-reuse sidecars before any retraining.

Prepared artifacts (ready locally, pushed to git):

- `scripts/prepare_pr1413_variants.py` — materializes both local record folders from real local refs
- `scripts/runpod_1413_batch.sh` — sequential batch runner for A/B/C/D/E
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase/` — faithful #1413 base mirror (16,719 code bytes)
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/` — stack variant with parallel-residual and n-gram hooks (17,390 code bytes)

On the next paid RunPod `8xH100 SXM` pod:

1. `git pull --ff-only` to sync the corrected branch
2. regenerate or patch the eval-time n-gram path so it matches the public causal correction (`#1437`-style token-only mode)
3. use the preserved `final_model.pt` checkpoints from the `D` archive for sidecars that do not require retraining
4. prepare export-only GPTQ refinements (CDQuant / OWC)
5. prepare eval-only TTT refinements (for example optimizer / freeze-policy changes)
6. start from the existing `D` stack, not a faithful `#1413` control rerun
7. run seed-0 proofs first:
   - corrected `E`
   - export-only GPTQ refinement
   - eval-only TTT refinement
8. only expand to the canonical seed pack `0,42,1234,1337,2025` if one of the seed-0 runs clearly beats plain `D`
9. keep Pegasus `07c1` TTT jobs in the background only

Remaining offline blockers:

- proof of actual execution on a real pod is still needed (no pod run yet)
- artifact-cap margin for B/D/E (code + model) is unmeasured until a real stack run completes
