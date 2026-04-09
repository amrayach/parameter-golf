# Session Handoff

Date: 2026-04-07

## Current Truths

- The completed strict RunPod `#1394` SP8192 baseline is the new stable base.
- Best measured script:
  - `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py`
- Best measured 3-seed result:
  - `1337`: `1.08471849` sliding, `1.10134414` roundtrip, `15,986,188` bytes
  - `42`: `1.08576707` sliding, `1.10263175` roundtrip, `15,987,537` bytes
  - `2025`: `1.08513825` sliding, `1.10167670` roundtrip, `15,986,526` bytes
  - mean sliding: `1.08520794`
- The preserved `#1394` tarball was recovered locally and verified:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256 `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- The extracted archive contents are present locally:
  - all three seed logs
  - `RUN_SUMMARY.txt`
  - `MANIFEST.tsv`
  - `SHA256SUMS.txt`
  - copied `train_gpt.py`
  - exact `2025` binaries
- The strict proof folder already exists:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- The old CPU-only recovery pod no longer contains unique evidence and may be terminated/deleted.
- `docs/campaign/AGENT_SYNC.md` remains the live source of truth if any summary disagrees.

## What Matters Now

- The next foreground branch is faithful `#1413`.
- Use a fresh paid RunPod `8xH100 SXM` pod, not the old CPU-only recovery pod.
- Sync repo state, then launch with:
  - `git pull --ff-only`
  - `bash scripts/runpod_1413.sh 0`
- `scripts/runpod_1413.sh` now handles:
  - fetching `pull/1413/head` into local ref `pr1413`
  - checking out only `records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828`
  - running `scripts/runpod_prepare_sp8192.sh`
  - forcing `DATA_DIR=${REPO_ROOT}/data`
  - applying the faithful `#1413` env contract:
    - `NCCL_NET=Socket`
    - `QK_GAIN_INIT=5.0`
    - `TTT_ENABLED=1`
    - `TTT_LR=0.005`
    - `TTT_EPOCHS=3`
  - archiving source files, metadata, `console.txt`, `logs/${RUN_ID}.txt`, `final_model.pt`, and `final_model.int6.ptz` into `/workspace/pr1413_archive_<stamp>/seed<seed>`
- Only after seed `0` is archived cleanly should later seeds be considered.

## Background Work

- `07c1` is now background evidence only.
- Latest checked Pegasus reruns:
  - `2740306` `07c1_ttt_s1337_fix`: `FAILED`
  - `2740307` `07c1_ttt_s2025_fix`: `FAILED`
- Do not let unresolved `07c1` TTT block the SP8192 line.

## Historical Note

The old `05c-plus` / `05f` / `05g` compression-era branch family is no longer live context. Keep it only for lineage and archival rationale. Do not treat those planning notes as current execution guidance.
