# Next Session

## Phase

**The strict RunPod `#1394` SP8192 baseline is complete for three seeds, the preserved archive has been fetched locally and checksum-verified, and the local strict proof folder now exists.**

**Strategy pivot (2026-04-07):** The foreground plan is no longer "run one faithful `#1413` seed next."  It is now: run the full A/B/C/D/E experiment suite from pre-built offline folders on the next paid RunPod pod.  The local variant folders are ready.

The old `07c1` foreground story is closed. `07c1` is now background evidence only.

## Immediate next action

1. Launch a fresh paid RunPod `8xH100 SXM` session.
2. Sync repo state:
   - `git pull --ff-only`
3. Run the A/B/C/D/E suite on seed `0`:
   - `bash scripts/runpod_1413_batch.sh 0`
   - Or individual runs: `bash scripts/runpod_1413_batch.sh 0 A B C`
4. The batch runner uses `FETCH_PAYLOAD=0` — no on-pod git fetch of upstream code needed.
5. Archive provenance for each run is written to `/workspace/pr1413_archive_<stamp>/seed<N>/`.
6. Run E requires the D checkpoint — the batch runner checks for it and exits clearly if missing.
7. The old CPU-only recovery pod may be terminated/deleted; it is no longer needed for `#1394` preservation.
8. Keep `07c1` in the background only:
   - `2740306` and `2740307` last checked as `FAILED`

## Prepared local folders

| Path | Code bytes | Purpose |
|------|-----------|---------|
| `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase/` | 16,719 | Faithful #1413 base mirror (runs A, C) |
| `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/` | 17,390 | Stack variant with parallel-residual and n-gram hooks (runs B, D, E) |

Builder: `python3 scripts/prepare_pr1413_variants.py --force` (re-materializes both folders from local git refs `pr1413` and `pr1437` with FORMAT_RAW roundtrip validation).

## Current best measured result

Completed `#1394` RunPod 3-seed baseline:

- `1337`: `1.08471849` sliding BPB, `1.10134414` roundtrip BPB, `15,986,188` bytes
- `42`: `1.08576707` sliding BPB, `1.10263175` roundtrip BPB, `15,987,537` bytes
- `2025`: `1.08513825` sliding BPB, `1.10167670` roundtrip BPB, `15,986,526` bytes

3-seed summary:

- mean sliding BPB: `1.08520794`
- mean roundtrip BPB: `1.10188420`
- max artifact: `15,987,537` bytes

## Locked findings

- faithful `#1394` reproduction on RunPod `8xH100 SXM` is complete enough to serve as the new stable base
- the packaging fix is validated on the real checkpoint:
  - counted code bytes: `17,821`
  - all three seeds are under cap
- the `#1394` archive is now fetched locally:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256 `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- the local strict proof folder exists:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- exact `2025` checkpoint is preserved; exact `1337` and `42` binaries are not
- archive recovery succeeded through the gateway PTY path; direct TCP forwarding was still refusing during recovery
- `07c1` remains useful evidence, but it is not the foreground branch anymore
- current open frontier order remains:
  - `#1394` clean base
  - `#1413` legal score-first TTT
  - `#1437` stacked frontier
  - `#1420` as component / legality ablation
  - `#1416` only later if needed

## Files to read first

1. `AGENTS.md`
2. `docs/campaign/AGENT_SYNC.md`
3. `CLAUDE.md`
4. `docs/codex-memory/project-state.md`
5. `docs/codex-memory/decisions.md`
6. `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py`
7. `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict/README.md`
8. `scripts/runpod_prepare_sp8192.sh`
9. `docs/campaign/results_log.jsonl`
