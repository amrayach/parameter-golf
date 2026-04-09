# Parameter Golf Campaign

This directory holds the campaign runbooks, prompts, artifacts, and historical session notes for this repo.

## Current status

For live execution state, always defer to:

- `AGENTS.md`
- `docs/campaign/AGENT_SYNC.md`
- `CLAUDE.md`

As of 2026-04-07:

- the faithful RunPod `#1394` SP8192 baseline is complete and is the new stable base
- the preserved `#1394` archive has already been fetched locally and verified
- the strict local proof folder already exists
- the next foreground step is faithful `#1413` on a fresh paid RunPod `8xH100 SXM` session
- the old CPU-only recovery pod no longer contains unique `#1394` evidence and may be terminated/deleted
- `07c1` is background evidence only
- the old `05c-plus` / GPTQ / width-compression branch family is historical, not active

## Active objective

Primary goal:

- use the recovered `#1394` baseline as the durable proof base and move the foreground lane to faithful `#1413`

Secondary goal:

- keep `07c1` and Pegasus-only TTT work strictly background until the SP8192 branch order is resolved

## Current branch order

- stable base: `#1394`
- next reviewer-facing branch: `#1413`
- later stacked frontier: `#1437`
- component / legality ablation only: `#1420`
- later higher-risk lane: `#1416`

## Current execution path

Use `docs/campaign/AGENT_SYNC.md` for the exact live steps.

The current RunPod foreground flow is:

1. launch a fresh paid `8xH100 SXM` pod
2. sync repo state:
   - `git pull --ff-only`
3. start faithful `#1413`:
   - `bash scripts/runpod_1413.sh 0`

The launcher handles:

- fetching `pull/1413/head`
- checking out the exact `#1413` record folder
- restoring SP8192 data/runtime deps
- forcing repo-root `DATA_DIR`
- archiving source files, metadata, logs, and final artifacts into `/workspace/pr1413_archive_*`

## Historical notes

The following materials remain useful for lineage and rationale, but they are **not** live source-of-truth docs:

- `sessions/`
- `prompts/`
- `docs/superpowers/plans/`
- older campaign artifacts describing `05c-plus`, `05f`, `05g`, GPTQ rescue attempts, or compression-path gating

If any historical file disagrees with `AGENT_SYNC.md`, trust `AGENT_SYNC.md`.

## Key references

- live campaign state: `docs/campaign/AGENT_SYNC.md`
- append-only measured results: `docs/campaign/results_log.jsonl`
- RunPod operational notes: `docs/campaign/RUNPOD_RUNBOOK.md`
- Pegasus operational notes: `docs/campaign/PEGASUS_H100_RUNBOOK.md`
- local strict proof bundle:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- recovered archive:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
