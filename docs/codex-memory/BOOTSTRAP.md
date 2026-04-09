# Agent Bootstrap

If this is a fresh session for Claude Code, Codex, or Antigravity, read these files in order:

1. `AGENTS.md`
2. `docs/campaign/AGENT_SYNC.md`
3. `CLAUDE.md`
4. `docs/codex-memory/project-state.md`
5. `docs/codex-memory/decisions.md`
6. `docs/codex-memory/next-session.md`

Then continue from the next pending foreground action.

## Current status

- The faithful RunPod `#1394` SP8192 baseline is complete and is now the stable base.
- Best measured code:
  - `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py`
- Best measured 3-seed result:
  - `1337`: sliding `1.08471849`, roundtrip `1.10134414`, bytes `15,986,188`
  - `42`: sliding `1.08576707`, roundtrip `1.10263175`, bytes `15,987,537`
  - `2025`: sliding `1.08513825`, roundtrip `1.10167670`, bytes `15,986,526`
  - mean sliding `1.08520794`
- The preserved `#1394` archive is already fetched locally:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256 `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- The extracted archive is present locally:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07`
- The strict local proof bundle exists:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- The exact `2025` binaries are preserved locally; exact `1337` and `42` binaries were already overwritten before archiving.
- The old CPU-only recovery pod no longer holds unique `#1394` evidence and may be terminated/deleted.
- The next foreground move is a fresh paid RunPod `8xH100 SXM` session for faithful `#1413`.
- Use:
  - `git pull --ff-only`
  - `bash scripts/runpod_1413.sh 0`
- `07c1` remains background evidence only. Latest checked Pegasus reruns:
  - `2740306`: `FAILED`
  - `2740307`: `FAILED`
- Do not spend foreground time on more `#1394` reruns, more `07c1` polish, SLOT, or the old `05c-plus` / GPTQ / mixed-int5 branch family.

## One-line resume prompt

```text
Read AGENTS.md, then docs/campaign/AGENT_SYNC.md, then CLAUDE.md. The recovered #1394 archive and strict proof bundle already exist locally, the old CPU recovery pod can be deleted, and the next paid H100 session should start faithful #1413 via bash scripts/runpod_1413.sh 0.
```
