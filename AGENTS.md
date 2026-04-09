# Shared Agent Entry Point

Start here for Claude Code, Codex, and Antigravity.

## Read First

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`

For deep context (campaign strategy, prior experiments, hardware state):

3. `docs/codex-memory/BOOTSTRAP.md` — full bootstrap reading list

## Purpose

`docs/campaign/AGENT_SYNC.md` is the mutable source of truth for:

- current objective
- current scope
- latest measured results
- next commands to run

`CLAUDE.md` contains the standing coordination rules for sessions, updates, and disagreement handling.

## Tool-Specific Config

| Tool | Config | Skills | Workflows |
|------|--------|--------|-----------|
| Claude Code | `~/.claude/settings.json` + `.claude/settings.local.json` | `~/.claude/skills/` | `~/.claude/commands/` |
| Codex | `~/.codex/config.toml` | `~/.codex/skills/` | N/A |
| Antigravity | `~/.gemini/antigravity/mcp_config.json` | `.agents/skills/` (project-level) | `.agents/workflows/` |

## Current Working Mode

- Active goal: use the fetched local `#1394` strict archive/proof bundle as the stable base and move the foreground branch to faithful `#1413` on the next paid H100 session.
- Best measured branch: `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py`
- Best measured result: completed `#1394` RunPod 3-seed baseline
  - `1337`: sliding s64 `1.08471849`, roundtrip exact `1.10134414`, bytes `15,986,188`
  - `42`: sliding s64 `1.08576707`, roundtrip exact `1.10263175`, bytes `15,987,537`
  - `2025`: sliding s64 `1.08513825`, roundtrip exact `1.10167670`, bytes `15,986,526`
  - mean sliding s64 `1.08520794`
- Local fetched archive:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256 `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- Local strict proof folder:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- Pegasus background jobs remain non-foreground evidence and last checked as failed:
  - `2740306` `07c1_ttt_s1337_fix`
  - `2740307` `07c1_ttt_s2025_fix`
- Fixed blockers already landed for the SP8192 RunPod lane:
  - packaging fix validated (`17,821` counted code bytes)
  - `scripts/runpod_prepare_sp8192.sh` restores data symlinks and runtime deps
  - `scripts/build_pr1394_runpod_bundle.py` has already materialized the local strict proof folder
- RunPod status: the archive is no longer stranded on the pod; direct TCP SSH was still refusing during recovery, but the gateway PTY path succeeded. The old CPU-only recovery pod no longer holds unique evidence and may be terminated/deleted. The next paid RunPod step should use a fresh `8xH100 SXM` pod.
- Out of scope for the next foreground step: more `#1394` seeds, more `07c1` polish, SLOT work, and reviving `05c-plus` / GPTQ / mixed int5-int6 as the mainline.
