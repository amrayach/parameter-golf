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

- Active goal: run Session 05g (XSA-8 throughput recovery) and compare against 05c-plus
- Best measured branch: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py` (05c-plus)
- Active candidate: `records/track_non_record_16mb/2026-03-31_05g_xsa8_throughput/train_gpt.py`
- 05c-plus result: quality-positive (sliding s64 `1.12558`, delta `-0.00347`), throughput regressed (`100.39 ms`, +9ms)
- 05f result: negative vs 05c-plus (sliding s64 `1.12661`, delta `+0.00103`)
- 05g change: `xsa_last_n` 11 → 8 on 05c-plus base
- GPTQ status: permanently parked
- Out of scope: FA3, TTT, SWA, GPTQ, Parallel Muon, Parameter Banking, Late QAT, LZMA
