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

- Active goal: finish compression-path feasibility on the saved 05c-plus / 05g artifacts, then decide whether to open one coherent larger fork
- Best measured branch: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py` (05c-plus)
- Current probe utility: `scripts/diagnostics/compress_probe.py`
- Current checkpoint utility: `scripts/diagnostics/diagnose_weights.py`
- 05c-plus result: quality-positive (sliding s64 `1.12558`, delta `-0.00347`), throughput regressed (`100.39 ms`, +9ms)
- 05f result: negative vs 05c-plus (sliding s64 `1.12661`, delta `+0.00103`)
- 05g result: negative vs 05c-plus despite modest throughput recovery; over cap on the old export path
- Local XSA / bigram micro-deltas are exhausted on this family
- GPTQ status: permanently parked
- Out of scope: FA3, TTT, SWA, GPTQ, local XSA/bigram micro-deltas, Parallel Muon, Parameter Banking, Late QAT, LZMA
