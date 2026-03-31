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

- Active goal: smoke and run Session 05f (BigramHash 3072x112 + warmdown 4000) on `8xH100`
- Code: `records/track_non_record_16mb/2026-03-31_05f_refine_bigram3072_warmdown4000/train_gpt.py`
- Base: 05c-plus (XSA-all + VE128 + warmdown 3500 + LeakyReLU(0.5)²)
- 05c-plus result: quality-positive (sliding s64 `1.12558`, delta `-0.00347`), throughput regressed (`100.39 ms`, +9ms)
- GPTQ status: permanently parked
- Next phase: 1xGPU smoke for 05f, then 8xH100 run if smoke passes
- Out of scope: FA3, TTT, SWA, GPTQ, Parallel Muon, Parameter Banking, Late QAT, LZMA
