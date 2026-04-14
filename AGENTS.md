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

- Active goal: reproduce PR `#1610` directly and layer a full-vocab posterior corrector to push below 1.070 BPB
- Execution plan: `docs/campaign/PLAN_PR1610_CORRECTOR.md` (locked Revision 3)
- Source base: `#1610` `train_gpt.py` at SHA `ca191953` (NOT patched D variant)
- Non-record PR `#1598` remains open and frozen; do not edit unless reviewers request changes
- Best measured result: canonical D 5-seed mean TTT BPB `1.08129` (sigma = 0.00059)
- Target: <= 1.070 BPB via #1610 reproduction + posterior corrector
- Budget: $212 RunPod (~35 runs), deadline Apr 30
- Fallback cascade defined in plan if corrector < 0.001 BPB gain
- Out of scope: more Pegasus resubmissions, paid OWC salvage on D stack, SLOT, pre-quant validation TTT, casefold tokenizers, D-variant patching
