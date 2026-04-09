# Shared Memory Pack

This directory is the shared, file-based memory layer for all agent tools (Claude Code, Codex, Antigravity).

Design goals:
- separate from each tool's built-in memory system
- durable across fresh sessions for any tool
- easy to inspect and edit as plain Markdown
- syncable into Codex's private memory area at `~/.codex/memories/parameter-golf/`

## Source of truth

Shared project memory lives here in the repo:
- `docs/codex-memory/`

Private Codex mirror lives here:
- `~/.codex/memories/parameter-golf/`

The repo copy is the collaborative source of truth.
The private Codex mirror is for fast reloading in future Codex sessions and to keep a Codex-owned note store separate from Claude's built-in state.

## Files

- `BOOTSTRAP.md` — what a fresh Codex session should read first
- `project-state.md` — current campaign state
- `decisions.md` — locked decisions and constraints
- `next-session.md` — exact next action after context reset
- `session-handoff.md` — current short-form handoff summary
- `leaderboard-techniques.md` — distilled competitive findings
- `hardware-and-constraints.md` — Pegasus, RunPod, and scheduling realities
- `rfn-and-attribution-assessment.md` — RFN judgment and attribution-graph update
- `sync-codex-memory.sh` — copy these notes into `~/.codex/memories/parameter-golf/`

Historical planning docs and older prompts may remain in the repo for lineage, but the live context should come from the files above plus `docs/campaign/AGENT_SYNC.md`.

## Basic Memory MCP status

This Codex environment now has a global `basic-memory` MCP server configured for:

```text
parameter-golf-codex
```

The configured command was originally:

```bash
codex mcp add basic-memory bash -c "uvx basic-memory mcp --project parameter-golf-codex"
```

Current fix:
- Codex now launches Basic Memory through a wrapper at
  `/home/amay/.codex/memories/parameter-golf/basic-memory-mcp.sh`
- the wrapper bypasses `uvx` and executes the cached `basic-memory` binary directly
- runtime state lives under
  `/home/amay/.codex/memories/parameter-golf/.basic-memory-runtime/`
- this avoids sandbox failures from `uvx` writes under `~/.cache/uv` and `~/.local/share/uv`
- the configured project is `parameter-golf-codex` and points at
  `~/.codex/memories/parameter-golf/`

Notes:
- it modifies `~/.codex/config.toml`
- if the cached `basic-memory` package is removed, the wrapper will fail until it is reinstalled
- the Markdown memory pack still works even if the MCP server is temporarily unavailable

## Fresh-session instruction

In a new Codex session, start with:

```text
Read @docs/codex-memory/BOOTSTRAP.md and continue from there.
```
