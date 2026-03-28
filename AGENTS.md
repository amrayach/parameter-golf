# Shared Agent Entry Point

Start here for both Claude Code and Codex.

## Read First

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`

## Purpose

`docs/campaign/AGENT_SYNC.md` is the mutable source of truth for:

- current objective
- current scope
- latest measured results
- next commands to run

`CLAUDE.md` contains the standing coordination rules for sessions, updates, and disagreement handling.

## Current Working Mode

- Active goal: A100 development evidence for a larger compute request
- Next runs: `a100_baseline_600s`, then `a100_lowerlr_600s`
- Out of scope: H100 parity claim, Session 03 anchor port, arbitrary strategy drift
