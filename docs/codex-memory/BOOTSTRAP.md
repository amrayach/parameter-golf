# Codex Bootstrap

If this is a fresh Codex session, read these files in order:

1. `@docs/codex-memory/project-state.md`
2. `@docs/codex-memory/decisions.md`
3. `@docs/codex-memory/next-session.md`
4. `@docs/codex-memory/session-handoff.md`
5. `@docs/campaign/artifacts/02a_pegasus_verification.md`
6. `@docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md`
7. `@docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
8. `@docs/campaign/README.md`

Then proceed with the next pending action.

## Current status

- campaign scaffolding exists under `docs/campaign/`
- Session 01 is complete
- Session 02 still has a mandatory live Pegasus verification gate before any training
- Pegasus verification is only partial as of `2026-03-27`
- the read-only root-script audit is complete:
  - `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
- Session 03 implementation has not started
- no baseline training should start until the verification artifact exists
- Basic Memory MCP is configured for Codex as `parameter-golf-codex`, but the repo files remain the clearest source of truth
- local project-relevant skills are installed for Codex:
  - `research-engineer`
  - `gptq`
  - `model-pruning`
  - `transformer-lens-interpretability`

## One-line resume prompt

Use this in a fresh Codex chat:

```text
Read @docs/codex-memory/BOOTSTRAP.md, then resume the Parameter Golf campaign from the current next step without re-planning from scratch. Use /research-engineer as the primary mode. Treat @docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md as the source of truth for the Session 03 root-script port scope.
```
