# Agent Bootstrap

If this is a fresh session (any tool: Claude Code, Codex, or Antigravity), read these files in order:

1. `AGENTS.md` — shared entry point, current working mode
2. `docs/campaign/AGENT_SYNC.md` — mutable source of truth (current objective, results, next steps)
3. `CLAUDE.md` — standing rules and operational constraints
4. `docs/codex-memory/project-state.md` — full project state
5. `docs/codex-memory/decisions.md` — locked decisions
6. `docs/codex-memory/next-session.md` — next actions

Then proceed with the next pending action.

## Current status

- **Session 05c-plus is the active objective** — training-quality bundle on Session 03 anchor
- Code implemented and pushed: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- Smoke test ready: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/smoke_test.sh`
- Next: run Pegasus 1xGPU smoke, then 8xH100 full run
- Session 03 anchor: sliding s64 `1.12904446`, roundtrip `1.15247273`, `91.37 ms/step`, `15751324` bytes
- Target: sliding s64 val_bpb < `1.126`
- GPTQ, FA3, TTT, SWA are all parked

## One-line resume prompt

```text
Read AGENTS.md, then docs/campaign/AGENT_SYNC.md, then CLAUDE.md. Session 05c-plus bundle is implemented and pushed. Run 1xGPU Pegasus smoke test, then 8xH100 full run. Code: records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
```
