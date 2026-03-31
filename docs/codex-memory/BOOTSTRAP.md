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

- **Session 05c-plus is the best measured branch, not the next run target**
- Best measured code: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`
- Session 05f and Session 05g are measured negatives; do not continue local bigram / XSA micro-deltas
- GPTQ probe on the 05c-plus architecture is complete and negative; GPTQ is permanently parked for this model family
- Diagnostics workflow exists:
  - `diagnostics/2026-03-31_05c_plus/`
  - `scripts/diagnostics/diagnose_weights.py`
  - `scripts/diagnostics/compress_probe.py`
- Next: follow the live gate in `docs/campaign/AGENT_SYNC.md`, rerun the corrected compression probe, then decide whether the next larger fork is compression+width or something less width-dependent
- Session 03 anchor: sliding s64 `1.12904446`, roundtrip `1.15247273`, `91.37 ms/step`, `15751324` bytes
- 05c-plus: sliding s64 `1.12557920`, roundtrip `1.14933197`, `100.39 ms/step`, `15589271` bytes
- GPTQ, FA3, TTT, SWA are all parked

## One-line resume prompt

```text
Read AGENTS.md, then docs/campaign/AGENT_SYNC.md, then CLAUDE.md. Session 05c-plus is the best measured branch, 05f and 05g are negative, and GPTQ is permanently parked after 05e. Use diagnostics/2026-03-31_05c_plus/ plus scripts/diagnostics/diagnose_weights.py and scripts/diagnostics/compress_probe.py, then follow AGENT_SYNC to decide the next larger fork.
```
