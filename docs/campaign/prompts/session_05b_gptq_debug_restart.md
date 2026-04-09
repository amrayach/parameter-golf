# Session 05b GPTQ Debug Restart Prompt

> Historical note (2026-03-30): Session 05b GPTQ on the current Session 03 anchor is parked after 7 conclusive ablations. Keep this prompt only as historical debugging context, not as the current mainline execution plan.

Paste this into a fresh Codex session.

---

You are resuming work in `/home/amay/Work/parameter-golf`.

## Read first

1. `AGENTS.md`
2. `docs/campaign/AGENT_SYNC.md`
3. `CLAUDE.md`
4. `docs/codex-memory/decisions.md`
5. `docs/codex-memory/project-state.md`
6. `docs/codex-memory/next-session.md`
7. `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`
8. `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`

## Goal

Fix the Session 05b Full Hessian GPTQ correctness bug.

Do **not** treat this as an open-ended research/planning session. Treat it as a code-debugging and code-porting session.

## Current ground truth

Session 03 anchor on Pegasus `8xH100`:
- sliding s64 `val_bpb=1.12904446`
- pre-quant EMA `val_bpb=1.14472403`
- roundtrip `val_bpb=1.15247273`
- step_avg `91.37 ms`
- artifact `15751324` bytes

Saved-container FA3 on Pegasus `8xH100` is a clean negative result:
- slower and worse than the anchor
- parked for now

Session 05b `1xH100` GPTQ smoke:
- stopped at `906` steps in `600202 ms`
- step_avg `662.47 ms`
- pre-quant EMA exact `1.47753094`
- roundtrip exact `1.68963326`
- `67` Hessians collected
- `66` GPTQ layers used
- `0` Cholesky fallbacks
- GPTQ quantization time `4236 ms`
- artifact total `7754877` bytes
- job hit time limit before sliding eval finished

Interpretation:
- the pipeline mechanics work
- the quantizer is still wrong
- the `1xH100` training-side numbers are not anchor-comparable because `WORLD_SIZE` changed

## Non-goals

- Do not chase FA3.
- Do not work on TTT.
- Do not do broad architecture redesign.
- Do not spend time on papers/websites before exhausting repo-local PR code.
- Do not run a full `8xH100` training job until the export path passes a corrected smoke gate.

## Source priority

For this task, use sources in this order:

1. `openai/parameter-golf` PR code
2. local repo code
3. papers only for ambiguous math

The most relevant PRs are:
- `#1060`
- `#1019`
- `#634`

## Likely divergences already known

These are real, but do not assume any single one is the sole root cause until you verify:

- local GPTQ uses only fixed `row_max / 31` scaling
- working PRs run the full GPTQ loop across percentiles `[0.9990, 0.9995, 0.9999, 0.99999, 1.0]` and keep best reconstruction MSE
- local export clamps to `[-32, 31]`
- working PRs clamp symmetrically to `[-31, 31]`
- local Hessian hook set includes an extra `bigram.proj` due to broad `.proj.` classification

## Required execution plan

1. Inspect the current implementation in:
   - `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`

2. Compare the local GPTQ helpers against working PR code, especially:
   - Hessian collection
   - quantizer inner loop
   - scale selection
   - clamp range
   - target tensor naming / lookup

3. Add an export-only A/B on the same checkpoint path:
   - naive int6 reconstruction MSE per layer
   - GPTQ reconstruction MSE per layer
   - print the worst offending layers

4. Add block-level diagnostics inside the GPTQ loop:
   - `max_err`
   - `max_residual`
   - enough logging to detect a cascade or obvious numerical blow-up

5. Run one ablation if needed:
   - `actorder=False`
   - `block_size=d_col`

6. If the bug is not obvious quickly, stop refining the local custom rewrite and transplant the GPTQ quantizer directly from PR code.

7. After correctness is restored:
   - add the 5-percentile search
   - switch to symmetric `[-31, 31]`
   - tighten the hook target set to only actually GPTQ-exported tensors

8. Re-run a `1xH100` smoke with enough post-train wallclock reserve.

9. Only after the smoke roundtrip gap is sane, prepare for a full `8xH100` run.

## Hard gates

- Do not run `8xH100` until the corrected `1xH100` smoke has a sane roundtrip gap.
- Do not leave the session without updating:
  - `docs/campaign/AGENT_SYNC.md`
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`
  - `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`

## Notes

- Official leaderboard entry is record-gated. Beating current `#5` quality is not enough.
- Keep the focus on restoring a trustworthy GPTQ export path first.

---

Start by reading the files above and comparing the local GPTQ quantizer against PR `#1060` and PR `#1019`. Then implement the smallest debugging instrumentation needed to identify whether the current local loop is salvageable or should be replaced wholesale.
