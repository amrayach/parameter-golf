# Session 07: Go Or No-Go Review

Preferred mode: Planning

Use `/research-engineer` if it exists locally and helps produce a disciplined final review.

Goal:
- Decide whether to continue the Parameter Golf campaign, pivot to a narrower non-record submission, or stop and redirect time.

Read these first:
- @docs/campaign/artifacts/01_lineage_and_environment_audit.md
- @docs/campaign/artifacts/02_pegasus_baseline_ladder_summary.md
- @docs/campaign/artifacts/03_pre_ttt_anchor_summary.md
- @docs/campaign/artifacts/04_targeted_delta_sweep.md
- @docs/campaign/artifacts/05_ttt_correctness_audit.md
- @docs/campaign/artifacts/06_rfn_sidecar_probe.md

Constraints:
- Be candid and quantitative.
- Prefer stopping over drifting if the evidence is weak.
- Tie the recommendation to actual measured outcomes, not vibes.

Required workflow:
1. Summarize the campaign state in one concise table.
2. Compare achieved results against the gates defined in `@docs/campaign/README.md`.
3. Choose one of these outcomes:
   - continue leaderboard push,
   - continue as non-record only,
   - stop this project and redirect effort.
4. Write the decision memo with the exact reasons and next action.

Deliverables:
- `docs/campaign/artifacts/07_go_no_go_review.md`

Definition of done:
- The memo names a single recommended path.
- The memo includes the strongest evidence for and against continuing.
- The memo makes the next action unambiguous.

Commit message:
- `docs(campaign): add go-no-go review template`
