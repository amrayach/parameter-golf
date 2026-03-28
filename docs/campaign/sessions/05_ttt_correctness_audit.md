# Session 05: TTT Correctness Audit

Preferred mode: Planning

Use `/research-engineer` if it exists locally and helps with audit structure.

Goal:
- Decide whether TTT should be integrated next by auditing correctness, legality, engineering cost, and expected upside.

Read these first:
- @README.md
- @docs/campaign/artifacts/04_targeted_delta_sweep.md
- @records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md
- @records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md
- @records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py

Constraints:
- This is an audit session, not an implementation session.
- Be strict about challenge legality and evaluation leakage.
- Separate score gain from engineering overhead.

Required workflow:
1. Inspect the public TTT implementations and trace exactly how they avoid leakage.
2. Document the score-first protocol and the evaluation-time cost budget.
3. Identify what parts are portable to your anchor stack and what parts are coupled to the public implementation.
4. Produce a recommendation: integrate now, defer, or skip.

Deliverables:
- `docs/campaign/artifacts/05_ttt_correctness_audit.md`

Definition of done:
- The audit explicitly states whether the public TTT path appears challenge-compliant.
- The audit states the estimated engineering cost to integrate it into your stack.
- The audit gives a yes or no recommendation for the next session.

Commit message:
- `docs(campaign): add ttt correctness audit`
