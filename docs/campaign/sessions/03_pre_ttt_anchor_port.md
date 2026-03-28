# Session 03: Pre-TTT Anchor Port

Preferred mode: Execution

Use `/research-engineer` if it exists locally and helps with narrowing the anchor, but execute the port directly.

Goal:
- Build your own reproducible pre-TTT anchor stack based on the strongest public non-TTT lineage and measure how close it gets on Pegasus.

Read these first:
- @README.md
- @docs/campaign/artifacts/01_lineage_and_environment_audit.md
- @docs/campaign/artifacts/02_pegasus_baseline_ladder_summary.md
- @records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md
- @records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md
- @records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md
- @records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train_gpt.py
- @records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py
- @records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py

Constraints:
- Do not pull in TTT in this session.
- Favor a narrow, explainable anchor over a kitchen-sink merge.
- Keep all work self-contained under `records/track_non_record_16mb/<today>_pre_ttt_anchor/`, where `<today>` is the current date in `YYYY-MM-DD` format.

Required workflow:
1. Diff the three target records and extract the stable core that most likely carries the score.
2. Build one anchor script in your own folder rather than editing public record folders.
3. Keep the README explicit about what was inherited, what was changed, and what was intentionally left out.
4. Run at least one measured Pegasus verification and record artifact size, step time, pre-quant, post-quant, and sliding `val_bpb`.
5. If the run is materially weak, identify one concrete bottleneck rather than broadening the experiment.

Deliverables:
- `records/track_non_record_16mb/<today>_pre_ttt_anchor/README.md`
- `records/track_non_record_16mb/<today>_pre_ttt_anchor/train_gpt.py`
- run logs and manifests in that folder
- `docs/campaign/artifacts/03_pre_ttt_anchor_summary.md`

Definition of done:
- The anchor script runs from its own folder.
- The README explains the inherited stack cleanly.
- The summary states whether the anchor is strong enough for targeted deltas.

Commit message:
- `feat(campaign): add pre-ttt anchor stack`
