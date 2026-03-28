# Session 01: Lineage And Environment Audit

Preferred mode: Planning

Use `/research-engineer` if it exists locally and helps with the analysis portion. If it is unavailable, proceed normally.

Goal:
- Produce one grounded document that maps the public winning lineage, the exact levers that mattered, the reproducibility risks, and the Pegasus execution protocol we will use.

Read these first:
- @README.md
- @train_gpt.py
- @docs/Pegasus_Server_documentation.txt
- @docs/dfki-nlp-pegasus-bridle-8a5edab282632443.txt
- @records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md
- @records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md
- @records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md
- @records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md
- @/home/amay/Work/BachExpGraph/README.md
- @/home/amay/Work/BachExpGraph/src/main.py
- @/home/amay/Work/BachExpGraph/src/explain.py

Constraints:
- Do not implement model changes in this session.
- Keep the output specific to this repo and this hardware.
- Be critical about the RFN idea. Do not sell it unless the evidence supports it.

Required workflow:
1. Inspect the target files and summarize the current stack progression from baseline to the strongest public non-TTT and TTT entries.
2. Identify the recurring high-yield levers, the lower-yield or risky ones, and any correctness caveats already visible in public writeups.
3. Confirm the exact Pegasus operating assumptions for single-node 8xH100-SXM work.
4. Compare the current RFN thesis implementation to what would be required for a transformer-compatible probe.
5. Write the audit to `docs/campaign/artifacts/01_lineage_and_environment_audit.md`.

Deliverables:
- `docs/campaign/artifacts/01_lineage_and_environment_audit.md`

Definition of done:
- The audit includes a ranked lever map.
- The audit includes a reproduction target for sessions 2 and 3.
- The audit includes a hard recommendation on RFN scope for this campaign.

Commit message:
- `docs(campaign): add lineage and environment audit`
