# Session 06: RFN Sidecar Probe

Preferred mode: Execution

Use `/research-engineer` if it exists locally. If not, proceed normally. Keep the probe small and falsifiable.

Goal:
- Test whether an RFN-style relevance signal provides useful guidance for compression or pruning sensitivity on a tiny transformer-compatible experiment.

Read these first:
- @docs/Abschlussarbeit_379315.pdf
- @/home/amay/Work/BachExpGraph/README.md
- @/home/amay/Work/BachExpGraph/src/main.py
- @/home/amay/Work/BachExpGraph/src/explain.py
- @/home/amay/Work/BachExpGraph/src/mlp.py
- @train_gpt.py
- @docs/campaign/artifacts/01_lineage_and_environment_audit.md

Constraints:
- This is a sidecar research probe, not a leaderboard push.
- Do not attempt full transformer RFN productionization in this session.
- Keep the scope to a tiny model, a small representative batch, and a simple comparison against magnitude-based heuristics.

Required workflow:
1. Decide the smallest viable transformer-compatible target inside this repo or a minimal side experiment.
2. Implement only enough instrumentation to rank modules, heads, channels, or blocks by relevance-like signal.
3. Compare that ranking against at least one simple baseline such as weight magnitude or activation magnitude.
4. Test whether the RFN-inspired ranking better predicts degradation under one small ablation or quantization perturbation.
5. Write a memo that says continue or stop.

Deliverables:
- minimal probe code in a clearly named non-record or research folder
- `docs/campaign/artifacts/06_rfn_sidecar_probe.md`

Definition of done:
- There is one explicit prediction task for the RFN signal.
- There is one baseline comparator.
- The memo gives a hard continue or stop judgment.

Commit message:
- `feat(campaign): add rfn sidecar probe`
