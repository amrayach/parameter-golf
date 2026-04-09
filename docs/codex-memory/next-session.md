# Next Session

## Phase

**The strict RunPod `#1394` SP8192 baseline is complete for three seeds, the `pr1413` RunPod archive is fetched locally, and the canonical local `D` bundle is now the clean foreground base.**

**Strategy pivot (2026-04-08):** The foreground plan is no longer "run the old A/B/C/D/E batch as prepared." It is now: use the measured `D` stack as the launch point and run a corrected, token-only, reviewer-defensible `E` / `#1437`-style eval-time n-gram layer on top of it.

The old `07c1` foreground story is closed. `07c1` is now background evidence only.

## Immediate next action

0. Refresh the local PR frontier cache before doing manual leaderboard review:
   - `python3 scripts/sync_pr_frontier.py sync`
   - cache: `artifacts/gh_frontier/openai_parameter-golf_pr_cache.json`
   - ranked CSV: `artifacts/gh_frontier/openai_parameter-golf_pr_ranked__combined__open.csv`
1. Launch a fresh paid RunPod `8xH100 SXM` session.
2. Sync repo state:
   - `git pull --ff-only`
3. Regenerate or patch the eval-time n-gram path so it matches the corrected public `#1437` token-only causal mode.
4. Start from the local `D` stack, not from a faithful `#1413` control rerun.
5. Reuse the preserved `final_model.pt` checkpoints from the `D` archive before introducing any retraining-side changes.
6. Prepare export-only GPTQ refinements (CDQuant / OWC).
7. Prepare eval-only TTT refinements (optimizer / freeze policy).
8. Run seed-0 proofs first:
   - corrected `E`
   - export-only GPTQ refinement
   - eval-only TTT refinement
9. Only expand to the canonical seed pack `0,42,1234,1337,2025` if one of those beats plain `D`.
10. Treat the old `pr1413_ngram_eval_s0` result only as an idea signal:
   - `1.08078425`
   - do not treat it as clean evidence because it predates the public causal-correction discussion
11. The old CPU-only recovery pod may be terminated/deleted; it is no longer needed for `#1394` preservation.
12. Keep `07c1` in the background only:
   - `2740306` and `2740307` last checked as `FAILED`

## Prepared local folders

| Path | Code bytes | Purpose |
|------|-----------|---------|
| `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase/` | 16,719 | Faithful #1413 base mirror (runs A, C) |
| `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/` | 17,390 | Stack variant with parallel-residual and n-gram hooks (runs B, D, E) |

Builder: `python3 scripts/prepare_pr1413_variants.py --force` (re-materializes both folders from local git refs `pr1413` and `pr1437` with FORMAT_RAW roundtrip validation).

## Current best measured result

Completed local `D` RunPod 5-seed canonical bundle:

- seeds `0,42,1234,1337,2025`
- mean score-first TTT BPB: `1.08128837`
- sample stddev: `0.00058943`
- max artifact: `15,992,638` bytes

Additional sixth seed:

- `7`: `1.08167555`
- all-6 mean: `1.08135290`

## Locked findings

- faithful `#1394` reproduction on RunPod `8xH100 SXM` is complete enough to serve as the new stable base
- the packaging fix is validated on the real checkpoint:
  - counted code bytes: `17,821`
  - all three seeds are under cap
- the `#1394` archive is now fetched locally:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - SHA256 `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- the local strict proof folder exists:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- exact `2025` checkpoint is preserved; exact `1337` and `42` binaries are not
- archive recovery succeeded through the gateway PTY path; direct TCP forwarding was still refusing during recovery
- `07c1` remains useful evidence, but it is not the foreground branch anymore
- clean open frontier order now reads:
  - local canonical `D`
  - corrected `#1437`
  - `#1420` only as a causal-bug / ablation reference
  - `#1416` / `#1423` are no longer clean anchors

## Files to read first

1. `AGENTS.md`
2. `docs/campaign/AGENT_SYNC.md`
3. `CLAUDE.md`
4. `docs/codex-memory/project-state.md`
5. `docs/codex-memory/decisions.md`
6. `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py`
7. `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict/README.md`
8. `scripts/runpod_prepare_sp8192.sh`
9. `docs/campaign/results_log.jsonl`
