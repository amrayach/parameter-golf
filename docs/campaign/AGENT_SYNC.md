# Agent Sync

Date: 2026-04-07

## Current Objective

The foreground baseline is now complete and the A/B/C/D/E offline prep suite is now the active focus:

1. preserve the completed strict RunPod `#1394` SP8192 baseline as the new stable base
2. use the fetched local `#1394` archive and strict proof folder as the durable proof base
3. **strategy pivot (2026-04-07):** rather than launching only a single faithful `#1413` seed on the next paid pod, materialize the full offline A/B/C/D/E experiment suite locally so it is ready for one-shot batch execution on RunPod — see `scripts/prepare_pr1413_variants.py` and `scripts/runpod_1413_batch.sh`
4. keep `07c1` evidence background-only; do not let the failed Pegasus reruns block SP8192
5. keep SLOT out of the mainline until the legality picture is explicit

A/B/C/D/E run contract:

| Run | Record folder | Key env overrides |
|-----|--------------|-------------------|
| A | `LocalBase` | none (faithful mirror) |
| B | `ParallelResid7_TiltPrep` | `PARALLEL_RESIDUAL_START=7` |
| C | `LocalBase` | `LOOP_START=3 LOOP_END=5` |
| D | `ParallelResid7_TiltPrep` | `PARALLEL_RESIDUAL_START=7 LOOP_START=3 LOOP_END=5` |
| E | `ParallelResid7_TiltPrep` | D env + `SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1` (requires D checkpoint) |

The mainline is no longer "polish `07c1` first." The mainline is now:

- completed clean SP8192 base: `#1394`
- next reviewer-defensible branch: `#1413`
- later stacked frontier: `#1437`
- `#1420` only as component / legality ablation
- `#1416` only after the lower-risk branches are understood

## Current Best Measured Result

Best current foreground result is the completed 3-seed RunPod `#1394` baseline on `8xH100 SXM`:

| Experiment | Script | Seed | sliding s64 BPB | roundtrip exact BPB | bytes_total | Status |
|-----------|--------|------|-----------------|---------------------|-------------|--------|
| `pr1394_sp8192_s1337` | `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py` | `1337` | **1.08471849** | 1.10134414 | 15,986,188 | valid under cap after packaging fix |
| `pr1394_sp8192_s42` | same | `42` | 1.08576707 | 1.10263175 | 15,987,537 | valid under cap |
| `pr1394_sp8192_s2025` | same | `2025` | 1.08513825 | 1.10167670 | 15,986,526 | valid under cap |

3-seed summary:

- mean sliding s64 BPB: `1.08520794`
- mean roundtrip exact BPB: `1.10188420`
- max artifact size: `15,987,537`
- code bytes counted after packaging fix: `17,821`

Previous best evidence line remains:

- `07c1_runpod_base_s2025_strict`
  - script: `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`
  - sliding s64 BPB: `1.10894203`
  - roundtrip exact: `1.11863096`
  - `val_loss = 1.87233216` nats
  - bytes total: `15,728,840`
  - train time: `598052ms`

## Competition Reality

Relevant open frontier references as of 2026-04-07:

- merged official `#1019`: `1.11473509` BPB / `1.88217853` nats
- open clean `#1394`: `1.08563` BPB (5-seed mean)
- open `#1412`: `1.08354` BPB
- open legal score-first TTT `#1413`: `1.08279` BPB
- open normalized causal tilt `#1420`: `1.08014` BPB
- open pre-quant TTT `#1416`: `1.07948` BPB
- open stacked frontier `#1437`: `1.07800` BPB
- open SLOT `#1333`: `1.0766` BPB, but latest visible legality discussion still argues leakage

Current gaps from our completed `#1394` 3-seed mean:

- vs open clean `#1394`: `-0.00042206` BPB
- vs `#1412`: `+0.00166794` BPB
- vs `#1413`: `+0.00241794` BPB
- vs `#1420`: `+0.00506794` BPB
- vs `#1416`: `+0.00572794` BPB
- vs `#1437`: `+0.00720794` BPB

Interpretation:

- the completed `#1394` baseline is competitive with the current clean SP8192 base tier
- it is no longer enough to reach the visible open frontier on score alone
- the next real move is not another `#1394` seed or more `07c1` polish; it is `#1413`

## `#1394` RunPod Baseline Status

What is now locked:

- faithful SP8192 reproduction completed on RunPod `8xH100 SXM`
- packaging fix validated on the actual RunPod checkpoint:
  - counted code bytes drop from the human-readable `58,367` to `17,821`
  - the best-seed `1337` artifact is under cap after the fix
- all three foreground seeds finished under cap
- the preserved archive has now been fetched locally and verified
- the local strict proof folder has been materialized
- the old CPU-only recovery pod no longer holds unique `#1394` evidence and may be terminated/deleted

What is preserved in the fetched local recovery copy:

- `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
- SHA256: `95e3a38cc89a160469d8417d0d4dbd40ef6a5106803d25ccdab5c0f86e2c0b07`
- extracted directory:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07`
- local strict proof folder:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`

What remains preserved on the current RunPod pod volume:

- `/workspace/pr1394_archive_2026-04-07`
- `/workspace/pr1394_archive_2026-04-07.tgz`

Archive contents:

- all three seed logs
- `RUN_SUMMARY.txt`
- `MANIFEST.tsv`
- `SHA256SUMS.txt`
- copied `train_gpt.py`
- exact `2025` checkpoints:
  - `final_model_s2025.int6.ptz`
  - `final_model_s2025.pt`

Important limitation:

- exact `1337` and `42` model binaries were overwritten by later runs
- only their logs and final metrics are preserved
- exact binary preservation currently exists only for `2025`

Archive recovery notes from the successful fetch:

- direct TCP `ssh` / `rsync` path was still refused during recovery
- gateway `scp` still failed because the subsystem was unavailable
- gateway non-PTY command execution still failed because the gateway required a PTY
- the successful path was a PTY-driven gateway shell with local base64 capture and checksum verification
- therefore the archive is no longer stranded on the pod

## `07c1` Background State

`07c1` remains useful evidence, but it is not the foreground record-play lane anymore.

Locked `07c1` findings:

- strict RunPod base proof now covers four seeds
- best strict seed: `1.10894203` BPB / `1.87233216` nats
- strict 4-seed significance versus merged `#1019` is still good:
  - delta: `-0.00796789` nats
  - Welch `t = -6.8667`
  - one-sided `p = 0.001785`
- falsified local levers:
  - `QK_GAIN=5.0`
  - `MLP_MULT=3.08`
- `MLP_MULT=3.5` is quality-positive but over the cap under the current export path
- TTT remains unresolved because the fixed reruns failed for operational reasons, not because the lever was cleanly measured

Pegasus TTT jobs stay background-only:

- `2740306` `07c1_ttt_s1337_fix`
- `2740307` `07c1_ttt_s2025_fix`

Latest Pegasus check:

- `2740306` state: `FAILED`
- `2740307` state: `FAILED`

Do not let `07c1` TTT block the SP8192 line.

## Current Interpretation

Locked campaign decisions:

- the `#1394` RunPod baseline is complete enough to serve as the new stable base
- the archive fetch and local strict proof/bundle step are now complete
- the next paid RunPod `8xH100` session should start directly on faithful `#1413`
- the current CPU pod no longer holds the only durable copy of the `#1394` evidence

Strategic implication:

- stop foreground time on more `#1394` seeds
- stop foreground time on more `07c1` packaging work
- use `#1394` as the stable launch point
- make `#1413` the next foreground target

## RunPod Status

- the archive recovery CPU pod was reachable through gateway user `xgb6r49j4xzjx7-64412169@ssh.runpod.io`
- direct TCP SSH forwarding remained broken during recovery even though `sshd` was listening inside the pod
- the gateway PTY path was sufficient to recover the tarball locally
- the current pod is no longer the only durable location of the `#1394` artifacts

Current RunPod policy:

- use the next paid RunPod session for one foreground thing only:
  1. start the faithful `#1413` reproduction on `8xH100 SXM`
- do not reuse the old CPU-only recovery pod for training; start fresh

## Immediate Next Commands

**Before launching the next paid pod**, confirm the local offline prep is complete:

```bash
# 1. verify the two prepared local record folders exist and pass roundtrip validation
python3 scripts/prepare_pr1413_variants.py --force

# 2. confirm the stack wrapper is in the correct upstream format
python3 -c "
with open('records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/train_gpt.py') as f:
    w = f.read()
assert 'format=L.FORMAT_RAW' in w, 'Stack wrapper missing FORMAT_RAW'
assert w.count('\\n') == 2, 'Expected 2-line wrapper'
print('OK')
"
```

Next paid RunPod `8xH100` session, in order:

```bash
# 1. sync the repo checkout so the launchers and prepared folders exist on the pod
git pull --ff-only

# 2. run the full A/B/C/D/E batch on seed 0 (or individual runs as needed)
bash scripts/runpod_1413_batch.sh 0        # runs A B C D E in sequence
# or selectively:
bash scripts/runpod_1413_batch.sh 0 A B C  # runs A then B then C

# 3. for Run E, the batch script guards for the D checkpoint automatically
#    (Run E requires final_model.int6.ptz from Run D in the stack folder)
```

Prepared folders (already committed locally, synced via `git pull`):

- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase/` — base mirror (16,719 code bytes)
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/` — stack variant (17,390 code bytes)

`scripts/runpod_1413_batch.sh` delegates each run to `scripts/runpod_1413.sh` with `FETCH_PAYLOAD=0`
(uses the pre-built local folders, no on-pod git fetch needed).  Per-run archive namespace:
`/workspace/pr1413_archive_<stamp>/seed<N>/`.

Pegasus background accounting snapshot:

```bash
ssh -F /dev/null ayach@login2.pegasus.kl.dfki.de \
  'sacct -j 2740306,2740307 --format=JobID,JobName%28,State,Elapsed,ExitCode'
```

## Out of Scope for the Next Session

- reopening archive-recovery work for `#1394`
- more `#1394` seed reruns
- more `07c1` sweeps
- SLOT work
- reviving `05c-plus`, GPTQ, or the old width/compression branch as the foreground line

## Canonical Files

- shared mutable state: `docs/campaign/AGENT_SYNC.md`
- append-only run log: `docs/campaign/results_log.jsonl`
- stable rules: `CLAUDE.md`
- current SP8192 base:
  - `records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py`
- background `07c1` line:
  - `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`
- local bundle builder:
  - `scripts/build_pr1394_runpod_bundle.py`
- RunPod data/deps prep helper:
  - `scripts/runpod_prepare_sp8192.sh`
- Codex memory:
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`

## Workspace

- local repo: `/home/amay/Work/parameter-golf`
- local fetched archive:
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07.tgz`
  - `artifacts/runpod_pull/pr1394_archive_2026-04-07/pr1394_archive_2026-04-07`
- local strict proof folder:
  - `records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict`
- Pegasus repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` by default.
