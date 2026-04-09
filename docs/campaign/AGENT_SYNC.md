# Agent Sync

Date: 2026-04-08

## Current Objective

The foreground baseline is now complete, the first A/B/C/D/E pod sweep is archived locally, and the corrected post-`D` attack is now the active focus:

1. preserve the completed strict RunPod `#1394` SP8192 baseline and the fetched `pr1413` archive as durable proof
2. treat the measured canonical `D` bundle as the new clean SP8192 foreground base
3. **strategy pivot (2026-04-08):** the next paid H100 attack is a corrected, token-only, reviewer-defensible `E`-style eval-time n-gram tilt on top of `D`
4. prioritize export-only and eval-only sidecars that can reuse preserved `D` checkpoints before any retraining sidecar
5. keep `07c1` evidence background-only; do not let failed Pegasus reruns block SP8192
6. keep SLOT, pre-quant validation fine-tuning, and full RFN / attribution-graph work out of the foreground lane

A/B/C/D/E run contract:

| Run | Record folder | Key env overrides |
|-----|--------------|-------------------|
| A | `LocalBase` | none (faithful mirror) |
| B | `ParallelResid7_TiltPrep` | `PARALLEL_RESIDUAL_START=7` |
| C | `LocalBase` | `LOOP_START=3 LOOP_END=5` |
| D | `ParallelResid7_TiltPrep` | `PARALLEL_RESIDUAL_START=7 LOOP_START=3 LOOP_END=5` |
| E | `ParallelResid7_TiltPrep` | D env + `SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1` (requires D checkpoint) |

The mainline is no longer "polish `07c1` first." The mainline is now:

- preserved clean SP8192 base: `#1394`
- measured clean stacked base: local canonical `D`
- next reviewer-defensible attack: corrected token-only `E` / `#1437`-style eval-time tilt on top of `D`
- highest-priority sidecars after corrected `E`: export-only CDQuant / OWC and eval-only TTT optimizer / layer-freeze variants on preserved `D` checkpoints
- training-side optimizer changes (for example Cautious Muon / AdaMuon-style variants) remain promising, but they are behind the cheaper checkpoint-reuse levers
- `#1420` only as a causal bug / ablation reference
- `#1416` and `#1423` are no longer clean anchor targets

## Current Best Measured Result

Best current foreground result is now the local RunPod `D` bundle from `pr1413_combo` on `8xH100 SXM`:

| Experiment | Script | Seed | sliding s64 BPB | score-first TTT BPB | bytes_total | Status |
|-----------|--------|------|-----------------|---------------------|-------------|--------|
| `pr1413_combo_s0` | `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/train_gpt.py` | `0` | 1.08261177 | **1.08093485** | 15,992,638 | valid under cap |
| `pr1413_combo_s42` | same | `42` | 1.08401205 | 1.08113936 | 15,990,501 | valid under cap |
| `pr1413_combo_s1234` | same | `1234` | 1.08247918 | **1.08091630** | 15,990,023 | valid under cap |
| `pr1413_combo_s1337` | same | `1337` | 1.08258969 | 1.08112499 | 15,989,185 | valid under cap |
| `pr1413_combo_s2025` | same | `2025` | 1.08379404 | 1.08232635 | 15,989,883 | valid under cap |

Canonical 5-seed summary (`0,42,1234,1337,2025`):

- mean score-first TTT BPB: `1.08128837`
- sample stddev: `0.00058943`
- max artifact size: `15,992,638`
- code bytes counted: `17,390`

Additional sixth seed:

- `7`: `1.08167555` TTT BPB, `15,994,511` bytes
- all-6 mean: `1.08135290`

Important note on the old eval-only `E` run:

- `pr1413_ngram_eval_s0` measured `1.08078425` on seed `0`, improving on `D` seed `0` by `0.00015060`
- that run predates the public causal correction discussion around `#1420` / `#1437`
- treat it as a promising idea signal, not as clean PR evidence

Previous best evidence line remains:

- `07c1_runpod_base_s2025_strict`
  - script: `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`
  - sliding s64 BPB: `1.10894203`
  - roundtrip exact: `1.11863096`
  - `val_loss = 1.87233216` nats
  - bytes total: `15,728,840`
  - train time: `598052ms`

## Competition Reality

Relevant open frontier references as of 2026-04-08:

- merged official `#1019`: `1.11473509` BPB / `1.88217853` nats
- open clean `#1394`: `1.08563` BPB (5-seed mean)
- open legal score-first TTT `#1413`: `1.08279` BPB
- open `#1420`: title now reports `1.08309` after the causal correction discussion; the old `1.08014` number is no longer a clean anchor
- open `#1416`: author admitted the pre-quant validation TTT violated the score-before-update reading and said they are stripping TTT
- open `#1423`: comment thread points out direct validation fine-tuning before quantization
- open `#1460`: `1.08269` BPB, but legality picture is still unresolved
- open stacked frontier `#1437`: `1.08091` BPB (causal-corrected 5-seed mean)

Current gaps from our canonical `D` 5-seed mean:

- vs open clean `#1394`: `-0.00434163` BPB
- vs `#1413`: `-0.00150163` BPB
- vs `#1460`: `-0.00140163` BPB
- vs `#1420` current title: `-0.00180163` BPB
- vs `#1437`: `+0.00037837` BPB

Interpretation:

- the local canonical `D` bundle is no longer "just an ablation"; it is one of the strongest reviewer-defensible open SP8192 results we have
- `#1416` and `#1423` should not be used as clean frontier anchors right now
- the old raw `#1420` `1.08014` number should not be used as the number to chase; the public thread itself now treats the causal correction as materially worse
- the clean public target is now `#1437` at `1.08091`
- the next real move is not more `#1394` seeds, not a faithful `#1413` rerun, and not a full RFN build; it is a corrected token-only `E`-style variant on top of `D`
- after corrected `E`, the most practical near-term sidecars are the ones that reuse preserved `final_model.pt` checkpoints:
  - export-only GPTQ refinements such as CDQuant / OWC
  - eval-only TTT refinements such as RMS+decay or middle-layer-only updates
- Fisher / Hessian sensitivity is now behind those checkpoint-reuse levers
- `torch.compile` is already used in the current training / eval path; do not treat "add compile" as a fresh mainline intervention

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

- the `#1394` RunPod baseline remains the durable proof base, but it is no longer the best measured foreground line
- the fetched `pr1413` archive is now the important local evidence bundle for SP8192 frontier work
- the local canonical `D` result is the new clean foreground base
- the next paid RunPod `8xH100` session should start from `D`, not from a faithful `#1413` restart
- the current CPU pod no longer holds the only durable copy of the evidence

Strategic implication:

- stop foreground time on more `#1394` seeds
- stop foreground time on more `07c1` packaging work
- use `D` as the stable launch point
- target the `#1437` gap first with a causally corrected, token-only, clearly auditable eval-time n-gram layer
- then spend the next effort on sidecars that do not force retraining:
  - export-only CDQuant / OWC against preserved `final_model.pt`
  - eval-only TTT optimizer / layer-freeze variants against preserved `final_model.pt`
- only after that should we consider training-side optimizer changes or Fisher-guided export

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

**Before launching the next paid pod**, confirm the local offline prep is complete and the corrected n-gram path is what we intend to run:

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
# 1. sync the repo checkout so the corrected launcher / record folder exist on the pod
git pull --ff-only

# 2. regenerate or patch the eval-time n-gram path so it matches the
#    corrected public #1437 token-only causal mode

# 3. launch from the existing D stack, not from a faithful #1413 restart

# 4. if there is engineering time left before the pod, prepare the
#    checkpoint-reuse sidecars first:
#    - export-only CDQuant / OWC
#    - eval-only TTT optimizer / layer-freeze variants

# 5. run seed-0 proof runs before expanding:
#    corrected E first
#    then the export-only / eval-only sidecars

# 6. treat training-side optimizer changes and Fisher-guided export as later
#    work unless the cheaper checkpoint-reuse levers stall out

# 7. only expand to the canonical seed pack if one of the above clearly beats plain D
```

Prepared folders (already committed locally, synced via `git pull`):

- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase/` — base mirror (16,719 code bytes)
- `records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/` — stack variant (17,390 code bytes)

The old `scripts/runpod_1413_batch.sh` workflow remains useful for reproducing the stale pre-correction suite,
but it is no longer the default next-pod command. The next default action is to patch / regenerate the eval-time
ngram path first, then rerun the corrected `E`-style lane from the `D` base. Because the archive preserves `final_model.pt`
for the `D` runs, the next sidecars should preferentially be checkpoint-reuse experiments (export-only CDQuant / OWC,
eval-only TTT refinements) before any retraining-side optimizer work.

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
