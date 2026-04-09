# 07c1 Record PR Readiness

Date: 2026-04-03

## Purpose

This note tracks what is required to turn the current `07c1` RunPod result into a
clean official record PR against the merged leaderboard, not just a good local
result.

The relevant source rules are in [README.md](../../README.md), especially the
record-track requirements:

1. beat the merged SOTA by at least `0.005` nats
2. provide enough logs to support `p < 0.01`
3. reproducibly run in under `10` minutes on `8xH100 SXM`
4. submit a self-contained folder under `records/track_10min_16mb/`

## Official Repo Findings

Direct GitHub checks on 2026-04-03:

- Merged official record `#1019`
  - exact 3-seed mean: `1.88217853` nats / `1.11473509` BPB
  - accepted because it clearly cleared the `0.005`-nat bar and looked
    reproducible
- Open PR `#1089`
  - exact 3-seed mean: `1.1091` BPB
  - reviewer comment shows that reproducibility and step-time mismatch are a
    live concern
- Open PR `#1212`
  - exact 5-seed mean: `1.8755` nats / `1.1108` BPB
  - strong README structure and full seed table
- Closed PR `#1242`
  - reported `1.0903` BPB but the author closed it because of a
    `candidate.meta.npz` byte-accounting bug
  - do not use `#1242` as the current real target

## Chronology Risk

Record submissions are reviewed chronologically, but the practical bar can still
move while a PR is waiting for review.

Important implication for `07c1`:

- against merged `#1019`, the current `07c1` base line is a real record candidate
- against open `#1212`, the current `07c1` mean is only about `0.0018` nats
  better, which is **not** enough to clear the official `0.005`-nat record bar
  if `#1212` merges first

So:

- official acceptance is still about the merged leaderboard
- but open older PRs still matter operationally because they can invalidate our
  record claim before review finishes

## What High-Quality Record PRs Have In Common

From merged `#1019` and strong open PRs `#1089` / `#1212`:

- the PR effectively only adds a new record folder
- the folder is self-contained and runnable
- `README.md` contains:
  - exact per-seed table
  - mean and usually std
  - exact delta vs merged SOTA in nats
  - artifact byte accounting
  - legality / reproducibility notes
  - clear run command
- `submission.json` contains:
  - name, author, github id, date
  - exact `val_loss` and `val_bpb`
  - `bytes_total`
  - concise blurb
- train logs are included for every claimed seed
- dependencies are documented; `requirements.txt` is included when nonstandard
  packages are needed

## Current 07c1 Base Status

Current RunPod base seeds:

- `42`: `1.87692713` nats / `1.11166353` BPB / `15,716,286` bytes
- `1337`: `1.87223752` nats / `1.10888598` BPB / `15,719,868` bytes
- `2025`: `1.87193268` nats / `1.10870543` BPB / `15,726,564` bytes

Current exact 3-seed mean:

- `1.87369911` nats / `1.10975165` BPB
- delta vs merged `#1019`: `-0.00847942` nats

This is strong enough on quality for a record claim against the merged
leaderboard.

Current strict RunPod proof seeds:

- `42`: `1.87679853` nats / `1.11158737` BPB / `15,723,725` bytes / `598053ms`
- `1337`: `1.87239786` nats / `1.10898094` BPB / `15,722,146` bytes / `598065ms`
- `2025`: `1.87233216` nats / `1.10894203` BPB / `15,728,840` bytes / `598052ms`
- `7`: `1.87531400` nats / `1.11070811` BPB / `15,731,988` bytes / `598027ms`

Current strict exact 4-seed mean:

- `1.87421064` nats / `1.11005461` BPB
- delta vs merged `#1019`: `-0.00796789` nats

Current locally pulled strict logs:

- `logs/runpod_remote/07c1_runpod_base_s42_strict.txt`
- `logs/runpod_remote/07c1_runpod_base_s1337_strict.txt`
- `logs/runpod_remote/07c1_runpod_base_s2025_strict.txt`
- `logs/runpod_remote/07c1_runpod_base_s7_strict.txt`
- `logs/runpod_remote/07c1_runpod_base_s42_strict.console.log`
- `logs/runpod_remote/07c1_runpod_base_s1337_strict.console.log`
- `logs/runpod_remote/07c1_runpod_base_s2025_strict.console.log`
- `logs/runpod_remote/07c1_runpod_base_s7_strict.console.log`
- `logs/runpod_remote/07c1_runpod_strict_env.env.txt`
- `logs/runpod_remote/runpod_07c1.sh`
- `logs/runpod_remote/train_gpt.py`

Note on `1337`:

- `logs/runpod_remote/07c1_runpod_base_s1337_strict.launch.log` contains an
  earlier SIGHUP-interrupted launch attempt with the same `RUN_ID`
- the completed proof is the matching `.txt` / `.console.log` pair, which
  contains the final exact metrics

## Current Gaps

### 1. Timing proof is now clean for four seeds

The strict RunPod proof set stayed under the nominal budget on all four base
seeds:

- `42`: `598053ms`
- `1337`: `598065ms`
- `2025`: `598052ms`
- `7`: `598027ms`

So timing hygiene is no longer the main blocker for the base path.

### 2. The current strict significance case is now strong enough

Using the current four RunPod seeds against merged `#1019` exact seed results:

- Welch `t = -6.8667`
- one-sided `p ~= 0.001785`

This now satisfies the written `p < 0.01` requirement.

### 3. Logs are now local, but should be preserved intentionally

`scripts/runpod_sync_repo.sh` excludes `logs/`, so exact train logs do not come
back automatically during repo sync.

We now have the current strict proof logs locally under `logs/runpod_remote/`.

### 4. Environment evidence is now captured in a stable local artifact

Because `#1089` already attracted reproducibility questions around step time and
CUDA / FA3 differences, we should preserve the exact RunPod environment used for
the strict proof set. That evidence is now local in:

- `logs/runpod_remote/07c1_runpod_strict_env.env.txt`

### 5. There is no record-track folder yet

The current work lives under:

- `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`

For the official PR, we need a new self-contained folder under
`records/track_10min_16mb/` containing only the files that belong in the
submission.

## Statistical Go / No-Go

The fourth strict seed already answered the open significance question:

- `s7` landed at `1.87531400` nats
- the strict 4-seed Welch test vs merged `#1019` is `p ~= 0.001785`

Implication:

- no additional strict RunPod seed is required for the merged official `p < 0.01` bar
- the remaining work is packaging, not more proof collection

## Required Submission Package

Before opening the PR, assemble a folder like:

- `records/track_10min_16mb/YYYY-MM-DD_<short_name>/README.md`
- `records/track_10min_16mb/YYYY-MM-DD_<short_name>/submission.json`
- `records/track_10min_16mb/YYYY-MM-DD_<short_name>/requirements.txt`
- `records/track_10min_16mb/YYYY-MM-DD_<short_name>/train_gpt.py`
- `records/track_10min_16mb/YYYY-MM-DD_<short_name>/train_seed*.log`

Recommended README content:

1. headline with exact mean BPB / nats / bytes / hardware
2. exact per-seed table
3. exact delta vs merged `#1019` in nats
4. significance statement with t-stat and p-value
5. what changed vs `#1212`
6. legality note:
   - no tokenizer or dataset edits
   - no TTT in the claimed base record
   - unchanged byte accounting path
7. artifact byte table
8. exact run command and package requirements

Recommended `requirements.txt` content:

- `sentencepiece`
- `brotli`
- any Flash Attention dependency note that is actually required by the saved
  script / environment

## Operational Checklist

1. Preserve the already-pulled strict logs under `logs/runpod_remote/`
2. Preserve the already-pulled environment snapshot under
   `logs/runpod_remote/07c1_runpod_strict_env.env.txt`
3. Preserve the locally copied launcher and exact `train_gpt.py` snapshot
4. Create the final `records/track_10min_16mb/...` folder
5. From local machine, the proof-collection step is already complete:

```bash
SSH_PORT=16006 bash scripts/runpod_fetch_logs.sh root@103.207.149.51
```

6. Keep the PR limited to that folder only

## Recommendation

Do not open the official PR until we have:

- strict timing logs
- enough strict seeds for the `p < 0.01` claim
- local copies of all claimed logs
- a self-contained record folder with `README.md`, `submission.json`,
  `requirements.txt`, and `train_gpt.py`

The first three bullets are now satisfied. Once the record folder is assembled,
this should be positioned as an official record PR
against merged `#1019`, while explicitly noting that open PRs `#1089` and
`#1212` remain stronger or comparable open frontier references on BPB alone.
