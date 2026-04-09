# RunPod Runbook

Date: 2026-04-03

This document defines the clean RunPod lane for this repo.

Use it for:
- fast foreground iteration when Pegasus is queued or slow
- exact single-node `8xH100 SXM` reproduction when we explicitly decide to spend credits

Do not use it to justify blind sweeps.
Pegasus remains the canonical validation target.

## Official Challenge Requirements

Source: repo `README.md`.

- Official leaderboard parity target:
  - `8xH100`
  - specifically the `SXM` variant
  - training must run in under `10 minutes`
  - evaluation must also run in under `10 minutes`
- Record submissions must:
  - beat the current official SOTA by at least `0.005` nats
  - provide enough logs to show `p < 0.01`
- Artifact cap:
  - `16,000,000` bytes decimal
  - counted as code bytes plus compressed model bytes
- Evaluation restrictions:
  - no cheating on validation loss
  - no training on validation tokens before those tokens are scored
  - test-time training is only legal on already-scored validation tokens
- Dependencies:
  - `requirements.txt` is allowed and supported
  - imported libraries are allowed so long as they do not violate challenge rules

Official RunPod instructions from the README:
- use the official Parameter Golf RunPod launch template:
  - `https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th`
- enable SSH terminal access
- after SSH, land in `/workspace`
- clone the repo there
- fetch cached FineWeb via `python3 data/cached_challenge_fineweb.py --variant sp1024`

## Official vs Frontier Setups

### Official parity setup

This is the setup to treat as challenge-parity:
- `1 pod`
- `8 GPUs`
- `NVIDIA H100 80GB HBM3`
- `H100 SXM`
- `600s` training cap
- single-node launch

For RunPod, this means:
- use H100 SXM, not PCIe, when claiming parity
- use one node with all 8 GPUs local to that node
- use `torchrun --standalone --nproc_per_node=8` on the pod

### Non-record / frontier setup

This is broader:
- still keep artifacts legal if we want a clean submission path
- may exceed the `10 minute` training/eval budget
- may use alternative tokenizers, TTT, n-gram sidecars, or other frontier methods

Non-record does not relax correctness.
It only relaxes the official timing/record gate.

## Current RunPod Account / MCP Surface

Checked via the connected RunPod MCP account on 2026-04-03.

Observed account facts:
- authenticated user exists and is reachable through MCP
- terms of service are signed
- MFA is currently disabled

Observed GPU inventory relevant to this project:
- `NVIDIA H100 80GB HBM3`
  - display name: `H100 SXM`
  - interpretation: on RunPod, `NVIDIA H100 80GB HBM3` is the GPU type id for the `H100 SXM` SKU, not a different parity target
  - max GPU count: `8`
  - currently visible cluster datacenters:
    - `EUR-IS-3`
    - `AP-IN-1`
  - note: the public REST schema currently rejects `AP-IN-1`, so the retry script defaults to `EUR-IS-3` and auto-drops invalid region ids if passed explicitly
- `NVIDIA A100-SXM4-80GB`
  - display name: `A100 SXM`
  - max GPU count: `8`
  - currently visible cluster datacenters:
    - `US-KS-2`
    - `US-MO-1`
    - `US-MD-1`

Interpretation:
- `H100 SXM` is the right parity target
- `A100 SXM` is acceptable for cheap development or environment checks, but not for final parity claims

## SSH Access

Yes, you can have SSH access to RunPod pods.

There are two requirements:
- the pod must be created with SSH enabled
- the RunPod account must have a valid SSH public key configured

Evidence:
- the official README explicitly says to set up an SSH key and enable SSH terminal access
- the RunPod MCP create-cluster schema exposes `start_ssh`
- the RunPod MCP user-settings schema exposes `pub_key`

Local machine status:
- local public keys exist:
  - `~/.ssh/id_ed25519.pub`
  - `~/.ssh/id_rsa.pub`

Important note:
- I did not overwrite the account-level RunPod SSH key automatically
- when you want, we can push `~/.ssh/id_ed25519.pub` to RunPod with the MCP

## Repo-Specific RunPod Lane

This repo now has five helper scripts:
- `scripts/runpod_sync_repo.sh`
- `scripts/runpod_prepare_env.sh`
- `scripts/runpod_07c1.sh`
- `scripts/runpod_1413.sh`
- `scripts/runpod_retry_h100.py`

### What each script does

`scripts/runpod_sync_repo.sh`
- clones or fast-forward pulls the repo on the pod
- optional `--rsync` overlays local uncommitted changes
- deliberately excludes dataset and tokenizer caches

`scripts/runpod_prepare_env.sh`
- checks CUDA visibility
- installs only the minimal missing Python packages for this lane
- hard-fails if `flash_attn_interface` is missing

`scripts/runpod_07c1.sh`
- `smoke`:
  - cheap `1xGPU` structural validation
- `base <seed>`:
  - exact `07c1` non-TTT `8xGPU` run
- `ttt <seed>`:
  - exact `07c1` TTT `8xGPU` run

`scripts/runpod_1413.sh`
- launches the faithful upstream `#1413` SP8192 legal score-first TTT lane
- fetches `pull/1413/head` into local ref `pr1413`
- checks out only the `2026-04-06_SP8192_QK5_LegalTTT_1.0828` record folder
- runs `scripts/runpod_prepare_sp8192.sh`
- injects `DATA_DIR=${REPO_ROOT}/data` so the launch does not rely on record-folder-local data paths
- archives run metadata, source files, console log, rank-0 logfile, and final artifacts into `/workspace/pr1413_archive_<stamp>/seed<seed>`

`scripts/runpod_retry_h100.py`
- retries allocation of a single `8x H100 SXM` pod via the official RunPod REST API
- requires `RUNPOD_API_KEY`
- uses `Any region` by default and only restricts datacenters if `--regions ...` is passed explicitly
- prints the resolved `ssh root@<ip> -p <port>` command once the pod is reachable
- also supports `status <pod-id>` and `delete <pod-id>` to manage pods after allocation

## Why the Environment Gate Is Strict

Our current `07c1` script depends on:
- `flash_attn_interface`
- `triton`
- `brotli`
- `sentencepiece`

The target script:
- `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`

The script hard-requires `brotli` and imports `flash_attn_interface` directly.

That means:
- a generic CUDA pod is not enough
- we should prefer the official Parameter Golf RunPod template
- if `flash_attn_interface` fails to import, stop and fix the image before any paid training

## Current 07c1 Exact Launch Contract

The exact `07c1` Pegasus reruns we want to mirror use:
- `NUM_LAYERS=12`
- `BIGRAM_VOCAB_SIZE=5120`
- `VE_DIM=128`
- `WINDOW_SIZE=512`
- `WINDOW_ATTN_LAYERS=2,4,6,8,10`
- `QK_GAIN_INIT=2.5`
- split LR banks:
  - `MATRIX_LR=0.024`
  - `MATRIX_LR_LATE=0.019`
  - `SCALAR_LR=0.020`
  - `SCALAR_LR_LATE=0.038`
  - `TIED_EMBED_LR=0.022`
- `MUON_MOMENTUM=0.985`
- `WARMDOWN_ITERS=4000`
- `TRAIN_BATCH_TOKENS=589824`
- mixed training lengths:
  - `SEQ_LENS_PER_GPU=2048,2048,2048,2048,2048,6144,6144,6144`
  - `SEQS_PER_GPU=36,36,36,36,36,10,10,10`
- `TRAIN_SEQ_LEN=2048`
- `EVAL_SEQ_LEN=6144`
- `EVAL_STRIDE=64`
- `MAX_WALLCLOCK_SECONDS=600`
- `GPTQ_RESERVE_MS=0`
- base:
  - `TTT_ENABLED=0`
- TTT:
  - `TTT_ENABLED=1`

Those values are already encoded in `scripts/runpod_07c1.sh`.

## Frontier Reference Points

These are not instructions to copy blindly.
They are reference setups showing what the frontier is actually using.

### PR #1212

Open PR, `1.1108` BPB, `1.8755` nats, `~15.73 MB`, `8xH100 SXM`, `600s`, no TTT.

Relevant environment/setup patterns:
- single-node `8xH100 SXM`
- Flash Attention 3 required
- `brotli` required for final artifact path
- mixed sequence-length training
- evaluation at long context (`6144`)

### PR #1089

Open PR, `1.1091` BPB, `~15.3 MB`, `8xH100 SXM`.

Relevant environment/setup patterns:
- single-node `8xH100 SXM`
- `brotli>=1.1`
- `torch>=2.11`
- `Python>=3.12`
- mixed-precision GPTQ and aggressive compression
- frontier setup is more fragile than our current `07c1` path

### PR #1242

Closed, unmerged, `1.0903` BPB, `8xH100 SXM`.

Relevant environment/setup patterns:
- alternative tokenizer
- n-gram rescoring
- legal TTT

This is frontier-relevant but out of scope for the current `07c1` campaign.

## Recommended RunPod Policy For This Repo

Use the following ladder.

### Tier 0: lane validation

Purpose:
- prove the pod image is correct
- prove SSH works
- prove the script imports and starts

Recommended hardware:
- `1x H100 SXM`

Commands:

```bash
export RUNPOD_API_KEY=...
python3 scripts/runpod_retry_h100.py acquire --name parameter-golf-07c1-h100
bash scripts/runpod_sync_repo.sh root@<pod-ip>
ssh root@<pod-ip> 'cd /workspace/parameter-golf && bash scripts/runpod_prepare_env.sh'
ssh root@<pod-ip> 'cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1'
ssh root@<pod-ip> 'cd /workspace/parameter-golf && bash scripts/runpod_07c1.sh smoke'
```

### Tier 1: exact foreground reproduction

Purpose:
- run one exact `07c1` base or TTT experiment
- only after we explicitly decide the spend is justified

Recommended hardware:
- `1x pod with 8x H100 SXM`

Commands:

```bash
ssh root@<pod-ip> 'cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80'
ssh root@<pod-ip> 'cd /workspace/parameter-golf && RUN_ID=07c1_runpod_base_s42 bash scripts/runpod_07c1.sh base 42'
ssh root@<pod-ip> 'cd /workspace/parameter-golf && RUN_ID=07c1_runpod_ttt_s2025 bash scripts/runpod_07c1.sh ttt 2025'
```

## What Not To Do

- do not claim parity from `H100 PCIe`
- do not claim parity from `A100`
- do not use RunPod for broad sweeps before the current `07c1` decision gate is answered
- do not auto-install a random FA3 stack until `flash_attn_interface` imports cleanly
- do not spend credits on `8xH100` if Pegasus is about to deliver the same signal

## Decision Standard

Run on RunPod when at least one of these is true:
- Pegasus is blocked long enough that waiting costs more than one decisive RunPod run
- we need same-day foreground validation on a known-good parity environment
- we are testing one high-EV exact experiment, not a sweep

Do not run on RunPod when:
- the exact same experiment is already queued and likely to start soon on Pegasus
- the result would not change the next branching decision
