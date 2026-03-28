# Project State

Date: 2026-03-27

## Objective

Primary:
- determine quickly whether there is a realistic path to a respectable Parameter Golf submission from this environment

Secondary:
- if yes, reproduce a strong public pre-TTT stack, test narrow deltas, and only then consider TTT or RFN-inspired ideas

Stretch:
- top-5 leaderboard ambition, but not assumed

## Current campaign state

- Codex created the campaign scaffolding under `docs/campaign/`
- Codex added a focused local skill bundle for this project into `~/.codex/skills/`:
  - `research-engineer`
  - `gptq`
  - `model-pruning`
  - `transformer-lens-interpretability`
- Claude executed Session 01 and wrote:
  - `docs/campaign/artifacts/01_lineage_and_environment_audit.md`
- Codex completed the read-only pre-TTT anchor diff analysis and wrote:
  - `docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md`
- Codex completed the root-script read-only port-gap audit and wrote:
  - `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
- Session 02 prompt was strengthened to require live Pegasus verification before any training
- Codex now has a configured `basic-memory` MCP server for `parameter-golf-codex`
- Claude also created:
  - `docs/campaign/sessions/06_attribution_graph_sidecar_probe.md` (reframed Session 06)
  - Claude's own plan at `~/.claude/plans/enumerated-drifting-crayon.md`
  - Claude memory at `~/.claude/projects/-home-amay-Work-parameter-golf/memory/`

## Public-state note checked on 2026-03-27

Merged public fact from `main`:

- `README.md` still lists the `2026-03-22` run (`1.1228`) as the top merged non-TTT result
- the top merged overall result remains the `2026-03-23` TTT run (`1.1194`)

Open public PR claims exist beyond that merged state, but they are not accepted leaderboard facts:

- PR `#693` claims `1.1186` non-TTT
- PR `#875` claims `1.0226` non-TTT with pure neural GDN
- PR `#910` is an open proposal with expected `~1.114-1.117`, not measured proof
- PR `#893` is a two-pass n-gram rescoring branch, not the pre-TTT anchor path

Interpretation:

- the merged leaderboard has not yet moved beyond the `03-22` non-TTT anchor lineage
- the frontier may already be moving in open PRs, but fresh sessions should treat those as horizon signals, not as locked ground truth

## Live Pegasus verification (PARTIAL as of 2026-03-27)

Completed:
- `sinfo`: H100/H100-RP/H100-SEE all show 8 GPUs per node, state=mix
- `sacctmgr`: empty output (no immediate rejection, actual limits unverified)
- `sshare`: user is in `compute-account`, zero recent usage
- V100 allocation succeeded (glendale) but PyTorch 2.11 does not support CC 7.0 — V100 not usable
- **A100-80GB allocation succeeded (serv-3333) — baseline smoke test PASSED**
  - 200 steps, train loss 6.93→3.68, val_bpb 2.14 (expected high for 200 steps)
  - int8+zlib export: 7.0 MB, all pipelines working end-to-end
  - AMP_DTYPE auto-detection confirmed (bf16 on A100)
- H100 allocation: queued but never allocated (cluster saturated during test window)

Still pending:
- 8-GPU H100 SXM allocation and nvidia-smi verification
- NVSwitch topology verification

## Important current reality

- Pegasus docs say the `H100` partition is `H100-SXM5`, 8 GPUs per node, NVSwitch
- A100-80GB smoke test proves the account, environment, and training pipeline work
- H100 parity gate still not satisfied — need one successful H100 allocation
- non-H100 development runs are valid grant-support evidence

## What has happened

- partial Pegasus verification artifact exists: `docs/campaign/artifacts/02a_pegasus_verification.md`
- 1 successful baseline smoke run on A100-80GB (200 steps, export pipeline confirmed)
- repo cloned to `/netscratch/ayach/parameter-golf` with V100-compat patch applied
- dataset downloaded (sp1024, 1 train shard)

## What has not happened yet

- no H100 SXM allocation or verification
- no full baseline run (13,000+ steps for val_bpb ~1.22)
- no pre-TTT anchor of your own
- no RFN or attribution-graph probe

## Compute-grant posture

- if reapplying for OpenAI/Runpod compute, prefer the `Development grant` tier, not `Advanced competitor`
- strongest application shape is:
  - `1` root baseline evidence run
  - `1` narrow clean-anchor smoke run
  - logs with GPU type, steps, wallclock, final `val_bpb`, artifact size, eval mode, and compile/export warnings
- until there is an owned competitive result, do not overclaim leaderboard position

## Latest locked Session 03 conclusion

- root `train_gpt.py` is a cleaner donor skeleton than the public record scripts, but it is not a near-anchor
- Session 03 must port multiple feature clusters, not just tune env vars
- stable core to port:
  - 11L / 512 / 8 heads / 4 KV heads with U-Net skip stack
  - 3x relu^2 MLP
  - SmearGate + BigramHash (`2048 x 128`)
  - XSA on the last 4 layers
  - EMA
  - partial RoPE `16/64`
  - layerwise LN scale
  - mixed int6 export + zstd
  - stride-64 sliding eval
- explicitly exclude from the first anchor port:
  - GPTQ-lite
  - VE
  - DTG
  - tight SWA
  - late QAT
  - MTP
  - any TTT path

## Shared campaign files

- `docs/campaign/README.md`
- `docs/campaign/PROMPT_TEMPLATE.md`
- `docs/campaign/PEGASUS_H100_RUNBOOK.md`
- `docs/campaign/sessions/01_lineage_and_environment_audit.md`
- `docs/campaign/sessions/02_pegasus_baseline_ladder.md`
- `docs/campaign/sessions/03_pre_ttt_anchor_port.md`
- `docs/campaign/sessions/04_targeted_delta_sweep.md`
- `docs/campaign/sessions/05_ttt_correctness_audit.md`
- `docs/campaign/sessions/06_rfn_sidecar_probe.md`
- `docs/campaign/sessions/07_go_no_go_review.md`

## Current best next move

- primary path:
  - run only the Pegasus live verification block from Session 02
  - do not start challenge-style baseline or anchor training until that passes
- while Pegasus remains saturated:
  - use `docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md` as the source of truth for the Session 03 anchor shape
  - use `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md` as the source of truth for which root `train_gpt.py` code paths Session 03 should actually touch
  - keep the next decision narrow: clean `2026-03-21`-style anchor first, or promote GPTQ-lite as the first post-anchor delta after anchor verification
- allowed secondary path if the explicit goal is compute-grant evidence rather than H100 parity:
  - run the `03b` evidence package on Pegasus `H200`, Pegasus `A100-80GB`, or remaining Runpod quick-start credit
  - treat those logs as development evidence only, not as leaderboard-parity validation
