# Session Handoff

Date: 2026-03-27

## Current Truths

- Pegasus verification is still partial.
- No training has been launched from this campaign.
- No Session 03 implementation has been started.
- The recommended first non-TTT anchor remains the clean `2026-03-21`-style stack, not the full `2026-03-22` kitchen sink.
- Local `README.md` still matches the merged public fact that `2026-03-22` `1.1228` is the top merged non-TTT result.
- Open PR claims after `2026-03-24` were checked and were not used to change the recommendation.
- The strongest open public non-TTT claims checked on 2026-03-27 were:
  - PR `#693`: `1.1186` non-TTT
  - PR `#875`: `1.0226` pure neural GDN
  - PR `#910`: expected `~1.114-1.117`, not measured proof
  - PR `#893`: two-pass n-gram rescoring branch, not the pre-TTT anchor path
- Those claims should be treated as horizon signals, not merged facts.

## Recently Completed

- Read the required memory and campaign files.
- Audited repo-root `train_gpt.py` directly against the recommended clean pre-TTT anchor.
- Wrote a new artifact:
  - `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
- Updated:
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/session-handoff.md`

## Main Conclusion

Root `train_gpt.py` is a usable donor skeleton, but it is not a near-anchor.

It already has:

- 512-dim GQA baseline skeleton
- U-Net skip stack
- relu^2 MLP path
- tied embeddings + logit softcap
- compiled DDP training loop
- export/eval scaffold

It still lacks the clean-anchor feature clusters that matter:

- SmearGate + BigramHash
- XSA on the last 4 layers
- partial RoPE `16/64`
- layerwise LN scale
- EMA
- Muon/Adam weight decay
- mixed int6 export + zstd
- stride-64 sliding eval

Therefore:

- Session 03 is not "set a few env vars and run"
- Session 03 is a controlled multi-cluster code port into a new non-record script

## Locked Session 03 Scope

### Stable core to port

- 11 layers, 512 dim, 8 heads / 4 KV heads, U-Net skip stack
- 3x relu^2 MLP
- SmearGate + BigramHash with `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=128`
- XSA on the last 4 layers
- EMA
- partial RoPE `16/64`
- layerwise LN scale
- mixed int6 export + zstd
- stride-64 sliding eval
- anchor launch defaults:
  - `TRAIN_SEQ_LEN=2048`
  - `TRAIN_BATCH_TOKENS=786432`
  - `MATRIX_LR=0.025`
  - `SCALAR_LR=0.025`
  - `TIED_EMBED_LR=0.035`
  - `MUON_WD=0.04`
  - `ADAM_WD=0.04`
  - `MUON_MOMENTUM=0.99`
  - `MUON_MOMENTUM_WARMUP_START=0.92`
  - `MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `GRAD_CLIP_NORM=0.3`
  - `WARMDOWN_ITERS=3000`

### Explicitly exclude from the first anchor port

- GPTQ-lite
- shared value embeddings / VE
- DTG
- tight SWA
- late QAT
- MTP
- any TTT path

## Exact Session 03 Implementation Order

1. Freeze anchor constants in the new Session 03 script.
2. Port SmearGate + BigramHash into the root token path.
3. Port partial RoPE + XSA + LN scale into attention/block/GPT wiring.
4. Add Muon/Adam weight decay and EMA.
5. Replace root int8+zlib export with mixed int6 + zstd roundtrip.
6. Add stride-64 sliding eval and final anchor logging.
7. Verify the excluded features remain absent.
8. Only then check whether attention-backend throughput is a real bottleneck.

## Relevant Risks

- `torch.compile(fullgraph=True)` means late-QAT-style runtime toggles remain structurally untrusted.
- Root uses `scaled_dot_product_attention`, while the `2026-03-21` record used `flash_attn_3_func`; this can distort wallclock-limited comparisons.
- Leaving export at int8+zlib would create a false negative.
- Leaving eval at non-overlapping eval would create a false comparison.
- Leaving anchor-defining settings as env knobs would recreate the stale-default problem from the public record files.

## Current Next Action

- Retry Pegasus allocation and finish live verification:
  - allocate a short H100-class job
  - capture `nvidia-smi -L`
  - capture `nvidia-smi topo -m`
- Do not start training until that verification is complete.

If the explicit goal changes from H100 parity to compute-grant support while Pegasus remains saturated:

- use the already-scoped two-run evidence package from `03b`
- preferred hardware order:
  - Pegasus `H200`
  - Pegasus `A100-80GB`
  - remaining Runpod quick-start fallback
- treat those runs as development evidence only, not as leaderboard-parity validation

## If Pegasus Stays Saturated

The read-only preparation is complete. The next non-read-only step is either:

- implement the narrow Session 03 anchor port exactly as locked in `03b`, or
- gather development evidence on `Pegasus H200`, then `Pegasus A100-80GB`, then remaining Runpod quick-start fallback

## Future Development Evidence Package

Two runs are already scoped in `03b`:

1. Root baseline evidence run
2. Narrow clean-anchor smoke port

For each run, capture:

- GPU type
- steps completed
- wallclock
- final `val_bpb`
- artifact size
- eval mode
- compile warnings
- export warnings

Purpose:

- establish operator readiness
- show end-to-end execution on available hardware
- prove disciplined reproduction progress before requesting more serious compute
- support a `Development grant` application, not an `Advanced competitor` claim

## Source Of Truth Files

- `docs/campaign/artifacts/02a_pegasus_verification.md`
- `docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md`
- `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
- `docs/codex-memory/project-state.md`
- `docs/codex-memory/next-session.md`
