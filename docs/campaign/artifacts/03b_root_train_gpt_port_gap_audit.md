# 03b Root `train_gpt.py` Port-Gap Audit

Date: 2026-03-27
Status: Complete

## Scope

This note audits the repo-root [`train_gpt.py`](../../../train_gpt.py) against the recommended clean pre-TTT anchor locked in [`03a_pre_ttt_anchor_diff_analysis.md`](./03a_pre_ttt_anchor_diff_analysis.md).

Anchor assumption used here:

- use the `2026-03-21`-style stack as the first anchor port
- do not promote `2026-03-22` kitchen-sink additions into the first anchor
- exclude GPTQ-lite, VE, DTG, tight SWA, late QAT, MTP, and any TTT path

Public-state note:

- merged leaderboard fact from local [`README.md`](../../../README.md): `2026-03-22` `1.1228` remains the top merged public non-TTT result
- open PR claims after `2026-03-24` do not change this audit because this is a root-script port-gap exercise, not a re-ranking of unmerged claims

## Bottom Line

The repo-root script is a viable donor skeleton, but it is not a near-anchor. It already contains the reusable baseline chassis:

- 512-dim GQA transformer skeleton
- U-Net skip stack
- relu^2 MLP path
- tied-embedding/logit-softcap baseline path
- DDP + `torch.compile` training loop
- post-train roundtrip export/eval scaffold

That is not enough to count as the clean pre-TTT anchor. Relative to the recommended anchor, root `train_gpt.py` still lacks five material feature clusters:

1. token-feature path: SmearGate + BigramHash
2. attention/path refinements: XSA on last 4 layers + partial RoPE
3. depth-scaling refinement: layerwise LN scale
4. training-side fidelity: EMA + weight decay + anchor launch defaults
5. export/eval fidelity: mixed int6 + zstd + stride-64 sliding eval

Diff scale confirms this is not an env-only port:

- root -> `2026-03-20`: `522` insertions / `93` deletions
- root -> `2026-03-21`: `559` insertions / `97` deletions
- root -> `2026-03-22`: `604` insertions / `328` deletions

Conclusion:

- Session 03 should treat root `train_gpt.py` as a cleaner starting point than the record scripts
- Session 03 should not pretend the port is "just change a few env vars"
- the correct smallest-first port is a controlled, multi-cluster code transplant into a new non-record script, not blind launch-time tuning of the root script

## Precise Gap Analysis

### Already Present In Root

| Anchor-related component | Root status | Evidence | Audit judgment |
|---|---|---|---|
| 512 model dim | present as default | root `Hyperparameters.model_dim=512` | aligned |
| 8 heads / 4 KV heads | present as default | root `num_heads=8`, `num_kv_heads=4` | aligned |
| U-Net skip stack | live code path | `GPT` stores encoder skips and reuses them in decoder | aligned |
| relu^2 MLP shape | live code path | `MLP.forward` is `relu(fc(x)).square()` then project | aligned |
| tied embeddings + logit softcap | live code path | `GPT.forward` uses tied projection and tanh softcap | aligned |
| compiled DDP training skeleton | live code path | root compiles `base_model` with `fullgraph=True` and trains under DDP | reusable |
| roundtrip export/eval scaffold | live code path | export -> compress -> load -> eval already exists | reusable but wrong export regime |

### Partially Present In Root

| Anchor component | Root status | Evidence | What is missing |
|---|---|---|---|
| 11-layer 512-dim stack | partially present | root has env-configurable `NUM_LAYERS`, but default is `9` | anchor default/config hardening |
| 3x MLP | partially present | root has env-configurable `MLP_MULT`, but default is `2` | anchor default/config hardening |
| seq 2048 / batch 786432 | partially present | root exposes `TRAIN_SEQ_LEN` and `TRAIN_BATCH_TOKENS`, but defaults are baseline-like | anchor default/config hardening |
| anchor LR/momentum schedule | partially present | root exposes `MATRIX_LR`, `SCALAR_LR`, `TIED_EMBED_LR`, Muon momentum warmup | anchor values differ materially; WD is absent |
| RoPE infrastructure | partially present | root has full-head RoPE machinery | no partial-RoPE split/pass-through path |
| quantized roundtrip | partially present | root exports per-row int8 + zlib and re-evaluates | no mixed int6 path, no zstd, no sliding eval |

### Missing From Root

| Anchor component | Root status | Evidence |
|---|---|---|
| SmearGate | missing | no `SmearGate` module or smear call in root |
| BigramHashEmbedding | missing | no bigram embedding path in root |
| XSA on last 4 layers | missing | no `use_xsa` flag or self-value subtraction path |
| partial RoPE `16/64` | missing | root applies RoPE to the entire head dim |
| layerwise LN scale | missing | no `1/sqrt(layer+1)` scaling in `Block` |
| EMA shadow averaging | missing | no EMA state, update, or pre-export apply path |
| Muon/Adam weight decay | missing | root has no `MUON_WD`, `ADAM_WD`, or optimizer `weight_decay` support |
| mixed int6 export | missing | root only supports int8 quantization |
| zstd compression | missing | root only compresses with `zlib.compress(..., level=9)` |
| stride-64 sliding eval | missing | root only calls non-overlapping `eval_val` |
| anchor eval logging | missing | no `final_int6_sliding_window_s64`-style log path |

### Present Under Misleading Or Baseline-Biased Surface

| Surface | Why it is misleading |
|---|---|
| root defaults | the root script defaults to baseline-like values: `9L`, `MLP_MULT=2`, `seq=1024`, `batch=524288`, `grad_clip=0.0`; launching this script "close to anchor" via a few env tweaks would still be materially off-anchor |
| root export success | the existing int8+zlib roundtrip can look like a complete challenge export path, but it is the wrong artifact regime for the clean anchor |
| root compile path | `torch.compile(..., fullgraph=True)` is safe for the current static root script, but becomes a trap if Session 03 accidentally imports dynamic late-QAT-style toggles later |

## Code-Backed Root vs Anchor Delta

### What root already gives you

Root `train_gpt.py` already provides:

- anchor-compatible width and GQA defaults in `Hyperparameters`
- a U-Net-shaped `GPT` block stack with learned skip weights
- relu^2 MLP blocks
- a compiled distributed training loop with warmup reset
- a working post-train serialization and roundtrip eval section

Key root locations:

- hyperparameters and baseline defaults: `train_gpt.py:39-87`
- current eval path: `train_gpt.py:220-278`
- current int8 quant/export path: `train_gpt.py:288-420`, `train_gpt.py:1076-1119`
- current RoPE + attention + MLP + block stack: `train_gpt.py:524-724`
- current optimizer/training loop: `train_gpt.py:826-1055`

### What the clean anchor has and root does not

Clean anchor evidence from `2026-03-21` lineage file:

- SmearGate + BigramHash are wired into the token path before the block stack
- XSA is a live attention path on the final 4 layers
- partial RoPE uses only a subset of head dims
- layerwise LN scale is applied in each block
- EMA is updated every step and applied before export
- export uses mixed int6 for `mlp` and `attn`, zstd compression, and sliding-window eval at stride `64`

Representative anchor locations:

- anchor hyperparameter surface: `2026-03-21/train_gpt.py:64-113`
- partial RoPE + XSA + SmearGate + BigramHash + LN scale wiring: `2026-03-21/train_gpt.py:560-860`
- EMA, late-QAT toggle, mixed int6 + zstd export, sliding eval: `2026-03-21/train_gpt.py:1386-1581`

The root gap is therefore architectural plus training plus export/eval. It is not just launch configuration drift.

## Stable Core, Optional Refinement, Misleading Knob / Likely Dead Path

### Stable Core Session 03 should port

- 11 layers, 512 dim, 8 heads / 4 KV heads, U-Net skip stack
- 3x MLP with relu^2 path
- SmearGate + BigramHash with `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=128`
- XSA on the last 4 layers
- EMA
- partial RoPE `16/64`
- layerwise LN scale
- mixed int6 export + zstd
- stride-64 sliding eval
- anchor launch defaults: `seq=2048`, `batch=786432`, `MATRIX_LR=0.025`, `SCALAR_LR=0.025`, `TIED_EMBED_LR=0.035`, `MUON_WD=0.04`, `ADAM_WD=0.04`, `MUON_MOMENTUM=0.99`, `MUON_MOMENTUM_WARMUP_START=0.92`, `MUON_MOMENTUM_WARMUP_STEPS=1500`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`

### Optional Refinement Session 03 should explicitly defer

- GPTQ-lite row-wise clip search from `2026-03-22`
- warmdown `3500`
- FlashAttention-3 parity as a first-pass requirement

Reason:

- GPTQ-lite and warmdown `3500` live in the same file that also adds VE, DTG, tighter SWA, and a different default surface
- kernel parity can matter for step budget, but it is not part of the locked clean-anchor feature set

### Misleading Knob / Likely Dead Path Session 03 should exclude

- late QAT
- SWA
- VE / shared value embeddings
- DTG
- MTP

Reason:

- `03a` already showed these are either excluded from the recommended anchor or structurally under-validated
- root does not currently contain them, which is a benefit, not a deficiency

## Exact Minimal Code Paths Session 03 Should Touch

Session 03 should touch only these root code clusters.

### 1. `Hyperparameters`

Root area:

- `train_gpt.py:39-87`

Required changes:

- harden anchor defaults for shape, batch, schedule, and optimizer values
- set `BIGRAM_VOCAB_SIZE=2048` and `BIGRAM_DIM=128` explicitly rather than inheriting the stale `4096` default from `2026-03-21`
- add only the anchor-specific controls that are truly needed in code:
  - `eval_stride`
  - `eval_seq_len` only if kept separate from `train_seq_len`
  - `muon_wd`
  - `adam_wd`
  - `xsa_last_n`
  - `rope_dims`
  - `ln_scale`
  - `ema_decay`
  - `bigram_vocab_size`
  - `bigram_dim`

Do not expose these as sweep knobs in the first port if they are meant to define the anchor.

### 2. Attention / RoPE / block stack

Root area:

- `train_gpt.py:524-724`

Required changes:

- extend `Rotary` and `apply_rotary_emb` for partial-RoPE split/pass-through
- add XSA support inside `CausalSelfAttention`
- add layerwise LN scaling in `Block`
- thread `xsa_last_n`, `rope_dims`, and `ln_scale` through `GPT`

### 3. Token feature path

Root area:

- `train_gpt.py:648-724`

Required changes:

- add `SmearGate`
- add `BigramHashEmbedding`
- wire them into `GPT.forward` before the block stack

### 4. Training / optimizer fidelity

Root areas:

- `train_gpt.py:96-163`
- `train_gpt.py:826-884`
- `train_gpt.py:1007-1034`

Required changes:

- add Muon weight decay support to the optimizer
- add Adam weight decay for token/scalar/head groups
- add EMA state creation, per-step EMA update, and pre-export EMA application
- set the anchor LR family, momentum warmup, grad clip, and warmdown defaults

### 5. Export / roundtrip path

Root areas:

- `train_gpt.py:288-420`
- `train_gpt.py:1062-1119`

Required changes:

- replace the clean-script int8-only export with mixed int6 for `mlp` and `attn`, int8 elsewhere
- swap zlib for zstd in the anchor export path
- keep the plain row-max int6 path for the first anchor port
- avoid `2026-03-22` GPTQ-lite clip search in Session 03

### 6. Eval path

Root areas:

- `train_gpt.py:220-278`
- `train_gpt.py:805-820`
- `train_gpt.py:980-991`
- `train_gpt.py:1102-1119`

Required changes:

- add sliding-window evaluation
- ensure final anchor logs include both roundtrip and stride-64 sliding metrics
- keep token-byte accounting identical to root so only the eval windowing changes

## Exact Env Vars Session 03 Should Touch

Keep the launch surface narrow.

### Runtime env vars to keep exposed

- `DATA_PATH`
- `TOKENIZER_PATH`
- `RUN_ID`
- `SEED`
- `ITERATIONS`
- `MAX_WALLCLOCK_SECONDS`

### Runtime env vars that still exist but should be treated as infrastructure, not experiment knobs

- `RANK`
- `WORLD_SIZE`
- `LOCAL_RANK`

### Anchor-defining values that Session 03 should set in code, not leave as first-pass launch knobs

- `NUM_LAYERS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- `TRAIN_SEQ_LEN`
- `TRAIN_BATCH_TOKENS`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `MUON_MOMENTUM`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `MUON_WD`
- `ADAM_WD`
- `GRAD_CLIP_NORM`
- `WARMDOWN_ITERS`
- `XSA_LAST_N`
- `ROPE_DIMS`
- `LN_SCALE`
- `BIGRAM_VOCAB_SIZE`
- `BIGRAM_DIM`
- `EVAL_STRIDE`

Reason:

- leaving these as launch-time knobs recreates the stale-default problem already seen in the public record files

## Exact Code Paths And Env Vars Session 03 Should Ignore Or Delete From Scope

### Code paths to ignore

- any late-QAT branch
- SWA bookkeeping
- `ValueEmbedding`
- DTG gate path
- MTP heads and MTP export special-casing
- GPTQ-lite percentile search
- any TTT logic

### Env vars to keep out of the first anchor scope

- `QAT_ENABLED`
- `LATE_QAT`
- `QAT_THRESHOLD`
- `LATE_QAT_THRESHOLD`
- `SWA_ENABLED`
- `SWA_EVERY`
- `VE_ENABLED`
- `VE_DIM`
- `VE_LAYERS`
- `DTG_ENABLED`
- `MTP_NUM_HEADS`
- `MTP_LOSS_WEIGHT`

### Broad baseline knobs that should not become tuning scope in Session 03

- `EMBED_LR`
- `HEAD_LR`
- `QK_GAIN_INIT`
- `BETA1`
- `BETA2`
- `ADAM_EPS`
- `VAL_BATCH_SIZE`
- `VAL_LOSS_EVERY`
- `TRAIN_LOG_EVERY`
- `CONTROL_TENSOR_NAME_PATTERNS`
- `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`

Reason:

- none of these should determine whether the clean pre-TTT anchor exists
- if any of them matter later, that is Session 04 delta-sweep territory, not Session 03 anchor definition

## Compile, Quantization, Export, And Eval Risks

### 1. Compile-risk if dynamic branches are reintroduced

Root already compiles both the Muon backend and the model with `fullgraph=True`.

Implication:

- if Session 03 imports late-QAT-style runtime toggles, the resulting branch behavior becomes untrustworthy
- the clean anchor avoids this by excluding late QAT entirely

### 2. Kernel-parity risk can distort wallclock conclusions

Root attention uses `torch.nn.functional.scaled_dot_product_attention`.

The `2026-03-21` record uses `flash_attn_3_func`.

Implication:

- if the first anchor port stays on root's attention backend, it may still be structurally correct
- but step count under a 600-second cap may differ enough to make a wallclock-limited comparison look worse than the real anchor

This is a throughput risk, not a reason to broaden Session 03 scope.

### 3. Quantization-regime mismatch will produce a false negative

If Session 03 leaves root on int8+zlib:

- model capacity under the 16 MB artifact cap will be misrepresented
- post-quant `val_bpb` will not be comparable to the clean anchor

Therefore:

- mixed int6 + zstd is not optional for a faithful anchor port

### 4. Export-averaging mismatch will produce a false negative

Root exports final raw weights.

The recommended anchor exports EMA-applied weights.

Therefore:

- no Session 03 result should be treated as an anchor reproduction unless EMA is applied before export

### 5. Eval-protocol mismatch will produce a false comparison

Root final metric is non-overlapping eval at `train_seq_len`.

The clean anchor comparison metric uses stride-64 sliding eval.

Therefore:

- do not compare root-style final `val_bpb` directly to public anchor numbers
- the Session 03 script must emit the sliding `s64` metric explicitly

### 6. Baseline defaults can silently poison the port

Root keeps a broad env-configurable baseline surface.

Therefore:

- the first anchor implementation should fix anchor constants in code
- it should not depend on a long launch command full of architecture-defining env overrides

## Smallest-First Session 03 Implementation Checklist

This is the exact implementation order that minimizes ambiguity.

1. Freeze the anchor constants in the new Session 03 script.
2. Port SmearGate + BigramHash into the root token path.
3. Port partial RoPE + XSA + LN scale into attention/block/GPT wiring.
4. Add Muon/Adam weight decay and EMA.
5. Replace the root int8+zlib export with mixed int6 + zstd roundtrip.
6. Add stride-64 sliding eval and the final anchor logging lines.
7. Verify that excluded features remain absent: GPTQ-lite, VE, DTG, SWA, late QAT, MTP, TTT.
8. Only after the code path is structurally clean, measure whether attention-backend throughput is a real bottleneck.

## What Counts As A Valid Anchor Smoke Test Once Training Is Allowed

A valid anchor smoke test is structural, not leaderboard-grade.

Minimum acceptable smoke evidence:

- the new Session 03 script launches without graph breaks or NaNs
- the script logs GPU type and training/eval backend details
- training runs long enough to prove the compiled train loop is stable and EMA updates execute
- export produces a mixed-int6 + zstd artifact successfully
- final logs include:
  - one roundtrip post-quant metric
  - one stride-64 sliding metric
  - final artifact size
  - any compile/export warnings

For a first smoke, competitive `val_bpb` is not required. Structural proof that the anchor code path is complete is required.

## Run Planning For A Future Compute-Grant Evidence Package

Pegasus H100 verification is still partial. These runs are development evidence, not parity claims.

### Run 1: Root baseline evidence run

Goal:

- prove this environment can execute the repo-root training/eval/export loop end-to-end and emit challenge-relevant logs

Preferred hardware order:

1. Pegasus H200
2. Pegasus A100-80GB
3. remaining Runpod quick-start credit as fallback

Exact metrics/logs to capture:

- GPU type
- total steps completed
- wallclock used
- final `val_bpb`
- final artifact size
- eval mode used
- any compile warnings
- any export warnings

Why this helps a Development grant application:

- it proves operator readiness on real accelerator hardware
- it demonstrates end-to-end ownership of training, export, and evaluation
- it produces a clean baseline reference even without H100 parity

### Run 2: Narrow clean-anchor smoke port

Goal:

- show that a root-derived, non-record anchor port can execute the correct clean pre-TTT feature stack without importing the `2026-03-22` kitchen sink

Preferred hardware order:

1. Pegasus H200
2. Pegasus A100-80GB
3. remaining Runpod quick-start credit as fallback

Exact metrics/logs to capture:

- GPU type
- total steps completed
- wallclock used
- final post-quant `val_bpb`
- final artifact size
- eval mode used, explicitly including stride-64 sliding eval
- any compile warnings
- any export warnings

Why this helps a Development grant application:

- it shows concrete reproduction progress toward the best public non-TTT lineage
- it demonstrates disciplined scope control by isolating the clean anchor rather than bundling speculative extras
- it gives reviewers evidence that future H100 time would be spent on a technically grounded plan rather than first-pass debugging

## Decision After This Audit

Recommended Session 03 port scope:

- root-script-derived non-record anchor
- include only the locked stable core
- exclude GPTQ-lite, VE, DTG, tight SWA, late QAT, MTP, and all TTT logic

Next decision point after the audit:

- once Pegasus verification or a development-hardware smoke run exists, decide whether the first post-anchor delta should be GPTQ-lite or whether wallclock/kernel parity must be addressed first
