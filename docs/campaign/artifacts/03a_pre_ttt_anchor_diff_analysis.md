# 03a Pre-TTT Anchor Diff Analysis

Date: 2026-03-27
Status: Complete

## Scope

This note compares the actual `train_gpt.py` code paths in the top 3 non-TTT public submissions:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train_gpt.py`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Method:

- compare live architecture, training, export, and eval code paths
- use READMEs and logs only to disambiguate launch settings or detect attribution drift
- do not treat README claims as true unless the code path exists

## 1. Concise Summary Of What Changed

### 2026-03-20 -> 2026-03-21

This is a narrow, clean patch.

Live code changes:

- `rope_dims` was added to `Hyperparameters`, `Rotary`, `CausalSelfAttention`, `Block`, and `GPT`, enabling partial RoPE rather than full-head RoPE.
- `ln_scale` was added to `Hyperparameters`, `Block`, and `GPT`, multiplying the normalized input to attention and MLP by `1/sqrt(layer_idx+1)`.
- `late_qat` was added to the training loop, toggling `CastedLinear._qat_enabled` once LR scale falls below `QAT_THRESHOLD`.

What did not materially change:

- the 11-layer U-Net-style stack
- SmearGate + BigramHash
- XSA on the last 4 layers
- EMA path
- mixed int6 export path
- sliding-window eval path

Interpretation:

- `2026-03-21` is mostly `2026-03-20` plus partial RoPE and LN scale.
- the late-QAT knob exists in code, but its effect is not trustworthy under `torch.compile`.

### 2026-03-21 -> 2026-03-22

This is not a similarly clean patch. It mixes a real quantization refinement with architecture and control-surface drift.

Live code changes:

- `quantize_int6_per_row` now searches five clip percentiles and picks the lowest-MSE reconstruction. This is the real GPTQ-lite change.
- `warmdown_iters` default changes from `1200` to `3500`, but the winning run still depends on explicit launch settings rather than file defaults.
- EMA becomes unconditional. `ema_enabled` and `ema_decay` knobs disappear; the file always builds `ema_state` and always applies it with literal `0.997`.
- SWA becomes active alongside EMA. In `2026-03-20` and `2026-03-21`, SWA is inert when EMA is enabled. In `2026-03-22`, tight SWA runs at `scale < 0.2` and `swa_every=50` even though EMA is also active.
- A new shared value-embedding path is added and enabled by default: `ValueEmbedding`, `ve_enabled=1`, `ve_dim=128`, `ve_layers="9,10"`, and value injection into attention via `v_embed`.
- A new `dtg` gate path is added but defaults off.
- Late QAT changes from `(late_qat, QAT_THRESHOLD)` to a single `LATE_QAT_THRESHOLD` path that is positive by default.

Interpretation:

- `2026-03-22` is not just `2026-03-21` plus GPTQ-lite.
- the file diff adds a live value-embedding mechanism and changes averaging behavior, so README attribution is incomplete at the file level.

## 2. Stable Core vs Optional Refinement vs Misleading Knob

### Stable Core

These are the smallest code-backed pieces that survive the top-3 lineage and should define the pre-TTT anchor:

| Component | Why it belongs in core |
|---|---|
| 11-layer, 512-dim, 8-head / 4-KV-head transformer with U-Net skip connections | Present and central across all three files. |
| 3x MLP with relu-squared | Present across all three files. |
| SmearGate + BigramHash token feature path | Present across all three files and part of the non-TTT lineage, not a one-off add-on. |
| XSA on the last 4 layers | Introduced by `2026-03-20`, retained by `2026-03-21`, defaulted on by `2026-03-22`. |
| EMA shadow weights applied before export | Present in all three, even though `2026-03-22` hardcodes it. |
| Partial RoPE (`16/64`) | Added cleanly in `2026-03-21`, retained in `2026-03-22`, zero-parameter and low-risk. |
| LN scale (`1/sqrt(layer+1)`) | Added cleanly in `2026-03-21`, retained in `2026-03-22`, zero-parameter and low-risk. |
| Mixed int6 export on MLP+attention, int8 elsewhere, zstd compression, roundtrip validation | Present across all three. |
| Sliding-window eval at stride 64 | Present across all three and required to compare against the public numbers. |
| The `0.025 / 0.025 / 0.035` LR family with `WD=0.04`, Muon momentum warmup, batch `786432`, seq `2048`, grad clip `0.3` | Repeated in the actual record launches even though `2026-03-20` and `2026-03-21` still carry stale defaults in-file. |

### Optional Refinement

These are real changes, but they are not the smallest credible anchor:

| Component | Status |
|---|---|
| GPTQ-lite row-wise clip search | Real and low-cost, but only appears in `2026-03-22`. Best treated as the first post-anchor export refinement. |
| Warmdown `3500` | Plausible, but only tested in the same file that also changes GPTQ-lite, SWA behavior, and value embeddings. |
| Tight SWA with EMA | Live in `2026-03-22`, but not part of the cleaner `2026-03-20` or `2026-03-21` core. |
| Shared value embeddings on layers `9,10` | Live in `2026-03-22`, but absent from the earlier two files and not isolated in the README claim. |

### Misleading Knob / Likely Dead Path

These should not be part of the Session 03 anchor:

| Component | Why it should be excluded |
|---|---|
| Late QAT in `2026-03-21` | The flag flips in logs, but the branch sits behind `torch.compile(fullgraph=True)` and the known constant-folding concern remains. Treat as untrusted. |
| Late QAT in `2026-03-22` | Same structural problem: `_qat_enabled` is read in `CastedLinear.forward`, the model is compiled before the later toggle, and there is no proof the compiled graph re-specializes. |
| `DTG_ENABLED` | New in `2026-03-22`, default-off, no evidence it contributed to the winning score. |
| `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` | The path exists in all three files but the winning runs use `0`; it is not part of the top non-TTT stack. |
| `late_k_layers` inside `mixed_quantize_int6` | Defined in all three files, used nowhere. |
| `2026-03-22` “Total submission size int8+zlib” log line | It prints the int6+zstd size again. It is a misleading log message, not a second export path. |
| Stale file defaults in `2026-03-20` and `2026-03-21` | Those files still default to baseline-ish values (`num_layers=9`, weaker LR/WD defaults, `bigram_vocab_size=4096`). They do not describe the record launch on their own. |

## 3. Exact Recommended Pre-TTT Anchor Feature Set

Recommendation:

- build Session 03 around the `2026-03-21` code shape
- carry the stable `2026-03-20` backbone
- include the clean `2026-03-21` additions
- leave `2026-03-22` refinements for later unless Pegasus verification is complete and the anchor is already reproducible

### Exact Anchor To Carry Forward

Architecture:

- 11 layers
- model dim `512`
- `8` attention heads, `4` KV heads
- MLP multiplier `3.0`
- U-Net skip connections
- SmearGate
- BigramHash with `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=128`
- XSA on the last `4` layers
- partial RoPE with `ROPE_DIMS=16`
- LN scale enabled
- tied embeddings
- logit softcap `30.0`

Training and averaging:

- train/eval seq len `2048`
- train batch tokens `786432`
- `ITERATIONS=9000`
- `MAX_WALLCLOCK_SECONDS=600`
- `MATRIX_LR=0.025`
- `SCALAR_LR=0.025`
- `TIED_EMBED_LR=0.035`
- `MUON_WD=0.04`
- `ADAM_WD=0.04`
- `MUON_MOMENTUM=0.99`
- `MUON_MOMENTUM_WARMUP_START=0.92`
- `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `GRAD_CLIP_NORM=0.3`
- EMA every step with decay `0.997`
- warmdown `3000`

Export and eval:

- mixed int6 export for MLP and attention matrices
- int8 export for embeddings and remaining large float tensors
- zstd level `22`
- roundtrip eval after dequantization
- sliding-window eval with `EVAL_STRIDE=64`

### Explicitly Leave Out Of The Anchor

- GPTQ-lite clip search
- warmdown `3500`
- tight SWA
- shared value embeddings
- DTG
- late QAT
- MTP

Rationale:

- this anchor is small enough to explain
- it preserves the strongest non-TTT lineage signal
- it avoids the `2026-03-22` attribution ambiguity
- it should still land in the `1.124-1.127` band if the environment and implementation are faithful

## 4. Implementation Risks

### 4.1 Do not port `2026-03-22` blindly

The `2026-03-22` file is a poor anchor template because it hides several decisions inside defaults and unconditional paths:

- EMA is hardcoded on
- SWA is also live
- value embeddings are live by default
- late QAT is live by threshold, despite unclear correctness under compile

That is not a narrow anchor. It is already a mixed bundle.

### 4.2 The `2026-03-22` parameter increase is a concrete warning sign

From the logs:

- `2026-03-21` model params: `26829913`
- `2026-03-22` model params: `26993756`

Delta: `163843` parameters.

That matches the newly added value-embedding path:

- `1024 * 128 = 131072` for the value-embedding table
- `128 * 256 = 32768` for the projection into KV space
- `1` shared scale
- `2` per-layer scales

Total: `163843`.

Conclusion:

- the `2026-03-22` score cannot be attributed to GPTQ-lite alone from file diff evidence

### 4.3 Late QAT remains under-validated

Evidence available:

- the logs show `late_qat:enabled ...`
- the code toggles `_qat_enabled`

What is still missing:

- proof that the compiled graph actually started executing the fake-quant branch after the toggle

Therefore:

- treat late QAT as ambiguous and exclude it from Session 03

### 4.4 Defaults are not trustworthy proxies for the record launches

`2026-03-20` and `2026-03-21` still default to:

- `num_layers=9`
- weaker LR values
- weaker WD values
- `bigram_vocab_size=4096`

But the record launches were clearly stronger than those defaults. Session 03 should not inherit those files as-is and then rely on defaults.

## 5. Minimal Env Vars And Code Paths For Session 03

### Code Paths To Keep

Keep only these code paths:

- `GPT` with U-Net skip connections
- `SmearGate`
- `BigramHashEmbedding`
- `CausalSelfAttention` with XSA support
- `Rotary` and `apply_rotary_emb` with partial-RoPE support
- `Block` with LN scale
- EMA shadow state and EMA application before export
- `mixed_quantize_int6` with the plain row-max int6 path
- roundtrip dequantization and sliding-window eval

### Code Paths To Drop

Drop these from the Session 03 anchor script:

- late QAT branch in `CastedLinear`
- SWA path
- `ValueEmbedding`
- `DTG`
- `MTP`
- GPTQ-lite clip-percentile search

### Minimal Env Surface

For the anchor port, prefer fixed code constants over a large env surface.

Carry forward only operator-facing env vars:

- `DATA_PATH`
- `TOKENIZER_PATH`
- `RUN_ID`
- `SEED`
- `MAX_WALLCLOCK_SECONDS`

Runtime-distributed env vars remain required but come from `torchrun`:

- `RANK`
- `WORLD_SIZE`
- `LOCAL_RANK`

If one extra training control must remain exposed, keep only:

- `ITERATIONS`

Everything else that defines the anchor should be fixed in the Session 03 script itself. That avoids the stale-default problem present in `2026-03-20` and `2026-03-21`.

## Bottom Line

Recommended anchor:

- use the `2026-03-20` backbone
- add the clean `2026-03-21` partial-RoPE and LN-scale changes
- keep EMA, XSA, SmearGate, BigramHash, mixed int6 export, and stride-64 sliding eval
- do not include GPTQ-lite, VE, DTG, SWA, or late QAT in the first anchor port

Next read-only decision point:

- once Pegasus verification is complete, decide whether Session 03 should stay with this clean `2026-03-21`-style anchor or whether GPTQ-lite should be promoted immediately as the first post-anchor export-only delta
