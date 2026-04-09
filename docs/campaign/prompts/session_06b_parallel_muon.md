---
session: 06b
date: 2026-04-01
author: Ammer Ayach
model: claude-sonnet-4-6
status: ready
---

> Historical note (2026-04-07): this prompt is archived context from the old `05c-plus` branch family. It is not current execution guidance. Use `AGENTS.md`, `docs/campaign/AGENT_SYNC.md`, and `docs/codex-memory/next-session.md` for the live `#1394` / `#1413` state.

# Session 06b — Parallel Muon + Banking + Quality Donors on 05c-plus

Execution-only. No strategy debate. Read the repo files listed below, then implement.

---

## 0. Read These Files First

1. `CLAUDE.md` — standing rules (Pegasus launcher, never torchrun, always --nodes=1, etc.)
2. `docs/campaign/AGENT_SYNC.md` — current state (05c-plus = best at 1.12558, 06a = failed, int5 = dead)
3. This file — the complete brief

---

## 1. Context

**Competition**: OpenAI Parameter Golf. Optimize BPB on FineWeb val. Artifact <= 16MB, train+eval each <= 10 min on 8xH100.

**Our best**: 05c-plus at `1.12558` BPB (sliding s64), 100ms/step, 5977 steps.

**What failed** (do not revisit):
- GPTQ (7 ablations, permanently parked)
- Mixed int5/int6 (gate failed: +0.006 BPB damage)
- Wider MLP 3.25x (throughput kills it: 137ms/step, 27% fewer steps)
- BigramHash tuning, XSA reduction

**What the leaders have that we don't**:
- PR #1089 (1.1086 BPB): Parallel Muon + parameter banking
- PR #1120 Rascal (1.1099 BPB): Parallel Muon + parameter banking + coprime loader + late QAT

**We already have a working Parallel Muon + banking implementation**:
`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
This achieved 83.3ms/step (faster than our 100ms) with the same architecture.

---

## 2. Task: Build 06b

Create: `records/track_non_record_16mb/2026-04-01_06b_parallel_muon_banking_brotli/train_gpt.py`

**Base**: Copy from `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py` (05c-plus)

**Donors**:
- Parallel Muon + parameter banking: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- Coprime loader + late QAT + brotli export: `records/track_non_record_16mb/2026-04-01_06a_width325_mixed_int5_coprime_lateqat/train_gpt.py`

---

## 3. Exact Deltas (7 changes on 05c-plus)

### Delta 1: Replace Muon optimizer with Parallel Muon

Port the `Muon` class from the 2026-03-23 script (lines 127-266). Key differences from 05c-plus's Muon:
- 3-phase: `launch_reduce_scatters()` → Adam steps → `step()` with overlapped all-gather
- Uses `dist.reduce_scatter_tensor()` + `dist.all_gather_into_tensor()` instead of `dist.all_reduce()`
- Operates on 3D bank parameters, not flattened 1D updates
- No DDP wrapping for bank params

### Delta 2: Add parameter banks to GPT model

Port from 2026-03-23 script (lines 819-827, 880-890). Add to the GPT `__init__`:

```python
head_dim = model_dim // num_heads
kv_dim = num_kv_heads * head_dim
mlp_dim = int(mlp_mult * model_dim)
self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
```

Banks replace the per-layer `CastedLinear` weights in attention (c_q, c_k, c_v, proj) and MLP (fc, proj). The Block forward must accept these as arguments.

Init (from 2026-03-23 `_init_weights`):
- Q, K, V: `orthogonal_(gain=1.0)`
- Out: `zeros_()` then `mul_(1/sqrt(2*num_layers))`
- MLP up: `orthogonal_(gain=1.0)`
- MLP down: `zeros_()` then `mul_(1/sqrt(2*num_layers))`

### Delta 3: Modify Block/Attention/MLP forward to use bank weights

Each Block receives its weights from the bank at the call site:
```python
for i in range(num_layers):
    x = self.blocks[i](x, ...,
        self.qo_bank[i],           # Q weight
        self.kv_bank[i],           # K weight
        self.kv_bank[n + i],       # V weight
        self.qo_bank[n + i],       # Out weight
        self.mlp_up_bank[i],       # MLP fc weight
        self.mlp_down_bank[i],     # MLP proj weight
        ...)
```

CastedLinear layers in attention/MLP become weight-free (just `F.linear(x, w.to(x.dtype))`). Keep CastedLinear for non-banked layers (bigram proj, VE proj, lm_head).

### Delta 4: Training loop — 3-phase optimizer step

Replace the current optimizer step pattern with:
```python
# After backward:
optimizer_muon.launch_reduce_scatters()   # Phase 1: async RS
# ... grad clip, Adam steps on tok/scalar params ...
optimizer_muon.step()                      # Phase 3: wait RS, NS5, AG
```

Do NOT wrap bank params in DDP. Only wrap non-bank params (if any non-bank 2D params exist, they stay in DDP).

### Delta 5: Export — unbank before quantization

Before `mixed_quantize_int6()`, convert bank params back to per-layer 2D weights:
```python
def _unbank_state_dict(sd):
    """Convert 3D bank params to per-layer 2D weights for quantization."""
    new_sd = {}
    n = num_layers  # from model config
    for k, v in sd.items():
        if k == "qo_bank":
            for i in range(n):
                new_sd[f"blocks.{i}.attn.c_q.weight"] = v[i]
                new_sd[f"blocks.{i}.attn.proj.weight"] = v[n + i]
        elif k == "kv_bank":
            for i in range(n):
                new_sd[f"blocks.{i}.attn.c_k.weight"] = v[i]
                new_sd[f"blocks.{i}.attn.c_v.weight"] = v[n + i]
        elif k == "mlp_up_bank":
            for i in range(n):
                new_sd[f"blocks.{i}.mlp.fc.weight"] = v[i]
        elif k == "mlp_down_bank":
            for i in range(n):
                new_sd[f"blocks.{i}.mlp.proj.weight"] = v[i]
        else:
            new_sd[k] = v
    return new_sd
```

Then quantize + export with uniform int6 + brotli-10 (from 06a, but WITHOUT int5):
```python
sd_unbanked = _unbank_state_dict(sd_cpu)
quant_result, quant_meta = mixed_quantize_int6(sd_unbanked, {"mlp", "attn"})
# ... custom_pack + brotli.compress(quality=10) ...
```

Keep the `_custom_pack`, `_custom_unpack`, `_byte_shuffle`, `_byte_unshuffle` inlined functions from 06a.

For the roundtrip eval, build a fresh model (with banks), unbank the dequantized state, then rebank into the eval model. OR: build a non-banked eval model variant. Simplest: just build a normal GPT (without banks) for eval — it only needs to forward, not optimize.

### Delta 6: Coprime loader (from 06a)

Port `choose_coprime_stride()` and `CoprimeDistributedTokenLoader` from 06a. Activate via `LOADER_MODE=coprime` env var. The lazy-loading version (header scan only at init, LRU shard cache).

### Delta 7: Late QAT (from 06a)

Port the `CastedLinear._qat_enabled` mechanism. BUT: in the banked model, CastedLinear is no longer used for the main attention/MLP weights — those are bank params. Late QAT needs to apply to the bank params directly instead.

Modify the QAT logic to operate on banks:
```python
if _qat_enabled and model.training:
    for bank in [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]:
        with torch.no_grad():
            b32 = bank.data.float()
            # Per-row int6 fake-quant across last dim
            row_max = b32.abs().amax(dim=-1, keepdim=True)
            scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
            b_q = (torch.clamp(torch.round(b32 / scale), -32, 31) * scale).to(bank.dtype)
        bank.data.copy_(bank.data + (b_q - bank.data).detach())  # NOT in-place on .data
```

Wait — the STE trick needs gradient flow. Apply it in the forward pass, not on `.data`. The banked forward already does `F.linear(x, weight.to(x.dtype))`. Add a wrapper:
```python
def _qat_weight(w):
    """STE fake-quant for int6 on a 2D or 3D weight."""
    if not _qat_enabled:
        return w
    with torch.no_grad():
        w32 = w.float()
        row_max = w32.abs().amax(dim=-1, keepdim=True)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
        w_q = (torch.clamp(torch.round(w32 / scale), -32, 31) * scale).to(w.dtype)
    return w + (w_q - w).detach()
```

Apply it when extracting bank slices in the forward pass.

---

## 4. What NOT to Change

- `mlp_mult = 3.0` — locked (3.25x throughput regression killed it)
- `vocab_size = 1024`, `num_layers = 11`, `model_dim = 512`, `num_heads = 8`, `num_kv_heads = 4`
- `xsa_last_n = 11`, `bigram_vocab_size = 2048`, `bigram_dim = 128`
- `ve_dim = 128`, `ve_layers = "9,10"`
- `warmdown_iters = 3500`, `leaky_relu_sq = 0.5`
- EMA decay 0.997
- No SWA (dead code in all PRs)
- No TTT (separate eval-time concern, not in this session)
- No GPTQ (permanently parked)
- No mixed int5/int6 (gate failed)
- `torchrun` — NEVER on Pegasus. Always `srun` with manual rank env vars.
- `| tail` on Slurm commands — NEVER. Always `PYTHONUNBUFFERED=1`.

---

## 5. Expected Outcome

The 2026-03-23 reference achieves **83.3ms/step** with banking + Parallel Muon (vs 05c-plus's 100ms/step). That's 17% throughput gain → ~7,200 steps vs 5,977 → ~20% more training. Combined with brotli saving ~150KB headroom, coprime loader, and late QAT, this should improve BPB.

The ablation table in the 2026-03-23 README shows banking + Parallel Muon alone is ±0.0000 BPB (no quality change, pure throughput). The gain comes from more steps.

---

## 6. Verification

1. `python -m py_compile` on the new script
2. Param count assertion: should be ~26.8M (same as 05c-plus, mlp_mult=3.0)
3. Check that bank shapes are correct: `qo_bank: (22, 512, 512)`, `kv_bank: (22, 256, 512)`, `mlp_up_bank: (11, 1536, 512)`, `mlp_down_bank: (11, 512, 1536)`
4. Check that `_unbank_state_dict` produces keys that match `mixed_quantize_int6` expectations

---

## 7. Pegasus Commands

**1xH100 smoke test:**
```bash
srun -p H100 --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 --mem=64G --time=00:15:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="/netscratch/ayach:/netscratch/ayach" \
  --container-workdir="/netscratch/ayach/parameter-golf" \
  bash -lc '
  export PYTHONUNBUFFERED=1
  export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
  export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
  export LATE_QAT_THRESHOLD=0.15
  export ITERATIONS=50 MAX_WALLCLOCK_SECONDS=120
  pip install --no-cache-dir brotli zstandard sentencepiece
  python -u records/track_non_record_16mb/2026-04-01_06b_parallel_muon_banking_brotli/train_gpt.py
  '
```

**Full 8xH100 training:**
```bash
srun -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none \
  --cpus-per-task=6 --mem=200G --time=02:00:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="/netscratch/ayach:/netscratch/ayach" \
  --container-workdir="/netscratch/ayach/parameter-golf" \
  bash -lc '
  export PYTHONUNBUFFERED=1
  export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
  export NCCL_IB_DISABLE=0 NCCL_DEBUG=WARN
  export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
  export LOADER_MODE=coprime LATE_QAT_THRESHOLD=0.15
  pip install --no-cache-dir brotli zstandard sentencepiece
  python -u records/track_non_record_16mb/2026-04-01_06b_parallel_muon_banking_brotli/train_gpt.py
  '
```

---

## 8. Kill Criteria

| Condition | Action |
|-----------|--------|
| `step_avg_ms > 110` (worse than 05c-plus + 10%) | Cancel, profile |
| `sliding_s64 > 1.133` at step 3000 | Cancel, investigate |
| `bytes_total > 16,000,000` at export | Abort, recheck export |

---

## 9. Key File Locations

```
Base (05c-plus):     records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
ParallelMuon donor:  records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py
06a (coprime/QAT):   records/track_non_record_16mb/2026-04-01_06a_width325_mixed_int5_coprime_lateqat/train_gpt.py
Output:              records/track_non_record_16mb/2026-04-01_06b_parallel_muon_banking_brotli/train_gpt.py
Stable rules:        CLAUDE.md
Mutable state:       docs/campaign/AGENT_SYNC.md
```

---

## 10. Post-Training

1. Append result to `docs/campaign/results_log.jsonl`
2. Update `docs/campaign/AGENT_SYNC.md` with Phase 9 result
3. If positive: design 06c (TTT eval-time adaptation, separate concern)

---

*End of session_06b_parallel_muon.md*
