"""CPU microbenchmark for PrefixNgramCorrector.

Gate criterion (PLAN_PR1610_CORRECTOR.md Phase 1B):
  Projected corrector overhead < 50s for a full eval pass.

Usage:
    python scripts/bench_corrector_cpu.py

Reports:
  - μs per get_logit_bias() call
  - μs per update() call
  - Projected total overhead for full eval (estimated 2M val tokens)
"""
import sys, types, pathlib, importlib.util, time

# ------------------------------------------------------------------
# Minimal stub for flash_attn (not present locally)
# ------------------------------------------------------------------
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

if "flash_attn_interface" not in sys.modules:
    _stub("flash_attn_interface",
          flash_attn_func=None,
          flash_attn_varlen_func=None)

# ------------------------------------------------------------------
# Load PrefixNgramCorrector from actual train_gpt.py
# ------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_TRAIN_GPT = _REPO_ROOT / "records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"

_spec = importlib.util.spec_from_file_location("_train_gpt_src", _TRAIN_GPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
PrefixNgramCorrector = _mod.PrefixNgramCorrector

# ------------------------------------------------------------------
# Benchmark parameters (matching production defaults)
# ------------------------------------------------------------------
B          = 64     # ttt_batch_size
V          = 8192   # actual vocab size
ORDERS     = [8]    # ablation run 1a (single order)
ALPHA      = 0.3
CHUNK_SIZE = 32     # ttt_chunk_size
SEQ_LEN    = 2048   # ttt_eval_seq_len

# Projection: estimated total val tokens for a full eval pass
# SP1024 fineweb val subset ≈ 1–2M tokens; use 2M as conservative upper bound
PROJECTED_VAL_TOKENS = 2_000_000

WARMUP_CHUNKS = 50   # chunks to feed before timing (warm CPU cache)
TIMED_CHUNKS  = 500  # chunks to time for get_logit_bias + stack
TIMED_UPDATES = 5000 # individual update() calls to time

import torch
import random

random.seed(42)

def make_correctors():
    return [PrefixNgramCorrector(V=V, alpha=ALPHA, orders=ORDERS) for _ in range(B)]

# Warm up: feed some tokens so hash tables are populated
correctors = make_correctors()
for _ in range(WARMUP_CHUNKS):
    for tok in range(CHUNK_SIZE):
        t = random.randrange(V)
        for c in correctors:
            c.update(t)

print(f"Benchmarking PrefixNgramCorrector  B={B}, V={V}, orders={ORDERS}, alpha={ALPHA}")
print(f"chunk_size={CHUNK_SIZE}, warmup_chunks={WARMUP_CHUNKS}")
print()

# ------------------------------------------------------------------
# Benchmark 1: get_logit_bias() per corrector + B-element stack
# ------------------------------------------------------------------
t0 = time.perf_counter()
for _ in range(TIMED_CHUNKS):
    biases = [correctors[b].get_logit_bias() for b in range(B)]
    _ = torch.stack(biases).unsqueeze(1)  # [B, 1, V]
t1 = time.perf_counter()

total_bias_us = (t1 - t0) * 1e6
per_chunk_bias_us = total_bias_us / TIMED_CHUNKS
per_corrector_bias_us = per_chunk_bias_us / B

print(f"get_logit_bias() + stack [B,1,V]:")
print(f"  {per_chunk_bias_us:.1f} μs / chunk   ({per_corrector_bias_us:.2f} μs per corrector)")

# ------------------------------------------------------------------
# Benchmark 2: update() per token
# ------------------------------------------------------------------
tokens_seq = [random.randrange(V) for _ in range(TIMED_UPDATES)]
c_single = PrefixNgramCorrector(V=V, alpha=ALPHA, orders=ORDERS)
# warm up the single corrector
for tok in range(100):
    c_single.update(random.randrange(V))

t0 = time.perf_counter()
for tok in tokens_seq:
    c_single.update(tok)
t1 = time.perf_counter()

total_update_us = (t1 - t0) * 1e6
per_update_us = total_update_us / TIMED_UPDATES

print(f"\nupdate():")
print(f"  {per_update_us:.3f} μs / token")

# ------------------------------------------------------------------
# Projected overhead for full eval
# ------------------------------------------------------------------
# Model: eval_val_ttt_phased processes B docs per batch.
#   wall-clock chunk steps = val_tokens / (B * chunk_size)
#     (each step: B simultaneous get_logit_bias() + B*chunk_size sequential update())
#   total update() calls   = val_tokens
#     (each token belongs to exactly one corrector — no B multiplier)
#
# per_chunk_bias_us already measured B=64 simultaneous calls + stack.
num_wc_chunk_steps = PROJECTED_VAL_TOKENS / (B * CHUNK_SIZE)
total_updates      = PROJECTED_VAL_TOKENS   # one update per token, one corrector

bias_overhead_s   = num_wc_chunk_steps * per_chunk_bias_us / 1e6
update_overhead_s = total_updates      * per_update_us    / 1e6
total_overhead_s  = bias_overhead_s + update_overhead_s

GATE_THRESHOLD_S = 50.0

print(f"\nProjection  (val_tokens={PROJECTED_VAL_TOKENS:,}, wall-clock chunk steps={num_wc_chunk_steps:,.0f}):")
print(f"  bias overhead:   {bias_overhead_s:.1f}s  (B={B} correctors × {num_wc_chunk_steps:,.0f} chunk steps)")
print(f"  update overhead: {update_overhead_s:.1f}s  ({PROJECTED_VAL_TOKENS:,} token updates, 1 per token)")
print(f"  TOTAL:           {total_overhead_s:.1f}s")
print()
if total_overhead_s < GATE_THRESHOLD_S:
    print(f"  GATE PASS  (< {GATE_THRESHOLD_S}s threshold)")
else:
    print(f"  GATE FAIL  (>= {GATE_THRESHOLD_S}s threshold) — optimize corrector before RunPod eval")
