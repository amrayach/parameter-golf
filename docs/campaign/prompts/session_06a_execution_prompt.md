---
session: 06a
date: 2026-04-01
author: Ammer Ayach
model: claude-opus-4-6
status: ready
---

> Historical note (2026-04-07): this prompt is archived context from the old `05c-plus` / mixed-int5 campaign branch. It is not the live execution source of truth. Use `AGENTS.md`, `docs/campaign/AGENT_SYNC.md`, and `docs/codex-memory/next-session.md` for the current `#1394` recovered / `#1413` next workflow.

# Session 06a — Execution Prompt

This is a complete, self-contained brief for a fresh Claude Opus session. Read everything in this file before touching code. Do not re-read AGENT_SYNC.md or decisions.md unless explicitly told to below — the relevant state is reproduced here.

---

## 0. Codex Memory Plugin — Session Startup Protocol

**MANDATORY: run these MCP searches at session start before any other action.**

You have access to `mcp__plugin_claude-mem_mcp-search__search` and `mcp__plugin_claude-mem_mcp-search__get_observations`. Use them as follows:

```
1. search("int5 probe conservative schedule tensor names")
   → retrieves the probe result observation with the 9 tensor names

2. search("export bpb gate result 06a")
   → retrieves gate result if it has already been run this session

3. search("late QAT Rascal CastedLinear implementation")
   → retrieves any prior implementation notes

4. search("coprime loader CoprimeDistributedTokenLoader")
   → retrieves any prior porting notes
```

After each major result (gate pass/fail, 06a training result), call `mcp__plugin_claude-mem_mcp-search__save_memory` with title + summary. This keeps the Codex memory in sync with measured outcomes.

If the probe JSON is not on the remote Pegasus yet (gate has not been run), use the verbatim tensor list from Section 5 below.

---

## 1. Competition Context

**Competition**: OpenAI Parameter Golf. Optimize bits-per-byte (BPB) on FineWeb val. Constraints:
- Artifact (code + compressed model) ≤ 16,000,000 bytes decimal
- Train + eval each ≤ 10 min on 8×H100
- Submission metric: sliding-window s=64 BPB

**Record bar**: Beat PR #1019 (1.1147 BPB, 3-seed mean 1.88218 nats) by ≥ 0.005 nats, p < 0.01.

**Open frontier** (unmerged, competitive reference):
- PR #1089 (`1.1086 BPB`, leader): Parallel Muon + EngramLite + mixed-precision GPTQ + brotli + 3.5x MLP
- PR #1060 (`1.1122 BPB`): GPTQ + brotli
- PR #1072 (`1.1170 BPB`): fused Triton kernel
- Rascal PR #1120 (`1.1099 BPB`): Parallel Muon + parameter banking + coprime loader + late QAT + brotli

**Our current best**: 05c-plus, `1.12558 BPB` (sliding s64), 8×H100 Pegasus, 2026-03-31.

---

## 2. Strategic Position

Local search around 05c-plus is **exhausted**. Three consecutive negative follow-up branches:
- 05e: GPTQ probe — parked permanently (44/66 layers worse than naive, on this model family)
- 05f: BigramHash 3072×112 + warmdown 4000 — negative (+0.00103 vs 05c-plus)
- 05g: XSA-8 throughput — negative (+0.00026 BPB, blew size cap)

**GPTQ is permanently parked.** Do not revisit.

The correct next move is a **coherent fork** with three validated levers:

| Lever | Mechanism | Status |
|-------|-----------|--------|
| Mixed int5/int6 export | 9 conservative tensors → int5, rest → int6 | Gate pending |
| Brotli-10 + custom serialization | Replaces torch.save + zstandard | Validated: saves 150KB on 05c-plus |
| Wider MLP (mlp_mult=3.25) | Unlocked by mixed-bit headroom | Contingent on gate |

Plus two quality donors from Rascal (#1120):
- **Coprime loader** — deterministic walk through data shards, avoids stride aliasing
- **Late QAT** — straight-through int6 fake-quant when LR schedule drops below threshold

Parameter banking + Parallel Muon are **06b scope** — do not bundle into 06a.

---

## 3. 05c-plus Anchor — Key Numbers

```
Base: records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py

sliding s64 val_bpb:    1.12557920
pre-quant EMA exact:    1.14186715
int6 roundtrip exact:   1.14933197
step_avg_ms:            100.39
steps:                  5977
bytes_total:            15,589,271
bytes_model_int6_zstd:  15,524,271 (before brotli upgrade)
bytes_model_brotli10:   ~14,893,228 (after custom+brotli, from probe)
```

Architecture (locked, do not change in 06a):
```
vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4
mlp_mult=3.0  ← changes to 3.25 in 06a
bigram_vocab_size=2048, bigram_dim=128
xsa_last_n=11, rope_dims=16, ln_scale=True
ve_dim=128, ve_layers="9,10"
tie_embeddings=True, tied_embed_init_std=0.005
logit_softcap=30.0, rope_base=10000.0
qk_gain_init=1.5, rope_train_seq_len=1024
```

---

## 4. Immediate Gate — Run BEFORE Any Training

**Script**: `scripts/diagnostics/export_bpb_ab.py`  
**Purpose**: Measure actual BPB damage from conservative int5/int6 export on the 05c-plus float checkpoint.  
**Status**: Written and syntax-checked. No training required. ~10-20 min on 1×GPU.

### Pegasus command

```bash
# First: verify probe JSON structure (run locally or on a login node)
python -c "
import json
d = json.load(open('/netscratch/ayach/parameter-golf/diagnostics/2026-03-31_int5_probe_05c_plus/int5_tolerance_probe.json'))
s = d.get('schedules', {})
print('schedules keys:', list(s.keys()))
c = s.get('conservative', {})
print('conservative keys:', list(c.keys()))
print('num tensor_names:', len(c.get('tensor_names', [])))
print('names:', c.get('tensor_names', []))
"

# Then: full gate job
srun -p batch --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 --mem=64G --time=00:25:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="/netscratch/ayach:/netscratch/ayach" \
  --container-workdir="/netscratch/ayach/parameter-golf" \
  bash -c '
  export PYTHONUNBUFFERED=1
  export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
  pip install --no-cache-dir brotli zstandard sentencepiece
  python -u scripts/diagnostics/export_bpb_ab.py \
    --float-checkpoint /netscratch/ayach/parameter-golf/diagnostics/2026-03-31_05c_plus/final_model.pt \
    --probe-json /netscratch/ayach/parameter-golf/diagnostics/2026-03-31_int5_probe_05c_plus/int5_tolerance_probe.json \
    --val-dir /netscratch/ayach/parameter-golf/data/datasets/fineweb10B_sp1024 \
    --tokenizer /netscratch/ayach/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
    --output-dir diagnostics/2026-04-01_export_ab
  '
```

### Gate criteria

```
GATE PASS:  delta_sw < 0.002 BPB  AND  total_bytes(B) < 16,000,000
GATE FAIL:  either condition violated
```

What `delta_sw` represents: `sliding_s64_bpb(int5/int6 export) - sliding_s64_bpb(uniform int6 export)`. Expected value from probe cosine analysis: ≤ 0.001.

### If gate fails

- If `delta_sw ≥ 0.002`: inspect per-tensor damage — the probe JSON ranks tensors by cos similarity. Drop the 2-3 worst int5 tensors from the schedule and re-run gate. Do NOT proceed to 3.25x training until gate passes.
- If size exceeds 16MB: impossible given probe savings of 924KB — this path means there's an export bug.
- If probe JSON key `schedules.conservative.tensor_names` does not exist: check actual probe JSON structure and update line 292 of export_bpb_ab.py accordingly.

---

## 5. Conservative Int5 Schedule — Verbatim Tensor Names

The int5_tolerance_probe.py ran on the 05c-plus checkpoint and identified these 9 tensors as safe for int5 (cosine similarity ≥ 0.9954, savings +924,000 compressed bytes):

```
# These are the expected names based on probe design. VERIFY against actual
# probe JSON before using. If probe JSON differs, use the probe JSON values.
transformer.h.0.mlp.fc.weight
transformer.h.1.mlp.fc.weight
transformer.h.2.mlp.fc.weight
transformer.h.3.mlp.fc.weight
transformer.h.4.mlp.fc.weight
transformer.h.5.mlp.fc.weight
transformer.h.6.mlp.fc.weight
transformer.h.7.mlp.fc.weight
transformer.h.8.mlp.fc.weight
```

**Critical**: Use actual names from the probe JSON, not this list. These are placeholders.
The probe JSON path: `/netscratch/ayach/parameter-golf/diagnostics/2026-03-31_int5_probe_05c_plus/int5_tolerance_probe.json`

---

## 6. 06a Training Fork — Implementation Sequence

**ONLY begin after gate passes.** Base file: `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

Output location: `records/track_non_record_16mb/2026-04-01_06a_width325_mixed_int5_coprime_lateqat/train_gpt.py`

Copy 05c-plus verbatim, then apply exactly these 5 deltas in order. No other changes. Do not touch architecture, optimizer, LR schedule, or eval logic.

---

### Delta 1: mlp_mult 3.0 → 3.25

In the `Hyperparameters` class, change the default:

```python
# BEFORE (05c-plus, line ~89):
mlp_mult: float = 3.0

# AFTER (06a):
mlp_mult: float = 3.25
```

This is the only architectural change. Verify: `hidden_dim = round(mlp_mult * model_dim) = round(3.25 * 512) = 1664`.

---

### Delta 2: Brotli-10 + custom serialization export

**Rule**: `train_gpt.py` must be self-contained. The submission artifact is `train_gpt.py` + `final_model.int6.ptz` only. Do NOT import from `compress_probe.py` at runtime — inline everything.

Add the following block verbatim to `train_gpt.py` (after the existing imports, before any functions). These are the exact functions from `compress_probe.py`:

```python
# ---------------------------------------------------------------------------
# Custom binary serialization + brotli export (inlined, no external deps)
# ---------------------------------------------------------------------------
import struct as _struct
import brotli as _brotli
import numpy as _np_pack

_DTYPE_TO_STR = {
    torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int8: "i8", torch.int16: "i16", torch.int32: "i32",
    torch.int64: "i64", torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}
_DTYPE_ELEM_SIZE = {
    "f32": 4, "f16": 2, "bf16": 2, "i8": 1,
    "i16": 2, "i32": 4, "i64": 8, "bool": 1,
}

def _byte_shuffle(data: bytes, elem_size: int) -> bytes:
    if elem_size <= 1:
        return data
    n = len(data) // elem_size
    arr = _np_pack.frombuffer(data, dtype=_np_pack.uint8).reshape(n, elem_size)
    return arr.T.copy().tobytes()

def _byte_unshuffle(data: bytes, elem_size: int, n_elements: int) -> bytes:
    if elem_size <= 1:
        return data
    arr = _np_pack.frombuffer(data, dtype=_np_pack.uint8).reshape(elem_size, n_elements)
    return arr.T.copy().tobytes()

def _custom_pack(state_dict: dict, meta: dict, shuffle: bool = True) -> bytes:
    header_entries = {}
    chunks = []
    offset = 0
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        dtype_str = _DTYPE_TO_STR[tensor.dtype]
        raw = tensor.numpy().tobytes() if tensor.dtype != torch.bfloat16 else tensor.float().numpy().tobytes()
        elem_size = _DTYPE_ELEM_SIZE[dtype_str]
        if shuffle and elem_size > 1:
            raw = _byte_shuffle(raw, elem_size)
        header_entries[name] = {"s": list(tensor.shape), "d": dtype_str,
                                "o": offset, "n": len(raw), "e": tensor.numel()}
        chunks.append(raw)
        offset += len(raw)
    import json as _json
    header_json = _json.dumps({"t": header_entries, "m": meta}, separators=(",", ":")).encode()
    return _struct.pack("<I", len(header_json)) + header_json + b"".join(chunks)

def _custom_unpack(blob: bytes, shuffle: bool = True) -> tuple[dict, dict]:
    import json as _json
    header_len = _struct.unpack("<I", blob[:4])[0]
    header = _json.loads(blob[4:4 + header_len])
    data_start = 4 + header_len
    state_dict = {}
    for name, info in header["t"].items():
        raw = blob[data_start + info["o"]:data_start + info["o"] + info["n"]]
        dtype_str = info["d"]
        elem_size = _DTYPE_ELEM_SIZE[dtype_str]
        if shuffle and elem_size > 1:
            raw = _byte_unshuffle(raw, elem_size, info["e"])
        dtype = _STR_TO_DTYPE[dtype_str]
        np_dtype = {"f32": _np_pack.float32, "f16": _np_pack.float16, "bf16": _np_pack.float32,
                    "i8": _np_pack.int8, "i16": _np_pack.int16, "i32": _np_pack.int32,
                    "i64": _np_pack.int64, "bool": _np_pack.bool_}[dtype_str]
        t = torch.from_numpy(_np_pack.frombuffer(bytearray(raw), dtype=np_dtype).copy()).reshape(info["s"])
        state_dict[name] = t.to(torch.bfloat16) if dtype == torch.bfloat16 else t
    return state_dict, header["m"]
```

Then update the export block (current lines ~1547–1564). Replace:
```python
quant_buf = io.BytesIO()
torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
quant_raw = quant_buf.getvalue()
if _COMPRESSOR == "zstd":
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
else:
    quant_blob = zlib.compress(quant_raw, 9)
```
With:
```python
# custom-shuffle + brotli-10: saves ~150KB vs zstandard-22 on this model
quant_blob = _brotli.compress(_custom_pack(quant_result, quant_meta, shuffle=True), quality=10)
```

Update the file write and size logging — the output file can keep the name `final_model.int6.ptz`.

Update the decompression/roundtrip block (lines ~1572–1578). Replace:
```python
quant_decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
```
With:
```python
quant_state, quant_meta_loaded = _custom_unpack(_brotli.decompress(quant_blob_disk), shuffle=True)
deq_state = dequantize_mixed_int6(quant_state, quant_meta_loaded, sd_cpu)
```

**Also inline `quantize_int5_per_row` and `mixed_quantize_int5_int6`** from `export_bpb_ab.py` into train_gpt.py (near the existing `quantize_int6_per_row`). The int5 tensor names must be a hardcoded constant — populated from `diagnostics/2026-04-01_export_ab/export_bpb_ab.json` field `int5_names` after the gate run:

```python
# Populated from gate run: diagnostics/2026-04-01_export_ab/export_bpb_ab.json → int5_names
INT5_TENSOR_NAMES: frozenset[str] = frozenset({
    "transformer.h.X.mlp.fc.weight",  # replace with actual 9 names from gate JSON
    # ...
})
```

Change the quantization call in the export block from `mixed_quantize_int6` to `mixed_quantize_int5_int6`:
```python
quant_result, quant_meta = mixed_quantize_int5_int6(sd_cpu, {"mlp", "attn"}, INT5_TENSOR_NAMES)
```

Remove the `zstandard` and `zlib` imports from train_gpt.py once the export path is fully replaced.

---

### Delta 3: Coprime data loader (verbatim Rascal port)

Rascal reference: `openai/parameter-golf` PR #1120, commit `e5c909f`, file `train_gpt.py`, lines 756–922.

Add the following verbatim to 06a's train_gpt.py, in the data loading section (after `load_data_shard`, before `DistributedDataLoader`):

```python
def choose_coprime_stride(modulus: int, salt: int) -> int:
    """Return a stride that is coprime to modulus, using salt to vary the choice."""
    import math
    candidates = list(range(modulus // 3, 2 * modulus // 3))
    candidates = [c for c in candidates if math.gcd(c, modulus) == 1]
    return candidates[salt % len(candidates)]


class CoprimeDistributedTokenLoader:
    """
    Deterministic coprime walk through shard blocks.
    Each rank gets a different starting offset via salt=rank.
    LRU cache of 4 shards to avoid repeated disk reads.
    Activated by LOADER_MODE=coprime env var (falls back to DistributedDataLoader).
    """
    def __init__(self, filenames: list[str], seq_len: int, rank: int, world_size: int):
        self.filenames = sorted(filenames)
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        # Load all shards to count total blocks
        self._shards: list[torch.Tensor] = []
        for f in self.filenames:
            self._shards.append(torch.from_numpy(
                np.fromfile(f, dtype=np.uint16).astype(np.int32)
            ))

        total_tokens = sum(s.numel() for s in self._shards)
        self.blocks_per_shard = [max(1, s.numel() // seq_len) for s in self._shards]
        self.total_blocks = sum(self.blocks_per_shard)

        self.stride = choose_coprime_stride(self.total_blocks, salt=rank)
        self.pos = (rank * self.stride) % self.total_blocks

        # LRU cache: shard_idx → (shard_tensor)
        from collections import OrderedDict
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._cache_max = 4

    def _get_block(self, global_block_idx: int) -> torch.Tensor:
        # Map global_block_idx → (shard_idx, local_block_idx)
        idx = global_block_idx % self.total_blocks
        shard_idx = 0
        for i, n in enumerate(self.blocks_per_shard):
            if idx < n:
                shard_idx = i
                break
            idx -= n
        local_block = idx

        # LRU shard fetch
        if shard_idx not in self._cache:
            if len(self._cache) >= self._cache_max:
                self._cache.popitem(last=False)
            self._cache[shard_idx] = self._shards[shard_idx]
        else:
            self._cache.move_to_end(shard_idx)

        shard = self._cache[shard_idx]
        start = local_block * self.seq_len
        end = min(start + self.seq_len + 1, shard.numel())
        block = shard[start:end]
        if block.numel() < self.seq_len + 1:
            # wrap
            block = torch.cat([block, shard[:self.seq_len + 1 - block.numel()]])
        return block[:self.seq_len + 1]

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        block = self._get_block(self.pos)
        self.pos = (self.pos + self.stride) % self.total_blocks
        return block
```

In the training setup, replace the data loader instantiation:

```python
# BEFORE (05c-plus):
train_loader = DistributedDataLoader(train_files, seq_len=args.train_seq_len,
                                     rank=rank, world_size=world_size)

# AFTER (06a):
_loader_mode = os.environ.get("LOADER_MODE", "default")
if _loader_mode == "coprime":
    train_loader = CoprimeDistributedTokenLoader(
        train_files, seq_len=args.train_seq_len, rank=rank, world_size=world_size
    )
    log0(f"loader:coprime stride={train_loader.stride} total_blocks={train_loader.total_blocks}")
else:
    train_loader = DistributedDataLoader(train_files, seq_len=args.train_seq_len,
                                         rank=rank, world_size=world_size)
```

Add `LOADER_MODE=coprime` to the Slurm run command (see Section 8).

---

### Delta 4: Late QAT (verbatim Rascal port)

**Rascal reference**: `openai/parameter-golf` PR #1120, commit `e5c909f`.

**Step 4a**: Modify `CastedLinear` to add the class-level flag and fake-quant forward:

```python
# In 05c-plus, CastedLinear is at line ~590. The class currently looks like:
class CastedLinear(nn.Linear):
    def __init__(self, ...):
        ...
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), ...)

# Replace forward with (verbatim from Rascal e5c909f):
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False  # class-level flag, toggled once during training

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31)
                       * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # straight-through estimator
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias else None)
```

**Step 4b**: Add `late_qat_threshold` to `Hyperparameters`:

```python
# In Hyperparameters class:
late_qat_threshold: float = float(os.environ.get("LATE_QAT_THRESHOLD", "0.15"))
```

**Step 4c**: Add the trigger in the training loop, after the LR schedule step (verbatim from Rascal):

```python
# In the training loop, find where scale (cosine LR) is computed.
# 05c-plus uses a cosine warmdown; scale is the LR multiplier in [0, 1].
# After computing scale, add:

if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
    CastedLinear._qat_enabled = True
    log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
```

The 05c-plus training loop computes `scale` as the cosine warmdown factor. Search for where `param_group['lr'] = args.learning_rate * scale` or similar — insert the trigger immediately after `scale` is set.

**What NOT to port from Rascal**:
- SWA: `swa_state` is accumulated but never applied in Rascal. It's dead code. Do not include it.
- Parameter banking / Parallel Muon: 06b scope.
- Any noise injection: Rascal's QAT is purely a straight-through estimator, no random noise.

---

### Delta 5: Anchor/experiment string update

At the top of train_gpt.py (line ~10), update the anchor comment:

```python
# BEFORE:
# Anchor: 05c-plus | SWA not included (dead code)

# AFTER:
# Anchor: 06a | mlp_mult=3.25 | mixed int5/int6 | brotli-10 | coprime loader | late QAT
```

Also update the `log0` feature line in the training setup (search for `features:smeargate`) to add `mlp_mult=3.25 qat=1 loader=coprime`.

---

## 7. Pre-flight Checklist Before 06a Training

Before submitting the 8×H100 job, verify all of the following locally:

```bash
# 1. Syntax check
python -m py_compile records/track_non_record_16mb/2026-04-01_06a_.../train_gpt.py

# 2. Import smoke test (no GPU, just checks imports + model instantiation)
python -c "
import sys
sys.path.insert(0, 'records/track_non_record_16mb/2026-04-01_06a_.../')
sys.path.insert(0, 'scripts/diagnostics/')
from train_gpt import GPT, Hyperparameters, CastedLinear, LOWP_DTYPE
import torch
h = Hyperparameters()
print('mlp_mult:', h.mlp_mult)  # must print 3.25
print('late_qat_threshold:', h.late_qat_threshold)  # must print 0.15
print('CastedLinear._qat_enabled:', CastedLinear._qat_enabled)  # must print False
m = GPT(vocab_size=1024, num_layers=2, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3.25, tie_embeddings=True)
print('hidden:', m.transformer.h[0].mlp.fc.weight.shape)  # must be (1664, 512)
"

# 3. Export smoke: load 05c-plus float ckpt, run mixed int5/int6 + brotli, check size < 16MB
# (only needed if you want to verify the export path before training; gate already validated this)
```

---

## 8. Pegasus 8×H100 Launch Command

```bash
srun -p batch --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none \
  --cpus-per-task=6 --mem=200G --time=02:00:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="/netscratch/ayach:/netscratch/ayach,/fscratch/ayach:/fscratch/ayach" \
  --container-workdir="/netscratch/ayach/parameter-golf" \
  bash -c '
  export PYTHONUNBUFFERED=1
  export MKL_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  export NCCL_IB_DISABLE=0
  export NCCL_DEBUG=WARN
  export LOCAL_RANK=$SLURM_LOCALID
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOADER_MODE=coprime
  export LATE_QAT_THRESHOLD=0.15

  pip install --no-cache-dir brotli zstandard sentencepiece

  # Stage data to fscratch for I/O speed
  rsync -a --progress \
    /netscratch/ayach/parameter-golf/data/datasets/fineweb10B_sp1024/ \
    /fscratch/ayach/fineweb10B_sp1024/ || true

  python -u records/track_non_record_16mb/2026-04-01_06a_width325_mixed_int5_coprime_lateqat/train_gpt.py \
    --data-dir /fscratch/ayach/fineweb10B_sp1024 \
    --tokenizer /netscratch/ayach/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
    --output-dir /netscratch/ayach/parameter-golf/records/track_non_record_16mb/2026-04-01_06a_width325_mixed_int5_coprime_lateqat
  '
```

**Env vars explained**:
- `LOADER_MODE=coprime`: activates CoprimeDistributedTokenLoader
- `LATE_QAT_THRESHOLD=0.15`: QAT engages when cosine LR scale < 0.15 (matches Rascal default)

---

## 9. Kill Criteria

Pre-define these before the run. Do not override them mid-run.

| Condition | Action |
|-----------|--------|
| `step_avg_ms > 115` (>14% regression vs 05c-plus 100ms) | Cancel after 500 steps, profile bottleneck |
| `sliding_s64 > 1.133` at step 3000 (worse than anchor at mid-run) | Cancel, investigate architecture regression |
| `bytes_total > 16,100,000` at export | Abort export, do not save — recheck INT5_TENSOR_NAMES and export logic |
| Gate did not pass | Do not train at 3.25x — stay at 3.0x and re-examine |

---

## 10. Post-Training Actions

After a successful 06a run:

1. **Re-run int5 probe on the 06a checkpoint** (the 3.25x model may have different per-tensor sensitivities than 3.0x — verify the conservative schedule still holds before final submission).

2. **Save result to codex memory**:
   ```
   mcp__plugin_claude-mem_mcp-search__save_memory(
     title="06a training result: mlp_mult=3.25 + int5/int6 + brotli + coprime + late QAT",
     content="[sliding_s64_bpb, step_avg_ms, bytes_total, gate delta_sw, gate_pass]"
   )
   ```

3. **Append to results_log.jsonl**:
   ```json
   {
     "session": "06a",
     "date": "2026-04-01",
     "branch": "2026-04-01_06a_width325_mixed_int5_coprime_lateqat",
     "sliding_s64_bpb": <measured>,
     "pre_quant_bpb": <measured>,
     "int6_roundtrip_bpb": <measured>,
     "step_avg_ms": <measured>,
     "steps": <measured>,
     "bytes_total": <measured>,
     "delta_vs_05c_plus": <measured - 1.12557920>
   }
   ```

4. **Update AGENT_SYNC.md** with Phase 7 measured result.

5. If 06a is positive (sliding_s64 < 1.1255): design 06b scope (parameter banking + Parallel Muon as one unit, see Section 11).

---

## 11. 06b Scope — Do Not Implement Now

**Trigger**: 06a produces a positive measured result.

06b adds the remaining Rascal structural changes as one bundled unit:

- **Parameter banking**: Reshape Q/Out/K/V/MLP weights into 3D banks:
  - `q_bank: (2*L, model_dim, head_dim)`  — L layers × Q/K
  - `v_bank: (2*L, model_dim, kv_head_dim)`
  - `out_bank: (L, model_dim, model_dim)`
  - `fc_bank: (L, hidden_dim, model_dim)`
  - `proj_bank: (L, model_dim, hidden_dim)`
  
- **Parallel Muon**: Single batched Newton-Schulz5 over the bank gradient, plus `reduce_scatter` instead of `all_reduce` for parameter banks.

Banking and Parallel Muon are tightly coupled — they must be ported as a unit. Do not attempt to port one without the other.

Reference: Rascal #1120 train_gpt.py, class `ParamBank` and `ParallelMuon`.

---

## 12. Key File Locations

```
Base script:     records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
06a output:      records/track_non_record_16mb/2026-04-01_06a_.../train_gpt.py
Gate script:     scripts/diagnostics/export_bpb_ab.py
Gate output:     diagnostics/2026-04-01_export_ab/export_bpb_ab.json
Probe JSON:      /netscratch/ayach/.../diagnostics/2026-03-31_int5_probe_05c_plus/int5_tolerance_probe.json
Compress utils:  scripts/diagnostics/compress_probe.py
05c-plus ckpt:   /netscratch/ayach/.../diagnostics/2026-03-31_05c_plus/final_model.pt
Decisions log:   docs/codex-memory/decisions.md
Results log:     docs/campaign/results_log.jsonl
Stable rules:    CLAUDE.md
Mutable state:   docs/campaign/AGENT_SYNC.md
```

---

## 13. What Must NOT Change in 06a

- `num_layers`, `model_dim`, `num_heads`, `num_kv_heads` — locked
- `vocab_size=1024` — locked
- `xsa_last_n=11` — locked (05g showed XSA reduction is negative)
- `bigram_vocab_size=2048, bigram_dim=128` — locked (05f showed no gain from changing this)
- `ve_dim=128, ve_layers="9,10"` — locked
- `warmdown_steps=3500` — locked
- `leaky_relu_sq=0.5` — locked
- Optimizer (Muon + AdamW split) — locked
- EMA decay 0.997 — locked
- `torchrun` usage — never; always `srun` with manual rank env vars
- `| tail -1` in Slurm commands — never; use `PYTHONUNBUFFERED=1`

---

*End of session_06a_execution_prompt.md*
