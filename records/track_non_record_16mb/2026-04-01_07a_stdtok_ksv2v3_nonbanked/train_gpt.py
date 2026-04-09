"""
Session 07a: Standard-Tokenizer KSV2/KSV3 Non-Banked Pivot

Stable first landing of the PR #1130 / #1212 family on the local standard-tokenizer,
non-TTT, non-banked chassis:
  1. 12 layers on the local skip-connected dense stack
  2. residual lambdas + cache/backout + MiLe
  3. split early/late LR groups + PR-style coprime loader
  4. compile-stable RoPE + eager eval
  5. optional window attention and optional mixed-seq training

Parameter banking, Turbo-Muon, EngramLite, GPTQ, and other #1089-only thesis pieces
remain out of scope for 07a.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

try:
    import brotli
    _BROTLI_AVAILABLE = True
except ImportError:
    brotli = None
    _BROTLI_AVAILABLE = False

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _FA3_AVAILABLE = True
except ImportError:
    flash_attn_3_func = None
    _FA3_AVAILABLE = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Pre-TTT anchor constants are hardcoded. Only operational env vars are exposed.

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 9000))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    amp_dtype = os.environ.get("AMP_DTYPE", "auto").strip().lower()

    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = 20
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 589_824))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 6144))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 2.5))

    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = 4
    model_dim = 512
    num_heads = 8
    mlp_mult = 3.0
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0

    embed_lr = 0.6
    head_lr = 0.008
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.022))
    tied_embed_init_std = 0.005
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.024))
    matrix_lr_late = float(os.environ.get("MATRIX_LR_LATE", os.environ.get("MATRIX_LR", "0.019")))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.020))
    scalar_lr_late = float(os.environ.get("SCALAR_LR_LATE", os.environ.get("SCALAR_LR", "0.038")))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.985))
    muon_backend_steps = 5
    muon_momentum_warmup_start = 0.92
    muon_momentum_warmup_steps = 1500
    beta1 = 0.9
    beta2 = 0.95
    adam_eps = 1e-8
    grad_clip_norm = 0.3
    muon_wd = 0.04
    adam_wd = 0.04
    ema_decay = 0.997
    eval_stride = int(os.environ.get("EVAL_STRIDE", 128))
    eval_also_stride64 = env_flag("EVAL_ALSO_STRIDE64", False)

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 5120))
    bigram_dim = 128
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 7))
    rope_dims = 16
    ln_scale = True
    rope_train_seq_len = 1024

    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "5,9,10")
    window_size = int(os.environ.get("WINDOW_SIZE", -1))
    window_attn_layers = os.environ.get("WINDOW_ATTN_LAYERS", "2,4,6,8,10")
    mile_gamma = float(os.environ.get("MILE_GAMMA", 0.75))
    mile_clamp_min = float(os.environ.get("MILE_CLAMP_MIN", 0.2))
    mile_peak_frac = float(os.environ.get("MILE_PEAK_FRAC", 0.4))
    cache_layer = int(os.environ.get("CACHE_LAYER", 7))
    backout_init = float(os.environ.get("BACKOUT_INIT", 0.1))


LOWP_DTYPE = torch.bfloat16
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_LOG_PATH = REPO_ROOT / "docs/campaign/results_log.jsonl"
_EXPORT_COMPRESSOR = "brotli10"

_DTYPE_TO_STR = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}
_DTYPE_ELEM_SIZE = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "bool": 1,
}


def _byte_shuffle(data: bytes, elem_size: int) -> bytes:
    if elem_size <= 1:
        return data
    n = len(data) // elem_size
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, elem_size)
    return arr.T.copy().tobytes()


def _byte_unshuffle(data: bytes, elem_size: int, n_elements: int) -> bytes:
    if elem_size <= 1:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(elem_size, n_elements)
    return arr.T.copy().tobytes()


def _custom_pack(state_dict: dict[str, torch.Tensor], meta: dict, shuffle: bool = True) -> bytes:
    header_entries = {}
    chunks = []
    offset = 0
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        dtype_str = _DTYPE_TO_STR.get(tensor.dtype)
        if dtype_str is None:
            raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")
        if tensor.dtype == torch.bfloat16:
            raw = tensor.float().numpy().tobytes()
            elem_size = 4
        else:
            raw = tensor.numpy().tobytes()
            elem_size = _DTYPE_ELEM_SIZE[dtype_str]
        if shuffle and elem_size > 1:
            raw = _byte_shuffle(raw, elem_size)
        header_entries[name] = {
            "s": list(tensor.shape),
            "d": dtype_str,
            "o": offset,
            "n": len(raw),
            "e": tensor.numel(),
        }
        chunks.append(raw)
        offset += len(raw)
    header_json = json.dumps({"t": header_entries, "m": meta}, separators=(",", ":")).encode()
    return struct.pack("<I", len(header_json)) + header_json + b"".join(chunks)


def _custom_unpack(blob: bytes, shuffle: bool = True) -> tuple[dict[str, torch.Tensor], dict]:
    header_len = struct.unpack("<I", blob[:4])[0]
    header = json.loads(blob[4:4 + header_len])
    data_start = 4 + header_len
    state_dict = {}
    for name, info in header["t"].items():
        raw = blob[data_start + info["o"]:data_start + info["o"] + info["n"]]
        dtype_str = info["d"]
        dtype = _STR_TO_DTYPE[dtype_str]
        elem_size = 4 if dtype == torch.bfloat16 else _DTYPE_ELEM_SIZE[dtype_str]
        if shuffle and elem_size > 1:
            raw = _byte_unshuffle(raw, elem_size, info["e"])
        if dtype == torch.bfloat16:
            tensor = (
                torch.from_numpy(np.frombuffer(bytearray(raw), dtype=np.float32).copy())
                .to(torch.bfloat16)
                .reshape(info["s"])
            )
        else:
            np_dtype = {
                "f32": np.float32,
                "f16": np.float16,
                "i8": np.int8,
                "i16": np.int16,
                "i32": np.int32,
                "i64": np.int64,
                "bool": np.bool_,
            }[dtype_str]
            tensor = torch.from_numpy(np.frombuffer(bytearray(raw), dtype=np_dtype).copy()).reshape(info["s"])
        state_dict[name] = tensor
    return state_dict, header["m"]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def maybe_compile(fn, *, enabled: bool, fullgraph: bool = False):
    return torch.compile(fn, dynamic=False, fullgraph=fullgraph) if enabled else fn


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def max_memory_mib(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated() // 1024 // 1024),
        int(torch.cuda.max_memory_reserved() // 1024 // 1024),
    )


def build_cpu_smoke_tokens(vocab_size: int, seq_len: int, total_sequences: int) -> Tensor:
    total_tokens = max(total_sequences * seq_len + 1, seq_len + 1)
    base = torch.arange(total_tokens, dtype=torch.int64) % vocab_size
    return base.to(dtype=torch.uint16).contiguous()


def build_cpu_smoke_luts(vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.ones(vocab_size, dtype=torch.int16, device=device),
        torch.zeros(vocab_size, dtype=torch.bool, device=device),
        torch.zeros(vocab_size, dtype=torch.bool, device=device),
    )


def append_results_log(payload: dict[str, object]) -> None:
    RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_LOG_PATH, "a", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def resolve_amp_dtype(amp_dtype_name: str) -> torch.dtype:
    name = amp_dtype_name.strip().lower()
    if name not in {"auto", "bf16", "fp16"}:
        raise ValueError(f"AMP_DTYPE must be one of auto, bf16, fp16; got {amp_dtype_name!r}")
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("AMP_DTYPE=bf16 requested, but this CUDA device does not support bfloat16")
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype).removeprefix("torch.")

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=LOWP_DTYPE)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=LOWP_DTYPE)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with autocast_ctx(device, LOWP_DTYPE):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION (mixed int6 + int8)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain",
    "skip_weight", "skip_weights", "smear",
    "ve_layer_scales", "ve_shared.scale",
    "backout_lambda", "resid_lambdas",
    "attn_gate", "vr_lambda", "dtg_gate",
)

INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 31.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_TOKEN_BYTES = np.dtype("<u2").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}


def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


def load_data_shard(file: Path) -> Tensor:
    num_tokens = _read_num_tokens(file)
    expected_size = _SHARD_HEADER_BYTES + num_tokens * _SHARD_TOKEN_BYTES
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=_SHARD_HEADER_BYTES)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class DistributedTokenLoader:
    """Coprime-stride + multi-shard loader from the public KSV family."""

    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        rank_batch_seqs: list[int] | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.rank_batch_seqs = rank_batch_seqs
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._num_tokens = np.array([_read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cfg: tuple[int, int, int] | None = None
        self._eligible_shards: np.ndarray | None = None
        self._base_block_counts: np.ndarray | None = None
        n = len(self.files)
        self._cursor_phase = np.zeros(n, dtype=np.int64)
        self._cursor_block_count = np.zeros(n, dtype=np.int64)
        self._cursor_next = np.zeros(n, dtype=np.int64)
        self._cursor_start = np.zeros(n, dtype=np.int64)
        self._cursor_stride = np.ones(n, dtype=np.int64)
        self._cursor_init = np.zeros(n, dtype=np.bool_)
        self._batches_built = 0
        self._rank_seq_offsets = None if rank_batch_seqs is None else np.cumsum([0] + rank_batch_seqs[:-1]).astype(np.int64)

    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            stride = int(self._rng.integers(1, n))
            if math.gcd(stride, n) == 1:
                return stride

    def _reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self._num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = (nt - 1 - phase) // seq_len
        self._cursor_phase[si] = phase
        self._cursor_block_count[si] = bc
        self._cursor_next[si] = 0
        self._cursor_start[si] = int(self._rng.integers(bc)) if bc > 1 else 0
        self._cursor_stride[si] = self._pick_coprime_stride(bc)
        self._cursor_init[si] = True

    def _ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self._cursor_init[si] or self._cursor_next[si] >= self._cursor_block_count[si]:
            self._reset_cursor(si, seq_len)

    def _take_from_shard(self, si: int, seq_len: int, count: int, out: list[tuple[int, int]]) -> None:
        remaining = count
        while remaining > 0:
            self._ensure_cursor(si, seq_len)
            bc = int(self._cursor_block_count[si])
            ni = int(self._cursor_next[si])
            take = min(remaining, bc - ni)
            phase = int(self._cursor_phase[si])
            start = int(self._cursor_start[si])
            stride = int(self._cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self._cursor_next[si] = ni + take
            remaining -= take

    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        if self.rank_batch_seqs is None:
            local_tokens = global_tokens // (self.world_size * grad_accum_steps)
            num_seqs = local_tokens // seq_len
            global_num_seqs = num_seqs * self.world_size
        else:
            num_seqs = int(self.rank_batch_seqs[self.rank])
            global_num_seqs = int(sum(self.rank_batch_seqs))
        self._cfg = (seq_len, num_seqs, global_num_seqs)
        bbc = (self._num_tokens - 1) // seq_len
        eligible = bbc > 0
        self._eligible_shards = np.nonzero(eligible)[0].astype(np.int64)
        self._base_block_counts = bbc[self._eligible_shards].astype(np.int64)

    def _sample_global_windows(self) -> list[tuple[int, int]]:
        assert self._cfg is not None and self._eligible_shards is not None and self._base_block_counts is not None
        seq_len, _, global_num_seqs = self._cfg
        eligible_count = int(self._eligible_shards.size)
        progress = min(self._batches_built / 1800.0, 1.0)
        remaining = np.empty(eligible_count, dtype=np.float64)
        for i, si in enumerate(self._eligible_shards.tolist()):
            if self._cursor_init[si]:
                rem = int(self._cursor_block_count[si]) - int(self._cursor_next[si])
                remaining[i] = float(max(rem, 1))
            else:
                remaining[i] = float(self._base_block_counts[i])
        alpha = 0.90 - 0.40 * progress
        weights = np.power(remaining, alpha)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.ones(eligible_count, dtype=np.float64)
            weight_sum = float(weights.sum())
        probs = weights / weight_sum
        low = min(max(8, self.world_size), eligible_count, global_num_seqs)
        high = min(max(32, self.world_size * 8), eligible_count, global_num_seqs)
        mix = max(1, min(int(round(low + progress * (high - low))), eligible_count, global_num_seqs))
        chosen_positions = self._rng.choice(eligible_count, size=mix, replace=False, p=probs)
        chosen_shards = self._eligible_shards[chosen_positions]
        chosen_probs = probs[chosen_positions].copy()
        chosen_probs /= chosen_probs.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = global_num_seqs - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, chosen_probs).astype(np.int64)
        perm = self._rng.permutation(mix)
        chosen_shards, counts = chosen_shards[perm], counts[perm]
        buckets: list[list[tuple[int, int]]] = []
        for si, cnt in zip(chosen_shards.tolist(), counts.tolist()):
            bucket: list[tuple[int, int]] = []
            self._take_from_shard(int(si), seq_len, int(cnt), bucket)
            if bucket:
                if len(bucket) > 1:
                    bucket_perm = self._rng.permutation(len(bucket))
                    bucket = [bucket[int(k)] for k in bucket_perm.tolist()]
                buckets.append(bucket)
        windows: list[tuple[int, int]] = []
        active = [i for i, bucket in enumerate(buckets) if bucket]
        while active:
            order = self._rng.permutation(len(active))
            new_active: list[int] = []
            for order_idx in order.tolist():
                bucket_idx = active[order_idx]
                if buckets[bucket_idx]:
                    windows.append(buckets[bucket_idx].pop())
                if buckets[bucket_idx]:
                    new_active.append(bucket_idx)
            active = new_active
        return windows

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        global_num_seqs = (
            int(sum(self.rank_batch_seqs))
            if self.rank_batch_seqs is not None
            else (global_tokens // (self.world_size * grad_accum_steps * seq_len)) * self.world_size
        )
        desired_cfg = (
            seq_len,
            (
                int(self.rank_batch_seqs[self.rank])
                if self.rank_batch_seqs is not None
                else global_tokens // (self.world_size * grad_accum_steps * seq_len)
            ),
            global_num_seqs,
        )
        if self._cfg != desired_cfg:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        assert self._cfg is not None
        _, num_seqs, _ = self._cfg
        global_windows = self._sample_global_windows()
        if self.rank_batch_seqs is None:
            local_windows = global_windows[self.rank::self.world_size]
        else:
            assert self._rank_seq_offsets is not None
            start = int(self._rank_seq_offsets[self.rank])
            local_windows = global_windows[start:start + num_seqs]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(local_windows):
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self._batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class SyntheticDistributedTokenLoader:
    def __init__(
        self,
        vocab_size: int,
        rank: int,
        world_size: int,
        device: torch.device,
        seed: int,
        rank_batch_seqs: list[int] | None = None,
    ):
        self.vocab_size = vocab_size
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.rank_batch_seqs = rank_batch_seqs
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self.rank_batch_seqs is None:
            local_tokens = global_tokens // (self.world_size * grad_accum_steps)
            local_batch_seqs = local_tokens // seq_len
        else:
            local_batch_seqs = int(self.rank_batch_seqs[self.rank])
        total_tokens = local_batch_seqs * seq_len + 1
        local = torch.randint(
            0,
            self.vocab_size,
            (total_tokens,),
            generator=self.generator,
            dtype=torch.int64,
        )
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device), y.to(self.device)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Compile-stable NTK-aware RoPE with precomputed tables for supported sequence lengths.
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        train_seq_len: int = 1024,
        rope_dims: int = 0,
        supported_seq_lens: tuple[int, ...] = (2048,),
    ):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.supported_seq_lens = tuple(sorted({int(s) for s in supported_seq_lens if int(s) > 0}))
        if not self.supported_seq_lens:
            raise ValueError("supported_seq_lens must be non-empty")
        self._seq_to_idx = {seq_len: idx for idx, seq_len in enumerate(self.supported_seq_lens)}
        max_seq_len = max(self.supported_seq_lens)
        rd = self.rope_dims
        base_inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        cos_tables = []
        sin_tables = []
        for seq_len in self.supported_seq_lens:
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
            else:
                inv_freq = base_inv_freq
            t = torch.arange(seq_len, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            cos_table = torch.zeros(max_seq_len, 1, rd // 2, dtype=torch.float32)
            sin_table = torch.zeros(max_seq_len, 1, rd // 2, dtype=torch.float32)
            cos_table[:seq_len] = freqs.cos().unsqueeze(1)
            sin_table[:seq_len] = freqs.sin().unsqueeze(1)
            cos_tables.append(cos_table)
            sin_tables.append(sin_table)
        self.register_buffer("cos_tables", torch.stack(cos_tables), persistent=False)
        self.register_buffer("sin_tables", torch.stack(sin_tables), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        seq_len = int(seq_len)
        idx = self._seq_to_idx.get(seq_len)
        if idx is None:
            raise ValueError(f"Unsupported seq_len={seq_len}; supported={self.supported_seq_lens}")
        cos = self.cos_tables[idx:idx + 1, :seq_len].to(dtype=dtype)
        sin = self.sin_tables[idx:idx + 1, :seq_len].to(dtype=dtype)
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rd = cos.size(-1) * 2
    if rd < x.size(-1):
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        rope_train_seq_len: int = 1024,
        supported_seq_lens: tuple[int, ...] = (2048,),
        window_size: int = -1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(
            self.head_dim,
            base=rope_base,
            train_seq_len=rope_train_seq_len,
            rope_dims=rope_dims,
            supported_seq_lens=supported_seq_lens,
        )
        self.use_xsa = False
        self.window_size = window_size

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection. y: (B, T, H, D), v: (B, T, Hkv, D)."""
        B, T, H, D = y.shape
        Hkv = v.size(2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(3)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        if self.window_size > 0:
            if not _FA3_AVAILABLE:
                raise RuntimeError(
                    "WINDOW_SIZE>0 requested but flash_attn_interface is unavailable. "
                    "Disable window attention or install the FA3 runtime."
                )
            if x.device.type != "cuda":
                raise RuntimeError("Window attention requires CUDA")
            y = flash_attn_3_func(q, k, v, causal=True, window_size=(self.window_size, 0))
        else:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
            y = y.transpose(1, 2).contiguous()
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers."""
    def __init__(self, vocab_size: int, ve_dim: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, kv_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.proj(self.embed(token_ids))
        return h * self.scale.to(dtype=h.dtype)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        rope_train_seq_len: int = 1024,
        supported_seq_lens: tuple[int, ...] = (2048,),
        layer_idx: int = 0,
        ln_scale: bool = False,
        window_size: int = -1,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            rope_dims=rope_dims,
            rope_train_seq_len=rope_train_seq_len,
            supported_seq_lens=supported_seq_lens,
            window_size=window_size,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        v_embed: Tensor | None = None,
        resid_lambda_attn: Tensor | None = None,
        resid_lambda_mlp: Tensor | None = None,
    ) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale_factor
        attn_out = self.attn(self.attn_norm(x_in) * s, v_embed=v_embed)
        if resid_lambda_attn is not None:
            x_out = resid_lambda_attn * x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x_out) * s)
        if resid_lambda_mlp is not None:
            x_out = resid_lambda_mlp * x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        return x_out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        rope_train_seq_len: int = 1024,
        ln_scale: bool = False,
        ve_dim: int = 0,
        ve_layers: str = "",
        mile_gamma: float = 0.75,
        mile_clamp_min: float = 0.2,
        cache_layer: int = 7,
        backout_init: float = 0.1,
        window_size: int = -1,
        window_attn_layers: str = "",
        supported_seq_lens: tuple[int, ...] = (2048, 6144),
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.register_buffer("mile_gamma_buf", torch.tensor(mile_gamma, dtype=torch.float32))
        self.mile_clamp_min = mile_clamp_min
        self.cache_layer = cache_layer
        self.backout_lambda = nn.Parameter(torch.tensor(backout_init, dtype=torch.float32))
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1 ** 0.5, dtype=torch.float32))
        window_layer_set = (
            {int(x) for x in window_attn_layers.split(",") if x.strip()}
            if window_size > 0 and window_attn_layers
            else set()
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    rope_dims=rope_dims,
                    rope_train_seq_len=rope_train_seq_len,
                    supported_seq_lens=supported_seq_lens,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    window_size=window_size if i in window_layer_set else -1,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        # Value Embedding
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_dim > 0 and ve_layers else []
        kv_dim = num_kv_heads * (model_dim // num_heads)
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if "ve" not in ve_cache:
            ve_cache["ve"] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache["ve"] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache["ve"].dtype)

    def _forward_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        x_cache: Tensor | None = None
        rl = self.resid_lambdas.to(dtype=x.dtype)

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](
                x,
                x0,
                v_embed=self._get_ve(i, input_ids, ve_cache),
                resid_lambda_attn=rl[i, 0],
                resid_lambda_mlp=rl[i, 1],
            )
            if i == self.cache_layer:
                x_cache = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            li = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[li](
                x,
                x0,
                v_embed=self._get_ve(li, input_ids, ve_cache),
                resid_lambda_attn=rl[li, 0],
                resid_lambda_mlp=rl[li, 1],
            )
            if li == self.cache_layer:
                x_cache = x
        if x_cache is not None:
            x = x - self.backout_lambda.to(dtype=x.dtype) * x_cache
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self._forward_hidden(input_ids)
        x = hidden.reshape(-1, hidden.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        logits_f = logits.float()
        if self.training:
            token_losses = F.cross_entropy(logits_f, targets, reduction="none")
            with torch.no_grad():
                probs = torch.softmax(logits_f, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                weights = ((1.0 - torch.exp(-entropy)) ** self.mile_gamma_buf).clamp(min=self.mile_clamp_min)
            return (token_losses * weights).mean()
        return F.cross_entropy(logits_f, targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self._forward_hidden(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# SLIDING WINDOW EVALUATION
# -----------------------------

def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with autocast_ctx(device, LOWP_DTYPE):
                logits = base_model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


# -----------------------------
# TRAINING
# -----------------------------

def apply_cpu_smoke_overrides(args: Hyperparameters) -> None:
    smoke_seq_len = int(os.environ.get("CPU_SMOKE_SEQ_LEN", 128))
    args.run_id = f"{args.run_id}-cpu-smoke"
    args.iterations = int(os.environ.get("CPU_SMOKE_ITERATIONS", 2))
    args.max_wallclock_seconds = float(os.environ.get("CPU_SMOKE_MAX_WALLCLOCK_SECONDS", 30.0))
    args.train_seq_len = smoke_seq_len
    args.eval_seq_len = smoke_seq_len
    args.rope_train_seq_len = min(args.rope_train_seq_len, smoke_seq_len)
    args.val_loss_every = 1
    args.train_log_every = 1
    args.val_batch_size = args.eval_seq_len
    args.train_batch_tokens = args.train_seq_len
    args.warmup_steps = 0
    args.eval_stride = 0
    args.window_size = -1
    args.window_attn_layers = ""


def parse_mixed_seq_config(
    args: Hyperparameters,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
) -> tuple[dict[str, object] | None, str | None]:
    seq_lens_raw = os.environ.get("SEQ_LENS_PER_GPU", "").strip()
    if not seq_lens_raw:
        return None, None
    seq_lens = [int(x) for x in seq_lens_raw.split(",") if x.strip()]
    if len(seq_lens) != world_size:
        raise ValueError(f"Need {world_size} seq lengths in SEQ_LENS_PER_GPU, got {len(seq_lens)}")
    seqs_raw = os.environ.get("SEQS_PER_GPU", "").strip()
    if not seqs_raw:
        raise RuntimeError(
            "07a mixed-seq requires explicit SEQS_PER_GPU. "
            "Speed-weighted auto-allocation is intentionally not landed in this first pass."
        )
    batch_seqs = [int(x) for x in seqs_raw.split(",") if x.strip()]
    if len(batch_seqs) != world_size:
        raise ValueError(f"Need {world_size} seq counts in SEQS_PER_GPU, got {len(batch_seqs)}")
    global_tokens = sum(sl * bs for sl, bs in zip(seq_lens, batch_seqs, strict=True)) * grad_accum_steps
    local_seq_len = seq_lens[rank]
    local_batch_seqs = batch_seqs[rank]
    local_tokens = local_seq_len * local_batch_seqs * grad_accum_steps
    loss_scale = world_size * local_tokens / max(global_tokens, 1)
    parts = [
        f"GPU{gpu_rank}={seq_lens[gpu_rank]}x{batch_seqs[gpu_rank]}="
        f"{seq_lens[gpu_rank] * batch_seqs[gpu_rank] * grad_accum_steps}tok"
        for gpu_rank in range(world_size)
    ]
    msg = f"mixed_seq_len: {' '.join(parts)} total={global_tokens}"
    return {
        "seq_lens": seq_lens,
        "batch_seqs": batch_seqs,
        "local_seq_len": local_seq_len,
        "local_batch_seqs": local_batch_seqs,
        "global_tokens": global_tokens,
        "loss_scale": loss_scale,
    }, msg


def collect_supported_seq_lens(args: Hyperparameters, mixed_cfg: dict[str, object] | None) -> tuple[int, ...]:
    seq_lens = {args.train_seq_len, args.eval_seq_len}
    if mixed_cfg is not None:
        seq_lens.update(int(x) for x in mixed_cfg["seq_lens"])
    return tuple(sorted(seq_len for seq_len in seq_lens if seq_len > 0))


def main() -> None:
    global LOWP_DTYPE

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    cpu_smoke = env_flag("CPU_SMOKE")
    if cpu_smoke:
        apply_cpu_smoke_overrides(args)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    compile_enabled = not cpu_smoke
    if cpu_smoke:
        distributed = False
        rank = 0
        world_size = 1
        local_rank = 0
        grad_accum_steps = 1
        device = torch.device("cpu")
        LOWP_DTYPE = torch.float32
    else:
        distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size <= 0:
            raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
        grad_accum_steps = 8 // world_size
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        LOWP_DTYPE = resolve_amp_dtype(args.amp_dtype)

    grad_scale = 1.0 / grad_accum_steps
    mixed_cfg, mixed_seq_msg = (None, None) if cpu_smoke else parse_mixed_seq_config(args, rank, world_size, grad_accum_steps)
    if mixed_cfg is not None:
        args.train_seq_len = int(mixed_cfg["local_seq_len"])
        args.train_batch_tokens = int(mixed_cfg["global_tokens"])

    if args.window_size > 0 and args.window_attn_layers and not _FA3_AVAILABLE:
        raise RuntimeError(
            "Window attention is enabled but flash_attn_interface is unavailable. "
            "Unset WINDOW_SIZE or run in the FA3 environment."
        )

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    if mixed_seq_msg is not None:
        log0(mixed_seq_msg)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    supported_seq_lens = collect_supported_seq_lens(args, mixed_cfg)
    if cpu_smoke:
        dataset_dir = Path("synthetic://cpu_smoke")
        actual_train_files = 0
        val_tokens = build_cpu_smoke_tokens(args.vocab_size, effective_eval_seq_len, total_sequences=4)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_cpu_smoke_luts(
            args.vocab_size, device
        )
        log0("val_bpb:enabled tokenizer_kind=synthetic cpu_smoke=1")
        log0("train_loader:dataset:synthetic_cpu_smoke train_shards:0")
        log0(f"val_loader:synthetic tokens:{val_tokens.numel() - 1}")
    else:
        if not args.tokenizer_path.endswith(".model"):
            raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        dataset_dir = Path(args.data_path).resolve()
        actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
        val_seq_len = max(supported_seq_lens)
        val_tokens = load_validation_tokens(args.val_files, val_seq_len)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            sp, args.vocab_size, device
        )
        log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
        log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
        log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        rope_train_seq_len=args.rope_train_seq_len,
        ln_scale=args.ln_scale,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        mile_gamma=args.mile_gamma,
        mile_clamp_min=args.mile_clamp_min,
        cache_layer=args.cache_layer,
        backout_init=args.backout_init,
        window_size=args.window_size,
        window_attn_layers=args.window_attn_layers,
        supported_seq_lens=supported_seq_lens,
    ).to(device=device, dtype=LOWP_DTYPE)
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = maybe_compile(base_model, enabled=compile_enabled and device.type == "cuda", fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split: early/late blocks for both matrices and control/scalar params.
    block_named_params = list(base_model.blocks.named_parameters())
    bank_split = args.num_layers // 2
    matrix_params_early: list[Tensor] = []
    matrix_params_late: list[Tensor] = []
    scalar_params_early: list[Tensor] = []
    scalar_params_late: list[Tensor] = []
    for name, p in block_named_params:
        block_idx = int(name.split(".")[0])
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            if block_idx < bank_split:
                matrix_params_early.append(p)
            else:
                matrix_params_late.append(p)
        else:
            if block_idx < bank_split:
                scalar_params_early.append(p)
            else:
                scalar_params_late.append(p)
    if base_model.skip_weights.numel() > 0:
        scalar_params_early.append(base_model.skip_weights)
    scalar_params_early.append(base_model.backout_lambda)
    scalar_params_early.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params_early.append(base_model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params_early.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        matrix_params_early.append(base_model.ve_shared.proj.weight)
        scalar_params_early.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params_early.append(s)

    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=device.type == "cuda",
    )
    optimizer_muon_early = Muon(
        matrix_params_early,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon_early.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_muon_late = Muon(
        matrix_params_late,
        lr=args.matrix_lr_late,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon_late.param_groups:
        group["base_lr"] = args.matrix_lr_late
    optimizer_scalar_early = torch.optim.AdamW(
        [{"params": scalar_params_early, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=device.type == "cuda",
    )
    optimizer_scalar_late = torch.optim.AdamW(
        [{"params": scalar_params_late, "lr": args.scalar_lr_late, "base_lr": args.scalar_lr_late}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=device.type == "cuda",
    )
    resid_lambda_lr = 5.0 * args.scalar_lr
    optimizer_resid_lambdas = torch.optim.AdamW(
        [{"params": [base_model.resid_lambdas], "lr": resid_lambda_lr, "base_lr": resid_lambda_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    optimizers: list[torch.optim.Optimizer] = [
        optimizer_tok,
        optimizer_muon_early,
        optimizer_muon_late,
        optimizer_scalar_early,
        optimizer_scalar_late,
        optimizer_resid_lambdas,
    ]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_wd,
            fused=device.type == "cuda",
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0("anchor:07a_stdtok_ksv2v3_nonbanked")
    log0(
        f"features:layers={args.num_layers} smeargate=1 bigram={args.bigram_vocab_size} "
        f"xsa={args.xsa_last_n} rope_dims={args.rope_dims} ln_scale={args.ln_scale} "
        f"ema={args.ema_decay} ve={args.ve_dim} ve_layers={args.ve_layers} "
        f"leaky_relu_sq=0.5 resid_lambdas=1 cache_layer={args.cache_layer}"
    )
    log0(f"weight_decay:muon_wd={args.muon_wd} adam_wd={args.adam_wd}")
    log0(f"export:mixed_int6+{_EXPORT_COMPRESSOR} eval_stride:{args.eval_stride}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    bf16_supported = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
    log0(f"amp_dtype:{amp_dtype_name(LOWP_DTYPE)} bf16_supported:{bf16_supported}")
    if device.type == "cuda":
        log0("sdp_backends:flash=True mem_efficient=False math=False")
    else:
        log0("sdp_backends:flash=False mem_efficient=False math=False cpu_smoke=1")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"window_size:{args.window_size} window_layers:{args.window_attn_layers or 'none'}"
    )
    log0(f"eval_also_stride64:{args.eval_also_stride64}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr}/{args.matrix_lr_late} "
        f"scalar_lr:{args.scalar_lr}/{args.scalar_lr_late} resid_lambda_lr:{resid_lambda_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # EMA STATE
    # -----------------------------

    ema_state: dict[str, Tensor] = {
        name: t.detach().float().clone() for name, t in base_model.state_dict().items()
    }
    log0(f"ema:enabled decay={args.ema_decay}")

    # Save VE init snapshot for smoke-test gradient validation
    if os.environ.get("SMOKE_SAVE_VE_INIT") == "1" and master_process and base_model.ve_shared is not None:
        ve_init = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
                   if "ve_shared" in k or "ve_layer_scales" in k}
        torch.save(ve_init, "ve_init_snapshot.pt")
        log0(f"smoke:saved VE init snapshot ({len(ve_init)} keys)")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    rank_batch_seqs = None if mixed_cfg is None else list(mixed_cfg["batch_seqs"])
    if cpu_smoke:
        train_loader = SyntheticDistributedTokenLoader(
            args.vocab_size, rank, world_size, device, seed=args.seed, rank_batch_seqs=rank_batch_seqs
        )
    else:
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device, rank_batch_seqs=rank_batch_seqs
        )
    local_loss_scale = 1.0 if mixed_cfg is None else float(mixed_cfg["loss_scale"])

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with autocast_ctx(device, LOWP_DTYPE):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale * local_loss_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        if cpu_smoke:
            train_loader = SyntheticDistributedTokenLoader(
                args.vocab_size, rank, world_size, device, seed=args.seed, rank_batch_seqs=rank_batch_seqs
            )
        else:
            train_loader = DistributedTokenLoader(
                args.train_files, rank, world_size, device, rank_batch_seqs=rank_batch_seqs
            )
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    sync_device(device)
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            sync_device(device)
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                eval_seq_len=effective_eval_seq_len,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            sync_device(device)
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with autocast_ctx(device, LOWP_DTYPE):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale * local_loss_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        if max_wallclock_ms is not None and max_wallclock_ms > 0 and args.mile_gamma > 0:
            progress = min(elapsed_ms / max_wallclock_ms, 1.0)
            peak = args.mile_peak_frac
            if progress < peak:
                cur_gamma = args.mile_gamma * (progress / peak)
            else:
                cur_gamma = args.mile_gamma * (1.0 - (progress - peak) / (1.0 - peak))
            base_model.mile_gamma_buf.fill_(max(cur_gamma, 0.0))
        for group in optimizer_muon_early.param_groups:
            group["momentum"] = muon_momentum
        for group in optimizer_muon_late.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update
        with torch.no_grad():
            d = args.ema_decay
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    peak_alloc_mib, peak_reserved_mib = max_memory_mib(device)
    log0(f"peak memory allocated: {peak_alloc_mib} MiB reserved: {peak_reserved_mib} MiB")

    # -----------------------------
    # APPLY EMA WEIGHTS
    # -----------------------------

    log0("ema:applying EMA weights")
    avg_state = {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in ema_state.items()}
    del ema_state
    base_model.load_state_dict(avg_state, strict=True)
    del avg_state

    # Pre-quant EMA eval
    sync_device(device)
    t_pre = time.perf_counter()
    pre_val_loss, pre_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    sync_device(device)
    log0(
        f"pre_quant_ema val_loss:{pre_val_loss:.4f} val_bpb:{pre_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_pre):.0f}ms"
    )
    log0(f"pre_quant_ema_exact val_loss:{pre_val_loss:.8f} val_bpb:{pre_val_bpb:.8f}")

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION (mixed int6 + custom-pack + brotli-10)
    # -----------------------------

    if not _BROTLI_AVAILABLE:
        raise RuntimeError("brotli is required for 07a export. Install it in the runtime environment.")

    code_bytes = len(code.encode("utf-8"))
    model_bytes = 0
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu().contiguous() for k, v in base_model.state_dict().items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    json_meta = {k: v if isinstance(v, dict) else str(v) for k, v in quant_meta.items()}
    packed = _custom_pack(quant_result, json_meta, shuffle=True)
    quant_blob = brotli.compress(packed, quality=10)
    quant_file_bytes = len(quant_blob)
    total_bytes = code_bytes + quant_file_bytes

    if master_process:
        with open("final_model.int6.br", "wb") as f:
            f.write(quant_blob)
        log0(f"Serialized model int6+brotli10: {quant_file_bytes} bytes")
        log0(f"bytes_code:{code_bytes}")
        log0(f"bytes_model_int6_brotli:{quant_file_bytes}")
        log0(f"bytes_total:{total_bytes}")
        log0(f"Total submission size int6+brotli10: {total_bytes} bytes")
        if total_bytes > 16_000_000:
            log0(f"WARNING: total submission size {total_bytes} exceeds 16,000,000 byte cap!")

    # Roundtrip: decompress + dequantize into fresh eval model
    if distributed:
        dist.barrier()
    with open("final_model.int6.br", "rb") as f:
        quant_blob_disk = f.read()
    quant_decompressed = brotli.decompress(quant_blob_disk)
    quant_result_rt, quant_meta_rt = _custom_unpack(quant_decompressed, shuffle=True)
    deq_state = dequantize_mixed_int6(quant_result_rt, quant_meta_rt, sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        rope_train_seq_len=args.rope_train_seq_len, ln_scale=args.ln_scale,
        ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        mile_gamma=args.mile_gamma, mile_clamp_min=args.mile_clamp_min,
        cache_layer=args.cache_layer, backout_init=args.backout_init,
        window_size=args.window_size, window_attn_layers=args.window_attn_layers,
        supported_seq_lens=supported_seq_lens,
    ).to(device=device, dtype=LOWP_DTYPE)
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    # Standard non-overlapping roundtrip eval
    sync_device(device)
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, eval_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    sync_device(device)
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval (submission score)
    sw_seq_len = effective_eval_seq_len
    sw_val_loss = None
    sw_val_bpb = None
    sw64_val_loss = None
    sw64_val_bpb = None
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        sync_device(device)
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        sync_device(device)
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        if args.eval_stride == 64:
            sw64_val_loss, sw64_val_bpb = sw_val_loss, sw_val_bpb
            log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    if args.eval_also_stride64 and args.eval_stride > 0 and args.eval_stride != 64 and 64 < sw_seq_len:
        sync_device(device)
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        sync_device(device)
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    if master_process:
        append_results_log(
            {
                "artifact_bytes_total": total_bytes,
                "bytes_code": code_bytes,
                "bytes_model_fp": model_bytes,
                "bytes_model_int6_brotli": quant_file_bytes,
                "device_type": device.type,
                "eval_stride": args.eval_stride,
                "experiment": "session_07a_stdtok_ksv2v3_nonbanked",
                "final_int6_roundtrip_val_bpb": q_val_bpb,
                "final_int6_roundtrip_val_loss": q_val_loss,
                "final_int6_sliding_window_s64_val_bpb": sw64_val_bpb,
                "final_int6_sliding_window_s64_val_loss": sw64_val_loss,
                "final_int6_sliding_window_val_bpb": sw_val_bpb,
                "final_int6_sliding_window_val_loss": sw_val_loss,
                "iterations_requested": args.iterations,
                "mode": "cpu_smoke" if cpu_smoke else "train",
                "peak_memory_allocated_mib": peak_alloc_mib,
                "peak_memory_reserved_mib": peak_reserved_mib,
                "pre_quant_ema_val_bpb": pre_val_bpb,
                "pre_quant_ema_val_loss": pre_val_loss,
                "results_log_version": 1,
                "run_id": args.run_id,
                "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "seed": args.seed,
                "step_avg_ms": round(training_time_ms / max(step, 1), 4),
                "steps_completed": step,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_seq_len": args.train_seq_len,
                "eval_seq_len": effective_eval_seq_len,
                "world_size": world_size,
            }
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
