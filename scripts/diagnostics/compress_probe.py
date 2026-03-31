#!/usr/bin/env python3
"""Offline compression-path feasibility probe.

Tests brotli + byte-shuffle vs current zstd pipeline on existing
quantized model artifacts. Verifies full roundtrip correctness.

Usage:
  python scripts/diagnostics/compress_probe.py <path_to_int6_ptz>

  # Example on Pegasus:
  python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz

Reports compressed sizes for multiple strategies and verifies
that every strategy round-trips to bit-identical tensors.
"""

import argparse
import io
import json
import struct
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional compressors — probe degrades gracefully
# ---------------------------------------------------------------------------
try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


# ---------------------------------------------------------------------------
# Byte-shuffle: reorder multi-byte elements for better entropy coding
# ---------------------------------------------------------------------------
def byte_shuffle(data: bytes, elem_size: int) -> bytes:
    """Transpose byte matrix: group all byte-0s, then byte-1s, etc."""
    if elem_size <= 1:
        return data
    n = len(data) // elem_size
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, elem_size)
    return arr.T.copy().tobytes()


def byte_unshuffle(data: bytes, elem_size: int, n_elements: int) -> bytes:
    """Inverse of byte_shuffle."""
    if elem_size <= 1:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(elem_size, n_elements)
    return arr.T.copy().tobytes()


# ---------------------------------------------------------------------------
# Custom binary serialization (no pickle/ZIP overhead)
# ---------------------------------------------------------------------------
DTYPE_TO_STR = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "bool",
}
STR_TO_DTYPE = {v: k for k, v in DTYPE_TO_STR.items()}
DTYPE_ELEM_SIZE = {
    "f32": 4, "f16": 2, "bf16": 2,
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "bool": 1,
}


def custom_pack(state_dict: dict[str, torch.Tensor], meta: dict,
                shuffle: bool = True) -> bytes:
    """Pack tensors into a compact binary format with optional byte-shuffle.

    Format:
      [4 bytes] header_len (uint32 LE)
      [header_len bytes] JSON header
      [remaining bytes] concatenated raw tensor data
    """
    header_entries = {}
    chunks = []
    offset = 0

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        dtype_str = DTYPE_TO_STR.get(tensor.dtype)
        if dtype_str is None:
            raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")

        raw = tensor.numpy().tobytes() if tensor.dtype != torch.bfloat16 else tensor.float().numpy().tobytes()
        elem_size = DTYPE_ELEM_SIZE[dtype_str]
        if shuffle and elem_size > 1:
            raw = byte_shuffle(raw, elem_size)

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


def custom_unpack(blob: bytes, shuffle: bool = True) -> tuple[dict[str, torch.Tensor], dict]:
    """Unpack custom binary format back to tensors."""
    header_len = struct.unpack("<I", blob[:4])[0]
    header = json.loads(blob[4:4 + header_len])
    data_start = 4 + header_len

    state_dict = {}
    for name, info in header["t"].items():
        raw = blob[data_start + info["o"]:data_start + info["o"] + info["n"]]
        dtype_str = info["d"]
        elem_size = DTYPE_ELEM_SIZE[dtype_str]

        if shuffle and elem_size > 1:
            raw = byte_unshuffle(raw, elem_size, info["e"])

        dtype = STR_TO_DTYPE[dtype_str]
        if dtype == torch.bfloat16:
            tensor = torch.from_numpy(np.frombuffer(bytearray(raw), dtype=np.float32).copy()).to(torch.bfloat16).reshape(info["s"])
        else:
            np_dtype = {
                "f32": np.float32, "f16": np.float16,
                "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
                "bool": np.bool_,
            }[dtype_str]
            tensor = torch.from_numpy(np.frombuffer(bytearray(raw), dtype=np_dtype).copy()).reshape(info["s"])
        state_dict[name] = tensor

    return state_dict, header["m"]


# ---------------------------------------------------------------------------
# Compression wrappers
# ---------------------------------------------------------------------------
def compress(data: bytes, method: str, level: int) -> bytes:
    if method == "zstd":
        return zstandard.ZstdCompressor(level=level).compress(data)
    elif method == "brotli":
        return brotli.compress(data, quality=level)
    else:
        raise ValueError(f"Unknown method: {method}")


def decompress(data: bytes, method: str) -> bytes:
    if method == "zstd":
        return zstandard.ZstdDecompressor().decompress(data)
    elif method == "brotli":
        return brotli.decompress(data)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Roundtrip verification
# ---------------------------------------------------------------------------
def verify_roundtrip(original: dict[str, torch.Tensor],
                     recovered: dict[str, torch.Tensor]) -> bool:
    if set(original.keys()) != set(recovered.keys()):
        missing = set(original.keys()) - set(recovered.keys())
        extra = set(recovered.keys()) - set(original.keys())
        print(f"  KEY MISMATCH: missing={missing}, extra={extra}")
        return False

    all_ok = True
    for name in sorted(original.keys()):
        a, b = original[name], recovered[name]
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH: {name} {a.shape} vs {b.shape}")
            all_ok = False
            continue
        if a.dtype != b.dtype:
            print(f"  DTYPE MISMATCH: {name} {a.dtype} vs {b.dtype}")
            all_ok = False
            continue
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            print(f"  VALUE MISMATCH: {name} max_diff={diff.max().item():.6e} "
                  f"mean_diff={diff.mean().item():.6e}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def analyze_tensor_composition(state_dict: dict[str, torch.Tensor], meta: dict):
    """Break down byte composition by tensor category."""
    categories = {"int8_quant": 0, "fp16_scale": 0, "fp32_ctrl": 0,
                  "fp16_pass": 0, "other": 0}
    for name, tensor in state_dict.items():
        nbytes = tensor.numel() * tensor.element_size()
        if name.endswith(".q"):
            categories["int8_quant"] += nbytes
        elif name.endswith(".scale"):
            categories["fp16_scale"] += nbytes
        elif tensor.dtype == torch.float32:
            categories["fp32_ctrl"] += nbytes
        elif tensor.dtype == torch.float16:
            categories["fp16_pass"] += nbytes
        else:
            categories["other"] += nbytes

    total = sum(categories.values())
    print("\n=== Tensor composition (raw uncompressed) ===")
    for cat, nbytes in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100.0 * nbytes / total if total else 0
        print(f"  {cat:15s}: {nbytes:>12,} bytes ({pct:5.1f}%)")
    print(f"  {'TOTAL':15s}: {total:>12,} bytes")
    return total


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compression-path feasibility probe")
    parser.add_argument("artifact", help="Path to .int6.ptz file")
    parser.add_argument("--code-bytes", type=int, default=69000,
                        help="Approximate code size for cap calculation (default: 69000)")
    parser.add_argument("--model-dim", type=int, default=512,
                        help="Model dimension (default: 512)")
    parser.add_argument("--mlp-mult", type=float, default=3.0,
                        help="Current MLP multiplier (default: 3.0)")
    args = parser.parse_args()

    cap = 16_000_000

    # Load existing artifact
    print(f"Loading: {args.artifact}")
    with open(args.artifact, "rb") as f:
        raw_blob = f.read()
    baseline_size = len(raw_blob)
    print(f"Baseline file size (zstd): {baseline_size:,} bytes")
    print(f"Baseline total (+ {args.code_bytes} code): {baseline_size + args.code_bytes:,} bytes")
    headroom = cap - (baseline_size + args.code_bytes)
    print(f"Headroom vs 16MB cap: {headroom:+,} bytes")

    # Decompress original
    print("\nDecompressing original (zstd)...")
    t0 = time.perf_counter()
    if HAS_ZSTD:
        raw_pickle = zstandard.ZstdDecompressor().decompress(raw_blob)
    else:
        print("ERROR: zstandard not available, cannot decompress .int6.ptz")
        sys.exit(1)
    t_decompress_zstd = time.perf_counter() - t0
    print(f"  Decompressed: {len(raw_pickle):,} bytes in {t_decompress_zstd:.2f}s")

    # Load torch state
    quant_state = torch.load(io.BytesIO(raw_pickle), map_location="cpu", weights_only=False)
    tensors = quant_state["w"]
    meta = quant_state["m"]
    print(f"  Loaded {len(tensors)} tensor entries, {len(meta)} meta entries")

    # Analyze composition
    raw_total = analyze_tensor_composition(tensors, meta)

    # Convert meta for JSON serialization (handle non-string values)
    json_meta = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            json_meta[k] = v
        else:
            json_meta[k] = str(v)

    # --- Run compression strategies ---
    results = []

    # Strategy 1: Baseline (existing zstd, already measured)
    results.append({
        "name": "baseline (torch.save + zstd-22)",
        "compressed": baseline_size,
        "compress_time": 0.0,
        "decompress_time": t_decompress_zstd,
        "roundtrip_ok": True,
    })

    # Strategy 2: torch.save + brotli (replace compressor only)
    if HAS_BROTLI:
        for bq in [9, 10, 11]:
            t0 = time.perf_counter()
            blob = brotli.compress(raw_pickle, quality=bq)
            ct = time.perf_counter() - t0
            t0 = time.perf_counter()
            recovered_pickle = brotli.decompress(blob)
            dt = time.perf_counter() - t0
            ok = (recovered_pickle == raw_pickle)
            results.append({
                "name": f"torch.save + brotli-{bq}",
                "compressed": len(blob),
                "compress_time": ct,
                "decompress_time": dt,
                "roundtrip_ok": ok,
            })

    # Strategy 3: Custom pack (no shuffle) + various compressors
    print("\nPacking custom format (no shuffle)...")
    t0 = time.perf_counter()
    packed_noshuffle = custom_pack(tensors, json_meta, shuffle=False)
    pack_time = time.perf_counter() - t0
    print(f"  Packed (no shuffle): {len(packed_noshuffle):,} bytes in {pack_time:.2f}s")

    for method, levels in [("zstd", [19, 22]), ("brotli", [9, 10, 11])]:
        if method == "zstd" and not HAS_ZSTD:
            continue
        if method == "brotli" and not HAS_BROTLI:
            continue
        for level in levels:
            t0 = time.perf_counter()
            blob = compress(packed_noshuffle, method, level)
            ct = time.perf_counter() - t0
            t0 = time.perf_counter()
            recovered_raw = decompress(blob, method)
            dt = time.perf_counter() - t0
            recovered_tensors, recovered_meta = custom_unpack(recovered_raw, shuffle=False)
            ok = verify_roundtrip(tensors, recovered_tensors)
            results.append({
                "name": f"custom-noshuffle + {method}-{level}",
                "compressed": len(blob),
                "compress_time": ct + pack_time,
                "decompress_time": dt,
                "roundtrip_ok": ok,
            })

    # Strategy 4: Custom pack (with byte-shuffle) + various compressors
    print("\nPacking custom format (with byte-shuffle)...")
    t0 = time.perf_counter()
    packed_shuffle = custom_pack(tensors, json_meta, shuffle=True)
    pack_time_s = time.perf_counter() - t0
    print(f"  Packed (shuffled): {len(packed_shuffle):,} bytes in {pack_time_s:.2f}s")

    for method, levels in [("zstd", [19, 22]), ("brotli", [9, 10, 11])]:
        if method == "zstd" and not HAS_ZSTD:
            continue
        if method == "brotli" and not HAS_BROTLI:
            continue
        for level in levels:
            t0 = time.perf_counter()
            blob = compress(packed_shuffle, method, level)
            ct = time.perf_counter() - t0
            t0 = time.perf_counter()
            recovered_raw = decompress(blob, method)
            dt = time.perf_counter() - t0
            recovered_tensors, recovered_meta = custom_unpack(recovered_raw, shuffle=True)
            ok = verify_roundtrip(tensors, recovered_tensors)
            results.append({
                "name": f"custom-shuffle + {method}-{level}",
                "compressed": len(blob),
                "compress_time": ct + pack_time_s,
                "decompress_time": dt,
                "roundtrip_ok": ok,
            })

    # --- Report ---
    print("\n" + "=" * 90)
    print(f"{'Strategy':<40s} {'Compressed':>12s} {'vs baseline':>12s} {'Total+code':>12s} "
          f"{'Headroom':>10s} {'OK':>4s}")
    print("=" * 90)

    for r in results:
        delta = r["compressed"] - baseline_size
        total = r["compressed"] + args.code_bytes
        headroom = cap - total
        status = "pass" if r["roundtrip_ok"] else "FAIL"
        marker = " ***" if headroom > 0 and r["name"] != results[0]["name"] else ""
        print(f"  {r['name']:<38s} {r['compressed']:>12,} {delta:>+12,} "
              f"{total:>12,} {headroom:>+10,} {status:>4s}{marker}")

    print("=" * 90)

    # Timing summary
    print(f"\n{'Strategy':<40s} {'Compress':>10s} {'Decompress':>10s}")
    print("-" * 62)
    for r in results:
        if r["compress_time"] > 0:
            print(f"  {r['name']:<38s} {r['compress_time']:>9.2f}s {r['decompress_time']:>9.2f}s")

    # Savings summary
    best = min(results, key=lambda r: r["compressed"])
    savings = baseline_size - best["compressed"]
    print(f"\nBest strategy: {best['name']}")
    print(f"  Savings vs baseline: {savings:,} bytes ({100*savings/baseline_size:.1f}%)")
    print(f"  Total with code: {best['compressed'] + args.code_bytes:,} bytes")
    print(f"  Headroom vs 16MB: {cap - best['compressed'] - args.code_bytes:+,} bytes")

    # Byte-shuffle contribution
    noshuffle_best = min((r for r in results if "noshuffle" in r["name"]), key=lambda r: r["compressed"], default=None)
    shuffle_best = min((r for r in results if "shuffle" in r["name"] and "noshuffle" not in r["name"]), key=lambda r: r["compressed"], default=None)
    if noshuffle_best and shuffle_best:
        shuffle_gain = noshuffle_best["compressed"] - shuffle_best["compressed"]
        print(f"\nByte-shuffle contribution: {shuffle_gain:,} bytes "
              f"({100*shuffle_gain/noshuffle_best['compressed']:.1f}% of custom-noshuffle best)")

    # Wider MLP simulation — accurate, scales only MLP tensors
    print("\n=== Wider MLP headroom simulation ===")
    if best["roundtrip_ok"]:
        # Identify MLP tensors and their raw bytes
        mlp_raw = 0
        non_mlp_raw = 0
        num_mlp_layers = 0
        mlp_layer_indices = set()
        for name, tensor in tensors.items():
            nbytes = tensor.numel() * tensor.element_size()
            if ".mlp." in name:
                mlp_raw += nbytes
                # Extract layer index to count MLP layers
                for part in name.split("."):
                    if part.isdigit():
                        mlp_layer_indices.add(int(part))
                        break
            else:
                non_mlp_raw += nbytes
        num_mlp_layers = len(mlp_layer_indices) if mlp_layer_indices else 11

        # Architecture params (from train_gpt.py args block)
        current_mlp_mult = args.mlp_mult
        model_dim = args.model_dim

        current_hidden = int(current_mlp_mult * model_dim)
        # Per-layer MLP raw: fc.q(hidden,dim) + fc.scale(hidden)*2 + proj.q(dim,hidden) + proj.scale(dim)*2
        per_layer_expected = (current_hidden * model_dim  # fc.q int8
                              + current_hidden * 2         # fc.scale fp16
                              + model_dim * current_hidden  # proj.q int8
                              + model_dim * 2)              # proj.scale fp16
        total_mlp_expected = per_layer_expected * num_mlp_layers

        cr = best["compressed"] / raw_total if raw_total > 0 else 1.0

        print(f"  Architecture: model_dim={model_dim}, current mlp_mult={current_mlp_mult}, "
              f"hidden={current_hidden}, {num_mlp_layers} MLP layers")
        print(f"  MLP raw bytes: {mlp_raw:,} (measured) / {total_mlp_expected:,} (expected)")
        print(f"  Non-MLP raw bytes: {non_mlp_raw:,}")
        print(f"  Overall compression ratio: {cr:.4f}")
        print()

        for target_mult in [3.1, 3.15, 3.2, 3.25, 3.5, 4.0]:
            target_hidden = int(target_mult * model_dim)
            new_per_layer = (target_hidden * model_dim + target_hidden * 2
                             + model_dim * target_hidden + model_dim * 2)
            new_mlp_raw = new_per_layer * num_mlp_layers
            extra_raw = new_mlp_raw - mlp_raw
            # Estimate: non-MLP bytes stay the same, MLP bytes change
            new_total_raw = non_mlp_raw + new_mlp_raw
            new_compressed_est = new_total_raw * cr
            new_total = new_compressed_est + args.code_bytes
            headroom = cap - new_total
            fits = "FITS" if headroom > 0 else "OVER"
            print(f"  MLP {target_mult:.2f}x (hidden={target_hidden}): "
                  f"extra_raw={extra_raw/1e6:+.2f}MB, "
                  f"est_compressed={new_compressed_est/1e6:.2f}MB, "
                  f"total={new_total/1e6:.2f}MB, "
                  f"headroom={headroom/1e6:+.3f}MB [{fits}]")


if __name__ == "__main__":
    main()
