import io
import json
import math
import re
import sys
from collections import defaultdict

import torch

try:
    import zstandard

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import zlib

    ZLIB_AVAILABLE = True
except ImportError:
    ZLIB_AVAILABLE = False


OUTPUT_PATH = "diagnostics_out.txt"
JSON_PATH = "diagnostics_out.json"
TOP_K = 15

BLOCK_GROUPS = [(0, 3), (4, 8), (9, 10)]
LATE_BLOCK_PREFIXES = ("blocks.9.", "blocks.10.", "ve_", "bigram.")


def block_group_label(name):
    """Extract block group label from tensor name, or None if not block-indexed."""
    m = re.match(r"blocks\.(\d+)\.", name)
    if not m:
        return None
    idx = int(m.group(1))
    for lo, hi in BLOCK_GROUPS:
        if lo <= idx <= hi:
            return f"blocks.{lo}-{hi}"
    return f"blocks.{idx}"


def is_late_block_tensor(name):
    return any(name.startswith(p) for p in LATE_BLOCK_PREFIXES)


def naive_dequantize_mixed_int6(result, meta):
    out = {}
    for name, info in meta.items():
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            out[name] = result[name].float()
            continue

        q_name = name + ".q"
        scale_name = name + ".scale"
        if q_name not in result or scale_name not in result:
            continue

        q = result[q_name]
        scale = result[scale_name]
        if scale.ndim > 0:
            view_shape = [q.shape[0]] + [1] * (q.ndim - 1)
            out[name] = q.float() * scale.float().view(*view_shape)
        else:
            out[name] = q.float() * float(scale.item())
    return out


def decompress_ptz_blob(blob):
    if ZSTD_AVAILABLE:
        try:
            return zstandard.ZstdDecompressor().decompress(blob)
        except zstandard.ZstdError:
            pass

    if ZLIB_AVAILABLE:
        try:
            return zlib.decompress(blob)
        except zlib.error:
            pass

    available = []
    if ZSTD_AVAILABLE:
        available.append("zstandard")
    if ZLIB_AVAILABLE:
        available.append("zlib")
    available_str = ", ".join(available) if available else "none"
    raise RuntimeError(
        f"Unable to decompress .ptz blob with available backends: {available_str}"
    )


def load_model_weights(model_path):
    print(f"Loading {model_path}...")
    if model_path.endswith(".ptz"):
        with open(model_path, "rb") as f:
            quant_blob_disk = f.read()
        quant_decompressed = decompress_ptz_blob(quant_blob_disk)
        print("Decompressed! Loading quantized dict...")
        quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
        return naive_dequantize_mixed_int6(quant_state["w"], quant_state["m"])

    state_dict = torch.load(model_path, map_location="cpu")
    if "w" in state_dict and "m" in state_dict:
        return naive_dequantize_mixed_int6(state_dict["w"], state_dict["m"])
    return state_dict


def tensor_family(name):
    if name.startswith("bigram."):
        return "bigram"
    if name.startswith("ve_") or name.startswith("ve_shared."):
        return "ve"
    if name.startswith("smear."):
        return "smear"
    if name.endswith("resid_mix"):
        return "resid_mix"
    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if name.endswith("_scale") or name.endswith(".scale") or name.endswith(".q_gain"):
        return "scales"
    if "embed.weight" in name or "lm_head" in name:
        return "embeddings"
    return "other"


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def analyze_tensor(name, tensor):
    t = tensor.float().cpu()
    t_abs = t.abs()
    numel = t.numel()
    if numel == 0:
        return {
            "name": name,
            "shape": tuple(t.shape),
            "numel": 0,
            "family": tensor_family(name),
            "empty": True,
        }

    abs_mean = t_abs.mean().item()
    abs_std = t_abs.std().item() if numel > 1 else 0.0
    outlier_threshold = abs_mean + 3.0 * abs_std
    outlier_count = (t_abs > outlier_threshold).sum().item() if abs_std > 0 else 0

    return {
        "name": name,
        "shape": tuple(t.shape),
        "numel": numel,
        "family": tensor_family(name),
        "empty": False,
        "mean": t.mean().item(),
        "std": t.std().item() if numel > 1 else 0.0,
        "min": t.min().item(),
        "max": t.max().item(),
        "max_mag": t_abs.max().item(),
        "mean_abs": abs_mean,
        "l2_norm": torch.linalg.norm(t).item(),
        "sparsity": (t == 0.0).sum().item() / numel,
        "outlier_threshold": outlier_threshold,
        "outlier_count": outlier_count,
        "outlier_frac": outlier_count / numel,
    }


def format_single_tensor(stats):
    if stats["empty"]:
        return f"--- {stats['name']} ---\n  EMPTY TENSOR"

    shape_str = "x".join(str(d) for d in stats["shape"])
    return "\n".join(
        [
            f"--- {stats['name']} [{shape_str}] ---",
            f"  Mean:      {stats['mean']:10.6f}     StdDev:     {stats['std']:10.6f}",
            f"  Min:       {stats['min']:10.6f}     Max:        {stats['max']:10.6f}",
            f"  Max Mag:   {stats['max_mag']:10.6f}     L2 Norm:    {stats['l2_norm']:10.6f}",
            (
                f"  Sparsity:  {stats['sparsity']:10.2%}     "
                f"Outlier >3σ:{stats['outlier_frac']:10.2%} ({stats['outlier_count']})"
            ),
        ]
    )


def summarize_single_model(all_stats, state_dict):
    families = build_single_family_summary(all_stats)

    lines = [
        "=" * 60,
        "FAMILY SUMMARY",
        "=" * 60,
        "",
    ]
    for family in sorted(families):
        bucket = families[family]
        lines.append(
            (
                f"{family:12s} tensors={bucket['tensors']:3d} "
                f"numel={bucket['numel']:10d} "
                f"mean_sparsity={bucket['mean_sparsity']:7.2%} "
                f"mean_outlier={bucket['mean_outlier']:7.2%} "
                f"max_mag={bucket['max_mag']:.6f} "
                f"max_norm_tensor={bucket['max_norm_name']}"
            )
        )

    lines.extend(["", "=" * 60, "COMPONENT ANALYSIS", "=" * 60, ""])
    component_analysis = build_component_analysis(state_dict)
    smear = component_analysis.get("smeargate")
    if smear is not None:
        lines.extend(
            [
                "[SMEARGATE ANALYSIS]",
                f"  Mean gate pass-through (1-g): {smear['mean_pass_through']:.4f}",
                f"  Max gate block: {smear['max_gate']:.4f}",
                f"  Min gate block: {smear['min_gate']:.4f}",
                "",
            ]
        )

    ve_scales = component_analysis.get("ve_layer_scales", {})
    if ve_scales:
        lines.append("[VE LAYER SCALES ANALYSIS]")
        for name, value in ve_scales.items():
            lines.append(f"  {name}: {value:.6f}")
        shared_scale = component_analysis.get("ve_shared_scale")
        if shared_scale is not None:
            lines.append(f"  ve_shared.scale: {shared_scale:.6f}")
        lines.append("")

    bigram_scale = component_analysis.get("bigram_scale")
    if bigram_scale is not None:
        lines.extend(
            [
                "[BIGRAM HASH]",
                f"  bigram.scale: {bigram_scale:.6f}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def compare_tensors(name, tensor_a, tensor_b):
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        return {
            "name": name,
            "family": tensor_family(name),
            "status": "shape_mismatch",
            "shape_a": tuple(tensor_a.shape),
            "shape_b": tuple(tensor_b.shape),
        }

    a = tensor_a.float().cpu().reshape(-1)
    b = tensor_b.float().cpu().reshape(-1)
    if a.numel() == 0:
        return {
            "name": name,
            "family": tensor_family(name),
            "status": "empty",
            "shape_a": tuple(tensor_a.shape),
            "shape_b": tuple(tensor_b.shape),
        }

    delta = a - b
    mse = (delta.square().mean()).item()
    rmse = math.sqrt(mse)
    norm_a = torch.linalg.norm(a).item()
    norm_b = torch.linalg.norm(b).item()
    rms_a = math.sqrt((a.square().mean()).item())
    max_abs_a = a.abs().max().item()
    max_abs_b = b.abs().max().item()
    peak = max(max_abs_a, max_abs_b)
    dot = (a * b).sum().item()
    cosine = dot / ((norm_a * norm_b) + 1e-12)
    rel_norm_error = abs(norm_a - norm_b) / (norm_a + 1e-12)
    rel_rmse = rmse / (rms_a + 1e-12)
    max_abs_delta = delta.abs().max().item()
    psnr = 10.0 * math.log10((peak * peak) / mse) if mse > 1e-12 and peak > 0.0 else float("inf")

    # --- quantization-specific metrics ---

    # sign flips: weights that changed sign (disproportionately damaging)
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)
    sign_flip_count = int(((sign_a != sign_b) & (sign_a != 0) & (sign_b != 0)).sum().item())
    sign_flip_frac = sign_flip_count / a.numel()

    # directional bias: mean(delta)/mean(|delta|), near 0 = symmetric, ±1 = biased
    mean_delta = delta.mean().item()
    mean_abs_delta_val = delta.abs().mean().item()
    directional_bias = mean_delta / (mean_abs_delta_val + 1e-12)

    # zero fraction change
    zero_frac_a = (a == 0.0).float().mean().item()
    zero_frac_b = (b == 0.0).float().mean().item()
    zero_frac_increase = zero_frac_b - zero_frac_a

    # per-row MSE concentration (only for 2D+ tensors)
    row_concentration = None
    worst_row_mse = None
    if tensor_a.ndim >= 2:
        rows = tensor_a.shape[0]
        delta_2d = delta.view(rows, -1)
        row_mse = delta_2d.square().mean(dim=1)
        total_row_sq = row_mse.sum().item()
        if total_row_sq > 1e-12 and rows >= 10:
            sorted_row_mse, _ = row_mse.sort(descending=True)
            top10pct = max(1, rows // 10)
            top10pct_frac = sorted_row_mse[:top10pct].sum().item() / total_row_sq
            row_concentration = top10pct_frac
        worst_row_mse = row_mse.max().item()

    return {
        "name": name,
        "family": tensor_family(name),
        "status": "ok",
        "shape_a": tuple(tensor_a.shape),
        "shape_b": tuple(tensor_b.shape),
        "numel": a.numel(),
        "mse": mse,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
        "cosine_drop": 1.0 - cosine,
        "rel_norm_error": rel_norm_error,
        "max_abs_delta": max_abs_delta,
        "psnr": psnr,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "sum_sq_error": delta.square().sum().item(),
        "sign_flip_count": sign_flip_count,
        "sign_flip_frac": sign_flip_frac,
        "directional_bias": directional_bias,
        "mean_abs_delta": mean_abs_delta_val,
        "zero_frac_a": zero_frac_a,
        "zero_frac_b": zero_frac_b,
        "zero_frac_increase": zero_frac_increase,
        "row_concentration": row_concentration,
        "worst_row_mse": worst_row_mse,
    }


def format_compare_tensor(stats):
    if stats["status"] == "shape_mismatch":
        shape_a = "x".join(str(d) for d in stats["shape_a"])
        shape_b = "x".join(str(d) for d in stats["shape_b"])
        return "\n".join(
            [
                f"--- {stats['name']} ---",
                f"  WARNING: SHAPE MISMATCH ({shape_a} vs {shape_b})",
            ]
        )

    if stats["status"] == "empty":
        return f"--- {stats['name']} ---\n  EMPTY TENSOR"

    lines = [
        f"--- {stats['name']} ---",
        f"  MSE:        {stats['mse']:10.8f}     RMSE:       {stats['rmse']:10.8f}",
        f"  Cos Sim:    {stats['cosine']:10.6f}     Rel RMSE:   {stats['rel_rmse']:10.6f}",
        f"  Rel L2 Err: {stats['rel_norm_error']:10.6f}     Max |Δ|:    {stats['max_abs_delta']:10.6f}",
        f"  PSNR:       {stats['psnr']:10.4f} dB     Norm A:     {stats['norm_a']:10.6f}",
        f"  Norm B:     {stats['norm_b']:10.6f}",
        f"  Sign Flips: {stats['sign_flip_count']:>10d} ({stats['sign_flip_frac']:7.2%})"
        f"     Dir Bias:  {stats['directional_bias']:+10.4f}",
        f"  Zero A:     {stats['zero_frac_a']:10.2%}     Zero B:     {stats['zero_frac_b']:10.2%}"
        f"     ΔZero:     {stats['zero_frac_increase']:+10.4f}",
    ]
    if stats.get("row_concentration") is not None:
        lines.append(
            f"  Top10% Row MSE: {stats['row_concentration']:7.2%} of total"
            f"     Worst Row MSE: {stats['worst_row_mse']:.8e}"
        )
    return "\n".join(lines)


def format_top_list(title, stats, key, reverse=True):
    lines = [title]
    if not stats:
        lines.append("  (none)")
        return lines

    ordered = sorted(stats, key=lambda item: item[key], reverse=reverse)[:TOP_K]
    for idx, item in enumerate(ordered, start=1):
        shape = "x".join(str(d) for d in item["shape_a"])
        lines.append(
            (
                f"  {idx:2d}. {item['name']} [{shape}] "
                f"mse={item['mse']:.8e} "
                f"rel_rmse={item['rel_rmse']:.6f} "
                f"cos={item['cosine']:.6f} "
                f"max_abs_delta={item['max_abs_delta']:.6f}"
            )
        )
    return lines


def summarize_comparison(compare_stats, model_a_name, model_b_name):
    ok_stats = [item for item in compare_stats if item["status"] == "ok"]
    mismatches = [item for item in compare_stats if item["status"] == "shape_mismatch"]

    total_numel = sum(item["numel"] for item in ok_stats)
    total_sq_error = sum(item["sum_sq_error"] for item in ok_stats)
    weighted_mse = safe_div(total_sq_error, total_numel)
    weighted_rmse = math.sqrt(weighted_mse) if weighted_mse > 0.0 else 0.0
    weighted_cosine = safe_div(
        sum(item["cosine"] * item["numel"] for item in ok_stats),
        total_numel,
    )
    weighted_rel_rmse = safe_div(
        sum(item["rel_rmse"] * item["numel"] for item in ok_stats),
        total_numel,
    )

    families = defaultdict(
        lambda: {
            "tensors": 0,
            "numel": 0,
            "sum_sq_error": 0.0,
            "weighted_cosine": 0.0,
            "weighted_rel_rmse": 0.0,
            "worst_mse_name": None,
            "worst_mse": -1.0,
        }
    )
    for item in ok_stats:
        bucket = families[item["family"]]
        bucket["tensors"] += 1
        bucket["numel"] += item["numel"]
        bucket["sum_sq_error"] += item["sum_sq_error"]
        bucket["weighted_cosine"] += item["cosine"] * item["numel"]
        bucket["weighted_rel_rmse"] += item["rel_rmse"] * item["numel"]
        if item["mse"] > bucket["worst_mse"]:
            bucket["worst_mse"] = item["mse"]
            bucket["worst_mse_name"] = item["name"]

    lines = [
        "=" * 60,
        "COMPARISON SUMMARY",
        "=" * 60,
        "",
        f"Model A: {model_a_name}",
        f"Model B: {model_b_name}",
        f"Shared tensors (same shape): {len(ok_stats)}",
        f"Shape mismatches: {len(mismatches)}",
        f"Weighted MSE: {weighted_mse:.8e}",
        f"Weighted RMSE: {weighted_rmse:.8e}",
        f"Weighted Cosine: {weighted_cosine:.8f}",
        f"Weighted Rel RMSE: {weighted_rel_rmse:.8f}",
        "",
        "=" * 60,
        "FAMILY DAMAGE SUMMARY",
        "=" * 60,
        "",
    ]

    for family in sorted(families):
        bucket = families[family]
        family_mse = safe_div(bucket["sum_sq_error"], bucket["numel"])
        family_cos = safe_div(bucket["weighted_cosine"], bucket["numel"])
        family_rel_rmse = safe_div(bucket["weighted_rel_rmse"], bucket["numel"])
        lines.append(
            (
                f"{family:12s} tensors={bucket['tensors']:3d} "
                f"numel={bucket['numel']:10d} "
                f"weighted_mse={family_mse:.8e} "
                f"weighted_rel_rmse={family_rel_rmse:.6f} "
                f"weighted_cos={family_cos:.6f} "
                f"worst_tensor={bucket['worst_mse_name']}"
            )
        )

    lines.extend(
        [
            "",
            "=" * 60,
            f"TOP {TOP_K} TENSORS",
            "=" * 60,
            "",
        ]
    )
    lines.extend(format_top_list("[By MSE]", ok_stats, "mse", reverse=True))
    lines.append("")
    lines.extend(format_top_list("[By Relative RMSE]", ok_stats, "rel_rmse", reverse=True))
    lines.append("")
    lines.extend(format_top_list("[By Cosine Drop]", ok_stats, "cosine_drop", reverse=True))
    lines.append("")
    lines.extend(format_top_list("[By Sign Flip Fraction]", ok_stats, "sign_flip_frac", reverse=True))

    # --- per-block aggregation ---
    block_buckets = defaultdict(
        lambda: {
            "tensors": 0,
            "numel": 0,
            "sum_sq_error": 0.0,
            "weighted_cosine": 0.0,
            "weighted_rel_rmse": 0.0,
            "total_sign_flips": 0,
            "worst_mse_name": None,
            "worst_mse": -1.0,
        }
    )
    for item in ok_stats:
        label = block_group_label(item["name"])
        if label is None:
            continue
        bucket = block_buckets[label]
        bucket["tensors"] += 1
        bucket["numel"] += item["numel"]
        bucket["sum_sq_error"] += item["sum_sq_error"]
        bucket["weighted_cosine"] += item["cosine"] * item["numel"]
        bucket["weighted_rel_rmse"] += item["rel_rmse"] * item["numel"]
        bucket["total_sign_flips"] += item["sign_flip_count"]
        if item["mse"] > bucket["worst_mse"]:
            bucket["worst_mse"] = item["mse"]
            bucket["worst_mse_name"] = item["name"]

    if block_buckets:
        lines.extend(
            [
                "",
                "=" * 60,
                "PER-BLOCK GROUP DAMAGE",
                "=" * 60,
                "",
            ]
        )
        for label in sorted(block_buckets):
            bucket = block_buckets[label]
            b_mse = safe_div(bucket["sum_sq_error"], bucket["numel"])
            b_cos = safe_div(bucket["weighted_cosine"], bucket["numel"])
            b_rel = safe_div(bucket["weighted_rel_rmse"], bucket["numel"])
            b_sf = safe_div(bucket["total_sign_flips"], bucket["numel"])
            lines.append(
                f"{label:14s} tensors={bucket['tensors']:3d} "
                f"numel={bucket['numel']:10d} "
                f"mse={b_mse:.8e} "
                f"rel_rmse={b_rel:.6f} "
                f"cos={b_cos:.6f} "
                f"sign_flip={b_sf:.4%} "
                f"worst={bucket['worst_mse_name']}"
            )

    # --- late-block top-k ---
    late_stats = [item for item in ok_stats if is_late_block_tensor(item["name"])]
    if late_stats:
        lines.extend(
            [
                "",
                "=" * 60,
                f"LATE-BLOCK TOP {TOP_K} (blocks.9-10, ve_*, bigram.*)",
                "=" * 60,
                "",
            ]
        )
        lines.extend(format_top_list("[Late By MSE]", late_stats, "mse", reverse=True))
        lines.append("")
        lines.extend(format_top_list("[Late By Rel RMSE]", late_stats, "rel_rmse", reverse=True))
        lines.append("")
        lines.extend(format_top_list("[Late By Cosine Drop]", late_stats, "cosine_drop", reverse=True))

    # --- quantization zeroing summary ---
    zeroing_stats = [item for item in ok_stats if item["zero_frac_increase"] > 0.001]
    if zeroing_stats:
        lines.extend(
            [
                "",
                "=" * 60,
                "QUANTIZATION ZEROING (tensors with >0.1% zero increase)",
                "=" * 60,
                "",
            ]
        )
        zeroing_stats.sort(key=lambda x: x["zero_frac_increase"], reverse=True)
        for item in zeroing_stats[:TOP_K]:
            rc_str = f"row_conc={item['row_concentration']:.2%}" if item.get("row_concentration") is not None else "row_conc=n/a"
            lines.append(
                f"  {item['name']}: "
                f"zero_A={item['zero_frac_a']:.2%} -> zero_B={item['zero_frac_b']:.2%} "
                f"(+{item['zero_frac_increase']:.4f}) "
                f"mean_abs_delta={item['mean_abs_delta']:.6e} "
                f"{rc_str}"
            )

    # --- directional bias outliers ---
    bias_outliers = [item for item in ok_stats if abs(item["directional_bias"]) > 0.1]
    if bias_outliers:
        lines.extend(
            [
                "",
                "=" * 60,
                "DIRECTIONAL BIAS OUTLIERS (|bias| > 0.1)",
                "=" * 60,
                "",
            ]
        )
        bias_outliers.sort(key=lambda x: abs(x["directional_bias"]), reverse=True)
        for item in bias_outliers[:TOP_K]:
            lines.append(
                f"  {item['name']}: bias={item['directional_bias']:+.4f} "
                f"mean_delta={item['directional_bias'] * item['mean_abs_delta']:+.6e} "
                f"mse={item['mse']:.8e}"
            )

    if mismatches:
        lines.extend(["", "=" * 60, "SHAPE MISMATCHES", "=" * 60, ""])
        for item in mismatches:
            shape_a = "x".join(str(d) for d in item["shape_a"])
            shape_b = "x".join(str(d) for d in item["shape_b"])
            lines.append(f"  {item['name']}: {shape_a} vs {shape_b}")

    return "\n".join(lines).rstrip() + "\n"


def _json_default(obj):
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return str(obj)
        return obj
    return obj


def build_single_family_summary(all_stats):
    families = defaultdict(
        lambda: {
            "tensors": 0,
            "numel": 0,
            "weighted_sparsity": 0.0,
            "weighted_outlier": 0.0,
            "max_mag": 0.0,
            "max_norm_name": None,
            "max_norm": -1.0,
        }
    )

    for stats in all_stats:
        if stats["empty"]:
            continue
        bucket = families[stats["family"]]
        bucket["tensors"] += 1
        bucket["numel"] += stats["numel"]
        bucket["weighted_sparsity"] += stats["sparsity"] * stats["numel"]
        bucket["weighted_outlier"] += stats["outlier_frac"] * stats["numel"]
        bucket["max_mag"] = max(bucket["max_mag"], stats["max_mag"])
        if stats["l2_norm"] > bucket["max_norm"]:
            bucket["max_norm"] = stats["l2_norm"]
            bucket["max_norm_name"] = stats["name"]

    summary = {}
    for family, bucket in families.items():
        summary[family] = {
            "tensors": bucket["tensors"],
            "numel": bucket["numel"],
            "mean_sparsity": safe_div(bucket["weighted_sparsity"], bucket["numel"]),
            "mean_outlier": safe_div(bucket["weighted_outlier"], bucket["numel"]),
            "max_mag": bucket["max_mag"],
            "max_norm_name": bucket["max_norm_name"],
            "max_norm": bucket["max_norm"],
        }
    return summary


def build_component_analysis(state_dict):
    analysis = {}

    smear_gate = state_dict.get("smear.gate")
    if smear_gate is not None:
        gate = smear_gate.float().cpu()
        if gate.min().item() < 0.0 or gate.max().item() > 1.0:
            gate = torch.sigmoid(gate)
        analysis["smeargate"] = {
            "mean_pass_through": 1.0 - gate.mean().item(),
            "max_gate": gate.max().item(),
            "min_gate": gate.min().item(),
        }

    ve_scale_names = sorted(name for name in state_dict if name.startswith("ve_layer_scales."))
    if ve_scale_names:
        analysis["ve_layer_scales"] = {
            name: state_dict[name].float().cpu().item() for name in ve_scale_names
        }
    if "ve_shared.scale" in state_dict:
        analysis["ve_shared_scale"] = state_dict["ve_shared.scale"].float().cpu().item()
    if "bigram.scale" in state_dict:
        analysis["bigram_scale"] = state_dict["bigram.scale"].float().cpu().item()

    return analysis


def _build_compare_json(compare_stats, model_a_name, model_b_name, missing_in_a, missing_in_b):
    ok_stats = [item for item in compare_stats if item["status"] == "ok"]

    # family summary
    family_summary = {}
    for item in ok_stats:
        fam = item["family"]
        if fam not in family_summary:
            family_summary[fam] = {"tensors": 0, "numel": 0, "sum_sq_error": 0.0,
                                   "weighted_cosine": 0.0, "total_sign_flips": 0}
        bucket = family_summary[fam]
        bucket["tensors"] += 1
        bucket["numel"] += item["numel"]
        bucket["sum_sq_error"] += item["sum_sq_error"]
        bucket["weighted_cosine"] += item["cosine"] * item["numel"]
        bucket["total_sign_flips"] += item["sign_flip_count"]
    for fam, bucket in family_summary.items():
        bucket["weighted_mse"] = safe_div(bucket["sum_sq_error"], bucket["numel"])
        bucket["weighted_cosine"] = safe_div(bucket["weighted_cosine"], bucket["numel"])
        bucket["sign_flip_rate"] = safe_div(bucket["total_sign_flips"], bucket["numel"])

    # block group summary
    block_summary = {}
    for item in ok_stats:
        label = block_group_label(item["name"])
        if label is None:
            continue
        if label not in block_summary:
            block_summary[label] = {"tensors": 0, "numel": 0, "sum_sq_error": 0.0,
                                    "weighted_cosine": 0.0, "total_sign_flips": 0}
        bucket = block_summary[label]
        bucket["tensors"] += 1
        bucket["numel"] += item["numel"]
        bucket["sum_sq_error"] += item["sum_sq_error"]
        bucket["weighted_cosine"] += item["cosine"] * item["numel"]
        bucket["total_sign_flips"] += item["sign_flip_count"]
    for label, bucket in block_summary.items():
        bucket["weighted_mse"] = safe_div(bucket["sum_sq_error"], bucket["numel"])
        bucket["weighted_cosine"] = safe_div(bucket["weighted_cosine"], bucket["numel"])
        bucket["sign_flip_rate"] = safe_div(bucket["total_sign_flips"], bucket["numel"])

    # rankings
    def top_k_by(key, reverse=True):
        return [item["name"] for item in sorted(ok_stats, key=lambda x: x[key], reverse=reverse)[:TOP_K]]

    rankings = {
        "by_mse": top_k_by("mse"),
        "by_rel_rmse": top_k_by("rel_rmse"),
        "by_cosine_drop": top_k_by("cosine_drop"),
        "by_sign_flip_frac": top_k_by("sign_flip_frac"),
    }

    # late-block rankings
    late = [item for item in ok_stats if is_late_block_tensor(item["name"])]
    if late:
        rankings["late_by_mse"] = [item["name"] for item in sorted(late, key=lambda x: x["mse"], reverse=True)[:TOP_K]]
        rankings["late_by_cosine_drop"] = [item["name"] for item in sorted(late, key=lambda x: x["cosine_drop"], reverse=True)[:TOP_K]]

    total_numel = sum(item["numel"] for item in ok_stats)
    total_sq = sum(item["sum_sq_error"] for item in ok_stats)

    return {
        "mode": "comparison",
        "model_a": model_a_name,
        "model_b": model_b_name,
        "global": {
            "shared_tensors": len(ok_stats),
            "shape_mismatches": len([s for s in compare_stats if s["status"] == "shape_mismatch"]),
            "weighted_mse": safe_div(total_sq, total_numel),
            "weighted_cosine": safe_div(sum(i["cosine"] * i["numel"] for i in ok_stats), total_numel),
        },
        "family_summary": family_summary,
        "block_summary": block_summary,
        "rankings": rankings,
        "tensors": compare_stats,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
    }


def write_single_report(state_dict, model_path):
    names = sorted(state_dict)
    all_stats = [analyze_tensor(name, state_dict[name]) for name in names]

    with open(OUTPUT_PATH, "w") as f:
        f.write(f"DIAGNOSTICS REPORT FOR: {model_path}\n")
        f.write("=" * 60 + "\n\n")
        for stats in all_stats:
            f.write(format_single_tensor(stats) + "\n\n")
        f.write(summarize_single_model(all_stats, state_dict))

    json_out = {
        "mode": "single",
        "model": model_path,
        "family_summary": build_single_family_summary(all_stats),
        "component_analysis": build_component_analysis(state_dict),
        "tensors": all_stats,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(sanitize_for_json(json_out), f, indent=2, allow_nan=False)
    print(f"JSON saved to {JSON_PATH}")


def write_compare_report(state_dict, compare_dict, model_path, compare_path):
    names = sorted(set(state_dict) | set(compare_dict))
    compare_stats = []
    missing_in_b = []
    missing_in_a = []

    for name in names:
        in_a = name in state_dict
        in_b = name in compare_dict
        if in_a and in_b:
            compare_stats.append(compare_tensors(name, state_dict[name], compare_dict[name]))
        elif in_a:
            missing_in_b.append(name)
        else:
            missing_in_a.append(name)

    with open(OUTPUT_PATH, "w") as f:
        f.write("QUANTIZATION / MODEL DAMAGE REPORT\n")
        f.write(f"Model A: {model_path}\n")
        f.write(f"Model B: {compare_path}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summarize_comparison(compare_stats, model_path, compare_path))

        if missing_in_b or missing_in_a:
            f.write("\n" + "=" * 60 + "\n")
            f.write("MISSING TENSORS\n")
            f.write("=" * 60 + "\n\n")
            if missing_in_b:
                f.write("Missing in Model B:\n")
                for name in missing_in_b:
                    f.write(f"  {name}\n")
                f.write("\n")
            if missing_in_a:
                f.write("Missing in Model A:\n")
                for name in missing_in_a:
                    f.write(f"  {name}\n")
                f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("PER-TENSOR DETAILS\n")
        f.write("=" * 60 + "\n\n")
        for stats in compare_stats:
            f.write(format_compare_tensor(stats) + "\n\n")

    json_out = _build_compare_json(compare_stats, model_path, compare_path, missing_in_a, missing_in_b)
    with open(JSON_PATH, "w") as f:
        json.dump(sanitize_for_json(json_out), f, indent=2, allow_nan=False)
    print(f"JSON saved to {JSON_PATH}")


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python scripts/diagnostics/diagnose_weights.py <model_a.pt|.ptz> [model_b.pt|.ptz]")
        sys.exit(1)

    model_path = sys.argv[1]
    state_dict = load_model_weights(model_path)

    if len(sys.argv) == 2:
        write_single_report(state_dict, model_path)
    else:
        compare_path = sys.argv[2]
        compare_dict = load_model_weights(compare_path)
        write_compare_report(state_dict, compare_dict, model_path, compare_path)

    print(f"Diagnostics saved to {OUTPUT_PATH}!")


if __name__ == "__main__":
    main()
