#!/usr/bin/env python3
"""Build a local PR #1394 RunPod submission bundle from finished seed logs.

This script parses the faithful SP8192 repro logs, recomputes code-byte
accounting using the current self-extracting wrapper path, and writes a
self-contained record folder with README/submission metadata plus copied logs.
"""

from __future__ import annotations

import argparse
import base64
import json
import lzma
import re
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / (
    "records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "records/track_10min_16mb/2026-04-07_pr1394_sp8192_runpod_strict"
)
MERGED_OFFICIAL_NATS = 1.88217853
OFFICIAL_RECORD_BAR_NATS = 1.87717853
MERGED_OFFICIAL_BPB = 1.11473509
OPEN_PR1394_MEAN_BPB = 1.08563


def _find_last(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise ValueError(f"Missing {label} in log")
    return matches[-1]


def _minified_code(source: str) -> str:
    compressed = lzma.compress(source.encode("utf-8"), preset=9)
    encoded = base64.b85encode(compressed)
    return (
        "import lzma as L,base64 as B\n"
        f"exec(L.decompress(B.b85decode({encoded!r})))\n"
    )


def _fmt_ms(ms: int) -> str:
    return f"{ms}ms"


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000:.2f} MB"


def _safe_sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


@dataclass
class SeedResult:
    seed: int
    steps_completed: int
    train_time_ms: int
    pre_quant_val_loss: float
    pre_quant_val_bpb: float
    final_int6_roundtrip_val_loss: float
    final_int6_roundtrip_val_bpb: float
    final_int6_sliding_window_val_loss: float
    final_int6_sliding_window_val_bpb: float
    compressed_model_bytes: int
    logged_total_bytes: int | None
    code_bytes_counted: int
    projected_total_bytes: int
    source_log: str


def parse_seed_log(log_path: Path, train_script: Path) -> SeedResult:
    text = log_path.read_text(encoding="utf-8")

    seed = int(_find_last(r"^\s*seed:\s*(\d+)\s*$", text, "seed").group(1))
    stop = _find_last(
        r"stopping_early:\s+wallclock_cap\s+train_time:\s*(\d+)ms\s+step:\s*(\d+)/",
        text,
        "stopping_early wallclock line",
    )
    pre_quant = _find_last(
        r"pre-quantization post-ema val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)",
        text,
        "pre-quantization post-ema",
    )
    quantized = _find_last(
        r"^quantized val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)",
        text,
        "quantized roundtrip metrics",
    )
    sliding = _find_last(
        r"^quantized_sliding_window val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)",
        text,
        "quantized sliding-window metrics",
    )
    model_bytes = int(
        _find_last(
            r"Serialized model quantized\+brotli:\s+(\d+)\s+bytes",
            text,
            "quantized model bytes",
        ).group(1)
    )

    logged_total_match = list(
        re.finditer(
            r"Total submission size quantized\+brotli:\s+(\d+)\s+bytes",
            text,
            flags=re.MULTILINE,
        )
    )
    logged_total = int(logged_total_match[-1].group(1)) if logged_total_match else None

    code_bytes = len(_minified_code(train_script.read_text(encoding="utf-8")).encode("utf-8"))
    projected_total = model_bytes + code_bytes

    return SeedResult(
        seed=seed,
        steps_completed=int(stop.group(2)),
        train_time_ms=int(stop.group(1)),
        pre_quant_val_loss=float(pre_quant.group(1)),
        pre_quant_val_bpb=float(pre_quant.group(2)),
        final_int6_roundtrip_val_loss=float(quantized.group(1)),
        final_int6_roundtrip_val_bpb=float(quantized.group(2)),
        final_int6_sliding_window_val_loss=float(sliding.group(1)),
        final_int6_sliding_window_val_bpb=float(sliding.group(2)),
        compressed_model_bytes=model_bytes,
        logged_total_bytes=logged_total,
        code_bytes_counted=code_bytes,
        projected_total_bytes=projected_total,
        source_log=str(log_path),
    )


def build_submission(results: list[SeedResult], date: str, author: str, github_id: str) -> dict:
    best = min(results, key=lambda r: (r.final_int6_sliding_window_val_bpb, r.projected_total_bytes))
    best_loss = best.final_int6_sliding_window_val_loss
    best_bpb = best.final_int6_sliding_window_val_bpb
    best_bytes = best.projected_total_bytes
    max_bytes = max(r.projected_total_bytes for r in results)
    mean_sliding = statistics.mean(r.final_int6_sliding_window_val_bpb for r in results)
    seed_count = len(results)

    return {
        "author": author,
        "github_id": github_id,
        "name": "Faithful PR #1394 SP8192 RunPod Strict Proof",
        "blurb": (
            "Faithful PR #1394 SP8192 reproduction on RunPod 8xH100 SXM, "
            f"using the best claimed seed ({best.seed}): {best_bpb:.8f} BPB / "
            f"{best_loss:.8f} nats / {best_bytes:,} bytes. "
            f"Current bundle summarizes {seed_count} strict seed run(s) with mean "
            f"sliding BPB {mean_sliding:.8f} and max artifact {max_bytes:,} bytes."
        ),
        "date": date,
        "val_loss": round(best_loss, 8),
        "val_bpb": round(best_bpb, 8),
        "bytes_total": best_bytes,
    }


def render_readme(results: list[SeedResult], submission: dict, train_script: Path) -> str:
    best = min(results, key=lambda r: (r.final_int6_sliding_window_val_bpb, r.projected_total_bytes))
    mean_sliding = statistics.mean(r.final_int6_sliding_window_val_bpb for r in results)
    mean_roundtrip = statistics.mean(r.final_int6_roundtrip_val_bpb for r in results)
    mean_nats = statistics.mean(r.final_int6_sliding_window_val_loss for r in results)
    std_sliding = _safe_sample_std([r.final_int6_sliding_window_val_bpb for r in results])
    max_bytes = max(r.projected_total_bytes for r in results)
    max_train_time_ms = max(r.train_time_ms for r in results)
    delta_bpb_vs_merged = mean_sliding - MERGED_OFFICIAL_BPB
    delta_bpb_vs_pr1394 = mean_sliding - OPEN_PR1394_MEAN_BPB

    lines = [
        "# Faithful PR #1394 SP8192 RunPod Strict Proof",
        "",
        f"**val_bpb: {mean_sliding:.4f}** ({len(results)}-seed mean sliding s64) | "
        f"**{_fmt_mb(max_bytes)}** max artifact | **8xH100 SXM**, "
        f"**{max_train_time_ms / 1000:.1f}s** max train time",
        "",
        f"Submitted artifact: **seed {best.seed}** with `val_loss = {best.final_int6_sliding_window_val_loss:.8f}`, "
        f"`val_bpb = {best.final_int6_sliding_window_val_bpb:.8f}`, "
        f"`bytes_total = {best.projected_total_bytes:,}`.",
        "",
        "## Summary",
        "",
        "This folder packages the faithful PR `#1394` SP8192 reproduction run on RunPod `8xH100 SXM`.",
        "It uses the current local `train_gpt.py` packaging path, which counts code bytes via the minified",
        "self-extracting wrapper instead of the larger human-readable source file.",
        "",
        "## Strict Multi-Seed Results",
        "",
        "| Seed | Steps | train_time | Pre-quant EMA BPB | Roundtrip exact BPB | Sliding s64 BPB | nats | model_bytes | bytes_total |",
        "|------|------:|-----------:|------------------:|--------------------:|----------------:|-----:|------------:|------------:|",
    ]

    for r in sorted(results, key=lambda x: x.seed):
        lines.append(
            f"| {r.seed} | {r.steps_completed} | {_fmt_ms(r.train_time_ms)} | "
            f"{r.pre_quant_val_bpb:.8f} | {r.final_int6_roundtrip_val_bpb:.8f} | "
            f"{r.final_int6_sliding_window_val_bpb:.8f} | {r.final_int6_sliding_window_val_loss:.8f} | "
            f"{r.compressed_model_bytes:,} | {r.projected_total_bytes:,} |"
        )

    lines.extend(
        [
            f"| **Mean** | | | **{statistics.mean(r.pre_quant_val_bpb for r in results):.8f}** | "
            f"**{mean_roundtrip:.8f}** | **{mean_sliding:.8f}** | **{mean_nats:.8f}** | | |",
            f"| **Std** | | | | | **{std_sliding:.8f}** | **{_safe_sample_std([r.final_int6_sliding_window_val_loss for r in results]):.8f}** | | |",
            "",
            "## BPB Reference Points",
            "",
            f"Merged official `#1019`: `{MERGED_OFFICIAL_BPB:.8f}` BPB",
            f"Open clean PR `#1394` mean: `{OPEN_PR1394_MEAN_BPB:.5f}` BPB",
            f"Mean delta vs merged `#1019`: `{delta_bpb_vs_merged:+.8f}` BPB",
            f"Mean delta vs open `#1394` reference: `{delta_bpb_vs_pr1394:+.8f}` BPB",
            "",
            "Because this bundle uses `SP8192`, its token-level `val_loss` nats are tokenizer-dependent.",
            "Cross-line comparison should therefore use **BPB**, not `val_loss` nats against the older SP1024 merged line.",
            "",
            "## Artifact Byte Accounting",
            "",
            "- `compressed_model_bytes` comes directly from the run log.",
            f"- `code_bytes_counted` is recomputed from `{train_script.name}` using the same lzma+base85 wrapper counted by the current export path.",
            "- `bytes_total` in this folder is therefore `compressed_model_bytes + code_bytes_counted`.",
            "",
            "## Environment Notes",
            "",
            "- hardware: RunPod `8xH100 SXM`",
            "- tokenizer/data: `fineweb10B_sp8192` + `fineweb_8192_bpe.model`",
            "- packaging note: the original 2026-04-07 seed-1337 smoke run was launched before the local code-byte fix,",
            "  so its logged `Total submission size` may exceed the recomputed `bytes_total` used here.",
            "",
            "## How to Run",
            "",
            "Install Python dependencies first:",
            "",
            "```bash",
            "pip install -r requirements.txt",
            "```",
            "",
            "Then run from the repo root:",
            "",
            "```bash",
            "PYTHONUNBUFFERED=1 \\",
            "DATA_DIR=./data \\",
            f"SEED={best.seed} \\",
            "RUN_ID=submission \\",
            "torchrun --standalone --nnodes=1 --nproc_per_node=8 train_gpt.py",
            "```",
            "",
            "## Included Files",
            "",
            "- `train_gpt.py`: faithful PR `#1394` repro snapshot with current packaging fix",
            "- `train_seed*.log`: one log per claimed seed",
            "- `submission.json`: leaderboard metadata for the best seed in this bundle",
        ]
    )

    return "\n".join(lines) + "\n"


def write_requirements(output_dir: Path) -> None:
    (output_dir / "requirements.txt").write_text(
        "\n".join(
            [
                "numpy",
                "torch==2.9.1",
                "sentencepiece==0.2.1",
                "brotli==1.2.0",
                "# flash_attn_interface (FA3) is expected from the container image;",
                "# it is not the PyPI flash-attn package.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-log", action="append", default=[], help="Path to a finished seed log")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--author", default="amay")
    parser.add_argument("--github-id", default="amrayach")
    parser.add_argument("--date", default="2026-04-07")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the parsed summary JSON even when writing a bundle",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only parse and print results; do not materialize the bundle directory",
    )
    args = parser.parse_args()

    if not args.seed_log:
        parser.error("At least one --seed-log is required")

    train_script = args.train_script.resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Missing train script: {train_script}")

    results = [parse_seed_log(Path(p).resolve(), train_script) for p in args.seed_log]
    results.sort(key=lambda r: r.seed)
    submission = build_submission(results, args.date, args.author, args.github_id)
    summary = {
        "submission": submission,
        "seeds": [asdict(r) for r in results],
        "mean_sliding_bpb": statistics.mean(r.final_int6_sliding_window_val_bpb for r in results),
        "mean_sliding_nats": statistics.mean(r.final_int6_sliding_window_val_loss for r in results),
    }

    if args.print_json or args.no_write:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if args.no_write:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_script, output_dir / "train_gpt.py")
    for result in results:
        shutil.copy2(result.source_log, output_dir / f"train_seed{result.seed}.log")
    write_requirements(output_dir)
    (output_dir / "submission.json").write_text(
        json.dumps(submission, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        render_readme(results, submission, train_script),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
