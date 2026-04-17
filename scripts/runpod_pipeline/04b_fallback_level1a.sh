#!/usr/bin/env bash
# Stage 4b: Level 1A export-side fallback — eval-only requantization variants
# No retraining. Loads seed-0 FP32 checkpoint, runs two export-side tweaks, evals each.
# Variant 1: int7 embeddings (EMBED_BITS=7, vs default 8)
# Variant 2: tighter matrix clip_sigmas (MATRIX_CLIP_SIGMAS=12.0, vs default 12.85)
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"
RUNS_DIR="${REPO_DIR}/runs"
CKPT_DIR="/workspace/checkpoints/seed0"
GATE_A_SUMMARY="${RUNS_DIR}/gate_a_summary.json"
PYTHON="/opt/pg-venv/bin/python"

mkdir -p "${RUNS_DIR}"
exec > >(tee -a "${RUNS_DIR}/04b_fallback.log") 2>&1

echo "=== Stage 4b: Fallback Level 1A (export-side requant) === $(date)"

cd "${REPO_DIR}"

[ -f "${GATE_A_SUMMARY}" ] || {
    echo "ERROR: ${GATE_A_SUMMARY} not found — run 02_gate_a.sh first" >&2; exit 1
}
[ -f "${CKPT_DIR}/final_model.pt" ] || {
    echo "ERROR: ${CKPT_DIR}/final_model.pt not found — run 02_gate_a.sh first" >&2; exit 1
}

BASELINE_BPB=$("${PYTHON}" -c "import json; print(json.load(open('${GATE_A_SUMMARY}'))['bpb'])")
echo "Baseline BPB (Gate A seed 0): ${BASELINE_BPB}"
echo "NOTE: export-side only — no GPU training cost beyond the eval pass"
echo ""

run_eval_variant() {
    local label="$1"
    shift
    # remaining args are KEY=VALUE env var pairs for the variant
    local variant_dir="${RUNS_DIR}/fallback_${label}"
    local log_file="${RUNS_DIR}/fallback_${label}_log.txt"
    mkdir -p "${variant_dir}"

    echo "--- Variant ${label}: $* ---"

    env "$@" \
    EVAL_ONLY_PATH="${CKPT_DIR}/final_model.pt" \
    ARTIFACT_DIR="${variant_dir}" \
    PHASED_TTT_ENABLED=1 \
    PHASED_TTT_PREFIX_DOCS=2000 \
    PYTHONUNBUFFERED=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NCCL_DEBUG=WARN \
        /opt/pg-venv/bin/torchrun \
        --standalone --nproc_per_node=8 \
        "${TRAIN_SCRIPT}" \
        2>&1 | tee "${log_file}"

    "${PYTHON}" - "${log_file}" "${label}" "${BASELINE_BPB}" <<'PY'
import re, sys, json, pathlib

log_path    = pathlib.Path(sys.argv[1])
label       = sys.argv[2]
baseline    = float(sys.argv[3])
log_text    = log_path.read_text()

# Accept both phased and lora result lines (fallback uses FP32 path)
m = re.search(
    r"quantized_ttt_(?:phased|lora) val_loss:[0-9.]+ val_bpb:([0-9.]+) eval_time:([0-9]+)ms",
    log_text
)
if not m:
    # Last-resort: any val_bpb line
    m = re.search(r"val_bpb:([0-9.]+) eval_time:([0-9]+)ms", log_text)
if not m:
    print(f"FATAL: variant {label}: no BPB found in log — fallback data invalid", flush=True)
    print(f"  Check: {log_path}", flush=True)
    sys.exit(1)

bpb     = float(m.group(1))
eval_ms = int(m.group(2))
delta   = baseline - bpb

ms = re.search(r"Total submission size quantized\+\S+: ([0-9]+) bytes", log_text)
artifact_bytes = int(ms.group(1)) if ms else 0

print(f"  bpb={bpb:.8f}  delta={delta:+.6f}  eval={eval_ms/1000:.1f}s  artifact={artifact_bytes}B")

out = log_path.parent / f"fallback_{label}.json"
out.write_text(json.dumps({
    "label": label, "bpb": bpb, "delta": delta,
    "eval_ms": eval_ms, "artifact_bytes": artifact_bytes
}, indent=2))
print(f"  Result saved: {out}")
PY
    echo ""
}

# Variant 1: int7 embeddings (EMBED_BITS env var read by train_gpt.py:165)
run_eval_variant "int7_emb" "EMBED_BITS=7"

# Variant 2: tighter matrix clip_sigmas (MATRIX_CLIP_SIGMAS env var read by train_gpt.py:166)
run_eval_variant "adaptive_clip" "MATRIX_CLIP_SIGMAS=12.0"

echo "=== Fallback Level 1A Summary ==="
"${PYTHON}" - "${RUNS_DIR}" "${BASELINE_BPB}" <<'PY'
import json, pathlib, sys

runs_dir     = pathlib.Path(sys.argv[1])
baseline_bpb = float(sys.argv[2])

results = {}
for f in sorted(runs_dir.glob("fallback_*.json")):
    if "_log" not in f.stem:
        d = json.loads(f.read_text())
        results[d["label"]] = d

# Hard failure: both variants must have produced valid results
required = {"int7_emb", "adaptive_clip"}
missing  = required - set(results.keys())
if missing:
    print(f"FATAL: fallback variants missing results: {missing}", flush=True)
    print(f"  Check logs in {runs_dir}/fallback_*_log.txt", flush=True)
    sys.exit(1)

print(f"  Baseline:  {baseline_bpb:.8f}")
print(f"  {'Variant':<18} {'BPB':>12} {'Delta':>10} {'Artifact':>12}")
print(f"  {'-'*56}")
for label in sorted(results):
    r = results[label]
    print(f"  {label:<18} {r['bpb']:>12.8f} {r['delta']:>+10.6f} {r.get('artifact_bytes',0):>12d}B")

best = max(results.values(), key=lambda r: r["delta"])
print(f"\n  Best variant: {best['label']} (delta={best['delta']:+.6f} BPB)")
if best["delta"] < 0.0005:
    print("  WARNING: best fallback delta < 0.0005 — marginal improvement")
    print("  Consider submitting clean Gate A baseline.")
PY

echo ""
echo "04b_fallback_level1a: DONE"
echo "Next: UPLOAD_TARGET=hf:<repo>:<path> bash scripts/runpod_pipeline/05_preserve_artifacts.sh"
