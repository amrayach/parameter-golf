#!/usr/bin/env bash
# Quick-exit script: pull PR #1413 batch results from RunPod, then terminate.
#
# Usage (on LOCAL machine):
#   SSH_PORT=<port> bash scripts/runpod_1413_pull_and_terminate.sh root@<pod-ip>
#
# What it does:
#   1. Captures a final environment snapshot on the pod
#   2. Captures pod repo provenance (HEAD, branch, remotes, status)
#   3. Copies those snapshots into the archive root
#   4. Compresses the entire archive dir on the pod into a single tarball
#   5. Pulls the tarball and sha256 locally via rsync
#   6. Extracts it into artifacts/runpod_pull/
#   7. Emits a local TSV/Markdown summary for PR writeup
#   8. Prints a summary of what was pulled
#   9. Reminds you to terminate the pod
#
# The archive on the pod is at:
#   /workspace/${ARCHIVE_NAME}/
#
# Optional env overrides:
#   ARCHIVE_NAME=pr1413_archive_<stamp>
#   D_TARGET_SEEDS=0,42,1234,1337,2025
set -euo pipefail

TARGET="${1:?Usage: SSH_PORT=<port> bash $0 root@<pod-ip>}"
SSH_PORT="${SSH_PORT:-}"

ARCHIVE_NAME="${ARCHIVE_NAME:-pr1413_archive_20260407_213205}"
REMOTE_ARCHIVE="/workspace/${ARCHIVE_NAME}"
REMOTE_TARBALL="/workspace/${ARCHIVE_NAME}.tar.gz"
REMOTE_SHA256="${REMOTE_TARBALL}.sha256"
ENV_LABEL="${ENV_LABEL:-${ARCHIVE_NAME}}"
REMOTE_ENV_FILE="/workspace/parameter-golf/logs/${ENV_LABEL}.env.txt"
REMOTE_REPO_STATE_FILE="/workspace/parameter-golf/logs/${ENV_LABEL}.repo_state.txt"
D_TARGET_SEEDS="${D_TARGET_SEEDS:-0,42,1234,1337,2025}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PULL_DIR="${REPO_ROOT}/artifacts/runpod_pull"
LOCAL_TARBALL="${LOCAL_PULL_DIR}/${ARCHIVE_NAME}.tar.gz"
LOCAL_SHA256="${LOCAL_TARBALL}.sha256"
LOCAL_EXTRACT_DIR="${LOCAL_PULL_DIR}/${ARCHIVE_NAME}"
LOCAL_SUMMARY_TSV="${LOCAL_EXTRACT_DIR}/run_summary.tsv"
LOCAL_SUMMARY_MD="${LOCAL_EXTRACT_DIR}/run_summary.md"
LOCAL_D_SUMMARY_TSV="${LOCAL_EXTRACT_DIR}/d_submission_summary.tsv"

SSH_CMD=(ssh)
RSYNC_SSH=(ssh)
if [[ -n "${SSH_PORT}" ]]; then
  SSH_CMD+=(-p "${SSH_PORT}")
  RSYNC_SSH+=(-p "${SSH_PORT}")
fi

echo "==> Step 1: Capture final environment snapshot on pod"
"${SSH_CMD[@]}" "${TARGET}" "cd /workspace/parameter-golf && if test -f scripts/runpod_capture_env.sh; then bash scripts/runpod_capture_env.sh ${ENV_LABEL}; else echo '  (skipping: scripts/runpod_capture_env.sh missing on pod)'; fi"

echo "==> Step 2: Capture repo provenance on pod"
"${SSH_CMD[@]}" "${TARGET}" "mkdir -p /workspace/parameter-golf/logs && cd /workspace/parameter-golf && { \
  echo timestamp_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ); \
  echo repo_head=\$(git rev-parse HEAD); \
  echo branch=\$(git branch --show-current); \
  echo; \
  echo '[git-remotes]'; git remote -v; \
  echo; \
  echo '[git-status]'; git status --short; \
  echo; \
  echo '[head-subject]'; git log -1 --decorate=short --stat --oneline; \
} > ${REMOTE_REPO_STATE_FILE}"

echo "==> Step 3: Copy env snapshot and repo provenance into archive root"
"${SSH_CMD[@]}" "${TARGET}" "test -f ${REMOTE_ENV_FILE} && cp ${REMOTE_ENV_FILE} ${REMOTE_ARCHIVE}/ || true; test -f ${REMOTE_REPO_STATE_FILE} && cp ${REMOTE_REPO_STATE_FILE} ${REMOTE_ARCHIVE}/ || true"

echo "==> Step 4: Compress archive on pod"
"${SSH_CMD[@]}" "${TARGET}" "cd /workspace && tar czf ${ARCHIVE_NAME}.tar.gz ${ARCHIVE_NAME}/ && sha256sum ${ARCHIVE_NAME}.tar.gz > ${ARCHIVE_NAME}.tar.gz.sha256"
echo "    Remote tarball created: ${REMOTE_TARBALL}"

echo "==> Step 5: Pull tarball locally"
mkdir -p "${LOCAL_PULL_DIR}"
rsync -ahz --progress \
  -e "$(printf '%q ' "${RSYNC_SSH[@]}")" \
  "${TARGET}:${REMOTE_TARBALL}" "${LOCAL_TARBALL}"
rsync -ahz \
  -e "$(printf '%q ' "${RSYNC_SSH[@]}")" \
  "${TARGET}:${REMOTE_SHA256}" "${LOCAL_SHA256}"
echo "    Saved to: ${LOCAL_TARBALL}"

echo "==> Step 6: Verify and extract locally"
(
  cd "${LOCAL_PULL_DIR}"
  sha256sum -c "$(basename "${LOCAL_SHA256}")"
)
tar xzf "${LOCAL_TARBALL}" -C "${LOCAL_PULL_DIR}/"
echo "    Extracted to: ${LOCAL_EXTRACT_DIR}/"

echo "==> Step 7: Build local run summary"
python3 - "${LOCAL_EXTRACT_DIR}" "${LOCAL_SUMMARY_TSV}" "${LOCAL_SUMMARY_MD}" "${LOCAL_D_SUMMARY_TSV}" "${D_TARGET_SEEDS}" <<'PY'
import pathlib
import re
import statistics
import sys

archive_dir = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
summary_md = pathlib.Path(sys.argv[3])
summary_d_tsv = pathlib.Path(sys.argv[4])
target_seed_list = [s.strip() for s in sys.argv[5].split(",") if s.strip()]

rows = []
for console_path in sorted(archive_dir.glob("seed*/pr1413_*/console.txt")):
    run_dir = console_path.parent
    run_id = run_dir.name
    seed = run_dir.parent.name.replace("seed", "")
    text = console_path.read_text(errors="replace")

    def grab(pattern):
        m = re.search(pattern, text, re.M)
        return m.group(1) if m else ""

    rows.append({
        "seed": seed,
        "run_id": run_id,
        "steps": grab(r"stopping_early: wallclock_cap train_time: \d+ms step: (\d+)/20000"),
        "train_ms": grab(r"stopping_early: wallclock_cap train_time: (\d+)ms"),
        "quant_bpb": grab(r"quantized val_loss:[^\n]* val_bpb:([0-9.]+)"),
        "sliding_bpb": grab(r"quantized_sliding_window val_loss:[^\n]* val_bpb:([0-9.]+)"),
        "sliding_eval_ms": grab(r"quantized_sliding_window val_loss:[^\n]* eval_time:([0-9]+)ms"),
        "ttt_bpb": grab(r"legal_ttt_exact val_loss:[^\n]* val_bpb:([0-9.]+)"),
        "ttt_eval_ms": grab(r"legal_ttt_exact val_loss:[^\n]* eval_time:([0-9]+)ms"),
        "total_bytes": grab(r"Total submission size quantized\+brotli: ([0-9]+) bytes"),
        "code_bytes": grab(r"Code size: ([0-9]+) bytes"),
        "model_bytes": grab(r"Serialized model quantized\+brotli: ([0-9]+) bytes"),
    })

rows.sort(key=lambda r: (int(r["seed"] or "0"), r["run_id"]))
headers = ["seed", "run_id", "steps", "train_ms", "quant_bpb", "sliding_bpb", "sliding_eval_ms", "ttt_bpb", "ttt_eval_ms", "code_bytes", "model_bytes", "total_bytes"]

summary_tsv.write_text(
    "\t".join(headers) + "\n" +
    "\n".join("\t".join(row.get(h, "") for h in headers) for row in rows) + "\n"
)

md_lines = [
    "# Run Summary",
    "",
    f"Archive: `{archive_dir.name}`",
    "",
    "| seed | run_id | steps | train_ms | quant_bpb | sliding_bpb | sliding_eval_ms | ttt_bpb | ttt_eval_ms | code_bytes | model_bytes | total_bytes |",
    "|------|--------|-------|----------|-----------|-------------|-----------------|---------|-------------|------------|-------------|-------------|",
]
for row in rows:
    md_lines.append(
        "| " + " | ".join(row.get(h, "") or "" for h in headers) + " |"
    )
summary_md.write_text("\n".join(md_lines) + "\n")

d_rows = [row for row in rows if row["run_id"].startswith("pr1413_combo_")]
d_lines = []
if d_rows:
    d_rows.sort(key=lambda row: int(row["seed"]))
    d_lines.extend([
        "",
        "## D Summary",
        "",
        "| seed | run_id | ttt_bpb | total_bytes | train_ms | ttt_eval_ms |",
        "|------|--------|---------|-------------|----------|-------------|",
    ])
    for row in d_rows:
        d_lines.append(
            f"| {row['seed']} | {row['run_id']} | {row['ttt_bpb']} | {row['total_bytes']} | {row['train_ms']} | {row['ttt_eval_ms']} |"
        )
    bpb_values = [float(row["ttt_bpb"]) for row in d_rows if row["ttt_bpb"]]
    byte_values = [int(row["total_bytes"]) for row in d_rows if row["total_bytes"]]
    seen_seed_set = {row["seed"] for row in d_rows}
    missing_seed_list = [seed for seed in target_seed_list if seed not in seen_seed_set]

    summary_d_tsv.write_text(
        "seed\trun_id\tttt_bpb\ttotal_bytes\ttrain_ms\tttt_eval_ms\n" +
        "\n".join(
            "\t".join([
                row["seed"],
                row["run_id"],
                row["ttt_bpb"],
                row["total_bytes"],
                row["train_ms"],
                row["ttt_eval_ms"],
            ])
            for row in d_rows
        ) + "\n"
    )

    if bpb_values:
        d_lines.extend([
            "",
            f"- seeds found: {len(d_rows)}",
            f"- mean ttt_bpb: {sum(bpb_values) / len(bpb_values):.8f}",
        ])
        if len(bpb_values) >= 2:
            d_lines.append(f"- sample stddev: {statistics.stdev(bpb_values):.8f}")
    if byte_values:
        worst = max(byte_values)
        d_lines.append(f"- worst total_bytes: {worst} (margin {16000000 - worst})")
    d_lines.append(f"- target seeds: {', '.join(target_seed_list)}")
    d_lines.append(f"- missing seeds: {', '.join(missing_seed_list) if missing_seed_list else '(none)'}")

    summary_md.write_text(summary_md.read_text() + "\n".join(d_lines) + "\n")
else:
    summary_d_tsv.write_text("seed\trun_id\tttt_bpb\ttotal_bytes\ttrain_ms\tttt_eval_ms\n")
PY
echo "    Wrote ${LOCAL_SUMMARY_TSV}"
echo "    Wrote ${LOCAL_SUMMARY_MD}"
echo "    Wrote ${LOCAL_D_SUMMARY_TSV}"

echo "==> Step 8: Summary"
echo "Archive contents:"
find "${LOCAL_EXTRACT_DIR}" -type f | sort | while read -r f; do
  size=$(stat --printf='%s' "$f" 2>/dev/null || stat -f '%z' "$f" 2>/dev/null)
  echo "  $(printf '%12s' "$size")  ${f#${LOCAL_PULL_DIR}/}"
done

echo ""
echo "==> Quick metrics extraction:"
for d in "${LOCAL_EXTRACT_DIR}"/seed*/pr1413_*/; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  echo "--- ${run} ---"
  grep 'legal_ttt_exact' "${d}/console.txt" 2>/dev/null || grep 'ttt_sliding:done' "${d}/console.txt" 2>/dev/null || echo "  (no TTT result)"
  grep 'Total submission size' "${d}/console.txt" 2>/dev/null || echo "  (no submission size)"
done

echo ""
echo "==> Step 9: D submission summary"
python3 - "${LOCAL_SUMMARY_TSV}" "${D_TARGET_SEEDS}" <<'PY'
import csv
import pathlib
import statistics
import sys

summary_tsv = pathlib.Path(sys.argv[1])
target_seed_list = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
rows = list(csv.DictReader(summary_tsv.open(), delimiter="\t"))
d_rows = [row for row in rows if row["run_id"].startswith("pr1413_combo_")]
if not d_rows:
    print("(no D combo runs found)")
    raise SystemExit(0)

d_rows.sort(key=lambda row: int(row["seed"]))
print("Run D (parallel_residual_start=7, loop_start=3, loop_end=5)")
print(f"Seeds found: {len(d_rows)}")
for row in d_rows:
    print(
        f"  seed{row['seed']}: bpb={row['ttt_bpb']} bytes={row['total_bytes']} "
        f"train={row['train_ms']} eval={row['ttt_eval_ms']}"
    )

bpb_values = [float(row["ttt_bpb"]) for row in d_rows if row["ttt_bpb"]]
byte_values = [int(row["total_bytes"]) for row in d_rows if row["total_bytes"]]
if bpb_values:
    print(f"  {len(bpb_values)}-seed mean BPB: {sum(bpb_values) / len(bpb_values):.8f}")
    if len(bpb_values) >= 2:
        print(f"  sample stddev: {statistics.stdev(bpb_values):.8f}")
if byte_values:
    worst = max(byte_values)
    print(f"  Worst-case artifact: {worst} bytes (cap: 16000000, margin: {16000000 - worst})")

seen_seed_set = {row["seed"] for row in d_rows}
missing_seed_list = [seed for seed in target_seed_list if seed not in seen_seed_set]
print(f"  Target seeds: {', '.join(target_seed_list)}")
print(f"  Missing seeds: {', '.join(missing_seed_list) if missing_seed_list else '(none)'}")

print()
if not missing_seed_list:
    print("  STATUS: target seed set complete. Ready for PR submission assessment.")
else:
    print(f"  STATUS: Only {len(d_rows)} seed(s). Need {len(missing_seed_list)} more for target set.")
PY

echo ""
echo "============================================"
echo "  Archive pulled successfully."
echo "  You can now TERMINATE the pod."
echo "============================================"
echo ""
echo "To terminate via RunPod CLI:"
echo "  runpodctl stop pod <POD_ID>"
echo "  runpodctl remove pod <POD_ID>"
