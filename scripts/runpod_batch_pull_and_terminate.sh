#!/usr/bin/env bash
set -euo pipefail

# Pull one or more optimization_batch_* result directories from a RunPod pod,
# preserve them locally, build a combined summary, and optionally delete the pod.
#
# Usage (on LOCAL machine):
#   SSH_PORT=<port> bash scripts/runpod_batch_pull_and_terminate.sh \
#     root@<pod-ip> optimization_batch_<stamp> [optimization_batch_<stamp> ...]
#
# Optional env:
#   BUNDLE_NAME=runpod_r_experiments_<stamp>    Local bundle folder / remote tarball stem
#   ENV_LABEL=<label>                           Label passed to scripts/runpod_capture_env.sh
#   DELETE_POD=1                                Delete the pod after successful pull/extract
#   POD_ID=<runpod-pod-id>                      Required when DELETE_POD=1
#   RUNPOD_API_KEY=<token>                      Required by scripts/runpod_retry_h100.py delete

usage() {
  cat <<'EOF'
Usage:
  SSH_PORT=<port> bash scripts/runpod_batch_pull_and_terminate.sh \
    root@<pod-ip> optimization_batch_<stamp> [optimization_batch_<stamp> ...]

Example:
  SSH_PORT=10530 POD_ID=gh5ldtpcft14va DELETE_POD=1 \
    bash scripts/runpod_batch_pull_and_terminate.sh \
    root@103.207.149.51 \
    optimization_batch_20260409_122546 \
    optimization_batch_20260409_131357
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

TARGET="$1"
shift
BATCH_DIRS=("$@")
SSH_PORT="${SSH_PORT:-}"
BUNDLE_NAME="${BUNDLE_NAME:-runpod_r_experiments_$(date +%Y%m%d_%H%M%S)}"
ENV_LABEL="${ENV_LABEL:-${BUNDLE_NAME}}"
DELETE_POD="${DELETE_POD:-0}"
POD_ID="${POD_ID:-}"

if [[ "${DELETE_POD}" == "1" ]]; then
  if [[ -z "${POD_ID}" ]]; then
    echo "DELETE_POD=1 requires POD_ID" >&2
    exit 1
  fi
  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "DELETE_POD=1 requires RUNPOD_API_KEY" >&2
    exit 1
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PULL_ROOT="${REPO_ROOT}/artifacts/runpod_pull"
LOCAL_BUNDLE_DIR="${LOCAL_PULL_ROOT}/${BUNDLE_NAME}"
LOCAL_TARBALL="${LOCAL_BUNDLE_DIR}/${BUNDLE_NAME}.tar.gz"
LOCAL_SHA256="${LOCAL_BUNDLE_DIR}/${BUNDLE_NAME}.tar.gz.sha256"
LOCAL_MANIFEST="${LOCAL_BUNDLE_DIR}/pull_manifest.txt"
LOCAL_SUMMARY_TSV="${LOCAL_BUNDLE_DIR}/combined_summary.tsv"
LOCAL_SUMMARY_MD="${LOCAL_BUNDLE_DIR}/combined_summary.md"
LOCAL_EVAL_RANKING_TSV="${LOCAL_BUNDLE_DIR}/eval_ranking.tsv"

REMOTE_TARBALL="/workspace/${BUNDLE_NAME}.tar.gz"
REMOTE_SHA256="${REMOTE_TARBALL}.sha256"
REMOTE_ENV_FILE="/workspace/parameter-golf/logs/${ENV_LABEL}.env.txt"
REMOTE_REPO_STATE_FILE="/workspace/parameter-golf/logs/${ENV_LABEL}.repo_state.txt"

SSH_CMD=(ssh)
SCP_CMD=(scp)
if [[ -n "${SSH_PORT}" ]]; then
  SSH_CMD+=(-p "${SSH_PORT}")
  SCP_CMD+=(-P "${SSH_PORT}")
fi

if [[ -d "${LOCAL_BUNDLE_DIR}" ]] && [[ -n "$(find "${LOCAL_BUNDLE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Local bundle dir already exists and is not empty: ${LOCAL_BUNDLE_DIR}" >&2
  echo "Choose a new BUNDLE_NAME or remove the existing directory first." >&2
  exit 1
fi
mkdir -p "${LOCAL_BUNDLE_DIR}"

echo "==> Step 1: Capture final environment snapshot on pod"
"${SSH_CMD[@]}" "${TARGET}" \
  "cd /workspace/parameter-golf && if test -f scripts/runpod_capture_env.sh; then bash scripts/runpod_capture_env.sh ${ENV_LABEL}; else echo '  (skipping: scripts/runpod_capture_env.sh missing on pod)'; fi"

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

echo "==> Step 3: Verify remote batch dirs and stamp them with env/provenance"
cat <<'EOF' | "${SSH_CMD[@]}" "${TARGET}" bash -s -- "${ENV_LABEL}" "${BUNDLE_NAME}" "${BATCH_DIRS[@]}"
set -euo pipefail
env_label="$1"
shift
bundle_name="$1"
shift

remote_env="/workspace/parameter-golf/logs/${env_label}.env.txt"
remote_repo_state="/workspace/parameter-golf/logs/${env_label}.repo_state.txt"

for batch in "$@"; do
  test -d "/workspace/${batch}"
  test -f "/workspace/${batch}/summary.tsv"
  test -f "/workspace/${batch}/final_ranking.txt"
  mkdir -p "/workspace/${batch}/repo_scripts"
  if [[ -f "${remote_env}" ]]; then
    cp "${remote_env}" "/workspace/${batch}/" || true
  fi
  if [[ -f "${remote_repo_state}" ]]; then
    cp "${remote_repo_state}" "/workspace/${batch}/" || true
  fi
  for script_name in run_pod_batch.sh runpod_1413.sh runpod_prepare_sp8192.sh runpod_capture_env.sh; do
    if [[ -f "/workspace/parameter-golf/scripts/${script_name}" ]]; then
      cp "/workspace/parameter-golf/scripts/${script_name}" "/workspace/${batch}/repo_scripts/" || true
    fi
  done
done

cd /workspace
rm -f "${bundle_name}.tar.gz" "${bundle_name}.tar.gz.sha256"
tar czf "${bundle_name}.tar.gz" "$@"
sha256sum "${bundle_name}.tar.gz" > "${bundle_name}.tar.gz.sha256"
EOF

echo "==> Step 4: Pull tarball and checksum locally"
"${SCP_CMD[@]}" "${TARGET}:${REMOTE_TARBALL}" "${LOCAL_TARBALL}"
"${SCP_CMD[@]}" "${TARGET}:${REMOTE_SHA256}" "${LOCAL_SHA256}"

echo "==> Step 5: Verify and extract locally"
(
  cd "${LOCAL_BUNDLE_DIR}"
  sha256sum -c "$(basename "${LOCAL_SHA256}")"
)
tar xzf "${LOCAL_TARBALL}" -C "${LOCAL_BUNDLE_DIR}"

{
  echo "pulled_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "target=${TARGET}"
  echo "ssh_port=${SSH_PORT:-22}"
  echo "bundle_name=${BUNDLE_NAME}"
  echo "batches=$(IFS=,; echo "${BATCH_DIRS[*]}")"
  echo "delete_pod_requested=${DELETE_POD}"
  if [[ -n "${POD_ID}" ]]; then
    echo "pod_id=${POD_ID}"
  fi
} > "${LOCAL_MANIFEST}"

echo "==> Step 6: Build combined local summary"
python3 - "${LOCAL_BUNDLE_DIR}" "${LOCAL_SUMMARY_TSV}" "${LOCAL_SUMMARY_MD}" "${LOCAL_EVAL_RANKING_TSV}" "${BATCH_DIRS[@]}" <<'PY'
import csv
import pathlib
import re
import sys

bundle_dir = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
summary_md = pathlib.Path(sys.argv[3])
eval_ranking_tsv = pathlib.Path(sys.argv[4])
batch_names_requested = sys.argv[5:]


def grab(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.M)
    return m.group(1) if m else ""


def run_kind(name: str) -> str:
    if name.endswith("_train"):
        return "tier2_train"
    if name.endswith("_eval"):
        return "tier2_eval"
    if name.startswith("R"):
        return "tier1_eval"
    return "unknown"


rows = []
for batch_name in batch_names_requested:
    batch_dir = bundle_dir / batch_name
    summary_path = batch_dir / "summary.tsv"
    if not summary_path.is_file():
        continue
    with summary_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            run_name = row.get("run_name", "")
            run_dir = batch_dir / run_name
            output_log = run_dir / "output.log"
            log_text = output_log.read_text(errors="replace") if output_log.is_file() else ""
            archive_ckpt = run_dir / "archive" / "final_model.int6.ptz"

            rows.append({
                "batch_name": batch_dir.name,
                "timestamp": row.get("timestamp", ""),
                "run_name": run_name,
                "run_kind": run_kind(run_name),
                "status": "FAILED" if row.get("bpb", "") == "FAILED" else "OK",
                "bpb": row.get("bpb", ""),
                "wall_seconds": row.get("wall_seconds", ""),
                "env_vars": row.get("env_vars", ""),
                "quant_bpb": grab(r"quantized val_loss:[^\n]* val_bpb:([0-9.]+)", log_text),
                "sliding_bpb": grab(r"quantized_sliding_window val_loss:[^\n]* val_bpb:([0-9.]+)", log_text),
                "ttt_bpb": (
                    grab(r"legal_ttt_exact val_loss:[^\n]* val_bpb:([0-9.]+)", log_text)
                    or grab(r"ttt_sliding:done val_loss=[^\n]* val_bpb=([0-9.]+)", log_text)
                ),
                "train_steps": grab(r"stopping_early: wallclock_cap train_time: \d+ms step: (\d+)/20000", log_text),
                "train_ms": grab(r"stopping_early: wallclock_cap train_time: (\d+)ms", log_text),
                "code_bytes": grab(r"Code size: ([0-9]+) bytes", log_text),
                "model_bytes": grab(r"Serialized model quantized\+brotli: ([0-9]+) bytes", log_text),
                "total_bytes": grab(r"Total submission size quantized\+brotli: ([0-9]+) bytes", log_text),
                "archive_ckpt_bytes": str(archive_ckpt.stat().st_size) if archive_ckpt.is_file() else "",
            })


def sort_key(row: dict[str, str]):
    if row["status"] != "OK":
        return (1, float("inf"), row["batch_name"], row["run_name"])
    try:
        bpb = float(row["bpb"])
    except ValueError:
        bpb = float("inf")
    return (0, bpb, row["batch_name"], row["run_name"])


rows.sort(key=sort_key)
headers = [
    "batch_name",
    "timestamp",
    "run_name",
    "run_kind",
    "status",
    "bpb",
    "wall_seconds",
    "quant_bpb",
    "sliding_bpb",
    "ttt_bpb",
    "train_steps",
    "train_ms",
    "code_bytes",
    "model_bytes",
    "total_bytes",
    "archive_ckpt_bytes",
    "env_vars",
]

summary_tsv.write_text(
    "\t".join(headers) + "\n" +
    "\n".join("\t".join(row.get(h, "") for h in headers) for row in rows) + "\n"
)

eval_rows = [row for row in rows if row["status"] == "OK" and row["run_kind"] in {"tier1_eval", "tier2_eval"}]
eval_ranking_tsv.write_text(
    "batch_name\trun_name\trun_kind\tbpb\twall_seconds\tenv_vars\n" +
    "\n".join(
        "\t".join([
            row["batch_name"],
            row["run_name"],
            row["run_kind"],
            row["bpb"],
            row["wall_seconds"],
            row["env_vars"],
        ])
        for row in eval_rows
    ) + "\n"
)

batch_names = list(batch_names_requested)

def best_of(kind: str):
    for row in rows:
        if row["status"] == "OK" and row["run_kind"] == kind:
            return row
    return None


best_overall = next((row for row in rows if row["status"] == "OK"), None)
best_tier1 = best_of("tier1_eval")
best_tier2 = best_of("tier2_eval")

md = [
    "# RunPod Batch Pull Summary",
    "",
    f"- bundle: `{bundle_dir.name}`",
    f"- batches: {', '.join(f'`{name}`' for name in batch_names) if batch_names else '(none)'}",
    f"- total runs: {len(rows)}",
    f"- successful runs: {sum(1 for row in rows if row['status'] == 'OK')}",
    f"- failed runs: {sum(1 for row in rows if row['status'] != 'OK')}",
    "",
]

if best_overall:
    md.append(f"- best overall: `{best_overall['run_name']}` in `{best_overall['batch_name']}` at `{best_overall['bpb']}` BPB")
if best_tier1:
    md.append(f"- best tier1 eval: `{best_tier1['run_name']}` at `{best_tier1['bpb']}` BPB")
if best_tier2:
    md.append(f"- best tier2 eval: `{best_tier2['run_name']}` at `{best_tier2['bpb']}` BPB")

md.extend([
    "",
    "## Eval Ranking",
    "",
    "| batch | run_name | run_kind | bpb | wall_seconds | env_vars |",
    "|-------|----------|----------|-----|--------------|----------|",
])
for row in eval_rows:
    md.append(
        f"| {row['batch_name']} | {row['run_name']} | {row['run_kind']} | {row['bpb']} | {row['wall_seconds']} | {row['env_vars']} |"
    )

md.extend([
    "",
    "## All Runs",
    "",
    "| batch | run_name | run_kind | status | bpb | wall_seconds | quant_bpb | sliding_bpb | ttt_bpb | total_bytes | archive_ckpt_bytes |",
    "|-------|----------|----------|--------|-----|--------------|-----------|-------------|---------|-------------|--------------------|",
])
for row in rows:
    md.append(
        f"| {row['batch_name']} | {row['run_name']} | {row['run_kind']} | {row['status']} | {row['bpb']} | {row['wall_seconds']} | {row['quant_bpb']} | {row['sliding_bpb']} | {row['ttt_bpb']} | {row['total_bytes']} | {row['archive_ckpt_bytes']} |"
    )

summary_md.write_text("\n".join(md) + "\n")
PY

echo "==> Local bundle ready"
echo "  ${LOCAL_BUNDLE_DIR}"
echo "  ${LOCAL_SUMMARY_TSV}"
echo "  ${LOCAL_SUMMARY_MD}"
echo "  ${LOCAL_EVAL_RANKING_TSV}"

if [[ "${DELETE_POD}" == "1" ]]; then
  echo "==> Step 7: Delete pod ${POD_ID}"
  python3 "${REPO_ROOT}/scripts/runpod_retry_h100.py" delete "${POD_ID}"
else
  echo "==> Pod deletion skipped"
  if [[ -n "${POD_ID}" ]]; then
    echo "  To delete later:"
    echo "  RUNPOD_API_KEY=... python3 scripts/runpod_retry_h100.py delete ${POD_ID}"
  fi
fi

echo ""
echo "============================================"
echo "  Pull complete. Local archive is preserved."
echo "============================================"
