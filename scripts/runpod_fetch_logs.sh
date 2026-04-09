#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_fetch_logs.sh <user@host> [remote_dir] [local_dir]

Defaults:
  remote_dir=/workspace/parameter-golf
  local_dir=logs/runpod_remote
  SSH_PORT unset (uses default SSH port 22)

This pulls the remote repo's logs/ directory into a local folder so exact train
logs and environment snapshots are preserved before opening a PR.
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 1
fi

TARGET="$1"
REMOTE_DIR="${2:-/workspace/parameter-golf}"
LOCAL_DIR_REL="${3:-logs/runpod_remote}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="${REPO_ROOT}/${LOCAL_DIR_REL}"
SSH_PORT="${SSH_PORT:-}"

RSYNC_SSH=(ssh)
if [[ -n "${SSH_PORT}" ]]; then
  RSYNC_SSH+=(-p "${SSH_PORT}")
fi

mkdir -p "${LOCAL_DIR}"
echo "==> Pulling remote logs from ${TARGET}:${REMOTE_DIR}/logs/ -> ${LOCAL_DIR}"
rsync -az \
  -e "$(printf '%q ' "${RSYNC_SSH[@]}")" \
  "${TARGET}:${REMOTE_DIR}/logs/" "${LOCAL_DIR}/"
echo "==> Log pull complete"
