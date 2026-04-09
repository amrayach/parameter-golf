#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_sync_repo.sh <user@host> [remote_dir] [--rsync]

Defaults:
  remote_dir=/workspace/parameter-golf
  SSH_PORT unset (uses default SSH port 22)

Behavior:
  1. Clone the current git origin onto the remote if the repo is absent.
  2. Otherwise fetch and fast-forward pull on the remote.
  3. If --rsync is passed, push the current working tree on top without touching
     remote dataset caches or .git metadata.
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 1
fi

TARGET="$1"
REMOTE_DIR="${2:-/workspace/parameter-golf}"
SYNC_MODE="${3:-}"

if [[ -n "${SYNC_MODE}" && "${SYNC_MODE}" != "--rsync" ]]; then
  echo "Unsupported sync mode: ${SYNC_MODE}" >&2
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGIN_URL="$(git -C "${REPO_ROOT}" remote get-url origin)"
REMOTE_PARENT="$(dirname "${REMOTE_DIR}")"
SSH_PORT="${SSH_PORT:-}"
SSH_CMD=(ssh)
if [[ -n "${SSH_PORT}" ]]; then
  SSH_CMD+=(-p "${SSH_PORT}")
fi

echo "==> Remote git sync: ${TARGET}:${REMOTE_DIR}"
"${SSH_CMD[@]}" "${TARGET}" "
  set -euo pipefail
  mkdir -p '${REMOTE_PARENT}'
  if [[ -d '${REMOTE_DIR}/.git' ]]; then
    cd '${REMOTE_DIR}'
    git fetch --all --prune
    git pull --ff-only
  elif [[ -d '${REMOTE_DIR}' ]] && [[ -n \"\$(ls -A '${REMOTE_DIR}')\" ]]; then
    echo '==> Remote dir exists but is not a git checkout; leaving it in place for rsync overlay'
  else
    git clone '${ORIGIN_URL}' '${REMOTE_DIR}'
  fi
"

if [[ "${SYNC_MODE}" == "--rsync" ]]; then
  echo "==> Overlay local working tree via rsync"
  RSYNC_SSH=(ssh)
  if [[ -n "${SSH_PORT}" ]]; then
    RSYNC_SSH+=(-p "${SSH_PORT}")
  fi
  rsync -az \
    -e "$(printf '%q ' "${RSYNC_SSH[@]}")" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'logs/' \
    --exclude 'data/datasets/' \
    --exclude 'data/tokenizers/' \
    "${REPO_ROOT}/" "${TARGET}:${REMOTE_DIR}/"
fi

echo "==> Sync complete"
