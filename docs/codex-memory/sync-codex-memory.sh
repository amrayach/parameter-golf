#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${HOME}/.codex/memories/parameter-golf"

mkdir -p "${DEST_DIR}"

for f in \
  README.md \
  BOOTSTRAP.md \
  project-state.md \
  decisions.md \
  next-session.md \
  leaderboard-techniques.md \
  hardware-and-constraints.md \
  rfn-and-attribution-assessment.md
do
  cp "${SRC_DIR}/${f}" "${DEST_DIR}/"
done

echo "Synced Codex memory to ${DEST_DIR}"
