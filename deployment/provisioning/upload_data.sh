#!/usr/bin/env bash
set -euo pipefail

: "${DROPLETS:?Set DROPLETS to a space-separated list of ssh targets}"
DATA_DIR="${DATA_DIR:-outputs/final_npy}"

for host in ${DROPLETS}; do
  ssh "$host" "sudo mkdir -p /opt/harmonia/data && sudo chown \$USER /opt/harmonia/data"
  scp "${DATA_DIR}"/*.npy "${DATA_DIR}"/metadata_*.csv "$host:/opt/harmonia/data/"
done
