#!/usr/bin/env bash
set -euo pipefail

MOUNT=/mnt/pendrive
mkdir -p "$MOUNT" && mount "$1" "$MOUNT"

[ -f "$MOUNT/video.mp4" ] || exit 1
[ -f "$MOUNT/imu.csv" ]   || exit 1

source /opt/build/backend/.venv/bin/activate

python3 /opt/build/backend/pipeline/run.py \
    --video "$MOUNT/video.mp4" \
    --imu   "$MOUNT/imu.csv"   \
    --out   "$MOUNT/output/"
