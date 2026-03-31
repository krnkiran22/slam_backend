"""
Main pipeline entrypoint.

Usage:
    python pipeline/run.py --video /path/to/video.mp4 --imu /path/to/imu.csv --out ./output/
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Callable

from pipeline.ingest import ingest
from pipeline.preprocess import iterate_frames
from pipeline.vio import run_vio
from pipeline.perception import perceive_frame
from pipeline.fuse import fuse_outputs

logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: str,
    imu_path: str,
    output_dir: str,
    backend: str = "openvins",
    progress_cb: Callable[[float], None] | None = None,
) -> dict:
    """
    Run the full VIO + perception pipeline.

    Returns dict with: frame_count, duration_s, rpe_rmse, poses
    """
    start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    depth_dir = out / "depth"

    # Stage 1 — Ingest
    logger.info("Stage 1: Ingesting data")
    ingest_result = ingest(video_path, imu_path)
    if progress_cb:
        progress_cb(10.0)

    # Stage 2+3 — VIO
    logger.info("Stage 2-3: Running VIO (%s)", backend)
    vio_result = run_vio(video_path, imu_path, backend=backend)
    if progress_cb:
        progress_cb(40.0)

    # Stage 4 — Perception (per-frame)
    logger.info("Stage 4: Running perception on %d frames", ingest_result.frame_count)
    perceptions = []
    total = ingest_result.frame_count
    for idx, ts, frame in iterate_frames(Path(video_path)):
        perc = perceive_frame(idx, frame, depth_output_dir=depth_dir)
        perceptions.append(perc)

        if progress_cb and total > 0:
            pct = 40.0 + (idx / total) * 50.0
            progress_cb(min(pct, 90.0))

    if progress_cb:
        progress_cb(90.0)

    # Stage 5 — Fusion
    logger.info("Stage 5: Fusing outputs")
    fused = fuse_outputs(video_path, vio_result.poses, perceptions, output_dir)
    if progress_cb:
        progress_cb(100.0)

    elapsed = time.time() - start
    logger.info("Pipeline complete in %.1fs — %d frames", elapsed, len(fused))

    return {
        "frame_count": len(fused),
        "duration_s": ingest_result.duration_s,
        "rpe_rmse": None,
        "poses": fused,
        "elapsed_s": elapsed,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build AI VIO + Perception Pipeline")
    parser.add_argument("--video", required=True, help="Path to video.mp4")
    parser.add_argument("--imu", required=True, help="Path to imu.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--backend", default="openvins", choices=["openvins", "basalt"])
    args = parser.parse_args()

    def print_progress(pct: float):
        print(f"\rProgress: {pct:.1f}%", end="", flush=True)

    result = run_pipeline(args.video, args.imu, args.out, args.backend, print_progress)
    print(f"\nDone — {result['frame_count']} frames in {result['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
