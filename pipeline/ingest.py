from __future__ import annotations
import csv
import logging
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    video_path: Path
    imu_path: Path
    frame_count: int
    fps: float
    duration_s: float
    width: int
    height: int
    imu_samples: int
    imu_data: np.ndarray  # (N, 7): timestamp, ax, ay, az, gx, gy, gz


def validate_video(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
        raise ValueError(f"Unsupported video format: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count == 0:
        raise ValueError("Video has no frames")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps > 0 else 0,
    }


def validate_imu(path: Path) -> np.ndarray:
    """Read IMU CSV: timestamp, ax, ay, az, gx, gy, gz."""
    if not path.exists():
        raise FileNotFoundError(f"IMU file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"IMU file must be CSV, got: {path.suffix}")

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            rows.append([float(v) for v in row[:7]])

    if not rows:
        raise ValueError("IMU CSV has no data rows")

    data = np.array(rows, dtype=np.float64)
    logger.info("Loaded %d IMU samples from %s", len(data), path)
    return data


def ingest(video_path: str, imu_path: str) -> IngestResult:
    vpath = Path(video_path)
    ipath = Path(imu_path)

    logger.info("Ingesting video=%s imu=%s", vpath, ipath)

    video_info = validate_video(vpath)
    imu_data = validate_imu(ipath)

    result = IngestResult(
        video_path=vpath,
        imu_path=ipath,
        frame_count=video_info["frame_count"],
        fps=video_info["fps"],
        duration_s=video_info["duration_s"],
        width=video_info["width"],
        height=video_info["height"],
        imu_samples=len(imu_data),
        imu_data=imu_data,
    )

    logger.info(
        "Ingested: %d frames @ %.1ffps (%.1fs), %d IMU samples, %dx%d",
        result.frame_count, result.fps, result.duration_s,
        result.imu_samples, result.width, result.height,
    )
    return result
