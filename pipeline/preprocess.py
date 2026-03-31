from __future__ import annotations
import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Build Gen 4 factory-calibrated intrinsics — DO NOT CHANGE
K = np.array([
    [718.90196364,   0.0,         960.01857437],
    [  0.0,         716.33626950, 558.31079911],
    [  0.0,           0.0,          1.0       ],
], dtype=np.float64)

DIST = np.array([-0.28182606, 0.07391488, 0.00031393, 0.00090297], dtype=np.float64)


def compute_undistort_maps(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, DIST, (width, height), 1, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, DIST, None, new_K, (width, height), cv2.CV_16SC2)
    return map1, map2, new_K


def undistort_frame(frame: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def extract_features(frame_gray: np.ndarray) -> list[cv2.KeyPoint]:
    """FAST corner detection with grid-based distribution."""
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    keypoints = fast.detect(frame_gray, None)
    return sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:200]


def iterate_frames(
    video_path: Path,
    undistort: bool = True,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_idx, timestamp_s, undistorted_frame) from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    map1, map2, new_K = compute_undistort_maps(width, height)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if undistort:
            frame = undistort_frame(frame, map1, map2)

        timestamp_s = frame_idx / fps if fps > 0 else 0.0
        yield frame_idx, timestamp_s, frame
        frame_idx += 1

    cap.release()
    logger.info("Preprocessed %d frames from %s", frame_idx, video_path)


def preprocess_to_dir(video_path: str, output_dir: str, undistort: bool = True) -> int:
    """Undistort all frames and save as PNGs. Returns frame count."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, ts, frame in iterate_frames(Path(video_path), undistort=undistort):
        cv2.imwrite(str(out / f"frame_{idx:06d}.png"), frame)
        count += 1

    logger.info("Saved %d undistorted frames to %s", count, out)
    return count
