from __future__ import annotations
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoseFrame:
    frame_id: int
    timestamp_s: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class VIOResult:
    poses: list[PoseFrame] = field(default_factory=list)
    backend: str = "openvins"


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to (roll, pitch, yaw) in radians."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


def parse_tum_trajectory(tum_file: Path) -> list[PoseFrame]:
    """Parse TUM-format trajectory: timestamp tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation

    poses = []
    frame_id = 0

    with open(tum_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            rot = Rotation.from_quat([qx, qy, qz, qw])
            roll, pitch, yaw = rot.as_euler("xyz")

            poses.append(PoseFrame(
                frame_id=frame_id,
                timestamp_s=ts,
                x=tx, y=ty, z=tz,
                roll=roll, pitch=pitch, yaw=yaw,
            ))
            frame_id += 1

    return poses


def run_openvins(
    video_path: str,
    imu_path: str,
    config_path: str = "configs/openvins_gen4.yaml",
) -> VIOResult:
    """
    Run OpenVINS on video + IMU data.

    Expects OpenVINS built and available at OPENVINS_BIN env var,
    or falls back to 'run_subscribe_msckf' on PATH.
    """
    import os
    openvins_bin = os.getenv("OPENVINS_BIN", "run_subscribe_msckf")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_tum = Path(tmpdir) / "trajectory.txt"

        cmd = [
            openvins_bin,
            "--config", config_path,
            "--video", video_path,
            "--imu", imu_path,
            "--output", str(output_tum),
        ]

        logger.info("Running OpenVINS: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode != 0:
                logger.error("OpenVINS stderr: %s", result.stderr)
                raise RuntimeError(f"OpenVINS failed with code {result.returncode}")

            if not output_tum.exists():
                raise FileNotFoundError(f"OpenVINS did not produce output at {output_tum}")

            poses = parse_tum_trajectory(output_tum)
            logger.info("OpenVINS produced %d poses", len(poses))
            return VIOResult(poses=poses, backend="openvins")

        except FileNotFoundError:
            logger.warning(
                "OpenVINS binary not found at '%s'. "
                "Falling back to stub VIO. Install OpenVINS for real results.",
                openvins_bin,
            )
            return _stub_vio(video_path)


def _stub_vio(video_path: str) -> VIOResult:
    """Placeholder VIO that generates identity poses. For dev/testing only."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.warning("Using STUB VIO — poses are identity. Install OpenVINS for real tracking.")

    poses = []
    for i in range(frame_count):
        poses.append(PoseFrame(
            frame_id=i,
            timestamp_s=i / fps,
            x=0.0, y=0.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
        ))

    return VIOResult(poses=poses, backend="stub")


def run_vio(
    video_path: str,
    imu_path: str,
    backend: str = "openvins",
    config_path: str | None = None,
) -> VIOResult:
    """Run VIO with the specified backend."""
    if backend == "openvins":
        cfg = config_path or "configs/openvins_gen4.yaml"
        return run_openvins(video_path, imu_path, cfg)
    elif backend == "basalt":
        cfg = config_path or "configs/basalt_gen4.json"
        logger.warning("BASALT backend not yet implemented, falling back to stub")
        return _stub_vio(video_path)
    else:
        raise ValueError(f"Unknown VIO backend: {backend}")
