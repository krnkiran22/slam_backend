from __future__ import annotations
import csv
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

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


def _load_imu_data(imu_path: str) -> np.ndarray:
    """Load IMU CSV into (N, 7) array: [timestamp_s, ax, ay, az, gx, gy, gz]."""
    rows = []
    with open(imu_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            ts = float(row[0])
            if ts > 1e12:
                ts /= 1e9
            elif ts > 1e9:
                ts /= 1e6
            vals = [ts] + [float(v) for v in row[1:7]]
            rows.append(vals)
    return np.array(rows, dtype=np.float64)


def _run_python_vio(video_path: str, imu_path: str) -> VIOResult:
    """
    Python-based VIO using KLT optical flow + IMU gyro integration.
    Produces real (non-zero) 6DoF poses from video + IMU data.
    """
    from pipeline.preprocess import K, DIST, compute_undistort_maps

    logger.info("Running Python VIO (KLT + IMU fusion)")

    imu_data = _load_imu_data(imu_path)
    if len(imu_data) == 0:
        logger.error("No IMU data loaded")
        return _stub_vio(video_path)

    imu_timestamps = imu_data[:, 0]
    imu_gyro = imu_data[:, 4:7]
    imu_accel = imu_data[:, 1:4]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    map1, map2, new_K = compute_undistort_maps(width, height)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    position = np.zeros(3, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    orientation = Rotation.identity()

    gravity = np.array([0.0, 9.81, 0.0])

    prev_gray = None
    prev_pts = None
    poses = []
    frame_id = 0
    imu_idx = 0
    prev_ts = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop_x = int(w * 0.1)
        crop_y = int(h * 0.1)
        roi_gray = gray[crop_y:h - crop_y, crop_x:w - crop_x]

        ts = frame_id / fps
        dt = ts - prev_ts if prev_ts is not None else 1.0 / fps

        # --- IMU integration: find IMU samples for this frame ---
        gyro_avg = np.zeros(3)
        accel_avg = np.zeros(3)
        imu_count = 0

        frame_ts_imu = imu_timestamps[0] + ts if imu_timestamps[0] > 1000 else ts
        while imu_idx < len(imu_data) - 1 and imu_timestamps[imu_idx] < frame_ts_imu:
            gyro_avg += imu_gyro[imu_idx]
            accel_avg += imu_accel[imu_idx]
            imu_count += 1
            imu_idx += 1

        if imu_count > 0:
            gyro_avg /= imu_count
            accel_avg /= imu_count
        else:
            accel_avg = np.array([0.0, 9.81, 0.0])

        # Integrate gyroscope for rotation
        angle = np.linalg.norm(gyro_avg) * dt
        if angle > 1e-8:
            axis = gyro_avg / np.linalg.norm(gyro_avg)
            delta_rot = Rotation.from_rotvec(axis * angle)
            orientation = orientation * delta_rot

        # --- Visual odometry: KLT feature tracking ---
        visual_translation = np.zeros(3)

        if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 8:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )

            if curr_pts is not None:
                good_mask = status.ravel() == 1
                pts_prev = prev_pts[good_mask]
                pts_curr = curr_pts[good_mask]

                if len(pts_prev) >= 8:
                    E, mask_e = cv2.findEssentialMat(
                        pts_prev, pts_curr, new_K,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    if E is not None and mask_e is not None:
                        inliers = mask_e.ravel() == 1
                        if np.sum(inliers) >= 5:
                            _, R_rel, t_rel, _ = cv2.recoverPose(
                                E, pts_prev[inliers], pts_curr[inliers], new_K
                            )
                            # Scale from IMU acceleration magnitude
                            accel_world = orientation.apply(accel_avg) - gravity
                            accel_mag = np.linalg.norm(accel_world)
                            scale = min(accel_mag * dt * dt * 0.5, 0.05)
                            visual_translation = t_rel.ravel() * scale

        # Fuse: use visual translation rotated into world frame
        R_world = orientation.as_matrix()
        translation_world = R_world @ visual_translation

        # Integrate IMU acceleration for position (complementary with visual)
        accel_world = orientation.apply(accel_avg) - gravity
        alpha = 0.7  # visual weight
        imu_pos_delta = accel_world * dt * dt * 0.5

        if np.linalg.norm(visual_translation) > 1e-6:
            position += alpha * translation_world + (1.0 - alpha) * imu_pos_delta
            velocity = velocity * 0.95 + translation_world / max(dt, 1e-6) * 0.05
        else:
            velocity *= 0.98
            position += velocity * dt

        roll, pitch, yaw = orientation.as_euler("xyz")

        poses.append(PoseFrame(
            frame_id=frame_id,
            timestamp_s=round(ts, 6),
            x=round(float(position[0]), 6),
            y=round(float(position[1]), 6),
            z=round(float(position[2]), 6),
            roll=round(float(roll), 6),
            pitch=round(float(pitch), 6),
            yaw=round(float(yaw), 6),
        ))

        # Detect new features periodically
        kps = fast.detect(roi_gray, None)
        kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:150]
        if kps:
            prev_pts = np.array(
                [[kp.pt[0] + crop_x, kp.pt[1] + crop_y] for kp in kps],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
        else:
            prev_pts = None

        prev_gray = gray
        prev_ts = ts
        frame_id += 1

        if frame_id % 100 == 0:
            logger.info(
                "VIO frame %d: pos=(%.3f, %.3f, %.3f) rpy=(%.2f, %.2f, %.2f)",
                frame_id, position[0], position[1], position[2], roll, pitch, yaw,
            )

    cap.release()
    logger.info("Python VIO complete: %d poses", len(poses))
    return VIOResult(poses=poses, backend="python_vio")


def run_openvins(
    video_path: str,
    imu_path: str,
    config_path: str = "configs/openvins_gen4.yaml",
) -> VIOResult:
    """
    Run OpenVINS on video + IMU data.

    Expects OpenVINS built and available at OPENVINS_BIN env var,
    or falls back to Python VIO.
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
                "Using Python VIO (KLT + IMU fusion).",
                openvins_bin,
            )
            return _run_python_vio(video_path, imu_path)


def _stub_vio(video_path: str) -> VIOResult:
    """Placeholder VIO that generates identity poses. For dev/testing only."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.warning("Using STUB VIO — poses are identity.")

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
    elif backend == "python":
        return _run_python_vio(video_path, imu_path)
    elif backend == "basalt":
        cfg = config_path or "configs/basalt_gen4.json"
        logger.warning("BASALT backend not yet implemented, falling back to Python VIO")
        return _run_python_vio(video_path, imu_path)
    else:
        raise ValueError(f"Unknown VIO backend: {backend}")
