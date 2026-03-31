from __future__ import annotations
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from pipeline.vio import PoseFrame
from pipeline.perception import FramePerception, Detection

logger = logging.getLogger(__name__)


def fuse_frame(pose: PoseFrame, perception: FramePerception | None) -> dict:
    """Merge VIO pose and perception outputs into a single frame entry."""
    entry = {
        "frame_id": pose.frame_id,
        "timestamp_s": pose.timestamp_s,
        "pose": {
            "position": {"x": pose.x, "y": pose.y, "z": pose.z},
            "orientation": {"roll": pose.roll, "pitch": pose.pitch, "yaw": pose.yaw},
        },
        "objects": [],
        "skeleton": None,
        "depth_map_path": None,
    }

    if perception:
        entry["objects"] = [
            {"class": d.cls, "conf": d.conf, "bbox": d.bbox}
            for d in perception.objects
        ]
        entry["skeleton"] = perception.skeleton
        entry["depth_map_path"] = perception.depth_map_path

    return entry


def write_poses_json(frames: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(frames, f, indent=2)
    logger.info("Wrote %d frames to %s", len(frames), output_path)


def draw_annotations(
    frame: np.ndarray,
    perception: FramePerception | None,
    pose: PoseFrame | None = None,
) -> np.ndarray:
    """Draw bounding boxes, skeleton, and pose info on a frame."""
    annotated = frame.copy()

    if perception:
        for det in perception.objects:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = (0, 255, 0) if det.cls == "person" else (255, 128, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.cls} {det.conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if perception.skeleton and perception.skeleton.get("keypoints"):
            for kp in perception.skeleton["keypoints"]:
                x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                if conf > 0.3:
                    cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

    if pose:
        info = f"pos=({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f})"
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def write_annotated_video(
    video_path: str,
    frames_data: list[dict],
    perceptions: dict[int, FramePerception],
    poses: dict[int, PoseFrame],
    output_path: Path,
) -> None:
    """Write annotated video with bounding boxes, skeleton, and pose overlay."""
    from pipeline.preprocess import iterate_frames

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for idx, ts, frame in iterate_frames(Path(video_path), undistort=True):
        perception = perceptions.get(idx)
        pose = poses.get(idx)
        annotated = draw_annotations(frame, perception, pose)
        writer.write(annotated)

    writer.release()
    logger.info("Wrote annotated video to %s", output_path)


def fuse_outputs(
    video_path: str,
    vio_poses: list[PoseFrame],
    perceptions: list[FramePerception],
    output_dir: str,
) -> list[dict]:
    """Fuse VIO + perception and write poses.json + annotated_video.mp4."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    perception_map: dict[int, FramePerception] = {p.frame_id: p for p in perceptions}
    pose_map: dict[int, PoseFrame] = {p.frame_id: p for p in vio_poses}

    fused_frames = []
    for pose in vio_poses:
        perception = perception_map.get(pose.frame_id)
        fused_frames.append(fuse_frame(pose, perception))

    write_poses_json(fused_frames, out / "poses.json")

    try:
        write_annotated_video(
            video_path, fused_frames, perception_map, pose_map,
            out / "annotated_video.mp4",
        )
    except Exception as e:
        logger.warning("Failed to write annotated video: %s", e)

    return fused_frames
