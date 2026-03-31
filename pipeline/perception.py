from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: list[float]


@dataclass
class FramePerception:
    frame_id: int
    objects: list[Detection] = field(default_factory=list)
    skeleton: dict | None = None
    depth_map: np.ndarray | None = None
    depth_map_path: str | None = None


# ── Object Detection (YOLOv8) ──────────────────────────────────────────

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
            logger.info("Loaded YOLOv8n model")
        except ImportError:
            logger.warning("ultralytics not installed — object detection disabled")
            return None
    return _yolo_model


def detect_objects(frame: np.ndarray) -> list[Detection]:
    model = _get_yolo()
    if model is None:
        return []

    results = model(frame, verbose=False)
    detections = []
    for box in results[0].boxes:
        detections.append(Detection(
            cls=model.names[int(box.cls)],
            conf=float(box.conf),
            bbox=box.xyxy[0].tolist(),
        ))
    return detections


# ── Body Skeleton (ViTPose) ────────────────────────────────────────────

_pose_model = None


def _get_vitpose():
    global _pose_model
    if _pose_model is None:
        try:
            from mmpose.apis import init_model
            _pose_model = init_model(
                "configs/body_2d_keypoint/topdown_heatmap/coco/"
                "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
                "vitpose-b.pth",
                device="cuda",
            )
            logger.info("Loaded ViTPose model")
        except (ImportError, FileNotFoundError):
            logger.warning("mmpose/ViTPose not available — skeleton estimation disabled")
            return None
    return _pose_model


def estimate_skeleton(frame: np.ndarray, bboxes: list[list[float]]) -> dict | None:
    model = _get_vitpose()
    if model is None or not bboxes:
        return None

    try:
        from mmpose.apis import inference_topdown
        results = inference_topdown(model, frame, bboxes)
        if results:
            keypoints = results[0].pred_instances.keypoints.tolist()
            scores = results[0].pred_instances.keypoint_scores.tolist()
            return {
                "keypoints": [
                    [kp[0], kp[1], float(s)]
                    for kp, s in zip(keypoints[0], scores[0])
                ],
            }
    except Exception as e:
        logger.warning("Skeleton estimation failed: %s", e)
    return None


# ── Depth Estimation (Depth Anything v2) ───────────────────────────────

_depth_model = None


def _get_depth_model():
    global _depth_model
    if _depth_model is None:
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2

            _depth_model = DepthAnythingV2(
                encoder="vitb", features=128,
                out_channels=[96, 192, 384, 768],
            )
            _depth_model.load_state_dict(
                torch.load("depth_anything_v2_vitb.pth", map_location="cpu")
            )
            _depth_model.eval()
            logger.info("Loaded Depth Anything v2 model")
        except (ImportError, FileNotFoundError):
            logger.warning("Depth Anything v2 not available — depth estimation disabled")
            return None
    return _depth_model


def estimate_depth(frame: np.ndarray) -> np.ndarray | None:
    model = _get_depth_model()
    if model is None:
        return None

    try:
        return model.infer_image(frame)
    except Exception as e:
        logger.warning("Depth estimation failed: %s", e)
        return None


# ── Per-Frame Perception ───────────────────────────────────────────────

def perceive_frame(
    frame_id: int,
    frame: np.ndarray,
    depth_output_dir: Path | None = None,
) -> FramePerception:
    """Run all perception models on a single frame."""

    objects = detect_objects(frame)

    person_bboxes = [d.bbox for d in objects if d.cls == "person"]
    skeleton = estimate_skeleton(frame, person_bboxes)

    depth_map = estimate_depth(frame)
    depth_path = None
    if depth_map is not None and depth_output_dir is not None:
        depth_output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = str(depth_output_dir / f"frame_{frame_id:06d}.npy")
        np.save(depth_path, depth_map)

    return FramePerception(
        frame_id=frame_id,
        objects=objects,
        skeleton=skeleton,
        depth_map=depth_map,
        depth_map_path=depth_path,
    )


def run_perception(
    frames: list[tuple[int, np.ndarray]],
    depth_output_dir: Path | None = None,
    max_workers: int = 1,
) -> list[FramePerception]:
    """Run perception on a batch of frames."""
    results = []
    for frame_id, frame in frames:
        result = perceive_frame(frame_id, frame, depth_output_dir)
        results.append(result)
    results.sort(key=lambda r: r.frame_id)
    return results
