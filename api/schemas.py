from __future__ import annotations
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class RunCreate(BaseModel):
    video_path: str
    imu_path: str


class PoseEntry(BaseModel):
    frame_id: int
    timestamp_s: float
    position: dict[str, float]
    orientation: dict[str, float]
    objects: list[dict] | None = None
    skeleton: dict | None = None
    depth_map_path: str | None = None


class RunResponse(BaseModel):
    id: UUID
    status: str
    video_path: str
    imu_path: str
    output_path: str | None
    rpe_rmse: float | None
    frame_count: int | None
    duration_s: float | None
    progress: float | None
    error_message: str | None
    created_at: datetime | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class RunListResponse(BaseModel):
    runs: list[RunResponse]
    total: int
