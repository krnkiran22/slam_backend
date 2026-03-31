import enum
import uuid
from sqlalchemy import Column, String, Float, Integer, Enum, DateTime, JSON, func
from sqlalchemy.dialects.postgresql import UUID
from api.database import Base


class RunStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class Run(Base):
    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(RunStatus), default=RunStatus.pending, nullable=False)
    video_path = Column(String, nullable=False)
    imu_path = Column(String, nullable=False)
    output_path = Column(String, nullable=True)
    rpe_rmse = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    duration_s = Column(Float, nullable=True)
    error_message = Column(String, nullable=True)
    progress = Column(Float, default=0.0)
    poses = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
