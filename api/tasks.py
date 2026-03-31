import os
import sys
import json
import logging
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv

# Ensure the backend root is on sys.path so `pipeline.*` imports work
# regardless of how the worker process is launched (Railway, local, etc.)
_backend_root = str(Path(__file__).resolve().parent.parent)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery = Celery("buildai", broker=REDIS_URL, backend=REDIS_URL)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def _update_run(run_id: str, **fields):
    from api.database import SessionLocal
    from api.models import Run

    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            for key, value in fields.items():
                setattr(run, key, value)
            db.commit()
    finally:
        db.close()


@celery.task(bind=True, max_retries=1)
def process_run(self, run_id: str, video_path: str, imu_path: str):
    from api.models import RunStatus

    output_dir = os.path.join(os.path.dirname(video_path), "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        _update_run(run_id, status=RunStatus.processing, progress=0.0)

        def on_progress(pct: float):
            _update_run(run_id, progress=pct)
            self.update_state(state="PROGRESS", meta={"progress": pct})

        from pipeline.run import run_pipeline
        result = run_pipeline(video_path, imu_path, output_dir, progress_cb=on_progress)

        _update_run(
            run_id,
            status=RunStatus.done,
            progress=100.0,
            output_path=output_dir,
            frame_count=result.get("frame_count"),
            duration_s=result.get("duration_s"),
            rpe_rmse=result.get("rpe_rmse"),
            poses=result.get("poses"),
        )
        return {"run_id": run_id, "status": "done"}

    except Exception as exc:
        logger.exception("Pipeline failed for run %s", run_id)
        _update_run(
            run_id,
            status=RunStatus.failed,
            error_message=str(exc),
        )
        raise self.retry(exc=exc, countdown=10)
