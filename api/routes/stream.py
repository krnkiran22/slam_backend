from __future__ import annotations
import json
import asyncio
from uuid import UUID
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session

from api.database import get_db, SessionLocal
from api.models import Run, RunStatus

router = APIRouter()


@router.websocket("/runs/{run_id}")
async def run_stream(websocket: WebSocket, run_id: UUID):
    await websocket.accept()

    try:
        while True:
            db = SessionLocal()
            try:
                run = db.query(Run).filter(Run.id == run_id).first()
                if not run:
                    await websocket.send_json({"error": "Run not found"})
                    break

                payload = {
                    "status": run.status.value if isinstance(run.status, RunStatus) else run.status,
                    "progress": run.progress or 0.0,
                    "frame_count": run.frame_count,
                    "rpe_rmse": run.rpe_rmse,
                }

                if run.status in (RunStatus.done, RunStatus.failed):
                    payload["error_message"] = run.error_message
                    await websocket.send_json(payload)
                    break

                await websocket.send_json(payload)
            finally:
                db.close()

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
