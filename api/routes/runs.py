from __future__ import annotations
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import Run, RunStatus
from api.schemas import RunCreate, RunResponse, RunListResponse

router = APIRouter()


@router.post("/runs", response_model=RunResponse, status_code=201)
def create_run(body: RunCreate, db: Session = Depends(get_db)):
    run = Run(
        video_path=body.video_path,
        imu_path=body.imu_path,
        status=RunStatus.pending,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    from api.tasks import process_run
    process_run.delay(str(run.id), body.video_path, body.imu_path)

    return run


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    skip: int = 0,
    limit: int = 50,
    status: RunStatus | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Run)
    if status:
        query = query.filter(Run.status == status)
    total = query.count()
    runs = query.order_by(Run.created_at.desc()).offset(skip).limit(limit).all()
    return RunListResponse(runs=runs, total=total)


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: UUID, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/poses")
def get_run_poses(
    run_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.poses:
        return {"frames": [], "total": 0}

    frames = run.poses[skip : skip + limit]
    return {"frames": frames, "total": len(run.poses)}


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: UUID, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete(run)
    db.commit()
