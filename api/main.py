import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.database import engine
from api.models import Base
from api.routes import runs, stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Build AI Pipeline API",
    description="VIO + Scene Perception Pipeline API",
    version="0.1.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://web-production-2f854.up.railway.app",
    "https://slam-frontend-seven.vercel.app",
    os.getenv("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api", tags=["runs"])
app.include_router(stream.router, prefix="/ws", tags=["stream"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
