import os
import logging
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import transcribe_async
from .middleware import timing_middleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _mb(env_key: str, default_mb: int) -> int:
    return int(os.getenv(env_key, str(default_mb))) * 1024 * 1024


MAX_FORM_MB = _mb("MAX_FORM_MB", 200)


app = FastAPI(
    title="FasterWhisperAPI",
    version="0.1.0",
    max_form_memory_size=MAX_FORM_MB,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 내부망이면 구체 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.middleware("http")(timing_middleware)

app.include_router(transcribe_async.router, prefix="")

# uvicorn app.main:app --host 0.0.0.0 --port 8000
