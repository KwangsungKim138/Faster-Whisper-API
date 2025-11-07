# app/jobs.py
from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict
import time, uuid, threading


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    progress: float = 0.0  # 0.0 ~ 1.0
    message: str = ""
    result: Optional[dict] = None
    request_id: Optional[str] = None


_STORE: Dict[str, Job] = {}
_LOCK = threading.Lock()


def create_job(request_id: Optional[str] = None) -> Job:
    j = Job(job_id=str(uuid.uuid4()), status=JobStatus.queued, request_id=request_id)
    with _LOCK:
        _STORE[j.job_id] = j
    return j


def get_job(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _STORE.get(job_id)


def update_job(job_id: str, **fields):
    with _LOCK:
        j = _STORE.get(job_id)
        if not j:
            return
        for k, v in fields.items():
            setattr(j, k, v)
