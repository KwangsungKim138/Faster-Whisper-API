from typing import List, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    index: int
    start: float
    end: float
    content: str
    avg_logprob: float


class TranscribeResult(BaseModel):
    request_id: Optional[str] = None
    language: str
    duration: float
    created_at: str
    result: dict  # {"text": str, "segments": List[Segment]}


class TranscribeQuery(BaseModel):
    request_id: Optional[str] = None
    language: str = "ko"
    is_video: bool = False
    start: int = 0
    end: int = 0
    vad: bool = True
    word_timestamps: bool = False
