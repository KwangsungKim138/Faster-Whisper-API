<div align="center">

# **Faster-Whisper-API**

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

A lean **FastAPI wrapper around `faster-whisper`** for local transcription on **GPU or CPU**.
Designed for **inference-only** on an internal server: upload an audio/video file → get full text + timestamped segments.

> Backend: [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) (CT2).
> Default: `large-v3`, `device="cuda"` if available (otherwise `cpu`), `compute_type="float16"` on GPU / `float32` on CPU.
</div>

---

## Features

* **Single process / single model instance** (warm & fast)
* **Audio or video input**
* **VAD filter** to suppress non-speech
* **Segment timestamps** (+ optional word timestamps)
* **Configurable language** (default `ko`)
* **File-size guardrails** (`MAX_AUDIO_BYTES`)
* Works on **CUDA GPUs** *and* **pure CPU** (no NVIDIA required)

---

## Project Structure (reference)

```
Faster-Whisper-API/
├─ app/
│  ├─ main.py
│  ├─ deps.py                 # WhisperModel singleton
│  ├─ routers/
│  │   └─ transcribe.py       # /transcribe, /health
│  ├─ schemas.py              # Pydantic I/O models
│  ├─ services/
│  │   ├─ transcriber.py
│  │   └─ audio_processor.py
│  └─ config/
│      └─ settings.py
├─ requirements.txt
└─ Dockerfile
```

---

## Requirements

* **Python 3.10+**
* **ffmpeg** installed on the system
* **GPU (optional)**: recent NVIDIA driver & CUDA for best performance
  → If no GPU is present, the API runs on **CPU** automatically when configured.

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
# or
pip install fastapi uvicorn[standard] pydub ffmpeg-python python-dotenv faster-whisper
```

Make sure `ffmpeg` is available on PATH.

### 2) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 3) Health Check

```bash
curl http://localhost:8000/health
```

---

## API

### POST `/transcribe`

**Form data**

* `file` (required): audio/video file (`.wav`, `.mp3`, `.mp4`, `.mkv`, `.mov`, `.m4a`, `.ogg`, …)
* `query` (JSON string, optional):

  * `request_id`: string
  * `language`: default `"ko"`
  * `is_video`: default `false`
  * `start`: int (seconds)
  * `end`: int (seconds)
  * `vad`: bool, default `true`
  * `word_timestamps`: bool, default `false`

**Response**

```json
{
  "id": "uuid",
  "request_id": "optional",
  "language": "ko",
  "duration": 123.45,
  "created_at": "YYYY-MM-DD HH:MM:SS.mmm",
  "result": {
    "text": "full transcription text ...",
    "segments": [
      { "index": 0, "start": 0.50, "end": 2.10, "content": "..." }
    ]
  }
}
```

**cURL example**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@/path/to/sample.wav" \
  -F 'query={"language":"ko","vad":true,"word_timestamps":false}'
```

---

## Configuration

`app/config/settings.py` (defaults)

* `MAX_AUDIO_BYTES`: e.g., `25 * 1024 * 1024`
* `DEFAULT_SR`: 16000
* `DEFAULT_BR`: `'96k'`
* `DEFAULT_CH`: 1
* `MAX_CHUNK_DURATION_MS`: e.g., 2h05m

`app/deps.py` (model init)

* `model_name`: `"large-v3"`, `"medium"`, etc.
* `device`: `"cuda"` **or** `"cpu"`
* `compute_type` (recommendations):

  * GPU (CUDA): `"float16"` *(fast & accurate)*, or `"int8_float16"` *(lower VRAM)*
  * CPU: `"float32"` *(default & stable)*, or `"int8"` / `"int8_float32"` *(lower RAM, faster on some CPUs)*

> **Tip (CPU)**: Set environment variable `OMP_NUM_THREADS` or `CT2_NUM_THREADS` to tune CPU threads.

---

## Selecting GPU vs CPU

You can force the backend via environment variables (read by `deps.py`) or by editing `WhisperFactory`:

```bash
# CPU mode (no CUDA required)
export FW_DEVICE=cpu
export FW_COMPUTE=float32   # or int8 / int8_float32
uvicorn app.main:app --host 0.0.0.0 --port 8000

# GPU mode (if CUDA available)
export FW_DEVICE=cuda
export FW_COMPUTE=float16   # or int8_float16
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example `deps.py` snippet:

```python
import os
from functools import lru_cache
from faster_whisper import WhisperModel

DEVICE = os.getenv("FW_DEVICE", "cuda")   # "cuda" or "cpu"
COMPUTE = os.getenv("FW_COMPUTE", "float16" if DEVICE == "cuda" else "float32")
MODEL = os.getenv("FW_MODEL", "large-v3")

@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    return WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
```

---

## Accuracy & Speed Notes

* Keep `temperature=0.0` to reduce repetition/hallucination.
* For long programs with music, `condition_on_previous_text=False` stabilizes first utterances after silence.
* **VAD** helps suppress long non-speech stretches.
* **Word timestamps** add latency—use only when needed.
* **CPU mode**: prefer `"float32"` first; try `"int8"` variants if you need to save RAM or speed up on large files.

---

## Docker

### CUDA (GPU)

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3-pip python3-dev ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
```

Run:

```bash
docker build -t faster-whisper-api .
docker run --gpus all -p 8000:8000 \
  -e FW_DEVICE=cuda -e FW_COMPUTE=float16 \
  faster-whisper-api
```

### CPU-only

If you don’t have NVIDIA GPUs, use a standard Python base image:

```dockerfile
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["bash","-lc","export FW_DEVICE=cpu FW_COMPUTE=float32 && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"]
```

---

## Security & Ops

* Internal use recommended; add **token auth** if exposed.
* Enforce upload size limits in app and reverse proxy (e.g., Nginx `client_max_body_size`).
* If multiple clients share one host, consider a queue; one worker keeps the model hot.

---

## Roadmap

* [ ] Streaming/chunked transcription endpoint
* [ ] Diarization hook (external)
* [ ] Batch job mode & webhook callback
* [ ] Optional language auto-detect

---

## Attribution

This project wraps **`faster-whisper`** (CT2).
See its license and the model licenses you deploy.

---

## License

MIT (project code). Models remain under their respective licenses.

---# Faster-Whisper-API
