FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FW_DEVICE=cuda \
    FW_COMPUTE=float16 \
    FW_MODEL=large-v3

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg pkg-config build-essential python3-dev \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavfilter-dev libswresample-dev libswscale-dev libavutil-dev \
 && rm -rf /var/lib/apt/lists/*

# generate venv
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
