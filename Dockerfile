FROM public.ecr.aws/docker/library/python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/models/parsbert \
    PORT=8000 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# Use the CPU wheel: this service is designed to run without CUDA.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.14.1"

COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --no-cache-dir --default-timeout=120 --retries=10 \
    --index-url https://package-mirror.liara.ir/repository/pypi/simple \
    -r requirements-api.txt

COPY app ./app
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /models/parsbert /home/app/.cache/huggingface \
    && chown -R app:app /app /models /home/app/.cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# One worker avoids loading an extra copy of the 500 MB-class model per process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

