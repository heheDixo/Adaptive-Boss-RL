# Dockerfile for the AdaptiveBoss OpenEnv server.
#
# Build context layout: the env directory's contents are at the build context
# root (this matches the Hugging Face Spaces deploy: the OpenEnv `push` CLI
# bundles the env directory's contents as the Space repo and relocates this
# Dockerfile from `server/Dockerfile` to the Space root).
#
# Local build (from inside adaptive_boss/):
#   cd adaptive_boss
#   docker build -t adaptive-boss-env -f server/Dockerfile .
#   docker run -p 8000:8000 adaptive-boss-env

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir git+https://github.com/meta-pytorch/OpenEnv.git

COPY server/requirements.txt /app/server-requirements.txt
RUN pip install --no-cache-dir -r /app/server-requirements.txt

# Server runtime needs models.py + game/ + server/. Skip training scripts,
# notebook, model weights, logs, blog assets — those bloat the image and
# aren't imported by the FastAPI app.
COPY models.py /app/models.py
COPY game/ /app/game/
COPY server/ /app/server/

ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV ENABLE_WEB_INTERFACE=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
