FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir git+https://github.com/meta-pytorch/OpenEnv.git

COPY adaptive_boss/server/requirements.txt /app/server-requirements.txt
RUN pip install --no-cache-dir -r /app/server-requirements.txt

COPY adaptive_boss/ /app/adaptive_boss/

ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV ENABLE_WEB_INTERFACE=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "adaptive_boss.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
