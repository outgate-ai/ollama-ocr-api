# CPU image — python-slim base, CPU-only torch.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_API_HOST=0.0.0.0 \
    OCR_API_PORT=11436 \
    OCR_API_DEVICE=cpu \
    HF_HOME=/home/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" "torchvision" \
    && pip install .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /home/app

EXPOSE 11436

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${OCR_API_PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
