# CPU image — python-slim base, CPU-only torch.
#
# Layer ordering matches Dockerfile.cuda: heavy, rarely-changing layers
# (torch, OS packages, user setup) come first; application code is last.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_API_HOST=0.0.0.0 \
    OCR_API_PORT=11436 \
    OCR_API_DEVICE=cpu \
    HF_HOME=/home/app/.cache/huggingface

# --- Layer 1: OS packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl gosu \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 2: torch + CPU wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" "torchvision"

# --- Layer 3: unprivileged user + entrypoint.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# --- Layer 4: project metadata.
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./

# --- Layer 5: application source + project pip install. Only this
# layer re-pushes on a typical code-change release.
COPY src ./src
RUN pip install .

WORKDIR /home/app

EXPOSE 11436

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${OCR_API_PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
