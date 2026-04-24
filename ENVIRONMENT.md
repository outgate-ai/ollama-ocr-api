# Environment variables

`ocr-api` is configured via CLI flags and environment variables. **CLI flags take precedence** over env vars, which take precedence over built-in defaults.

## Server-level variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_HOST` | `127.0.0.1` (bare) / `0.0.0.0` (Docker) | Host interface to bind. |
| `OCR_API_PORT` | `11436` | TCP port. Ollama uses `11434`, `openai-privacy-filter-api` uses `11435`; we use `11436` so multiple Ollama-compatible services can run side by side. |
| `OCR_API_DEVICE` | `auto` | `auto` lets `accelerate`/`transformers` pick (`device_map="auto"` — GPU if available, else CPU). Force with `cpu` or `cuda`. |
| `OCR_API_DTYPE` | `auto` | Model dtype. `auto` uses the dtype from the checkpoint config. Override with `bfloat16`, `float16`, or `float32`. |
| `OCR_API_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warning`, or `error`. |

## Model variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_MODEL_ID` | `zai-org/GLM-OCR` | Hugging Face repo id for the vision model. Any checkpoint loadable via `AutoModelForImageTextToText` should work (GLM-OCR, Qwen2-VL, GOT-OCR2, etc.) but only GLM-OCR is tested. The model is downloaded to `$HF_HOME` on first startup. |
| `OCR_API_MODEL_NAME` | `glm-ocr` | Name advertised on `GET /api/tags` **and required** as the `model` field on `POST /api/chat`. Requests with any other `model` value return HTTP 404. Change this if you want to hide behind a different identifier for your clients. |

## Request-behavior variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_DEFAULT_PROMPT` | `Text Recognition:` | Prompt substituted when a client sends only images with empty `content`. GLM-OCR accepts `Text Recognition:`, `Formula Recognition:`, `Table Recognition:`, or a JSON schema for info extraction. |
| `OCR_API_MAX_NEW_TOKENS` | `8192` | Default generation cap when the client doesn't set `options.num_predict`. Applies per-request. |
| `OCR_API_MAX_IMAGE_BYTES` | `20971520` (20 MiB) | Maximum decoded-image size in bytes per image in a request. Requests with oversized images return HTTP 400. |

## Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_AUTH_TOKEN` | unset (open) | Shared secret required by all `/api/*` endpoints when set. Clients send `Authorization: Bearer <token>` or `X-API-Key: <token>`. Missing/wrong tokens return HTTP 401 with `WWW-Authenticate: Bearer`. `/health` is always open. Constant-time comparison; failed attempts logged at `warning`. |

## Upstream Hugging Face variables (passed through)

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Cache location for downloaded model weights. Docker images set this to `/home/app/.cache/huggingface`. Mount a volume here to persist downloads across container restarts. |
| `HF_TOKEN` | Required only if the model repo is gated/private. GLM-OCR is public — leave unset. |
| `HF_HUB_OFFLINE` | Set to `1` to forbid any outbound HF request — only uses already-cached weights. |
| `TRANSFORMERS_OFFLINE` | Same idea, older flag. |

## Precedence summary

```
CLI flag  >  OCR_API_* env var  >  built-in default
```

Per-request `options.*` fields (`num_predict`, `temperature`, `top_p`, `top_k`) override the server-side `OCR_API_MAX_NEW_TOKENS` and sampling defaults for that one request only. Unknown option keys are silently ignored.

## Examples

Bare:
```bash
OCR_API_DEVICE=cpu ocr-api
```

Docker with a persistent model cache and auth:
```bash
docker run -d --name ocr-api --restart unless-stopped --gpus all \
  -p 11436:11436 \
  -v ocr-models:/home/app/.cache/huggingface \
  -e OCR_API_AUTH_TOKEN="$(openssl rand -hex 32)" \
  ghcr.io/outgate-ai/ollama-ocr-api:latest-cuda
```
