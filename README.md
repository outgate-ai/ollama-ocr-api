# ollama-ocr-api

Ollama-compatible HTTP server for vision/OCR models. Ships with [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B params, MIT) as the default. Drop-in replacement for Ollama's `/api/chat` when the "model" is actually an OCR model — send a base64-encoded image, get back the recognized text.

Built by [@outgate-ai](https://github.com/outgate-ai). Sibling project to [openai-privacy-filter-api](https://github.com/outgate-ai/openai-privacy-filter-api).

[![ci](https://github.com/outgate-ai/ollama-ocr-api/actions/workflows/ci.yml/badge.svg)](https://github.com/outgate-ai/ollama-ocr-api/actions/workflows/ci.yml)
[![docker](https://github.com/outgate-ai/ollama-ocr-api/actions/workflows/docker.yml/badge.svg)](https://github.com/outgate-ai/ollama-ocr-api/actions/workflows/docker.yml)

## Quick start

### Docker (recommended)

GPU (NVIDIA):
```bash
docker run -d --name ocr-api --restart unless-stopped --gpus all \
  -p 11436:11436 \
  -v ocr-models:/home/app/.cache/huggingface \
  ghcr.io/outgate-ai/ollama-ocr-api:latest-cuda
```

CPU:
```bash
docker run -d --name ocr-api --restart unless-stopped \
  -p 11436:11436 \
  -v ocr-models:/home/app/.cache/huggingface \
  ghcr.io/outgate-ai/ollama-ocr-api:latest
```

The named volume persists the ~2GB GLM-OCR weights across container restarts; first startup downloads them from Hugging Face.

### Python

```bash
pip install git+https://github.com/outgate-ai/ollama-ocr-api.git
ocr-api --device auto
```

## Call it

```bash
IMG_B64=$(base64 -i ./document.png | tr -d '\n')

curl -s http://127.0.0.1:11436/api/chat \
  -H 'content-type: application/json' \
  -d "$(cat <<EOF
{
  "model": "glm-ocr",
  "messages": [
    {"role": "user", "content": "Text Recognition:", "images": ["$IMG_B64"]}
  ],
  "stream": false
}
EOF
)"
```

Response:
```json
{
  "model": "glm-ocr",
  "created_at": "2026-04-24T13:10:12.123456Z",
  "message": {
    "role": "assistant",
    "content": "The recognized text from the image..."
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 1243582100,
  "prompt_eval_count": 128,
  "eval_count": 457
}
```

## Contract

### `POST /api/chat`

- **Request shape** — Ollama-compatible: `{ model, messages, stream, options? }`
- **Images** live on the last user message: `messages[-1].images: [base64, ...]`. At least one image is required; each can optionally use a `data:image/...;base64,` prefix.
- **`content`** is used as the prompt. If empty, falls back to `OCR_API_DEFAULT_PROMPT` (default `"Text Recognition:"`).
- **`model`** must equal the configured server name (default `glm-ocr`, set via `OCR_API_MODEL_NAME`). Any other value returns **HTTP 404**. This prevents clients from sending requests intended for `llama3` / etc. and getting silently answered by an OCR model.
- **`stream: true`** → **HTTP 400**. Sync responses only for v0.1.
- **`options`** supported: `num_predict`, `temperature`, `top_p`, `top_k`. Anything else is silently ignored.
- **`message.content`** in the response is the generated text, plain string — not fenced, not JSON.

### Supported GLM-OCR prompts

| `content` | What it does |
|-----------|--------------|
| `Text Recognition:` | Full document text extraction (default) |
| `Formula Recognition:` | LaTeX/math formula extraction |
| `Table Recognition:` | Table-structured output (HTML/Markdown) |
| JSON schema string | Structured information extraction into the schema |

See the [model card](https://huggingface.co/zai-org/GLM-OCR) for details.

### Other endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/tags` | Lists the one advertised model — for Ollama client compatibility. |
| `GET /api/version` | Server version. |
| `GET /health` | Returns engine state: `loading` / `ready` / `error`. Never requires auth. |
| `POST /api/generate` | **400** — use `/api/chat`. |
| `POST /api/embeddings` | **400** — not supported. |

### Headers

- `x-request-id` — echoed back if set by the caller, otherwise a UUID is generated and returned.
- `authorization: Bearer <token>` or `x-api-key: <token>` — **required on all `/api/*` endpoints when `OCR_API_AUTH_TOKEN` is set.** Unauthenticated requests return HTTP 401 with `WWW-Authenticate: Bearer`.

## Configuration

All settings are available as CLI flags and env vars. **CLI > env > default.** See [ENVIRONMENT.md](ENVIRONMENT.md) for the full table.

| Flag | Env var | Default |
|------|---------|---------|
| `--host` | `OCR_API_HOST` | `127.0.0.1` |
| `--port` | `OCR_API_PORT` | `11436` |
| `--device` | `OCR_API_DEVICE` | `auto` (`cpu`/`cuda`/`auto`) |
| `--dtype` | `OCR_API_DTYPE` | `auto` |
| `--model-id` | `OCR_API_MODEL_ID` | `zai-org/GLM-OCR` |
| `--model-name` | `OCR_API_MODEL_NAME` | `glm-ocr` |
| `--default-prompt` | `OCR_API_DEFAULT_PROMPT` | `Text Recognition:` |
| `--max-new-tokens` | `OCR_API_MAX_NEW_TOKENS` | `8192` |
| `--max-image-bytes` | `OCR_API_MAX_IMAGE_BYTES` | `20971520` (20 MiB) |
| `--log-level` | `OCR_API_LOG_LEVEL` | `info` |
| `--auth-token` | `OCR_API_AUTH_TOKEN` | unset (open) |

## Production deployment

```bash
sudo docker run -d --name ocr-api --restart unless-stopped --gpus all \
  -p 11436:11436 \
  -v ocr-models:/home/app/.cache/huggingface \
  -e OCR_API_AUTH_TOKEN="$(openssl rand -hex 32)" \
  ghcr.io/outgate-ai/ollama-ocr-api:latest-cuda

sudo docker logs -f ocr-api   # wait for "vision model ready"
```

**Upgrade without re-downloading the model:**
```bash
sudo docker pull ghcr.io/outgate-ai/ollama-ocr-api:latest-cuda
sudo docker stop ocr-api && sudo docker rm ocr-api
# re-run the original docker run — the named volume is reused
```

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

The test suite uses a fake engine — runs in under a second, no model download.

## Notes & limitations

- **Streaming is not implemented.** Every request is sync. OCR outputs are typically short enough (<1k tokens) that this is fine. Streaming may land in a future release.
- **Single-request serialization.** Generation calls are serialized with a lock; one GPU, one request at a time. For throughput, run multiple containers behind a load balancer.
- **Default model is tested; others are not.** The engine uses `AutoModelForImageTextToText`, so other compatible models (Qwen2-VL, GOT-OCR2, etc.) should load via `OCR_API_MODEL_ID`, but they may need a different `OCR_API_DEFAULT_PROMPT` and haven't been verified.

## License

Apache 2.0. The default model (GLM-OCR) ships under MIT + Apache 2.0 (layout component) — see its [model card](https://huggingface.co/zai-org/GLM-OCR).
