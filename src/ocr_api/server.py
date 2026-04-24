"""FastAPI app exposing Ollama-compatible endpoints backed by a vision model."""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import DEFAULT_OCR_PROMPT, MODEL_ID_DEFAULT, MODEL_NAME_DEFAULT, __version__
from .engine import Engine, VisionEngine, decode_base64_image
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    TagModel,
    TagsResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _last_user_message(messages: Iterable[ChatMessage]) -> ChatMessage | None:
    for msg in reversed(list(messages)):
        if msg.role == "user":
            return msg
    return None


def _extract_presented_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None
    api_key = request.headers.get("x-api-key", "").strip()
    return api_key or None


def _make_auth_dependency(expected_token: str | None):
    if not expected_token:
        async def _noop() -> None:
            return None

        return _noop

    async def _require_auth(request: Request) -> None:
        presented = _extract_presented_token(request)
        if presented is None or not hmac.compare_digest(presented, expected_token):
            client_host = request.client.host if request.client else "?"
            logger.warning(
                "auth failed rid=%s ip=%s path=%s reason=%s",
                getattr(request.state, "request_id", "?"),
                client_host,
                request.url.path,
                "missing" if presented is None else "mismatch",
            )
            raise HTTPException(
                status_code=401,
                detail="invalid or missing authentication token",
                headers={"WWW-Authenticate": 'Bearer realm="ocr-api"'},
            )

    return _require_auth


def _coerce_int_option(options: dict[str, Any], key: str) -> int | None:
    value = options.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_option(options: dict[str, Any], key: str) -> float | None:
    value = options.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_app(
    *,
    engine: Engine | None = None,
    model_id: str = MODEL_ID_DEFAULT,
    model_name: str = MODEL_NAME_DEFAULT,
    device: str = "auto",
    dtype: str = "auto",
    default_prompt: str = DEFAULT_OCR_PROMPT,
    max_new_tokens_default: int = 8192,
    max_image_bytes: int = 20 * 1024 * 1024,
    auth_token: str | None = None,
) -> FastAPI:
    owned_engine = engine is None
    if engine is None:
        engine = VisionEngine(  # type: ignore[arg-type]
            model_id=model_id,
            device=device,  # type: ignore[arg-type]
            dtype=dtype,  # type: ignore[arg-type]
            max_image_bytes=max_image_bytes,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if owned_engine:
            engine.load()
        yield

    app = FastAPI(
        title="Ollama OCR API",
        version=__version__,
        description="Ollama-compatible /api/chat server for vision/OCR models.",
        lifespan=lifespan,
    )

    require_auth = _make_auth_dependency(auth_token)
    if auth_token:
        logger.info("authentication enabled on /api/* endpoints")
    else:
        logger.info("authentication disabled (set OCR_API_AUTH_TOKEN to enable)")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed rid=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rid=%s %s %s -> %d %.1fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health")
    async def health():
        return {"status": engine.state, "error": engine.error}

    @app.get("/api/version", dependencies=[Depends(require_auth)])
    async def api_version() -> VersionResponse:
        return VersionResponse(version=__version__)

    @app.get("/api/tags", dependencies=[Depends(require_auth)])
    async def api_tags() -> TagsResponse:
        return TagsResponse(
            models=[TagModel(name=model_name, model=model_name, modified_at=_iso_now())]
        )

    @app.post("/api/chat", dependencies=[Depends(require_auth)])
    async def api_chat(req: ChatRequest):
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported; set stream=false",
            )

        if not req.model or req.model != model_name:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"model {req.model!r} not found, "
                    f"this server only serves {model_name!r}"
                ),
            )

        if engine.state != "ready":
            raise HTTPException(
                status_code=503, detail=f"engine not ready (state={engine.state})"
            )

        last = _last_user_message(req.messages)
        if last is None:
            raise HTTPException(status_code=400, detail="no user message found in request")

        images = last.images or []
        if not images:
            raise HTTPException(
                status_code=400,
                detail="user message must include at least one image (messages[].images)",
            )

        # Validate base64/size at the HTTP layer so callers get a clean 400
        # regardless of which engine implementation is wired up.
        for idx, b64 in enumerate(images):
            try:
                decode_base64_image(b64, max_image_bytes)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"images[{idx}]: {exc}",
                ) from exc

        prompt = last.content.strip() if last.content else ""
        if not prompt:
            prompt = default_prompt

        num_predict = _coerce_int_option(req.options, "num_predict")
        max_new_tokens = (
            num_predict if num_predict and num_predict > 0 else max_new_tokens_default
        )
        temperature = _coerce_float_option(req.options, "temperature")
        top_p = _coerce_float_option(req.options, "top_p")
        top_k = _coerce_int_option(req.options, "top_k")

        t0 = time.perf_counter_ns()
        try:
            result = engine.generate(
                prompt=prompt,
                images_b64=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        except ValueError as exc:
            # Bad image payload, too-large, etc. — client error, safe to surface the message.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("generation failed")
            raise HTTPException(
                status_code=500, detail="internal error during generation"
            ) from exc
        total_duration = time.perf_counter_ns() - t0

        response = ChatResponse(
            model=model_name,
            created_at=_iso_now(),
            message=ChatResponseMessage(role="assistant", content=result.text),
            done=True,
            done_reason="stop",
            total_duration=total_duration,
            prompt_eval_count=result.prompt_tokens,
            eval_count=result.output_tokens,
        )
        return JSONResponse(content=response.model_dump())

    @app.post("/api/generate")
    async def api_generate_unsupported():
        raise HTTPException(
            status_code=400, detail="/api/generate is not supported; use /api/chat"
        )

    @app.post("/api/embeddings")
    async def api_embeddings_unsupported():
        raise HTTPException(status_code=400, detail="/api/embeddings is not supported")

    return app
