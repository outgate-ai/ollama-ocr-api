"""Runtime configuration resolved from CLI args and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from . import MODEL_ID_DEFAULT, MODEL_NAME_DEFAULT


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"env var {key} must be an integer (got {value!r})") from exc


def _coerce_device(value: str) -> Literal["cpu", "cuda", "auto"]:
    if value not in ("cpu", "cuda", "auto"):
        raise ValueError(f"device must be 'cpu', 'cuda', or 'auto' (got {value!r})")
    return value  # type: ignore[return-value]


def _coerce_dtype(value: str) -> Literal["auto", "bfloat16", "float16", "float32"]:
    if value not in ("auto", "bfloat16", "float16", "float32"):
        raise ValueError(
            f"dtype must be auto|bfloat16|float16|float32 (got {value!r})"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    device: Literal["cpu", "cuda", "auto"]
    dtype: Literal["auto", "bfloat16", "float16", "float32"]
    model_id: str
    model_name: str
    default_prompt: str
    max_new_tokens_default: int
    max_image_bytes: int
    log_level: str
    auth_token: str | None

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            host=_env_str("OCR_API_HOST", "127.0.0.1"),
            port=_env_int("OCR_API_PORT", 11436),
            device=_coerce_device(_env_str("OCR_API_DEVICE", "auto")),
            dtype=_coerce_dtype(_env_str("OCR_API_DTYPE", "auto")),
            model_id=_env_str("OCR_API_MODEL_ID", MODEL_ID_DEFAULT),
            model_name=_env_str("OCR_API_MODEL_NAME", MODEL_NAME_DEFAULT),
            default_prompt=_env_str("OCR_API_DEFAULT_PROMPT", "Text Recognition:"),
            max_new_tokens_default=_env_int("OCR_API_MAX_NEW_TOKENS", 8192),
            max_image_bytes=_env_int(
                "OCR_API_MAX_IMAGE_BYTES", 20 * 1024 * 1024
            ),
            log_level=_env_str("OCR_API_LOG_LEVEL", "info"),
            auth_token=os.environ.get("OCR_API_AUTH_TOKEN") or None,
        )

    def override(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        device: str | None = None,
        dtype: str | None = None,
        model_id: str | None = None,
        model_name: str | None = None,
        default_prompt: str | None = None,
        max_new_tokens_default: int | None = None,
        max_image_bytes: int | None = None,
        log_level: str | None = None,
        auth_token: str | None = None,
    ) -> Config:
        return Config(
            host=host if host is not None else self.host,
            port=port if port is not None else self.port,
            device=_coerce_device(device) if device is not None else self.device,
            dtype=_coerce_dtype(dtype) if dtype is not None else self.dtype,
            model_id=model_id if model_id is not None else self.model_id,
            model_name=model_name if model_name is not None else self.model_name,
            default_prompt=(
                default_prompt if default_prompt is not None else self.default_prompt
            ),
            max_new_tokens_default=(
                max_new_tokens_default
                if max_new_tokens_default is not None
                else self.max_new_tokens_default
            ),
            max_image_bytes=(
                max_image_bytes if max_image_bytes is not None else self.max_image_bytes
            ),
            log_level=log_level if log_level is not None else self.log_level,
            auth_token=auth_token if auth_token is not None else self.auth_token,
        )
