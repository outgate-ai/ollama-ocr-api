"""Config resolution from env vars and CLI overrides."""

from __future__ import annotations

import pytest

from ocr_api.config import Config

_ENV_KEYS = (
    "OCR_API_HOST",
    "OCR_API_PORT",
    "OCR_API_DEVICE",
    "OCR_API_DTYPE",
    "OCR_API_MODEL_ID",
    "OCR_API_MODEL_NAME",
    "OCR_API_DEFAULT_PROMPT",
    "OCR_API_MAX_NEW_TOKENS",
    "OCR_API_MAX_IMAGE_BYTES",
    "OCR_API_LOG_LEVEL",
    "OCR_API_AUTH_TOKEN",
)


def test_defaults_when_no_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    cfg = Config.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11436
    assert cfg.device == "auto"
    assert cfg.dtype == "auto"
    assert cfg.model_id == "zai-org/GLM-OCR"
    assert cfg.model_name == "glm-ocr"
    assert cfg.default_prompt == "Text Recognition:"
    assert cfg.max_new_tokens_default == 8192
    assert cfg.max_image_bytes == 20 * 1024 * 1024
    assert cfg.log_level == "info"
    assert cfg.auth_token is None


def test_env_values_applied(monkeypatch):
    monkeypatch.setenv("OCR_API_HOST", "0.0.0.0")
    monkeypatch.setenv("OCR_API_PORT", "8080")
    monkeypatch.setenv("OCR_API_DEVICE", "cpu")
    monkeypatch.setenv("OCR_API_DTYPE", "float16")
    monkeypatch.setenv("OCR_API_MODEL_ID", "other/model")
    monkeypatch.setenv("OCR_API_MODEL_NAME", "other-model")
    monkeypatch.setenv("OCR_API_AUTH_TOKEN", "t0k")
    cfg = Config.from_env()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080
    assert cfg.device == "cpu"
    assert cfg.dtype == "float16"
    assert cfg.model_id == "other/model"
    assert cfg.model_name == "other-model"
    assert cfg.auth_token == "t0k"


def test_invalid_device_rejected(monkeypatch):
    monkeypatch.setenv("OCR_API_DEVICE", "tpu")
    with pytest.raises(ValueError):
        Config.from_env()


def test_invalid_dtype_rejected(monkeypatch):
    monkeypatch.setenv("OCR_API_DTYPE", "bf32")
    with pytest.raises(ValueError):
        Config.from_env()


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("OCR_API_PORT", "9000")
    monkeypatch.setenv("OCR_API_DEVICE", "cpu")
    cfg = Config.from_env().override(port=1234, device="cuda")
    assert cfg.port == 1234
    assert cfg.device == "cuda"
