"""Authentication tests — Bearer + X-API-Key, off-by-default."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_api.server import create_app

from .conftest import FakeEngine

TOKEN = "s3cret-abc123"


@pytest.fixture
def auth_client() -> TestClient:
    app = create_app(engine=FakeEngine(), model_name="glm-ocr", auth_token=TOKEN)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def png_b64() -> str:
    img = Image.new("RGB", (4, 4))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _chat_payload(png_b64: str) -> dict:
    return {
        "model": "glm-ocr",
        "messages": [
            {"role": "user", "content": "Text Recognition:", "images": [png_b64]}
        ],
        "stream": False,
    }


def test_missing_token_rejected(auth_client, png_b64):
    resp = auth_client.post("/api/chat", json=_chat_payload(png_b64))
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].lower().startswith("bearer")


def test_wrong_token_rejected(auth_client, png_b64):
    resp = auth_client.post(
        "/api/chat",
        json=_chat_payload(png_b64),
        headers={"authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_bearer_token_accepted(auth_client, png_b64):
    resp = auth_client.post(
        "/api/chat",
        json=_chat_payload(png_b64),
        headers={"authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200


def test_api_key_header_accepted(auth_client, png_b64):
    resp = auth_client.post(
        "/api/chat",
        json=_chat_payload(png_b64),
        headers={"x-api-key": TOKEN},
    )
    assert resp.status_code == 200


def test_tags_requires_auth(auth_client):
    assert auth_client.get("/api/tags").status_code == 401
    assert (
        auth_client.get(
            "/api/tags", headers={"authorization": f"Bearer {TOKEN}"}
        ).status_code
        == 200
    )


def test_health_open(auth_client):
    assert auth_client.get("/health").status_code == 200


def test_no_token_configured_means_open(client):
    assert client.get("/api/tags").status_code == 200
