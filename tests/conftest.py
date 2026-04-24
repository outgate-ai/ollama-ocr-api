"""Shared fixtures with a fake engine so tests don't load real model weights."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_api.engine import OcrResult
from ocr_api.server import create_app


class FakeEngine:
    def __init__(self, response_text: str = "HELLO WORLD") -> None:
        self._response_text = response_text
        self.state = "ready"
        self.error: str | None = None
        self.load_calls = 0
        self.calls: list[dict] = []

    def load(self) -> None:
        self.load_calls += 1
        self.state = "ready"

    def generate(
        self,
        *,
        prompt: str,
        images_b64: list[str],
        max_new_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> OcrResult:
        if self.state != "ready":
            raise RuntimeError(f"engine not ready (state={self.state})")
        self.calls.append(
            {
                "prompt": prompt,
                "num_images": len(images_b64),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        return OcrResult(
            text=self._response_text,
            prompt_tokens=len(prompt),
            output_tokens=len(self._response_text),
        )


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(fake_engine: FakeEngine) -> TestClient:
    app = create_app(engine=fake_engine, model_name="glm-ocr")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tiny_png_b64() -> str:
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
