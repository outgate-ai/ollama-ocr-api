"""Unit tests for the base64 image decoding helper."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from ocr_api.engine import decode_base64_image


def _png_bytes() -> bytes:
    img = Image.new("RGB", (2, 2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_plain_base64():
    b64 = base64.b64encode(_png_bytes()).decode("ascii")
    decoded = decode_base64_image(b64, max_bytes=1_000_000)
    assert decoded.data == _png_bytes()


def test_decode_data_url():
    b64 = base64.b64encode(_png_bytes()).decode("ascii")
    decoded = decode_base64_image(f"data:image/png;base64,{b64}", max_bytes=1_000_000)
    assert decoded.data == _png_bytes()
    assert decoded.mime == "image/png"


def test_decode_invalid_base64_raises():
    with pytest.raises(ValueError):
        decode_base64_image("!!!", max_bytes=1_000_000)


def test_decode_rejects_empty():
    with pytest.raises(ValueError):
        decode_base64_image("", max_bytes=1_000_000)


def test_decode_rejects_oversize():
    b64 = base64.b64encode(_png_bytes()).decode("ascii")
    with pytest.raises(ValueError):
        decode_base64_image(b64, max_bytes=1)
