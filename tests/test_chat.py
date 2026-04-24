"""Ollama-compatible /api/chat contract tests."""

from __future__ import annotations


def _request(
    *,
    images: list[str] | None,
    content: str = "Text Recognition:",
    stream: bool = False,
    model: str = "glm-ocr",
    options: dict | None = None,
):
    msg: dict = {"role": "user", "content": content}
    if images is not None:
        msg["images"] = images
    return {
        "model": model,
        "messages": [msg],
        "stream": stream,
        "options": options or {},
    }


def test_chat_returns_plain_text_content(client, tiny_png_b64):
    resp = client.post("/api/chat", json=_request(images=[tiny_png_b64]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "HELLO WORLD"
    assert body["model"] == "glm-ocr"
    assert body["prompt_eval_count"] > 0
    assert body["eval_count"] > 0
    assert body["total_duration"] >= 0


def test_chat_passes_prompt_through(client, fake_engine, tiny_png_b64):
    client.post("/api/chat", json=_request(images=[tiny_png_b64], content="Table Recognition:"))
    assert fake_engine.calls[-1]["prompt"] == "Table Recognition:"


def test_chat_defaults_prompt_when_content_empty(client, fake_engine, tiny_png_b64):
    client.post("/api/chat", json=_request(images=[tiny_png_b64], content=""))
    assert fake_engine.calls[-1]["prompt"] == "Text Recognition:"


def test_chat_rejects_stream_true(client, tiny_png_b64):
    resp = client.post("/api/chat", json=_request(images=[tiny_png_b64], stream=True))
    assert resp.status_code == 400


def test_chat_rejects_unknown_model(client, tiny_png_b64):
    resp = client.post(
        "/api/chat", json=_request(images=[tiny_png_b64], model="qwen-vl")
    )
    assert resp.status_code == 404


def test_chat_requires_images(client):
    resp = client.post("/api/chat", json=_request(images=None))
    assert resp.status_code == 400
    resp = client.post("/api/chat", json=_request(images=[]))
    assert resp.status_code == 400


def test_chat_rejects_invalid_base64(client):
    resp = client.post("/api/chat", json=_request(images=["!!!not-base64!!!"]))
    assert resp.status_code == 400


def test_chat_honors_num_predict(client, fake_engine, tiny_png_b64):
    client.post(
        "/api/chat",
        json=_request(images=[tiny_png_b64], options={"num_predict": 256}),
    )
    assert fake_engine.calls[-1]["max_new_tokens"] == 256


def test_chat_honors_sampling_options(client, fake_engine, tiny_png_b64):
    client.post(
        "/api/chat",
        json=_request(
            images=[tiny_png_b64],
            options={"temperature": 0.7, "top_p": 0.9, "top_k": 40},
        ),
    )
    call = fake_engine.calls[-1]
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.9
    assert call["top_k"] == 40


def test_chat_ignores_unknown_options(client, tiny_png_b64):
    resp = client.post(
        "/api/chat",
        json=_request(images=[tiny_png_b64], options={"mirostat": 2, "wat": True}),
    )
    assert resp.status_code == 200


def test_chat_503_when_engine_not_ready(client, fake_engine, tiny_png_b64):
    fake_engine.state = "loading"
    resp = client.post("/api/chat", json=_request(images=[tiny_png_b64]))
    assert resp.status_code == 503


def test_chat_500_does_not_leak_internals(client, fake_engine, tiny_png_b64):
    def _boom(**_kwargs):
        raise RuntimeError("SECRET_PATH=/etc/private/key.pem")

    fake_engine.generate = _boom
    resp = client.post("/api/chat", json=_request(images=[tiny_png_b64]))
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "SECRET_PATH" not in detail
    assert detail == "internal error during generation"


def test_chat_echoes_request_id(client, tiny_png_b64):
    resp = client.post(
        "/api/chat",
        json=_request(images=[tiny_png_b64]),
        headers={"x-request-id": "trace-xyz"},
    )
    assert resp.headers["x-request-id"] == "trace-xyz"


def test_chat_accepts_data_url_prefix(client, fake_engine, tiny_png_b64):
    data_url = f"data:image/png;base64,{tiny_png_b64}"
    resp = client.post("/api/chat", json=_request(images=[data_url]))
    assert resp.status_code == 200
