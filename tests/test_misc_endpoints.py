"""/api/tags, /api/version, /health, and rejection of unsupported endpoints."""

from __future__ import annotations


def test_health_reports_ready(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_version_returns_version_string(client):
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_tags_advertises_model(client):
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) == 1
    assert models[0]["name"] == "glm-ocr"


def test_generate_rejected(client):
    assert client.post("/api/generate", json={}).status_code == 400


def test_embeddings_rejected(client):
    assert client.post("/api/embeddings", json={}).status_code == 400
