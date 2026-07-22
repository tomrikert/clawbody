import os

os.environ.setdefault("ASMO_VOICE_REFERENCE", __file__)

from fastapi.testclient import TestClient

import app as voice_app


class FakeRegistry:
    ready = True
    device = "cuda"
    stt_model_name = "distil-large-v3"
    loaded_at = 1.0


def test_health_contract(monkeypatch):
    monkeypatch.setattr(voice_app, "models", FakeRegistry())
    client = TestClient(voice_app.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["stt_model"] == "distil-large-v3"


def test_speech_rejects_unknown_voice(monkeypatch):
    monkeypatch.setattr(voice_app, "models", FakeRegistry())
    client = TestClient(voice_app.app)
    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "somebody-else"},
    )
    assert response.status_code == 400
