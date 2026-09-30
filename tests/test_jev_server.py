"""Test the Jev HTTP contract without loading GPU model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from jev_monitor.jev_server import create_app


class FakeRuntime:
    model_id = "test/jev-omni"

    def __init__(self) -> None:
        self.loaded = False
        self.last_media_exists = False

    def load(self) -> None:
        self.loaded = True

    def predict(
        self,
        *,
        state: str,
        question: str,
        options: list[str],
        media: Path,
    ) -> dict[str, Any]:
        self.last_media_exists = media.is_file()
        return {
            "prediction": options[1],
            "confidence": 0.8,
            "probabilities": {options[0]: 0.2, options[1]: 0.8},
        }


def test_classify_contract() -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/classify",
            data={
                "state": "Three ordered pet-camera frames.",
                "question": "Does this require further analysis?",
                "options_json": '["safe", "suspicious"]',
            },
            files={"media": ("window.jpg", b"image-bytes", "image/jpeg")},
        )

    assert runtime.loaded
    assert runtime.last_media_exists
    assert response.status_code == 200
    assert response.json()["probabilities"] == {"safe": 0.2, "suspicious": 0.8}


def test_classify_rejects_invalid_options() -> None:
    with TestClient(create_app(FakeRuntime())) as client:
        response = client.post(
            "/v1/classify",
            data={"state": "state", "question": "question", "options_json": '["safe"]'},
            files={"media": ("window.jpg", b"image-bytes", "image/jpeg")},
        )

    assert response.status_code == 422
