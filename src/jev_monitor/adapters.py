"""Implement motion, model, media, and event-storage adapters."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageStat

from .domain import (
    AnomalyAnalysis,
    ClassificationResult,
    Event,
    MotionResult,
    Severity,
    TemporalWindow,
)

# calculate the difference between three frames and return a motion score
class PillowMotionGate:
    def __init__(self, threshold: float, force_every: int = 10) -> None:
        self.threshold = threshold
        self.force_every = max(force_every, 1)
        self._seen = 0

    def evaluate(self, window: TemporalWindow) -> MotionResult:
        images = [Image.open(frame.path).convert("L").resize((160, 90)) for frame in window.frames]
        pair_scores = []
        for left, right in zip(images, images[1:]):
            mean = ImageStat.Stat(ImageChops.difference(left, right)).mean[0]
            pair_scores.append(mean / 255.0)
        score = max(pair_scores)
        self._seen += 1
        forced = self._seen % self.force_every == 0
        return MotionResult(score=round(score, 6), should_classify=score >= self.threshold or forced, forced=forced)

# combine three frames into a single contact sheet for classification
def build_contact_sheet(window: TemporalWindow) -> Path:
    opened = [Image.open(frame.path).convert("RGB") for frame in window.frames]
    tile_width, tile_height = 512, 288
    canvas = Image.new("RGB", (tile_width * 3, tile_height + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(opened, start=1):
        image.thumbnail((tile_width, tile_height))
        x = (index - 1) * tile_width + (tile_width - image.width) // 2
        y = (tile_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(((index - 1) * tile_width + 12, tile_height + 10), f"t{index}", fill="black")
    handle = NamedTemporaryFile(prefix="jev-window-", suffix=".jpg", delete=False)
    handle.close()
    path = Path(handle.name)
    canvas.save(path, quality=90)
    return path

# mock implementation for Jev Calssification locally
class MockJevClassifier:
    name = "mock-jev"

    def classify(self, window: TemporalWindow, contact_sheet: Path) -> ClassificationResult:
        images = [Image.open(frame.path).convert("L").resize((160, 90)) for frame in window.frames]
        change = max(
            ImageStat.Stat(ImageChops.difference(left, right)).mean[0] / 255.0
            for left, right in zip(images, images[1:])
        )
        suspicious = min(0.99, 0.10 + change * 4.5)
        return ClassificationResult(
            safe=round(1.0 - suspicious, 4),
            suspicious=round(suspicious, 4),
            provider=self.name,
        )

# call Jev api
class HttpJevClassifier:
    name = "http-jev"

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def classify(self, window: TemporalWindow, contact_sheet: Path) -> ClassificationResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with contact_sheet.open("rb") as media:
            response = httpx.post(
                self.endpoint,
                headers=headers,
                data={
                    "state": f"Three ordered frames from camera {window.camera_id}.",
                    "question": "Does this sequence contain an event requiring further analysis?",
                    "options_json": json.dumps(["safe", "suspicious"]),
                },
                files={"media": (contact_sheet.name, media, "image/jpeg")},
                timeout=90,
            )
        response.raise_for_status()
        data = response.json()
        probabilities = data["probabilities"]
        return ClassificationResult(
            safe=float(probabilities["safe"]),
            suspicious=float(probabilities["suspicious"]),
            provider=str(data.get("model", self.name)),
        )

# mock VLM for analysis
class MockVisionAnalyzer:
    name = "mock-vlm"

    def analyze(self, window: TemporalWindow) -> AnomalyAnalysis:
        return AnomalyAnalysis(
            anomaly_type="unusual_motion",
            severity=Severity.MEDIUM,
            summary="The sequence contains a large visual change that needs review.",
            confidence=0.78,
            evidence=["large change across the ordered frames"],
            recommended_action="queue for review",
            provider=self.name,
        )


ANOMALY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "anomaly_type",
        "severity",
        "summary",
        "confidence",
        "evidence",
        "recommended_action",
    ],
    "properties": {
        "anomaly_type": {"type": "string"},
        "severity": {"enum": [item.value for item in Severity]},
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string"},
    },
}

# call VLM api
class HttpVisionAnalyzer:
    name = "http-vlm"

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def analyze(self, window: TemporalWindow) -> AnomalyAnalysis:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        frames = []
        for frame in window.frames:
            frames.append(
                {
                    "mime_type": "image/jpeg",
                    "data_base64": base64.b64encode(frame.path.read_bytes()).decode("ascii"),
                }
            )
        response = httpx.post(
            self.endpoint,
            headers=headers,
            json={
                "frames": frames,
                "context": {
                    "camera_id": window.camera_id,
                    "captured_at": window.frames[-1].captured_at.isoformat(),
                },
                "response_schema": ANOMALY_SCHEMA,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return AnomalyAnalysis(
            anomaly_type=data["anomaly_type"],
            severity=Severity(data["severity"]),
            summary=data["summary"],
            confidence=float(data["confidence"]),
            evidence=list(data["evidence"]),
            recommended_action=data["recommended_action"],
            provider=self.name,
        )

# JSONL log event
class JsonlEventSink:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: Event) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
