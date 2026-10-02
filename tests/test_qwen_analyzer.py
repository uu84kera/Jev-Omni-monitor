"""Test Qwen incident preparation without loading the model."""

from __future__ import annotations

import json
from pathlib import Path

from jev_monitor import qwen_analyzer


class FakeAnalyzer:
    model_id = "fake-qwen"

    def analyze(self, clip_path: Path, use_audio: bool) -> dict:
        assert clip_path.is_file()
        assert use_audio is True
        return {
            "is_anomaly": False,
            "anomaly_type": "none",
            "severity": "none",
            "summary": "Ordinary pet behavior.",
            "confidence": 0.9,
            "evidence": ["The dog is standing normally."],
            "recommended_action": "No action.",
        }


def test_parse_json_response_accepts_wrapped_json() -> None:
    response = "Result:\n```json\n" + json.dumps(
        {
            "is_anomaly": True,
            "anomaly_type": "fall",
            "severity": "high",
            "summary": "A fall is visible.",
            "confidence": 0.85,
            "evidence": ["Standing, then on floor."],
            "recommended_action": "Notify the owner.",
        }
    ) + "\n```"

    parsed = qwen_analyzer.parse_json_response(response)

    assert parsed["anomaly_type"] == "fall"
    assert parsed["severity"] == "high"


def test_incident_clip_is_capped_around_peak() -> None:
    bounds = qwen_analyzer.incident_clip_bounds(
        {
            "started_at_seconds": 10.0,
            "ended_at_seconds": 30.0,
            "peak_window_start_seconds": 20.0,
        },
        pre_roll_seconds=1.0,
        post_roll_seconds=1.0,
        max_clip_seconds=8.0,
    )

    assert bounds.start == 16.5
    assert bounds.end == 24.5


def test_analyze_incident_file_writes_structured_output(tmp_path: Path, monkeypatch) -> None:
    incidents = tmp_path / "incidents.jsonl"
    incidents.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "camera_id": "video-1",
                "started_at_seconds": 2.0,
                "ended_at_seconds": 4.0,
                "peak_window_start_seconds": 2.5,
                "status": "pending_analysis",
            }
        )
        + "\n"
    )
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "video-1.mp4").write_bytes(b"video")

    def fake_extract(video_path: Path, output_path: Path, bounds) -> None:
        output_path.write_bytes(b"clip")

    monkeypatch.setattr(qwen_analyzer, "extract_clip", fake_extract)
    monkeypatch.setattr(qwen_analyzer, "video_has_audio", lambda path: True)
    output_path = tmp_path / "output.jsonl"

    output = qwen_analyzer.analyze_incident_file(
        incidents,
        video_dir,
        output_path,
        FakeAnalyzer(),
    )

    assert output[0]["status"] == "analyzed"
    assert output[0]["action"] == "log_only"
    assert output[0]["analysis"]["provider"] == "fake-qwen"
    assert output[0]["media"]["audio_used"] is True
    assert json.loads(output_path.read_text())["analysis"]["is_anomaly"] is False
