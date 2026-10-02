"""Test static-window filtering and suspicious-window escalation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from jev_monitor.adapters import (
    JEV_QUESTION,
    JEV_STATE_TEMPLATE,
    JsonlEventSink,
    MockJevClassifier,
    MockVisionAnalyzer,
    PillowMotionGate,
)
from jev_monitor.domain import (
    Action,
    AnomalyAnalysis,
    ClassificationResult,
    Frame,
    MotionResult,
    Severity,
    TemporalWindow,
)
from jev_monitor.incidents import JsonlIncidentSink
from jev_monitor.pipeline import MonitoringPipeline


def make_window(tmp_path: Path, colors: tuple[str, str, str]) -> TemporalWindow:
    frames = []
    for index, color in enumerate(colors):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (64, 64), color).save(path)
        frames.append(Frame(path=path, captured_at=datetime.now(timezone.utc)))
    return TemporalWindow(camera_id="test", frames=tuple(frames))  # type: ignore[arg-type]


def test_static_window_is_ignored(tmp_path: Path) -> None:
    
    pipeline = MonitoringPipeline(
        motion_gate=PillowMotionGate(threshold=0.02, force_every=10),
        classifier=MockJevClassifier(),
        analyzer=MockVisionAnalyzer(),
        sink=JsonlEventSink(tmp_path / "events.jsonl"),
        suspicious_threshold=0.70,
    )
    event = pipeline.process(make_window(tmp_path, ("white", "white", "white")))
    assert event.action == Action.IGNORED
    assert event.classification is None


def test_large_change_escalates(tmp_path: Path) -> None:
    pipeline = MonitoringPipeline(
        motion_gate=PillowMotionGate(threshold=0.02),
        classifier=MockJevClassifier(),
        analyzer=MockVisionAnalyzer(),
        sink=JsonlEventSink(tmp_path / "events.jsonl"),
        suspicious_threshold=0.70,
    )
    event = pipeline.process(make_window(tmp_path, ("white", "black", "white")))
    assert event.action == Action.QUEUE_REVIEW
    assert event.analysis is not None


def test_jev_prompt_covers_normal_motion_and_static_anomalies() -> None:
    prompt = f"{JEV_STATE_TEMPLATE} {JEV_QUESTION}".lower()

    assert "jumping toward the camera" in prompt
    assert "feces" in prompt
    assert "static scene condition" in prompt


class AlwaysMotionGate:
    def evaluate(self, window: TemporalWindow) -> MotionResult:
        return MotionResult(score=0.2, should_classify=True)


class SequenceClassifier:
    name = "sequence"

    def __init__(self, suspicious: list[float]) -> None:
        self.suspicious = iter(suspicious)

    def classify(self, window: TemporalWindow, contact_sheet: Path) -> ClassificationResult:
        score = next(self.suspicious)
        return ClassificationResult(safe=1 - score, suspicious=score, provider=self.name)


class CountingAnalyzer:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, window: TemporalWindow) -> AnomalyAnalysis:
        self.calls += 1
        return AnomalyAnalysis(
            anomaly_type="test",
            severity=Severity.MEDIUM,
            summary="test incident",
            confidence=0.8,
            evidence=["test"],
            recommended_action="review",
            provider=self.name,
        )


def test_stream_analyzes_one_peak_window_per_incident(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), "white").save(image_path)
    analyzer = CountingAnalyzer()
    event_path = tmp_path / "events.jsonl"
    incident_path = tmp_path / "incidents.jsonl"
    pipeline = MonitoringPipeline(
        motion_gate=AlwaysMotionGate(),
        classifier=SequenceClassifier([0.20, 0.40, 0.05, 0.05, 0.05, 0.05, 0.80]),
        analyzer=analyzer,
        sink=JsonlEventSink(event_path),
        suspicious_threshold=0.15,
        incident_sink=JsonlIncidentSink(incident_path),
        incident_max_gap_seconds=2.0,
    )
    stream = pipeline.incident_stream(sample_interval_seconds=0.5)
    started_at = datetime.now(timezone.utc)

    emitted = []
    window_ids = []
    for index in range(7):
        captured_at = started_at + timedelta(seconds=index * 0.5)
        window = TemporalWindow(
            camera_id="test",
            frames=(
                Frame(image_path, captured_at),
                Frame(image_path, captured_at),
                Frame(image_path, captured_at),
            ),
        )
        window_ids.append(window.id)
        emitted.extend(stream.process(window))
    emitted.extend(stream.flush())

    assert analyzer.calls == 2
    assert stream.incident_count == 2
    assert stream.strong_vlm_calls == 2
    assert len(emitted) == 7
    assert [event.window_id for event in emitted] == window_ids
    assert sum(event.action == Action.CANDIDATE_MERGED for event in emitted) == 1
    incidents = [json.loads(line) for line in incident_path.read_text().splitlines()]
    assert [incident["window_count"] for incident in incidents] == [2, 1]
    assert all(incident["status"] == "analyzed" for incident in incidents)


def test_stream_can_defer_incident_analysis(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), "white").save(image_path)
    analyzer = CountingAnalyzer()
    incident_path = tmp_path / "incidents.jsonl"
    pipeline = MonitoringPipeline(
        motion_gate=AlwaysMotionGate(),
        classifier=SequenceClassifier([0.80]),
        analyzer=analyzer,
        sink=JsonlEventSink(tmp_path / "events.jsonl"),
        suspicious_threshold=0.15,
        incident_sink=JsonlIncidentSink(incident_path),
        analysis_mode="deferred",
    )
    stream = pipeline.incident_stream(sample_interval_seconds=0.5)
    captured_at = datetime.now(timezone.utc)
    window = TemporalWindow(
        camera_id="test",
        frames=(
            Frame(image_path, captured_at),
            Frame(image_path, captured_at),
            Frame(image_path, captured_at),
        ),
    )

    stream.process(window)
    events = stream.flush()

    assert analyzer.calls == 0
    assert stream.strong_vlm_calls == 0
    assert events[0].action == Action.QUEUE_REVIEW
    assert events[0].analysis is None
    incident = json.loads(incident_path.read_text())
    assert incident["status"] == "pending_analysis"
    assert incident["analysis"] is None
