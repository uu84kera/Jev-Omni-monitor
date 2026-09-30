"""Test static-window filtering and suspicious-window escalation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from jev_monitor.adapters import JsonlEventSink, MockJevClassifier, MockVisionAnalyzer, PillowMotionGate
from jev_monitor.domain import Action, Frame, TemporalWindow
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

