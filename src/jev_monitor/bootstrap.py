"""Construct the monitoring pipeline from application settings."""

from __future__ import annotations

from .adapters import (
    HttpJevClassifier,
    HttpVisionAnalyzer,
    JsonlEventSink,
    MockJevClassifier,
    MockVisionAnalyzer,
    PillowMotionGate,
)
from .config import Settings
from .pipeline import MonitoringPipeline


def build_pipeline(settings: Settings) -> tuple[MonitoringPipeline, JsonlEventSink]:
    classifier = (
        HttpJevClassifier(settings.jev_endpoint, settings.jev_api_key)
        if settings.jev_provider == "http"
        else MockJevClassifier()
    )
    analyzer = (
        HttpVisionAnalyzer(settings.vlm_endpoint, settings.vlm_api_key)
        if settings.vlm_provider == "http"
        else MockVisionAnalyzer()
    )
    sink = JsonlEventSink(settings.event_log)
    pipeline = MonitoringPipeline(
        motion_gate=PillowMotionGate(settings.motion_threshold, settings.force_classify_every),
        classifier=classifier,
        analyzer=analyzer,
        sink=sink,
        suspicious_threshold=settings.suspicious_threshold,
    )
    return pipeline, sink

