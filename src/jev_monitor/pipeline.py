"""Orchestrate motion gating, classification, escalation, policy, and logging."""

from __future__ import annotations

from pathlib import Path

from .adapters import build_contact_sheet
from .domain import Action, Event, ScreeningResult, Severity, TemporalWindow
from .incidents import IncidentStreamProcessor, JsonlIncidentSink
from .ports import (
    DecisionClassifier,
    EventSink,
    MotionGate,
    StrongVisionAnalyzer,
)


class MonitoringPipeline:
    def __init__(
        self,
        motion_gate: MotionGate,
        classifier: DecisionClassifier,
        analyzer: StrongVisionAnalyzer,
        sink: EventSink,
        suspicious_threshold: float,
        incident_sink: JsonlIncidentSink | None = None,
        incident_max_gap_seconds: float = 2.0,
        analysis_mode: str = "immediate",
    ) -> None:
        if analysis_mode not in {"immediate", "deferred"}:
            raise ValueError("Analysis mode must be 'immediate' or 'deferred'")
        self.motion_gate = motion_gate
        self.classifier = classifier
        self.analyzer = analyzer
        self.sink = sink
        self.suspicious_threshold = suspicious_threshold
        self.incident_sink = incident_sink
        self.incident_max_gap_seconds = incident_max_gap_seconds
        self.analysis_mode = analysis_mode

    def process(self, window: TemporalWindow) -> Event:
        """Process one independent window, including immediate escalation."""
        screening = self.screen(window)
        if self.is_candidate(screening):
            return self.record(self.analyzed_event(screening, incident_id=window.id))
        return self.record(self.screening_event(screening))

    def screen(self, window: TemporalWindow) -> ScreeningResult:
        motion = self.motion_gate.evaluate(window)

        if not motion.should_classify:
            return ScreeningResult(window=window, motion=motion)

        contact_sheet: Path | None = None
        try:
            contact_sheet = build_contact_sheet(window)
            classification = self.classifier.classify(
                window,
                contact_sheet,
            )
        finally:
            if contact_sheet is not None:
                contact_sheet.unlink(missing_ok=True)

        return ScreeningResult(
            window=window,
            motion=motion,
            classification=classification,
        )

    def is_candidate(self, screening: ScreeningResult) -> bool:
        return (
            screening.classification is not None
            and screening.classification.suspicious >= self.suspicious_threshold
        )

    def screening_event(
        self,
        screening: ScreeningResult,
        *,
        action: Action | None = None,
        incident_id: str | None = None,
    ) -> Event:
        if action is None:
            action = (
                Action.IGNORED
                if screening.classification is None
                else Action.SAFE
            )
        return Event(
            window_id=screening.window.id,
            camera_id=screening.window.camera_id,
            captured_at=screening.window.frames[-1].captured_at,
            motion=screening.motion,
            classification=screening.classification,
            action=action,
            incident_id=incident_id,
        )

    def analyzed_event(self, screening: ScreeningResult, incident_id: str) -> Event:
        if screening.classification is None:
            raise ValueError("Cannot analyze a window without a classification")
        analysis = self.analyzer.analyze(screening.window)
        action = {
            Severity.CRITICAL: Action.ALERT_IMMEDIATELY,
            Severity.HIGH: Action.NOTIFY,
            Severity.MEDIUM: Action.QUEUE_REVIEW,
            Severity.LOW: Action.LOG_ONLY,
            Severity.NONE: Action.LOG_ONLY,
        }[analysis.severity]
        return Event(
            window_id=screening.window.id,
            camera_id=screening.window.camera_id,
            captured_at=screening.window.frames[-1].captured_at,
            motion=screening.motion,
            classification=screening.classification,
            analysis=analysis,
            action=action,
            incident_id=incident_id,
        )

    def incident_stream(self, sample_interval_seconds: float) -> IncidentStreamProcessor:
        if self.incident_sink is None:
            raise RuntimeError("An incident sink is required for stream processing")
        return IncidentStreamProcessor(
            pipeline=self,
            sink=self.incident_sink,
            sample_interval_seconds=sample_interval_seconds,
            max_gap_seconds=self.incident_max_gap_seconds,
            defer_analysis=self.analysis_mode == "deferred",
        )

    def record(self, event: Event) -> Event:
        self.sink.append(event)
        return event
