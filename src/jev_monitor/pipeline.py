"""Orchestrate motion gating, classification, escalation, policy, and logging."""

from __future__ import annotations

from pathlib import Path

from .adapters import build_contact_sheet
from .domain import Action, Event, Severity, TemporalWindow
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
    ) -> None:
        self.motion_gate = motion_gate
        self.classifier = classifier
        self.analyzer = analyzer
        self.sink = sink
        self.suspicious_threshold = suspicious_threshold

    def process(self, window: TemporalWindow) -> Event:
        motion = self.motion_gate.evaluate(window)

        # Skip low-motion windows unless the motion gate requests classification.
        if not motion.should_classify:
            return self._record(
                Event(
                    window_id=window.id,
                    camera_id=window.camera_id,
                    captured_at=window.frames[-1].captured_at,
                    motion=motion,
                    action=Action.IGNORED,
                )
            )

        # Jev-Omni receives the three ordered frames as one contact sheet.
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

        # Safe windows do not need the more expensive vision model.
        if classification.suspicious < self.suspicious_threshold:
            return self._record(
                Event(
                    window_id=window.id,
                    camera_id=window.camera_id,
                    captured_at=window.frames[-1].captured_at,
                    motion=motion,
                    classification=classification,
                    action=Action.SAFE,
                )
            )

        # Suspicious windows are escalated for structured anomaly analysis.
        analysis = self.analyzer.analyze(window)

        action = {
            Severity.CRITICAL: Action.ALERT_IMMEDIATELY,
            Severity.HIGH: Action.NOTIFY,
            Severity.MEDIUM: Action.QUEUE_REVIEW,
            Severity.LOW: Action.LOG_ONLY,
            Severity.NONE: Action.LOG_ONLY,
        }[analysis.severity]

        return self._record(
            Event(
                window_id=window.id,
                camera_id=window.camera_id,
                captured_at=window.frames[-1].captured_at,
                motion=motion,
                classification=classification,
                analysis=analysis,
                action=action,
            )
        )

    def _record(self, event: Event) -> Event:
        self.sink.append(event)
        return event