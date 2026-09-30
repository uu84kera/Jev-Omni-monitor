"""Define replaceable interfaces for pipeline components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .domain import AnomalyAnalysis, ClassificationResult, Event, MotionResult, TemporalWindow


class MotionGate(Protocol):
    def evaluate(self, window: TemporalWindow) -> MotionResult: ...

# Jev used here
class DecisionClassifier(Protocol):
    """Classify a temporal window as safe or suspicious."""
    @property
    def name(self) -> str: ...

    def classify(self, window: TemporalWindow, contact_sheet: Path) -> ClassificationResult: ...

# VLM you want
class StrongVisionAnalyzer(Protocol):
    """Produce a structured anomaly analysis for escalated windows."""
    @property
    def name(self) -> str: ...

    def analyze(self, window: TemporalWindow) -> AnomalyAnalysis: ...


class EventSink(Protocol):
    def append(self, event: Event) -> None: ...

