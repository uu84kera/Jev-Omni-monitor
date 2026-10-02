"""Define the domain models shared across the monitoring pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class Severity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(StrEnum):
    IGNORED = "ignored"
    SAFE = "safe"
    CANDIDATE_MERGED = "candidate_merged"
    LOG_ONLY = "log_only"
    QUEUE_REVIEW = "queue_review"
    NOTIFY = "notify"
    ALERT_IMMEDIATELY = "alert_immediately"


@dataclass(frozen=True)
class Frame:
    path: Path
    captured_at: datetime


@dataclass(frozen=True)
class TemporalWindow:
    camera_id: str
    frames: tuple[Frame, Frame, Frame]
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class MotionResult:
    score: float
    should_classify: bool
    forced: bool = False


@dataclass(frozen=True)
class ClassificationResult:
    safe: float
    suspicious: float
    provider: str


@dataclass(frozen=True)
class AnomalyAnalysis:
    anomaly_type: str
    severity: Severity
    summary: str
    confidence: float
    evidence: list[str]
    recommended_action: str
    provider: str


@dataclass(frozen=True)
class Event:
    window_id: str
    camera_id: str
    captured_at: datetime
    motion: MotionResult
    action: Action
    incident_id: str | None = None
    classification: ClassificationResult | None = None
    analysis: AnomalyAnalysis | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            return value

        return encode(asdict(self))


@dataclass(frozen=True)
class ScreeningResult:
    window: TemporalWindow
    motion: MotionResult
    classification: ClassificationResult | None = None
