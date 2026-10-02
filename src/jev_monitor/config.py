"""Load monitor configuration from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"JEV_MONITOR_{name}", default)


@dataclass(frozen=True)
class Settings:
    jev_provider: str = "mock"
    jev_endpoint: str = "http://localhost:8100/v1/classify"
    jev_api_key: str = ""
    vlm_provider: str = "mock"
    vlm_endpoint: str = "http://localhost:8200/v1/analyze"
    vlm_api_key: str = ""
    suspicious_threshold: float = 0.70
    motion_threshold: float = 0.025
    force_classify_every: int = 10
    event_log: Path = Path("data/events/events.jsonl")
    incident_log: Path = Path("data/events/incidents.jsonl")
    incident_max_gap_seconds: float = 2.0
    analysis_mode: str = "immediate"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            jev_provider=_env("JEV_PROVIDER", "mock"),
            jev_endpoint=_env("JEV_ENDPOINT", "http://localhost:8100/v1/classify"),
            jev_api_key=_env("JEV_API_KEY", ""),
            vlm_provider=_env("VLM_PROVIDER", "mock"),
            vlm_endpoint=_env("VLM_ENDPOINT", "http://localhost:8200/v1/analyze"),
            vlm_api_key=_env("VLM_API_KEY", ""),
            suspicious_threshold=float(_env("SUSPICIOUS_THRESHOLD", "0.70")),
            motion_threshold=float(_env("MOTION_THRESHOLD", "0.025")),
            force_classify_every=int(_env("FORCE_CLASSIFY_EVERY", "10")),
            event_log=Path(_env("EVENT_LOG", "data/events/events.jsonl")),
            incident_log=Path(_env("INCIDENT_LOG", "data/events/incidents.jsonl")),
            incident_max_gap_seconds=float(_env("INCIDENT_MAX_GAP_SECONDS", "2.0")),
            analysis_mode=_env("ANALYSIS_MODE", "immediate"),
        )
