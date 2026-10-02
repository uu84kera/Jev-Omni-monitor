"""Aggregate suspicious window events into candidate incidents."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .domain import Action, Event, ScreeningResult, TemporalWindow


@dataclass(frozen=True)
class CandidateWindow:
    index: int
    window_id: str
    camera_id: str
    captured_at: str
    suspicious_probability: float


@dataclass(frozen=True)
class CandidateIncident:
    incident_id: str
    camera_id: str
    started_at_seconds: float
    ended_at_seconds: float
    window_count: int
    window_ids: list[str]
    peak_window_id: str
    peak_window_start_seconds: float
    peak_suspicious_probability: float
    peak_captured_at: str
    action: str | None = None
    analysis: dict[str, Any] | None = None
    status: str = "pending_analysis"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class IncidentPipeline(Protocol):
    def screen(self, window: TemporalWindow) -> ScreeningResult: ...

    def is_candidate(self, screening: ScreeningResult) -> bool: ...

    def screening_event(
        self,
        screening: ScreeningResult,
        *,
        action: Action | None = None,
        incident_id: str | None = None,
    ) -> Event: ...

    def analyzed_event(self, screening: ScreeningResult, incident_id: str) -> Event: ...

    def record(self, event: Event) -> Event: ...


class JsonlIncidentSink:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, incident: CandidateIncident) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(incident.to_dict()) + "\n")


class IncidentStreamProcessor:
    """Buffer candidate windows and analyze one peak window per incident."""

    def __init__(
        self,
        pipeline: IncidentPipeline,
        sink: JsonlIncidentSink,
        sample_interval_seconds: float,
        max_gap_seconds: float,
        defer_analysis: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.sink = sink
        self.sample_interval_seconds = sample_interval_seconds
        self.max_gap_seconds = max_gap_seconds
        self.defer_analysis = defer_analysis
        self.incident_count = 0
        self.strong_vlm_calls = 0
        self._window_index = 0
        self._active: list[tuple[int, ScreeningResult]] = []
        self._pending: list[tuple[int, ScreeningResult]] = []

    def process(self, window: TemporalWindow) -> list[Event]:
        index = self._window_index
        self._window_index += 1
        screening = self.pipeline.screen(window)
        emitted: list[Event] = []

        if self.pipeline.is_candidate(screening):
            if self._active and self._gap_from_last(index) > self.max_gap_seconds:
                emitted.extend(self._flush())
            self._active.append((index, screening))
            self._pending.append((index, screening))
            return emitted

        if self._active:
            if self._gap_from_last(index) > self.max_gap_seconds:
                emitted.extend(self._flush())
            else:
                self._pending.append((index, screening))
                return emitted
        event = self.pipeline.screening_event(screening)
        emitted.append(self.pipeline.record(event))
        return emitted

    def flush(self) -> list[Event]:
        return self._flush()

    def _gap_from_last(self, index: int) -> float:
        return (index - self._active[-1][0]) * self.sample_interval_seconds

    def _flush(self) -> list[Event]:
        if not self._active:
            return []

        incident_id = str(uuid4())
        peak_index, peak = max(
            self._active,
            key=lambda item: item[1].classification.suspicious,  # type: ignore[union-attr]
        )
        analyzed = (
            self.pipeline.screening_event(
                peak,
                action=Action.QUEUE_REVIEW,
                incident_id=incident_id,
            )
            if self.defer_analysis
            else self.pipeline.analyzed_event(peak, incident_id)
        )
        candidate_ids = {screening.window.id for _, screening in self._active}
        emitted = []
        for _, screening in self._pending:
            event = (
                analyzed
                if screening.window.id == peak.window.id
                else self.pipeline.screening_event(
                    screening,
                    action=(
                        Action.CANDIDATE_MERGED
                        if screening.window.id in candidate_ids
                        else None
                    ),
                    incident_id=(
                        incident_id if screening.window.id in candidate_ids else None
                    ),
                )
            )
            emitted.append(self.pipeline.record(event))

        first_index, first = self._active[0]
        last_index, _ = self._active[-1]
        classification = peak.classification
        if classification is None:
            raise RuntimeError("Incident peak has no classification")
        event_data = analyzed.to_dict()
        incident = CandidateIncident(
            incident_id=incident_id,
            camera_id=first.window.camera_id,
            started_at_seconds=round(first_index * self.sample_interval_seconds, 3),
            ended_at_seconds=round((last_index + 2) * self.sample_interval_seconds, 3),
            window_count=len(self._active),
            window_ids=[screening.window.id for _, screening in self._active],
            peak_window_id=peak.window.id,
            peak_window_start_seconds=round(
                peak_index * self.sample_interval_seconds,
                3,
            ),
            peak_suspicious_probability=classification.suspicious,
            peak_captured_at=peak.window.frames[-1].captured_at.isoformat(),
            action=analyzed.action.value,
            analysis=event_data["analysis"],
            status="pending_analysis" if self.defer_analysis else "analyzed",
        )
        self.sink.append(incident)
        self.incident_count += 1
        if not self.defer_analysis:
            self.strong_vlm_calls += 1
        self._active = []
        self._pending = []
        return emitted


def _candidate(index: int, event: dict[str, Any], threshold: float) -> CandidateWindow | None:
    classification = event.get("classification")
    if not isinstance(classification, dict):
        return None
    probability = float(classification["suspicious"])
    if probability < threshold:
        return None
    return CandidateWindow(
        index=index,
        window_id=str(event["window_id"]),
        camera_id=str(event["camera_id"]),
        captured_at=str(event["captured_at"]),
        suspicious_probability=probability,
    )


def aggregate_events(
    events: list[dict[str, Any]],
    threshold: float,
    max_gap_seconds: float,
    sample_interval_seconds: float,
) -> list[CandidateIncident]:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if max_gap_seconds < 0:
        raise ValueError("Maximum incident gap cannot be negative.")
    if sample_interval_seconds <= 0:
        raise ValueError("Sample interval must be greater than zero.")

    candidates = [
        candidate
        for index, event in enumerate(events)
        if (candidate := _candidate(index, event, threshold)) is not None
    ]
    groups: list[list[CandidateWindow]] = []
    for candidate in candidates:
        if not groups:
            groups.append([candidate])
            continue
        previous = groups[-1][-1]
        gap = (candidate.index - previous.index) * sample_interval_seconds
        if gap <= max_gap_seconds:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    incidents = []
    for group in groups:
        camera_ids = {candidate.camera_id for candidate in group}
        if len(camera_ids) != 1:
            raise ValueError("An incident cannot contain events from multiple cameras.")
        peak = max(group, key=lambda candidate: candidate.suspicious_probability)
        incidents.append(
            CandidateIncident(
                incident_id=str(uuid4()),
                camera_id=group[0].camera_id,
                started_at_seconds=round(
                    group[0].index * sample_interval_seconds,
                    3,
                ),
                ended_at_seconds=round(
                    (group[-1].index + 2) * sample_interval_seconds,
                    3,
                ),
                window_count=len(group),
                window_ids=[candidate.window_id for candidate in group],
                peak_window_id=peak.window_id,
                peak_window_start_seconds=round(
                    peak.index * sample_interval_seconds,
                    3,
                ),
                peak_suspicious_probability=peak.suspicious_probability,
                peak_captured_at=peak.captured_at,
            )
        )
    return incidents


def aggregate_file(
    event_path: Path,
    output_path: Path,
    threshold: float,
    max_gap_seconds: float,
    sample_interval_seconds: float,
) -> list[CandidateIncident]:
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    incidents = aggregate_events(
        events,
        threshold=threshold,
        max_gap_seconds=max_gap_seconds,
        sample_interval_seconds=sample_interval_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for incident in incidents:
            handle.write(json.dumps(incident.to_dict()) + "\n")
    return incidents


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate JevGuard candidate incidents")
    parser.add_argument("--events-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    args = parser.parse_args()

    paths = sorted(args.events_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No event JSONL files found in {args.events_dir}")

    total_events = 0
    total_incidents = 0
    videos = []
    for path in paths:
        incidents = aggregate_file(
            event_path=path,
            output_path=args.output_dir / path.name,
            threshold=args.threshold,
            max_gap_seconds=args.max_gap_seconds,
            sample_interval_seconds=args.sample_interval,
        )
        event_count = len(path.read_text(encoding="utf-8").splitlines())
        total_events += event_count
        total_incidents += len(incidents)
        videos.append(
            {
                "video_id": path.stem,
                "events": event_count,
                "incidents": len(incidents),
            }
        )

    summary = {
        "threshold": args.threshold,
        "max_gap_seconds": args.max_gap_seconds,
        "sample_interval_seconds": args.sample_interval,
        "total_events": total_events,
        "total_incidents": total_incidents,
        "videos": videos,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


def summarize_incidents(input_dir: Path, output_path: Path) -> dict[str, object]:
    videos = []
    total_incidents = 0
    total_analyzed = 0
    for path in sorted(input_dir.glob("*.jsonl")):
        incidents = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        analyzed = sum(incident.get("status") == "analyzed" for incident in incidents)
        total_incidents += len(incidents)
        total_analyzed += analyzed
        videos.append(
            {
                "video_id": path.stem,
                "incidents": len(incidents),
                "analyzed": analyzed,
            }
        )
    summary = {
        "total_incidents": total_incidents,
        "total_analyzed": total_analyzed,
        "videos": videos,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def summary_main() -> int:
    parser = argparse.ArgumentParser(description="Summarize analyzed JevGuard incidents")
    parser.add_argument("--incidents-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize_incidents(args.incidents_dir, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
