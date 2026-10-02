"""Sample videos into overlapping temporal windows and summarize events."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Protocol

import cv2

from .domain import Action, Event, Frame, TemporalWindow


class WindowProcessor(Protocol):
    def process(self, window: TemporalWindow) -> Event: ...


@dataclass(frozen=True)
class VideoSummary:
    video_path: str
    camera_id: str
    sample_interval_seconds: float
    sampled_frames: int
    total_windows: int
    ignored_windows: int
    safe_windows: int
    suspicious_windows: int
    incidents: int
    strong_vlm_calls: int
    max_suspicious_probability: float | None
    decision: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sampled_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float,
) -> Iterator[Frame]:
    if interval_seconds <= 0:
        raise ValueError("Sample interval must be greater than zero.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError(f"Video reports an invalid frame rate: {video_path}")

    started_at = datetime.now(timezone.utc)
    frame_index = 0
    sample_index = 0
    next_sample_time = 0.0

    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break

            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + 1e-9 < next_sample_time:
                continue

            path = output_dir / f"sample-{sample_index:08d}.jpg"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Unable to write sampled frame: {path}")

            yield Frame(
                path=path,
                captured_at=started_at + timedelta(seconds=timestamp),
            )
            sample_index += 1
            next_sample_time += interval_seconds
    finally:
        capture.release()


def process_video(
    video_path: Path,
    camera_id: str,
    interval_seconds: float,
    pipeline: WindowProcessor,
) -> VideoSummary:
    if not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path}")

    sampled_count = 0
    total_windows = 0
    ignored_windows = 0
    safe_windows = 0
    suspicious_windows = 0
    suspicious_probabilities: list[float] = []
    window_frames: deque[Frame] = deque(maxlen=3)
    stream_factory = getattr(pipeline, "incident_stream", None)
    incident_stream = (
        stream_factory(interval_seconds)
        if callable(stream_factory)
        else None
    )

    def collect(event: Event) -> None:
        nonlocal ignored_windows, safe_windows, suspicious_windows
        if event.classification is not None:
            suspicious_probabilities.append(event.classification.suspicious)
        if event.action == Action.IGNORED:
            ignored_windows += 1
        elif event.action == Action.SAFE:
            safe_windows += 1
        else:
            suspicious_windows += 1

    with TemporaryDirectory(prefix="jev-monitor-video-") as directory:
        output_dir = Path(directory)
        for frame in sampled_frames(video_path, output_dir, interval_seconds):
            sampled_count += 1
            window_frames.append(frame)
            if len(window_frames) < 3:
                continue

            window = TemporalWindow(
                camera_id=camera_id,
                frames=tuple(window_frames),  # type: ignore[arg-type]
            )
            total_windows += 1
            events = (
                incident_stream.process(window)
                if incident_stream is not None
                else [pipeline.process(window)]
            )
            for event in events:
                collect(event)

        if incident_stream is not None:
            for event in incident_stream.flush():
                collect(event)

    if sampled_count < 3:
        raise ValueError(
            f"Video produced only {sampled_count} sampled frames; at least three are required."
        )

    return VideoSummary(
        video_path=str(video_path),
        camera_id=camera_id,
        sample_interval_seconds=interval_seconds,
        sampled_frames=sampled_count,
        total_windows=total_windows,
        ignored_windows=ignored_windows,
        safe_windows=safe_windows,
        suspicious_windows=suspicious_windows,
        incidents=(incident_stream.incident_count if incident_stream is not None else 0),
        strong_vlm_calls=(
            incident_stream.strong_vlm_calls if incident_stream is not None else suspicious_windows
        ),
        max_suspicious_probability=(
            round(max(suspicious_probabilities), 4)
            if suspicious_probabilities
            else None
        ),
        decision="suspicious" if suspicious_windows else "safe",
    )
