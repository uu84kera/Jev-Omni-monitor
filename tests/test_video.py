"""Test video sampling, overlapping windows, and summary generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from jev_monitor.domain import Action, ClassificationResult, Event, MotionResult, TemporalWindow
from jev_monitor.video import process_video


class RecordingPipeline:
    def __init__(self) -> None:
        self.windows: list[TemporalWindow] = []

    def process(self, window: TemporalWindow) -> Event:
        self.windows.append(window)
        return Event(
            window_id=window.id,
            camera_id=window.camera_id,
            captured_at=window.frames[-1].captured_at,
            motion=MotionResult(score=0.1, should_classify=True),
            classification=ClassificationResult(
                safe=0.8,
                suspicious=0.2,
                provider="test",
            ),
            action=Action.SAFE,
        )


def create_video(path: Path, frame_count: int = 5, fps: float = 2.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            image = np.full((48, 64, 3), index * 40, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def test_video_uses_overlapping_three_frame_windows(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    create_video(video_path)
    pipeline = RecordingPipeline()

    summary = process_video(
        video_path=video_path,
        camera_id="test-camera",
        interval_seconds=0.5,
        pipeline=pipeline,
    )

    assert summary.sampled_frames == 5
    assert summary.total_windows == 3
    assert summary.safe_windows == 3
    assert summary.suspicious_windows == 0
    assert summary.decision == "safe"
    assert len(pipeline.windows) == 3
    assert pipeline.windows[0].frames[1].captured_at == pipeline.windows[1].frames[0].captured_at
    assert isinstance(pipeline.windows[-1].frames[-1].captured_at, datetime)
