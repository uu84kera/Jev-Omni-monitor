"""Provide demo, folder, and video command-line entry points."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from .bootstrap import build_pipeline
from .config import Settings
from .domain import Frame, TemporalWindow
from .video import process_video


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _window(paths: list[Path], camera_id: str) -> TemporalWindow:
    start = datetime.now(timezone.utc)
    frames = tuple(Frame(path=path, captured_at=start + timedelta(milliseconds=500 * i)) for i, path in enumerate(paths))
    return TemporalWindow(camera_id=camera_id, frames=frames)  # type: ignore[arg-type]


def run_demo() -> int:
    with TemporaryDirectory(prefix="jev-monitor-demo-") as directory:
        paths = []
        for index, x in enumerate((15, 90, 190), start=1):
            image = Image.new("RGB", (320, 180), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((x, 55, x + 60, 125), fill="black")
            path = Path(directory) / f"frame-{index}.jpg"
            image.save(path)
            paths.append(path)
        pipeline, _ = build_pipeline(Settings())
        event = pipeline.process(_window(paths, "demo-camera"))
        print(json.dumps(event.to_dict(), indent=2))
    return 0


def run_folder(folder: Path, camera_id: str) -> int:
    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < 3:
        raise SystemExit("The folder must contain at least three ordered images.")
    pipeline, _ = build_pipeline(Settings.from_env())
    for index in range(0, len(paths) - 2, 3):
        group = paths[index:index + 3]
        if len(group) < 3:
            break
        event = pipeline.process(_window(group, camera_id))
        print(json.dumps(event.to_dict()))
    return 0


def run_video(video_path: Path, camera_id: str | None, sample_interval: float) -> int:
    pipeline, _ = build_pipeline(Settings.from_env())
    try:
        summary = process_video(
            video_path=video_path,
            camera_id=camera_id or video_path.stem,
            interval_seconds=sample_interval,
            pipeline=pipeline,
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Jev-Omni hierarchical anomaly monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run a synthetic three-frame window")
    folder = commands.add_parser("run-folder", help="Process ordered images in groups of three")
    folder.add_argument("path", type=Path)
    folder.add_argument("--camera-id", default="folder-camera")
    video = commands.add_parser("run-video", help="Process overlapping windows sampled from a video")
    video.add_argument("path", type=Path)
    video.add_argument("--camera-id")
    video.add_argument("--sample-interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "demo":
        return run_demo()
    if args.command == "run-folder":
        return run_folder(args.path, args.camera_id)
    return run_video(args.path, args.camera_id, args.sample_interval)


if __name__ == "__main__":
    raise SystemExit(main())
