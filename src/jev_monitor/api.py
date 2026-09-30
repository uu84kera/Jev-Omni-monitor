"""Expose the monitoring pipeline through a FastAPI service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .bootstrap import build_pipeline
from .config import Settings
from .domain import Frame, TemporalWindow


settings = Settings.from_env()
pipeline, sink = build_pipeline(settings)
app = FastAPI(title="Jev-Omni Monitor", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "jev_provider": settings.jev_provider,
        "vlm_provider": settings.vlm_provider,
    }


@app.get("/v1/events")
def events(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return sink.recent(limit)


@app.post("/v1/analyze-window")
async def analyze_window(
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...),
    frame3: UploadFile = File(...),
    camera_id: str = "api-camera",
) -> dict:
    uploads = (frame1, frame2, frame3)
    if any(not (upload.content_type or "").startswith("image/") for upload in uploads):
        raise HTTPException(status_code=415, detail="All three files must be images.")
    with TemporaryDirectory(prefix="jev-monitor-api-") as directory:
        paths = []
        for index, upload in enumerate(uploads, start=1):
            path = Path(directory) / f"frame-{index}.jpg"
            path.write_bytes(await upload.read())
            paths.append(path)
        start = datetime.now(timezone.utc)
        window = TemporalWindow(
            camera_id=camera_id,
            frames=tuple(
                Frame(path=path, captured_at=start + timedelta(milliseconds=500 * index))
                for index, path in enumerate(paths)
            ),  # type: ignore[arg-type]
        )
        try:
            return pipeline.process(window).to_dict()
        except Exception as error:
            raise HTTPException(status_code=502, detail=f"Pipeline failed: {error}") from error

