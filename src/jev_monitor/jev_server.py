"""Serve Jev-Omni image classification behind the monitor HTTP contract."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


DEFAULT_MODEL_ID = "akhilaaa3/Jev-Omni"
MAX_MEDIA_BYTES = 25 * 1024 * 1024


class PredictionRuntime(Protocol):
    model_id: str

    def load(self) -> None: ...

    def predict(
        self,
        *,
        state: str,
        question: str,
        options: list[str],
        media: Path,
    ) -> dict[str, Any]: ...


class JevOmniRuntime:
    """Lazily import the model repository and keep one classifier on the GPU."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._classifier: Any | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._classifier is not None

    def load(self) -> None:
        if self.loaded:
            return

        from huggingface_hub import hf_hub_download

        module_path = Path(hf_hub_download(self.model_id, "jev_omni.py"))
        spec = importlib.util.spec_from_file_location("jev_omni_runtime", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import Jev-Omni loader from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._classifier = module.load_jev_omni(model_id=self.model_id)

    def predict(
        self,
        *,
        state: str,
        question: str,
        options: list[str],
        media: Path,
    ) -> dict[str, Any]:
        if self._classifier is None:
            raise RuntimeError("Jev-Omni is not loaded")

        with self._lock:
            return self._classifier.predict(
                state=state,
                question=question,
                options=options,
                media=str(media),
                modality="image",
            )


def _parse_options(options_json: str) -> list[str]:
    try:
        value = json.loads(options_json)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="options_json must be valid JSON") from error

    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise HTTPException(status_code=422, detail="options_json must be a list of non-empty strings")
    if not 2 <= len(value) <= 20:
        raise HTTPException(status_code=422, detail="Jev-Omni requires 2 to 20 options")
    if len(set(value)) != len(value):
        raise HTTPException(status_code=422, detail="options must be unique")
    return value


def _save_upload(media: UploadFile) -> Path:
    suffix = Path(media.filename or "media.jpg").suffix or ".jpg"
    handle = NamedTemporaryFile(prefix="jev-upload-", suffix=suffix, delete=False)
    try:
        with handle:
            shutil.copyfileobj(media.file, handle)
        path = Path(handle.name)
        if path.stat().st_size > MAX_MEDIA_BYTES:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail="media exceeds the 25 MB limit")
        return path
    finally:
        media.file.close()


def create_app(runtime: PredictionRuntime | None = None) -> FastAPI:
    selected_runtime = runtime or JevOmniRuntime(
        os.getenv("JEV_OMNI_MODEL_ID", DEFAULT_MODEL_ID)
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        selected_runtime.load()
        yield

    app = FastAPI(title="Jev-Omni inference service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": selected_runtime.model_id}

    @app.post("/v1/classify")
    def classify(
        state: str = Form(...),
        question: str = Form(...),
        options_json: str = Form(...),
        media: UploadFile = File(...),
    ) -> dict[str, Any]:
        options = _parse_options(options_json)
        media_path = _save_upload(media)
        try:
            result = selected_runtime.predict(
                state=state,
                question=question,
                options=options,
                media=media_path,
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        finally:
            media_path.unlink(missing_ok=True)

        probabilities = result.get("probabilities")
        if not isinstance(probabilities, dict) or any(option not in probabilities for option in options):
            raise HTTPException(status_code=500, detail="Jev-Omni returned invalid probabilities")
        return {
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "probabilities": {option: float(probabilities[option]) for option in options},
            "model": selected_runtime.model_id,
        }

    return app


app = create_app()
