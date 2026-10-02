"""Test dataset-level labels, metrics, and report artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from jev_monitor.evaluation import evaluate, expected_label


def _write_summary(directory: Path, video_id: str, score: float) -> None:
    with (directory / f"{video_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"camera_id": video_id, "max_suspicious_probability": score},
            handle,
        )


def test_expected_label_maps_vague_abnormal_to_suspicious() -> None:
    assert expected_label("Normal") == "safe"
    assert expected_label("Abnormal") == "suspicious"
    assert expected_label("Vague Abnormal") == "suspicious"


def test_evaluate_writes_confusion_matrix_and_threshold_sweep(tmp_path: Path) -> None:
    summaries = tmp_path / "summaries"
    output = tmp_path / "evaluation"
    summaries.mkdir()
    _write_summary(summaries, "normal-correct", 0.20)
    _write_summary(summaries, "normal-false-alarm", 0.90)
    _write_summary(summaries, "anomaly-correct", 0.85)
    _write_summary(summaries, "anomaly-missed", 0.40)

    annotations = tmp_path / "annotations.csv"
    with annotations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Title", "Anomaly Tag"])
        writer.writeheader()
        writer.writerows(
            [
                {"Title": "normal-correct", "Anomaly Tag": "Normal"},
                {"Title": "normal-false-alarm", "Anomaly Tag": "Normal"},
                {"Title": "anomaly-correct", "Anomaly Tag": "Abnormal"},
                {"Title": "anomaly-missed", "Anomaly Tag": "Vague Abnormal"},
            ]
        )

    report = evaluate(summaries, annotations, output, threshold=0.70)

    assert report["metrics"] == {
        "videos": 4,
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall_sensitivity": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
        "balanced_accuracy": 0.5,
    }
    assert report["false_negatives"] == ["anomaly-missed"]
    assert (output / "evaluation.json").is_file()
    assert (output / "per_video.csv").is_file()
    assert (output / "confusion_matrix.csv").is_file()
    assert len((output / "threshold_sweep.csv").read_text().splitlines()) == 102
