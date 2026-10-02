"""Evaluate video summaries against SmartHome-Bench anomaly annotations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SAFE = "safe"
SUSPICIOUS = "suspicious"


@dataclass(frozen=True)
class VideoPrediction:
    video_id: str
    anomaly_tag: str
    expected: str
    predicted: str
    max_suspicious_probability: float | None
    correct: bool


def expected_label(anomaly_tag: str) -> str:
    normalized = " ".join(anomaly_tag.strip().lower().split())
    if normalized == "normal":
        return SAFE
    if normalized in {"abnormal", "vague abnormal"}:
        return SUSPICIOUS
    raise ValueError(f"Unsupported anomaly tag: {anomaly_tag!r}")


def predicted_label(score: float | None, threshold: float) -> str:
    return SUSPICIOUS if score is not None and score >= threshold else SAFE


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def calculate_metrics(predictions: Iterable[VideoPrediction]) -> dict[str, object]:
    rows = list(predictions)
    tp = sum(row.expected == SUSPICIOUS and row.predicted == SUSPICIOUS for row in rows)
    tn = sum(row.expected == SAFE and row.predicted == SAFE for row in rows)
    fp = sum(row.expected == SAFE and row.predicted == SUSPICIOUS for row in rows)
    fn = sum(row.expected == SUSPICIOUS and row.predicted == SAFE for row in rows)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    balanced_accuracy = (
        round((recall + specificity) / 2, 4)
        if recall is not None and specificity is not None
        else None
    )
    return {
        "videos": len(rows),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": _ratio(tp + tn, len(rows)),
        "precision": _ratio(tp, tp + fp),
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": balanced_accuracy,
    }


def _load_scores(summaries_dir: Path) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for path in sorted(summaries_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        video_id = str(summary.get("camera_id") or path.stem)
        value = summary.get("max_suspicious_probability")
        scores[video_id] = float(value) if value is not None else None
    if not scores:
        raise ValueError(f"No summary JSON files found in {summaries_dir}")
    return scores


def load_predictions(
    summaries_dir: Path,
    annotations_path: Path,
    threshold: float,
) -> list[VideoPrediction]:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    scores = _load_scores(summaries_dir)
    predictions: list[VideoPrediction] = []
    with annotations_path.open(encoding="utf-8-sig", newline="") as handle:
        for annotation in csv.DictReader(handle):
            video_id = annotation["Title"].strip()
            if video_id not in scores:
                raise ValueError(f"Missing summary for annotated video: {video_id}")
            anomaly_tag = annotation["Anomaly Tag"].strip()
            expected = expected_label(anomaly_tag)
            predicted = predicted_label(scores[video_id], threshold)
            predictions.append(
                VideoPrediction(
                    video_id=video_id,
                    anomaly_tag=anomaly_tag,
                    expected=expected,
                    predicted=predicted,
                    max_suspicious_probability=scores[video_id],
                    correct=expected == predicted,
                )
            )

    annotated_ids = {row.video_id for row in predictions}
    extras = sorted(set(scores) - annotated_ids)
    if extras:
        raise ValueError(f"Summaries have no matching annotation: {', '.join(extras)}")
    return sorted(predictions, key=lambda row: row.video_id)


def _threshold_rows(
    scores_and_labels: list[tuple[float | None, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(101):
        threshold = step / 100
        predictions = []
        for index, (score, expected) in enumerate(scores_and_labels):
            predicted = predicted_label(score, threshold)
            predictions.append(
                VideoPrediction(
                    video_id=str(index),
                    anomaly_tag="",
                    expected=expected,
                    predicted=predicted,
                    max_suspicious_probability=score,
                    correct=expected == predicted,
                )
            )
        rows.append({"threshold": round(threshold, 2), **calculate_metrics(predictions)})
    return rows


def _metric_value(row: dict[str, object], name: str) -> float:
    value = row[name]
    return float(value) if value is not None else -1.0


def evaluate(
    summaries_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    threshold: float = 0.70,
) -> dict[str, object]:
    predictions = load_predictions(summaries_dir, annotations_path, threshold)
    metrics = calculate_metrics(predictions)
    sweep = _threshold_rows(
        [(row.max_suspicious_probability, row.expected) for row in predictions]
    )
    recommended = max(
        sweep,
        key=lambda row: (
            _metric_value(row, "f1"),
            _metric_value(row, "balanced_accuracy"),
            float(row["threshold"]),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "threshold": threshold,
        "label_mapping": {
            "Normal": SAFE,
            "Abnormal": SUSPICIOUS,
            "Vague Abnormal": SUSPICIOUS,
        },
        "metrics": metrics,
        "confusion_matrix": {
            "labels": [SAFE, SUSPICIOUS],
            "rows_actual_columns_predicted": [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ],
        },
        "false_negatives": [
            row.video_id
            for row in predictions
            if row.expected == SUSPICIOUS and row.predicted == SAFE
        ],
        "anomaly_videos": [
            {
                "video_id": row.video_id,
                "anomaly_tag": row.anomaly_tag,
                "score": row.max_suspicious_probability,
                "screened_as_suspicious": row.predicted == SUSPICIOUS,
            }
            for row in predictions
            if row.expected == SUSPICIOUS
        ],
        "exploratory_threshold_recommendation": {
            "threshold": recommended["threshold"],
            "f1": recommended["f1"],
            "balanced_accuracy": recommended["balanced_accuracy"],
            "note": "Selected on this eight-video sample; validate on a separate set before adoption.",
        },
    }

    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    with (output_dir / "per_video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(predictions[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in predictions)

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", SAFE, SUSPICIOUS])
        writer.writerow([SAFE, metrics["true_negative"], metrics["false_positive"]])
        writer.writerow([SUSPICIOUS, metrics["false_negative"], metrics["true_positive"]])

    with (output_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate JevGuard video summaries")
    parser.add_argument("--summaries-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    try:
        report = evaluate(
            summaries_dir=args.summaries_dir,
            annotations_path=args.annotations,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
