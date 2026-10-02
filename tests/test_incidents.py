"""Test suspicious-window incident aggregation."""

from __future__ import annotations

from jev_monitor.incidents import aggregate_events


def event(index: int, suspicious: float | None) -> dict:
    return {
        "window_id": f"window-{index}",
        "camera_id": "camera-1",
        "captured_at": f"2026-10-01T00:00:{index:02d}+00:00",
        "classification": (
            {"safe": 1 - suspicious, "suspicious": suspicious, "provider": "test"}
            if suspicious is not None
            else None
        ),
    }


def test_aggregate_events_merges_nearby_candidates_and_keeps_singletons() -> None:
    events = [
        event(0, 0.10),
        event(1, 0.20),
        event(2, 0.40),
        event(3, None),
        event(4, 0.30),
        event(5, 0.05),
        event(6, 0.05),
        event(7, 0.80),
    ]

    incidents = aggregate_events(
        events,
        threshold=0.15,
        max_gap_seconds=1.0,
        sample_interval_seconds=0.5,
    )

    assert len(incidents) == 2
    first, second = incidents
    assert first.window_count == 3
    assert first.window_ids == ["window-1", "window-2", "window-4"]
    assert first.peak_window_id == "window-2"
    assert first.peak_suspicious_probability == 0.40
    assert first.started_at_seconds == 0.5
    assert first.ended_at_seconds == 3.0
    assert second.window_count == 1
    assert second.peak_window_id == "window-7"
    assert second.status == "pending_analysis"
