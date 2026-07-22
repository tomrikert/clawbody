from __future__ import annotations

from reachy_mini_openclaw.attention import TapDetector, format_attention


def imu(acceleration: list[float], gyroscope: list[float] | None = None) -> dict:
    return {
        "accelerometer": acceleration,
        "gyroscope": gyroscope or [0.0, 0.0, 0.0],
    }


def test_double_tap_detector_requires_two_distinct_impacts() -> None:
    detector = TapDetector(acceleration_delta=2.0)
    assert not detector.feed(imu([0.0, 0.0, 9.8]), 0.0)
    assert not detector.feed(imu([0.0, 0.0, 13.0]), 1.0)
    assert not detector.feed(imu([0.0, 0.0, 13.0]), 1.05)
    assert detector.feed(imu([0.0, 0.0, 13.0]), 1.35)


def test_taps_too_far_apart_do_not_trigger() -> None:
    detector = TapDetector(acceleration_delta=2.0, max_gap=0.8)
    detector.feed(imu([0.0, 0.0, 9.8]), 0.0)
    assert not detector.feed(imu([0.0, 0.0, 13.0]), 1.0)
    assert not detector.feed(imu([0.0, 0.0, 13.0]), 2.0)


def test_gyroscope_can_detect_deliberate_pat() -> None:
    detector = TapDetector(gyroscope_threshold=0.5)
    detector.feed(imu([0.0, 0.0, 9.8]), 0.0)
    assert not detector.feed(imu([0.0, 0.0, 9.8], [0.7, 0.0, 0.0]), 1.0)
    assert detector.feed(imu([0.0, 0.0, 9.8], [0.7, 0.0, 0.0]), 1.4)


def test_attention_summary_reads_both_sections() -> None:
    text = format_attention(
        {
            "needsAttention": [
                {
                    "source": "gmail",
                    "sender": "Security",
                    "subject": "New login",
                },
                {
                    "source": "annexus",
                    "sender": "Jules",
                    "subject": "Quote status",
                },
            ],
            "actionsTaken": ["Marked 3 low-priority Gmail messages as read."],
        }
    )
    assert "Gmail, from Security: New login" in text
    assert "Annexus, from Jules: Quote status" in text
    assert "Actions taken" in text
    assert "Marked 3" in text
