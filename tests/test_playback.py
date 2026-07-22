import numpy as np

from reachy_mini_openclaw.audio.playback import apply_output_gain


def test_output_gain_preserves_full_scale_by_default() -> None:
    samples = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(apply_output_gain(samples, 1.0), samples)


def test_output_gain_boosts_and_clips_safely() -> None:
    samples = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float32)
    np.testing.assert_allclose(
        apply_output_gain(samples, 2.0),
        np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32),
    )
