"""Audio preparation helpers for Reachy speaker playback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_output_gain(samples: NDArray[np.floating], gain: float) -> NDArray[np.float32]:
    """Apply software gain and safely constrain samples to Reachy's float range."""
    audio = np.asarray(samples, dtype=np.float32)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32, copy=False)
