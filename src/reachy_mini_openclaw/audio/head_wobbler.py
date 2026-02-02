"""Audio-driven head movement for natural speech animation.

This module analyzes audio output in real-time and generates subtle head
movements that make the robot appear more expressive and alive while speaking.

The wobble is generated based on:
- Audio amplitude (volume) -> vertical movement
- Frequency content -> horizontal sway
- Speech rhythm -> timing of movements

Design:
- Runs in a separate thread to avoid blocking the main audio pipeline
- Uses a circular buffer for smooth interpolation
- Generates offsets that are added to the primary pose by MovementManager
"""

import base64
import logging
import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias for speech offsets: (x, y, z, roll, pitch, yaw)
SpeechOffsets = Tuple[float, float, float, float, float, float]


class HeadWobbler:
    """Generate audio-driven head movements for expressive speech.
    
    The wobbler analyzes incoming audio and produces subtle head movements
    that are synchronized with speech patterns, making the robot appear
    more natural and engaged during conversation.
    
    Example:
        def apply_offsets(offsets):
            movement_manager.set_speech_offsets(offsets)
        
        wobbler = HeadWobbler(set_speech_offsets=apply_offsets)
        wobbler.start()
        
        # Feed audio as it's played
        wobbler.feed(base64_audio_chunk)
        
        wobbler.stop()
    """
    
    def __init__(
        self,
        set_speech_offsets: Callable[[SpeechOffsets], None],
        sample_rate: int = 24000,
        update_rate: float = 30.0,  # Hz
    ):
        """Initialize the head wobbler.
        
        Args:
            set_speech_offsets: Callback to apply offsets to the movement system
            sample_rate: Expected audio sample rate (Hz)
            update_rate: How often to update offsets (Hz)
        """
        self.set_speech_offsets = set_speech_offsets
        self.sample_rate = sample_rate
        self.update_period = 1.0 / update_rate
        
        # Audio analysis parameters
        self.amplitude_scale = 0.008  # Max displacement in meters
        self.roll_scale = 0.15  # Max roll in radians
        self.pitch_scale = 0.08  # Max pitch in radians
        self.smoothing = 0.3  # Smoothing factor (0-1)
        
        # State
        self._audio_buffer: deque[NDArray[np.float32]] = deque(maxlen=10)
        self._buffer_lock = threading.Lock()
        self._current_amplitude = 0.0
        self._current_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_feed_time = 0.0
        self._is_speaking = False
        
        # Decay parameters for smooth return to neutral
        self._decay_rate = 3.0  # How fast to decay when not speaking
        self._speech_timeout = 0.3  # Seconds of silence before decay starts
        
    def start(self) -> None:
        """Start the wobbler thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("HeadWobbler already running")
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.debug("HeadWobbler started")
        
    def stop(self) -> None:
        """Stop the wobbler thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        # Reset to neutral
        self.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        logger.debug("HeadWobbler stopped")
        
    def reset(self) -> None:
        """Reset the wobbler state (call when speech ends or is interrupted)."""
        with self._buffer_lock:
            self._audio_buffer.clear()
        self._current_amplitude = 0.0
        self._is_speaking = False
        self.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        logger.debug("HeadWobbler reset")
        
    def feed(self, audio_b64: str) -> None:
        """Feed audio data to the wobbler.
        
        Args:
            audio_b64: Base64-encoded PCM audio (int16)
        """
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            with self._buffer_lock:
                self._audio_buffer.append(audio_float)
                
            self._last_feed_time = time.monotonic()
            self._is_speaking = True
            
        except Exception as e:
            logger.debug("Error feeding audio to wobbler: %s", e)
            
    def _compute_amplitude(self) -> float:
        """Compute current audio amplitude from buffer."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return 0.0
            
            # Concatenate recent audio
            audio = np.concatenate(list(self._audio_buffer))
            
        # RMS amplitude
        rms = np.sqrt(np.mean(audio ** 2))
        return min(1.0, rms * 3.0)  # Scale and clamp
        
    def _compute_offsets(self, amplitude: float, t: float) -> SpeechOffsets:
        """Compute head offsets based on amplitude and time.
        
        Args:
            amplitude: Current audio amplitude (0-1)
            t: Current time for oscillation
            
        Returns:
            Tuple of (x, y, z, roll, pitch, yaw) offsets
        """
        if amplitude < 0.01:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        # Vertical bob based on amplitude
        z_offset = amplitude * self.amplitude_scale * np.sin(t * 8.0)
        
        # Subtle roll sway
        roll_offset = amplitude * self.roll_scale * np.sin(t * 3.0)
        
        # Pitch variation
        pitch_offset = amplitude * self.pitch_scale * np.sin(t * 5.0 + 0.5)
        
        # Small yaw drift
        yaw_offset = amplitude * 0.05 * np.sin(t * 2.0)
        
        return (0.0, 0.0, z_offset, roll_offset, pitch_offset, yaw_offset)
        
    def _run_loop(self) -> None:
        """Main wobbler loop."""
        start_time = time.monotonic()
        
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            t = loop_start - start_time
            
            # Check if we're still receiving audio
            silence_duration = loop_start - self._last_feed_time
            
            if silence_duration > self._speech_timeout:
                # Decay amplitude when not speaking
                self._current_amplitude *= np.exp(-self._decay_rate * self.update_period)
                self._is_speaking = False
            else:
                # Compute new amplitude with smoothing
                raw_amplitude = self._compute_amplitude()
                self._current_amplitude = (
                    self.smoothing * raw_amplitude +
                    (1 - self.smoothing) * self._current_amplitude
                )
            
            # Compute and apply offsets
            offsets = self._compute_offsets(self._current_amplitude, t)
            
            # Smooth transition between offsets
            new_offsets = tuple(
                self.smoothing * new + (1 - self.smoothing) * old
                for new, old in zip(offsets, self._current_offsets)
            )
            self._current_offsets = new_offsets
            
            # Apply to movement system
            self.set_speech_offsets(new_offsets)
            
            # Maintain update rate
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self.update_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
