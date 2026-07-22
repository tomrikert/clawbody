"""Camera worker thread with frame buffering and face tracking.

Provides:
- 30Hz+ camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation
- Room scanning when no face is detected
- Latest frame always available for tools
- Smooth return to neutral when face is lost

Based on pollen-robotics/reachy_mini_conversation_app camera worker.
"""

import time
import logging
import threading
import math
from io import BytesIO
from typing import Any, List, Tuple, Optional

import httpx
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking.
    
    State machine for face tracking:
        SCANNING  -- no face known, sweeping the room to find one
        TRACKING  -- face detected, following it with head offsets
        WAITING   -- face just lost, holding position briefly
        RETURNING -- interpolating back to neutral before scanning again
    """

    def __init__(
        self,
        reachy_mini: ReachyMini,
        head_tracker: Any = None,
        *,
        daemon_tracking: bool = False,
        daemon_url: str | None = None,
        tracking_weight: float = 0.7,
        sound_tracking: bool = True,
    ) -> None:
        """Initialize camera worker.
        
        Args:
            reachy_mini: Connected ReachyMini instance
            head_tracker: Optional head tracker (YOLO or MediaPipe)
        """
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker
        self.daemon_tracking = daemon_tracking
        self.daemon_url = daemon_url.rstrip("/") if daemon_url else None
        self.tracking_weight = min(1.0, max(0.0, tracking_weight))
        self.sound_tracking = sound_tracking
        self._http = httpx.Client(timeout=0.5) if self.daemon_url else None

        # Thread-safe frame storage
        self.latest_frame: Optional[NDArray[np.uint8]] = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()
        self.face_detected = False
        self.speech_detected = False
        self.doa_angle: Optional[float] = None
        self._last_attention_poll = 0.0
        self._sound_yaw = 0.0
        self._ever_received_frame = False

        # Face tracking timing (for smooth interpolation back to neutral)
        self.last_face_detected_time: Optional[float] = None
        self.interpolation_start_time: Optional[float] = None
        self.interpolation_start_pose: Optional[NDArray[np.float32]] = None
        self.face_lost_delay = 2.0  # seconds to wait before starting interpolation
        self.interpolation_duration = 1.0  # seconds to interpolate back to neutral

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled
        
        # Tracking scale factor (proportional gain for the camera-head servo loop).
        # 0.85 provides accurate convergence via closed-loop feedback while
        # avoiding single-frame overshoot that causes jitter.
        self.tracking_scale = 0.85
        
        # Smoothing factor for exponential moving average (0.0-1.0)
        # At 25Hz with alpha=0.25, 95% convergence ~0.5s -- smooth enough to
        # filter detection noise, responsive enough to feel like eye contact.
        self.smoothing_alpha = 0.25
        
        # Previous smoothed offsets for EMA calculation
        self._smoothed_offsets: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # --- Room scanning state ---
        # When no face is visible, the robot periodically sweeps the room.
        self._scanning = False
        self._scanning_start_time = 0.0
        # Scanning pattern: sinusoidal yaw sweep
        self._scan_yaw_amplitude = np.deg2rad(35)  # ±35 degrees
        self._scan_period = 8.0  # seconds for a full left-right-left cycle
        self._scan_pitch_offset = np.deg2rad(3)  # slight upward tilt while scanning
        # Start scanning immediately at boot (before any face has ever been seen)
        self._ever_seen_face = False

    def get_latest_frame(self) -> Optional[NDArray[np.uint8]]:
        """Get the latest frame (thread-safe).
        
        Returns:
            Copy of latest frame in BGR format, or None if no frame available
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_latest_jpeg(self, quality: int = 80) -> bytes | None:
        """Encode the latest BGR camera frame for an OpenClaw image turn."""
        frame = self.get_latest_frame()
        if frame is None:
            return None
        output = BytesIO()
        Image.fromarray(frame[:, :, ::-1]).save(output, format="JPEG", quality=quality)
        return output.getvalue()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe).
        
        Returns:
            Tuple of (x, y, z, roll, pitch, yaw) offsets
        """
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking.
        
        Args:
            enabled: Whether to enable face tracking
        """
        if enabled and not self.is_head_tracking_enabled:
            # Reset smoothed offsets so tracking converges quickly from scratch
            self._smoothed_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # Start scanning immediately when re-enabled
            self._start_scanning()
        self.is_head_tracking_enabled = enabled
        if self.daemon_tracking:
            if enabled:
                self.reachy_mini.start_head_tracking(weight=self.tracking_weight)
            else:
                self.reachy_mini.stop_head_tracking()
        logger.info("Head tracking %s", "enabled" if enabled else "disabled")

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True)
        self._thread.start()
        logger.info("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.daemon_tracking:
            try:
                self.reachy_mini.stop_head_tracking()
            except Exception:
                logger.debug("Could not stop daemon head tracking", exc_info=True)
        if self._http is not None:
            self._http.close()
        logger.info("Camera worker stopped")

    def get_attention_state(self) -> dict[str, Any]:
        """Return the latest daemon face and microphone-array state."""
        with self.face_tracking_lock:
            return {
                "face_detected": self.face_detected,
                "speech_detected": self.speech_detected,
                "doa_angle": self.doa_angle,
            }

    def _update_daemon_attention(self, current_time: float) -> None:
        """Blend daemon face tracking with a conservative sound-direction cue."""
        if current_time - self._last_attention_poll < 0.1:
            return
        self._last_attention_poll = current_time

        try:
            face = self.reachy_mini.get_tracked_face(wait=False)
            face_detected = bool(face and face.detected)
        except Exception:
            face_detected = False

        doa_angle: Optional[float] = None
        speech_detected = False
        if self.sound_tracking and self._http is not None:
            try:
                payload = self._http.get(f"{self.daemon_url}/api/state/doa").json()
                if payload:
                    doa_angle = float(payload["angle"])
                    speech_detected = bool(payload["speech_detected"])
            except Exception:
                logger.debug("DoA poll failed", exc_info=True)

        # Visual tracking runs inside the daemon and wins whenever a face is
        # visible. Otherwise, softly bias the normal app pose toward speech.
        if face_detected:
            target_yaw = 0.0
        elif speech_detected and doa_angle is not None:
            target_yaw = float(np.clip((math.pi / 2) - doa_angle, -0.75, 0.75))
        else:
            target_yaw = 0.0

        alpha = 0.35 if speech_detected and not face_detected else 0.18
        self._sound_yaw = alpha * target_yaw + (1.0 - alpha) * self._sound_yaw
        if abs(self._sound_yaw) < 0.01:
            self._sound_yaw = 0.0

        with self.face_tracking_lock:
            self.face_detected = face_detected
            self.speech_detected = speech_detected
            self.doa_angle = doa_angle
            self.face_tracking_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, self._sound_yaw]

    # ------------------------------------------------------------------
    # Scanning helpers
    # ------------------------------------------------------------------

    def _start_scanning(self) -> None:
        """Begin the room-scanning sweep."""
        if not self._scanning:
            self._scanning = True
            self._scanning_start_time = time.time()
            logger.debug("Started room scanning")

    def _stop_scanning(self) -> None:
        """Stop the room-scanning sweep."""
        if self._scanning:
            self._scanning = False
            logger.debug("Stopped room scanning")

    def _update_scanning_offsets(self, current_time: float) -> None:
        """Compute scanning offsets -- a slow yaw sweep with slight pitch up.
        
        The sweep is sinusoidal so the head slows at the extremes (more natural)
        and the face detector gets a chance to catch faces at the edges.
        """
        t = current_time - self._scanning_start_time
        
        yaw = float(self._scan_yaw_amplitude * np.sin(2 * np.pi * t / self._scan_period))
        pitch = float(self._scan_pitch_offset)
        
        with self.face_tracking_lock:
            self.face_tracking_offsets = [0.0, 0.0, 0.0, 0.0, pitch, yaw]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _working_loop(self) -> None:
        """Main camera worker loop.
        
        Runs at ~25Hz, captures frames and processes face tracking.
        """
        logger.debug("Starting camera working loop")

        # Neutral pose for interpolation target
        neutral_pose = np.eye(4, dtype=np.float32)
        self.previous_head_tracking_state = self.is_head_tracking_enabled
        
        # Begin scanning right away so the robot looks for a face on startup
        if self.is_head_tracking_enabled and self.head_tracker is not None:
            self._start_scanning()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Get frame from robot
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    if not self._ever_received_frame:
                        self._ever_received_frame = True
                        logger.info("First camera frame received: shape=%s", frame.shape)
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame

                    if self.daemon_tracking and self.is_head_tracking_enabled:
                        self._update_daemon_attention(current_time)

                    # Check if face tracking was just disabled
                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        # Face tracking was just disabled - start interpolation to neutral
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None
                        self._stop_scanning()

                    # Update tracking state
                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    # Handle face tracking if enabled and head tracker available
                    if self.daemon_tracking:
                        pass
                    elif self.is_head_tracking_enabled and self.head_tracker is not None:
                        self._process_face_tracking(frame, current_time, neutral_pose)
                    elif self.last_face_detected_time is not None:
                        # Handle interpolation back to neutral when tracking disabled
                        self._interpolate_to_neutral(current_time, neutral_pose)

                # Sleep to maintain ~25Hz
                time.sleep(0.04)

            except Exception as e:
                logger.error("Camera worker error: %s", e)
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")

    def _process_face_tracking(
        self, 
        frame: NDArray[np.uint8], 
        current_time: float,
        neutral_pose: NDArray[np.float32]
    ) -> None:
        """Process face tracking from frame.
        
        Args:
            frame: Current camera frame
            current_time: Current timestamp
            neutral_pose: Neutral pose matrix for interpolation
        """
        eye_center, _ = self.head_tracker.get_head_position(frame)

        if eye_center is not None:
            # Face detected!
            if not self._ever_seen_face:
                self._ever_seen_face = True
                logger.info("Face detected for the first time")
            
            # Stop scanning if we were scanning
            if self._scanning:
                self._stop_scanning()
                # Seed the EMA from current scanning offsets for smooth transition
                with self.face_tracking_lock:
                    self._smoothed_offsets = list(self.face_tracking_offsets)

            self.last_face_detected_time = current_time
            self.interpolation_start_time = None  # Stop any interpolation

            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            eye_center_norm = (eye_center + 1) / 2
            eye_center_pixels = [
                eye_center_norm[0] * w,
                eye_center_norm[1] * h,
            ]

            # Get the head pose needed to look at the target
            target_pose = self.reachy_mini.look_at_image(
                eye_center_pixels[0],
                eye_center_pixels[1],
                duration=0.0,
                perform_movement=False,
            )

            # Extract translation and rotation from the target pose
            translation = target_pose[:3, 3]
            rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

            # Scale for smoother closed-loop convergence
            translation *= self.tracking_scale
            rotation *= self.tracking_scale

            # Apply exponential moving average (EMA) smoothing to reduce jitter
            # new_smoothed = alpha * new_value + (1 - alpha) * old_smoothed
            alpha = self.smoothing_alpha
            new_offsets = [
                translation[0], translation[1], translation[2],
                rotation[0], rotation[1], rotation[2],
            ]
            
            smoothed = [
                alpha * new_offsets[i] + (1 - alpha) * self._smoothed_offsets[i]
                for i in range(6)
            ]
            self._smoothed_offsets = smoothed

            # Thread-safe update of face tracking offsets
            with self.face_tracking_lock:
                self.face_tracking_offsets = smoothed

        else:
            # No face detected
            if self._scanning:
                # Already scanning -- keep sweeping the room
                self._update_scanning_offsets(current_time)
            else:
                # Not scanning yet -- go through the wait/return/scan sequence
                self._interpolate_to_neutral(current_time, neutral_pose)

    def _interpolate_to_neutral(
        self, 
        current_time: float,
        neutral_pose: NDArray[np.float32]
    ) -> None:
        """Interpolate face tracking offsets back to neutral when face is lost.
        
        Once interpolation completes, automatically starts room scanning.
        
        Args:
            current_time: Current timestamp
            neutral_pose: Target neutral pose matrix
        """
        if self.last_face_detected_time is None:
            # Never seen a face -- go straight to scanning
            self._start_scanning()
            return

        time_since_face_lost = current_time - self.last_face_detected_time

        if time_since_face_lost >= self.face_lost_delay:
            # Start interpolation if not already started
            if self.interpolation_start_time is None:
                self.interpolation_start_time = current_time
                # Capture current pose as start of interpolation
                with self.face_tracking_lock:
                    current_translation = self.face_tracking_offsets[:3]
                    current_rotation_euler = self.face_tracking_offsets[3:]
                    # Convert to 4x4 pose matrix
                    pose_matrix = np.eye(4, dtype=np.float32)
                    pose_matrix[:3, 3] = current_translation
                    pose_matrix[:3, :3] = R.from_euler(
                        "xyz", current_rotation_euler
                    ).as_matrix()
                    self.interpolation_start_pose = pose_matrix

            # Calculate interpolation progress (t from 0 to 1)
            elapsed_interpolation = current_time - self.interpolation_start_time
            t = min(1.0, elapsed_interpolation / self.interpolation_duration)

            # Interpolate between current pose and neutral pose
            interpolated_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                neutral_pose,
                t,
            )

            # Extract translation and rotation from interpolated pose
            translation = interpolated_pose[:3, 3]
            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

            # Thread-safe update of face tracking offsets
            with self.face_tracking_lock:
                self.face_tracking_offsets = [
                    translation[0], translation[1], translation[2],
                    rotation[0], rotation[1], rotation[2],
                ]

            # If interpolation is complete, start scanning the room
            if t >= 1.0:
                self.last_face_detected_time = None
                self.interpolation_start_time = None
                self.interpolation_start_pose = None
                self._smoothed_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self._start_scanning()
