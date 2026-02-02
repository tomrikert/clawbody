"""Movement system for expressive robot control.

This module provides a 100Hz control loop for managing robot movements,
combining sequential primary moves (dances, emotions, head movements) with
additive secondary moves (speech wobble, face tracking).

Architecture:
- Primary moves are queued and executed sequentially
- Secondary moves are additive offsets applied on top
- Single control point via set_target at 100Hz
- Automatic breathing animation when idle

Based on the movement systems from:
- pollen-robotics/reachy_mini_conversation_app
- eoai-dev/moltbot_body
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset, linear_pose_interpolation

logger = logging.getLogger(__name__)

# Configuration
CONTROL_LOOP_FREQUENCY_HZ = 100.0

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]
SpeechOffsets = Tuple[float, float, float, float, float, float]


class BreathingMove(Move):
    """Continuous breathing animation for idle state."""
    
    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize breathing move.
        
        Args:
            interpolation_start_pose: Current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions
            interpolation_duration: Time to blend to neutral (seconds)
        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration
        
        # Target neutral pose
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])
        
        # Breathing parameters
        self.breathing_z_amplitude = 0.005  # 5mm gentle movement
        self.breathing_frequency = 0.1  # Hz
        self.antenna_sway_amplitude = np.deg2rad(15)  # degrees
        self.antenna_frequency = 0.5  # Hz
        
    @property
    def duration(self) -> float:
        """Duration of the move (infinite for breathing)."""
        return float("inf")
        
    def evaluate(self, t: float) -> tuple:
        """Evaluate the breathing pose at time t."""
        if t < self.interpolation_duration:
            # Interpolate to neutral
            alpha = t / self.interpolation_duration
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, 
                self.neutral_head_pose, 
                alpha
            )
            antennas = (1 - alpha) * self.interpolation_start_antennas + alpha * self.neutral_antennas
            antennas = antennas.astype(np.float64)
        else:
            # Breathing pattern
            breathing_t = t - self.interpolation_duration
            
            z_offset = self.breathing_z_amplitude * np.sin(
                2 * np.pi * self.breathing_frequency * breathing_t
            )
            head_pose = create_head_pose(
                x=0, y=0, z=z_offset, 
                roll=0, pitch=0, yaw=0, 
                degrees=True, mm=False
            )
            
            antenna_sway = self.antenna_sway_amplitude * np.sin(
                2 * np.pi * self.antenna_frequency * breathing_t
            )
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)
            
        return (head_pose, antennas, 0.0)


class HeadLookMove(Move):
    """Move to look in a specific direction."""
    
    DIRECTIONS = {
        "left": (0, 0, 0, 0, 0, 30),      # yaw left
        "right": (0, 0, 0, 0, 0, -30),    # yaw right
        "up": (0, 0, 10, 0, 15, 0),       # pitch up, z up
        "down": (0, 0, -5, 0, -15, 0),    # pitch down, z down
        "front": (0, 0, 0, 0, 0, 0),      # neutral
    }
    
    def __init__(
        self,
        direction: str,
        start_pose: NDArray[np.float32],
        start_antennas: Tuple[float, float],
        duration: float = 1.0,
    ):
        """Initialize head look move.
        
        Args:
            direction: One of 'left', 'right', 'up', 'down', 'front'
            start_pose: Current head pose
            start_antennas: Current antenna positions
            duration: Move duration in seconds
        """
        self.direction = direction
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas)
        self._duration = duration
        
        # Get target pose from direction
        params = self.DIRECTIONS.get(direction, self.DIRECTIONS["front"])
        self.target_pose = create_head_pose(
            x=params[0], y=params[1], z=params[2],
            roll=params[3], pitch=params[4], yaw=params[5],
            degrees=True, mm=True
        )
        self.target_antennas = np.array([0.0, 0.0])
        
    @property
    def duration(self) -> float:
        return self._duration
        
    def evaluate(self, t: float) -> tuple:
        """Evaluate pose at time t."""
        alpha = min(1.0, t / self._duration)
        # Smooth easing
        alpha = alpha * alpha * (3 - 2 * alpha)
        
        head_pose = linear_pose_interpolation(
            self.start_pose,
            self.target_pose,
            alpha
        )
        antennas = (1 - alpha) * self.start_antennas + alpha * self.target_antennas
        
        return (head_pose, antennas.astype(np.float64), 0.0)


def combine_full_body(primary: FullBodyPose, secondary: FullBodyPose) -> FullBodyPose:
    """Combine primary pose with secondary offsets."""
    primary_head, primary_ant, primary_yaw = primary
    secondary_head, secondary_ant, secondary_yaw = secondary
    
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)
    combined_ant = (
        primary_ant[0] + secondary_ant[0],
        primary_ant[1] + secondary_ant[1],
    )
    combined_yaw = primary_yaw + secondary_yaw
    
    return (combined_head, combined_ant, combined_yaw)


def clone_pose(pose: FullBodyPose) -> FullBodyPose:
    """Deep copy a full body pose."""
    head, ant, yaw = pose
    return (head.copy(), (float(ant[0]), float(ant[1])), float(yaw))


@dataclass
class MovementState:
    """State for the movement system."""
    current_move: Optional[Move] = None
    move_start_time: Optional[float] = None
    last_activity_time: float = 0.0
    speech_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    face_tracking_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_primary_pose: Optional[FullBodyPose] = None
    
    def update_activity(self) -> None:
        self.last_activity_time = time.monotonic()


class MovementManager:
    """Coordinate robot movements at 100Hz.
    
    This class manages:
    - Sequential primary moves (dances, emotions, head movements)
    - Additive secondary offsets (speech wobble, face tracking)
    - Automatic idle breathing animation
    - Thread-safe communication with other components
    
    Example:
        manager = MovementManager(robot)
        manager.start()
        
        # Queue a head movement
        manager.queue_move(HeadLookMove("left", ...))
        
        # Set speech offsets (called by HeadWobbler)
        manager.set_speech_offsets((0, 0, 0.01, 0.1, 0, 0))
        
        manager.stop()
    """
    
    def __init__(
        self,
        current_robot: ReachyMini,
        camera_worker: Any = None,
    ):
        """Initialize movement manager.
        
        Args:
            current_robot: Connected ReachyMini instance
            camera_worker: Optional camera worker for face tracking
        """
        self.current_robot = current_robot
        self.camera_worker = camera_worker
        
        self._now = time.monotonic
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        
        # Initialize neutral pose
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral, (0.0, 0.0), 0.0)
        
        # Move queue
        self.move_queue: deque[Move] = deque()
        
        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds before breathing starts
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency
        
        # Thread state
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_listening = False
        self._breathing_active = False
        
        # Last commanded pose for smooth transitions
        self._last_commanded_pose = clone_pose(self.state.last_primary_pose)
        self._listening_antennas = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4
        
        # Cross-thread communication
        self._command_queue: Queue[Tuple[str, Any]] = Queue()
        
        # Speech offsets (thread-safe)
        self._speech_lock = threading.Lock()
        self._pending_speech_offsets: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._speech_dirty = False
        
        # Shared state lock
        self._shared_lock = threading.Lock()
        self._shared_last_activity = self.state.last_activity_time
        self._shared_is_listening = False
        
    def queue_move(self, move: Move) -> None:
        """Queue a primary move. Thread-safe."""
        self._command_queue.put(("queue_move", move))
        
    def clear_move_queue(self) -> None:
        """Clear all queued moves. Thread-safe."""
        self._command_queue.put(("clear_queue", None))
        
    def set_speech_offsets(self, offsets: SpeechOffsets) -> None:
        """Update speech-driven offsets. Thread-safe."""
        with self._speech_lock:
            self._pending_speech_offsets = offsets
            self._speech_dirty = True
            
    def set_listening(self, listening: bool) -> None:
        """Set listening state (freezes antennas). Thread-safe."""
        self._command_queue.put(("set_listening", listening))
        
    def is_idle(self) -> bool:
        """Check if robot has been idle. Thread-safe."""
        with self._shared_lock:
            if self._shared_is_listening:
                return False
            return self._now() - self._shared_last_activity >= self.idle_inactivity_delay
            
    def _poll_signals(self, current_time: float) -> None:
        """Process queued commands and pending offsets."""
        # Apply speech offsets
        with self._speech_lock:
            if self._speech_dirty:
                self.state.speech_offsets = self._pending_speech_offsets
                self._speech_dirty = False
                self.state.update_activity()
                
        # Process commands
        while True:
            try:
                cmd, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(cmd, payload, current_time)
            
    def _handle_command(self, cmd: str, payload: Any, current_time: float) -> None:
        """Handle a single command."""
        if cmd == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                logger.debug("Queued move, queue size: %d", len(self.move_queue))
        elif cmd == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.info("Cleared move queue")
        elif cmd == "set_listening":
            desired = bool(payload)
            if self._is_listening != desired:
                self._is_listening = desired
                if desired:
                    self._listening_antennas = self._last_commanded_pose[1]
                    self._antenna_unfreeze_blend = 0.0
                else:
                    self._antenna_unfreeze_blend = 0.0
                self.state.update_activity()
                
    def _manage_move_queue(self, current_time: float) -> None:
        """Advance the move queue."""
        # Check if current move is done
        if self.state.current_move is not None and self.state.move_start_time is not None:
            elapsed = current_time - self.state.move_start_time
            if elapsed >= self.state.current_move.duration:
                self.state.current_move = None
                self.state.move_start_time = None
                
        # Start next move if available
        if self.state.current_move is None and self.move_queue:
            self.state.current_move = self.move_queue.popleft()
            self.state.move_start_time = current_time
            self._breathing_active = isinstance(self.state.current_move, BreathingMove)
            logger.debug("Starting move with duration: %s", self.state.current_move.duration)
            
    def _manage_breathing(self, current_time: float) -> None:
        """Start breathing when idle."""
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    _, current_ant = self.current_robot.get_current_joint_positions()
                    current_head = self.current_robot.get_current_head_pose()
                    
                    breathing = BreathingMove(
                        interpolation_start_pose=current_head,
                        interpolation_start_antennas=current_ant,
                        interpolation_duration=1.0,
                    )
                    self.move_queue.append(breathing)
                    self._breathing_active = True
                    self.state.update_activity()
                    logger.debug("Started breathing after %.1fs idle", idle_for)
                except Exception as e:
                    logger.error("Failed to start breathing: %s", e)
                    
        # Stop breathing if new moves queued
        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            
    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get current primary pose from move or last pose."""
        if self.state.current_move is not None and self.state.move_start_time is not None:
            t = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(t)
            
            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            if body_yaw is None:
                body_yaw = 0.0
                
            pose = (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))
            self.state.last_primary_pose = clone_pose(pose)
            return pose
            
        if self.state.last_primary_pose is not None:
            return clone_pose(self.state.last_primary_pose)
            
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        return (neutral, (0.0, 0.0), 0.0)
        
    def _get_secondary_pose(self) -> FullBodyPose:
        """Get secondary offsets."""
        offsets = [
            self.state.speech_offsets[i] + self.state.face_tracking_offsets[i]
            for i in range(6)
        ]
        
        secondary_head = create_head_pose(
            x=offsets[0], y=offsets[1], z=offsets[2],
            roll=offsets[3], pitch=offsets[4], yaw=offsets[5],
            degrees=False, mm=False
        )
        return (secondary_head, (0.0, 0.0), 0.0)
        
    def _compose_pose(self, current_time: float) -> FullBodyPose:
        """Compose final pose from primary and secondary."""
        primary = self._get_primary_pose(current_time)
        secondary = self._get_secondary_pose()
        return combine_full_body(primary, secondary)
        
    def _blend_antennas(self, target: Tuple[float, float]) -> Tuple[float, float]:
        """Blend antennas with listening freeze state."""
        if self._is_listening:
            return self._listening_antennas
            
        # Blend back from freeze
        blend = min(1.0, self._antenna_unfreeze_blend + self.target_period / self._antenna_blend_duration)
        self._antenna_unfreeze_blend = blend
        
        return (
            self._listening_antennas[0] * (1 - blend) + target[0] * blend,
            self._listening_antennas[1] * (1 - blend) + target[1] * blend,
        )
        
    def _issue_command(self, head: NDArray, antennas: Tuple[float, float], body_yaw: float) -> None:
        """Send command to robot."""
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            self._last_commanded_pose = (head.copy(), antennas, body_yaw)
        except Exception as e:
            logger.debug("set_target failed: %s", e)
            
    def _publish_shared_state(self) -> None:
        """Update shared state for external queries."""
        with self._shared_lock:
            self._shared_last_activity = self.state.last_activity_time
            self._shared_is_listening = self._is_listening
            
    def start(self) -> None:
        """Start the control loop thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MovementManager already running")
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("MovementManager started")
        
    def stop(self) -> None:
        """Stop the control loop and reset to neutral."""
        if self._thread is None or not self._thread.is_alive():
            return
            
        logger.info("Stopping MovementManager...")
        self.clear_move_queue()
        
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        
        # Reset to neutral
        try:
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            self.current_robot.goto_target(
                head=neutral,
                antennas=[0.0, 0.0],
                duration=2.0,
                body_yaw=0.0,
            )
            logger.info("Reset to neutral position")
        except Exception as e:
            logger.error("Failed to reset: %s", e)
            
    def _run_loop(self) -> None:
        """Main control loop at 100Hz."""
        logger.debug("Starting 100Hz control loop")
        
        while not self._stop_event.is_set():
            loop_start = self._now()
            
            # Process signals
            self._poll_signals(loop_start)
            
            # Manage moves
            self._manage_move_queue(loop_start)
            self._manage_breathing(loop_start)
            
            # Compose pose
            head, antennas, body_yaw = self._compose_pose(loop_start)
            
            # Blend antennas for listening
            antennas = self._blend_antennas(antennas)
            
            # Send to robot
            self._issue_command(head, antennas, body_yaw)
            
            # Update shared state
            self._publish_shared_state()
            
            # Maintain timing
            elapsed = self._now() - loop_start
            sleep_time = max(0.0, self.target_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        logger.debug("Control loop stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status for debugging."""
        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "last_commanded_pose": {
                "head": self._last_commanded_pose[0].tolist(),
                "antennas": self._last_commanded_pose[1],
                "body_yaw": self._last_commanded_pose[2],
            },
        }
