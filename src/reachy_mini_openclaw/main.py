"""ClawBody - Give your OpenClaw AI agent a physical robot body.

This module provides the main application that connects:
- OpenAI Realtime API for voice I/O (speech recognition + TTS)
- OpenClaw Gateway for AI intelligence (Clawson's brain)
- Reachy Mini robot for physical embodiment

Usage:
    # Console mode (direct audio)
    clawbody

    # With Gradio UI
    clawbody --gradio

    # With debug logging
    clawbody --debug
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dotenv import load_dotenv

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

# Load environment from project root (override=True ensures .env takes precedence)
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env", override=True)

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application.
    
    Args:
        debug: Enable debug level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Reduce noise from libraries
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="ClawBody - Give your OpenClaw AI agent a physical robot body",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run in console mode
    clawbody

    # Run with Gradio web UI
    clawbody --gradio

    # Connect to specific robot
    clawbody --robot-name my-reachy

    # Use different OpenClaw gateway
    clawbody --gateway-url http://192.168.1.100:18790
        """
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio web UI instead of console mode"
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        help="Robot name for connection (default: auto-discover)"
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789"),
        help="OpenClaw gateway URL (from OPENCLAW_GATEWAY_URL env or default)"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera functionality"
    )
    parser.add_argument(
        "--no-openclaw",
        action="store_true",
        help="Disable OpenClaw integration"
    )
    parser.add_argument(
        "--no-face-tracking",
        action="store_true",
        help="Disable face tracking"
    )
    parser.add_argument(
        "--local-vision",
        action="store_true",
        help="Enable local vision processing with SmolVLM2"
    )
    parser.add_argument(
        "--local-ai",
        action="store_true",
        help="Use local AI pipeline (faster-whisper + Ollama + Kokoro) instead of OpenAI Realtime"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Custom personality profile to use"
    )
    
    return parser.parse_args()


class ClawBodyCore:
    """ClawBody core application controller.
    
    This class orchestrates all components:
    - Reachy Mini robot connection and movement control
    - OpenAI Realtime API for voice I/O
    - OpenClaw gateway bridge for AI intelligence
    - Audio input/output loops
    """
    
    def __init__(
        self,
        gateway_url: str = "ws://localhost:18789",
        robot_name: Optional[str] = None,
        enable_camera: bool = True,
        enable_openclaw: bool = True,
        enable_face_tracking: Optional[bool] = None,
        head_tracker_type: Optional[str] = None,
        robot: Optional["ReachyMini"] = None,
        external_stop_event: Optional[threading.Event] = None,
    ):
        """Initialize the application.
        
        Args:
            gateway_url: OpenClaw gateway URL
            robot_name: Optional robot name for connection
            enable_camera: Whether to enable camera functionality
            enable_openclaw: Whether to enable OpenClaw integration
            enable_face_tracking: Optional override for face tracking
            head_tracker_type: Optional override for tracker type
            robot: Optional pre-initialized robot (for app framework)
            external_stop_event: Optional external stop event
        """
        from reachy_mini import ReachyMini
        from reachy_mini_openclaw.config import config
        from reachy_mini_openclaw.moves import MovementManager
        from reachy_mini_openclaw.audio.head_wobbler import HeadWobbler
        from reachy_mini_openclaw.openclaw_bridge import OpenClawBridge
        from reachy_mini_openclaw.tools.core_tools import ToolDependencies
        
        self.gateway_url = gateway_url
        self._external_stop_event = external_stop_event
        self._owns_robot = robot is None
        
        # Validate configuration
        errors = config.validate()
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            sys.exit(1)
        
        # Connect to robot
        if robot is not None:
            self.robot = robot
            logger.info("Using provided Reachy Mini instance")
        else:
            logger.info("Connecting to Reachy Mini...")
            robot_kwargs = {}
            if robot_name:
                robot_kwargs["robot_name"] = robot_name
                
            try:
                self.robot = ReachyMini(**robot_kwargs)
            except TimeoutError as e:
                logger.error("Connection timeout: %s", e)
                logger.error("Check that the robot is powered on and reachable.")
                sys.exit(1)
            except Exception as e:
                logger.error("Robot connection failed: %s", e)
                sys.exit(1)
                
            logger.info("Connected to robot: %s", self.robot.client.get_status())
        
        # Initialize movement system
        logger.info("Initializing movement system...")
        self.movement_manager = MovementManager(current_robot=self.robot)
        self.head_wobbler = HeadWobbler(
            set_speech_offsets=self.movement_manager.set_speech_offsets
        )
        
        # Initialize OpenClaw bridge
        self.openclaw_bridge = None
        if enable_openclaw:
            logger.info("Initializing OpenClaw bridge...")
            self.openclaw_bridge = OpenClawBridge(
                gateway_url=gateway_url,
                gateway_token=config.OPENCLAW_TOKEN,
            )
        
        # Camera worker for video streaming and frame capture
        self.camera_worker = None
        self.head_tracker = None
        self.vision_manager = None
        
        if enable_camera:
            logger.info("Initializing camera worker...")
            from reachy_mini_openclaw.camera_worker import CameraWorker
            
            # Initialize head tracker for local face tracking
            tracking_enabled = config.ENABLE_FACE_TRACKING if enable_face_tracking is None else enable_face_tracking
            tracker = head_tracker_type or config.HEAD_TRACKER_TYPE
            if tracking_enabled:
                self.head_tracker = self._initialize_head_tracker(tracker)
            
            # Initialize camera worker with head tracker
            self.camera_worker = CameraWorker(
                reachy_mini=self.robot,
                head_tracker=self.head_tracker,
            )
            
            # Enable/disable head tracking based on whether we have a tracker
            self.camera_worker.set_head_tracking_enabled(self.head_tracker is not None)
            
            # Initialize local vision processor if enabled
            if config.ENABLE_LOCAL_VISION:
                self.vision_manager = self._initialize_vision_manager()
        
        # Create tool dependencies
        self.deps = ToolDependencies(
            movement_manager=self.movement_manager,
            head_wobbler=self.head_wobbler,
            robot=self.robot,
            camera_worker=self.camera_worker,
            openclaw_bridge=self.openclaw_bridge,
            vision_manager=self.vision_manager,
        )
        
        # Initialize voice handler
        if config.AI_PROVIDER == "local":
            from reachy_mini_openclaw.local_voice import LocalVoiceHandler

            logger.info("Using local AI provider")
            self.handler = LocalVoiceHandler(
                deps=self.deps,
                openclaw_bridge=self.openclaw_bridge,
            )
            handler_task_name = "local-voice-handler"
        else:
            from reachy_mini_openclaw.openai_realtime import OpenAIRealtimeHandler

            logger.info("Using OpenAI Realtime provider")
            self.handler = OpenAIRealtimeHandler(
                deps=self.deps,
                openclaw_bridge=self.openclaw_bridge,
            )
            handler_task_name = "openai-handler"
        self._handler_task_name = handler_task_name
        
        # State
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def _wait_for_voice_ready(
        self,
        handler_task: asyncio.Task,
        timeout: float = 90.0,
    ) -> None:
        """Wait until the selected voice handler can accept audio."""
        ready_event = getattr(self.handler, "_ready", None)
        if ready_event is None:
            ready_event = getattr(self.handler, "_connected_event", None)

        if ready_event is None:
            return

        ready_task = asyncio.create_task(ready_event.wait(), name="voice-ready-wait")
        try:
            done, _pending = await asyncio.wait(
                {ready_task, handler_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if handler_task in done:
                exc = handler_task.exception()
                if exc is not None:
                    raise exc
                logger.warning("Voice handler exited before reporting ready")
                return

            if ready_task in done:
                return

            logger.warning(
                "Voice handler not ready after %.0f seconds; starting audio anyway",
                timeout,
            )
        finally:
            if not ready_task.done():
                ready_task.cancel()
        
    def _initialize_vision_manager(self) -> Optional[Any]:
        """Initialize local vision processor (SmolVLM2).
        
        Returns:
            VisionManager instance or None if initialization fails
        """
        if self.camera_worker is None:
            logger.warning("Cannot initialize vision manager without camera worker")
            return None
        
        try:
            from reachy_mini_openclaw.vision.processors import (
                VisionConfig, 
                initialize_vision_manager,
            )
            from reachy_mini_openclaw.config import config
            
            vision_config = VisionConfig(
                model_path=config.LOCAL_VISION_MODEL,
                device_preference=config.VISION_DEVICE,
                hf_home=config.HF_HOME,
            )
            
            logger.info("Initializing local vision processor (SmolVLM2)...")
            vision_manager = initialize_vision_manager(self.camera_worker, vision_config)
            
            if vision_manager is not None:
                logger.info("Local vision processor initialized")
            else:
                logger.warning("Local vision processor failed to initialize")
            
            return vision_manager
            
        except ImportError as e:
            logger.warning(f"Local vision not available: {e}")
            logger.warning("Install with: pip install torch transformers")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize vision manager: {e}")
            return None
    
    def _initialize_head_tracker(self, tracker_type: Optional[str] = None) -> Optional[Any]:
        """Initialize head tracker for local face tracking.
        
        Args:
            tracker_type: Type of tracker ("yolo", "mediapipe", or None for auto)
            
        Returns:
            Initialized head tracker or None if initialization fails
        """
        # Default to YOLO if not specified
        if tracker_type is None:
            tracker_type = "yolo"
        
        if tracker_type == "yolo":
            try:
                from reachy_mini_openclaw.vision.yolo_head_tracker import HeadTracker
                logger.info("Initializing YOLO face tracker...")
                tracker = HeadTracker(device="cpu")  # CPU is fast enough for face detection
                logger.info("YOLO face tracker initialized")
                return tracker
            except ImportError as e:
                logger.warning(f"YOLO tracker not available: {e}")
                logger.warning("Install with: pip install ultralytics supervision")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO tracker: {e}")
        
        elif tracker_type == "mediapipe":
            try:
                from reachy_mini_openclaw.vision.mediapipe_tracker import HeadTracker
                logger.info("Initializing MediaPipe face tracker...")
                tracker = HeadTracker()
                logger.info("MediaPipe face tracker initialized")
                return tracker
            except ImportError as e:
                logger.warning(f"MediaPipe tracker not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe tracker: {e}")
        
        logger.warning("No face tracker available - face tracking disabled")
        return None
        
    def _should_stop(self) -> bool:
        """Check if we should stop."""
        if self._stop_event.is_set():
            return True
        if self._external_stop_event is not None and self._external_stop_event.is_set():
            return True
        return False
        
    async def record_loop(self) -> None:
        """Read audio from robot microphone and send to handler."""
        input_sr = self.robot.media.get_input_audio_samplerate()
        logger.info("Recording at %d Hz", input_sr)
        
        while not self._should_stop():
            audio_frame = self.robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sr, audio_frame))
            await asyncio.sleep(0.01)
            
    async def play_loop(self) -> None:
        """Play audio from handler through robot speakers."""
        output_sr = self.robot.media.get_output_audio_samplerate()
        logger.info("Playing at %d Hz", output_sr)
        
        while not self._should_stop():
            output = await self.handler.emit()
            if output is not None:
                if isinstance(output, tuple):
                    input_sr, audio_data = output
                    
                    # Convert to float32 and normalize (OpenAI sends int16)
                    audio_data = audio_data.flatten().astype("float32") / 32768.0
                    
                    # Reduce volume to prevent distortion (0.5 = 50% volume)
                    audio_data = audio_data * 0.5
                    
                    # Resample if needed
                    if input_sr != output_sr:
                        from scipy.signal import resample
                        num_samples = int(len(audio_data) * output_sr / input_sr)
                        audio_data = resample(audio_data, num_samples).astype("float32")
                        
                    self.robot.media.push_audio_sample(audio_data)
                # else: it's an AdditionalOutputs (transcript) - handle in UI mode
                
            await asyncio.sleep(0.01)
            
    async def run(self) -> None:
        """Run the main application loop."""
        # Test OpenClaw connection
        if self.openclaw_bridge is not None:
            connected = await self.openclaw_bridge.connect()
            if connected:
                logger.info("OpenClaw gateway connected")
            else:
                logger.warning("OpenClaw gateway not available - some features disabled")
        
        # Enable motors and move to neutral pose
        logger.info("Enabling motors and moving to neutral position...")
        try:
            self.robot.enable_motors()
            from reachy_mini.utils import create_head_pose
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            self.robot.goto_target(
                head=neutral,
                antennas=[0.0, 0.0],
                duration=2.0,
                body_yaw=0.0,
            )
            time.sleep(2)  # Wait for goto to complete
            logger.info("Robot at neutral position with motors enabled")
        except Exception as e:
            logger.error("Failed to initialize robot pose: %s", e)
        
        # Wire up camera worker to movement manager for face tracking
        if self.camera_worker is not None:
            self.movement_manager.camera_worker = self.camera_worker
            logger.info("Face tracking connected to movement system")
        
        # Start movement system
        logger.info("Starting movement system...")
        self.movement_manager.start()
        self.head_wobbler.start()
        
        # Start camera worker for video streaming
        if self.camera_worker is not None:
            logger.info("Starting camera worker...")
            self.camera_worker.start()
        
        # Start local vision processor if available
        if self.vision_manager is not None:
            logger.info("Starting local vision processor...")
            self.vision_manager.start()
        
        # Start voice handler before audio so readiness logs reflect real availability.
        logger.info("Starting voice handler...")
        handler_task = asyncio.create_task(self.handler.start_up(), name=self._handler_task_name)
        await self._wait_for_voice_ready(handler_task)

        # Start audio
        logger.info("Starting audio...")
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        time.sleep(1)  # Let pipelines initialize
        
        logger.info("Ready! Speak to me...")
        
        # Start audio loops
        self._tasks = [
            handler_task,
            asyncio.create_task(self.record_loop(), name="record-loop"),
            asyncio.create_task(self.play_loop(), name="play-loop"),
        ]
        
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")
            
    def stop(self) -> None:
        """Stop everything."""
        logger.info("Stopping...")
        self._stop_event.set()
        
        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Stop voice handler
        try:
            asyncio.get_event_loop().run_until_complete(self.handler.shutdown())
        except Exception as e:
            logger.debug("Voice handler shutdown: %s", e)
                
        # Stop movement system
        self.head_wobbler.stop()
        self.movement_manager.stop()
        
        # Stop vision manager
        if self.vision_manager is not None:
            self.vision_manager.stop()
        
        # Stop camera worker
        if self.camera_worker is not None:
            self.camera_worker.stop()
        
        # Disconnect OpenClaw bridge
        if self.openclaw_bridge is not None:
            try:
                asyncio.get_event_loop().run_until_complete(
                    self.openclaw_bridge.disconnect()
                )
            except Exception as e:
                logger.debug("OpenClaw disconnect: %s", e)
        
        # Close resources if we own them
        if self._owns_robot:
            try:
                self.robot.media.close()
            except Exception as e:
                logger.debug("Media close: %s", e)
            self.robot.client.disconnect()
            
        logger.info("Stopped")


class ClawBodyApp:
    """ClawBody - Reachy Mini Apps entry point.
    
    This class allows ClawBody to be installed and run from
    the Reachy Mini dashboard as a Reachy Mini App.
    """
    
    # No custom settings UI
    custom_app_url: Optional[str] = None
    
    def run(self, reachy_mini, stop_event: threading.Event) -> None:
        """Run ClawBody as a Reachy Mini App.
        
        Args:
            reachy_mini: Pre-initialized ReachyMini instance
            stop_event: Threading event to signal stop
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        gateway_url = os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789")
        
        app = ClawBodyCore(
            gateway_url=gateway_url,
            robot=reachy_mini,
            external_stop_event=stop_event,
        )
        
        try:
            loop.run_until_complete(app.run())
        except Exception as e:
            logger.error("Error running app: %s", e)
        finally:
            app.stop()
            loop.close()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.debug)
    
    # Set custom profile if specified
    if args.profile:
        from reachy_mini_openclaw.config import set_custom_profile
        set_custom_profile(args.profile)
    
    # Configure provider, face tracking, and local vision from args
    from reachy_mini_openclaw.config import (
        config,
        set_face_tracking_enabled, 
        set_local_vision_enabled,
    )
    if args.local_ai:
        config.AI_PROVIDER = "local"
    if args.no_face_tracking:
        set_face_tracking_enabled(False)
    if args.local_vision:
        set_local_vision_enabled(True)
    
    if args.gradio:
        # Launch Gradio UI
        logger.info("Starting Gradio UI...")
        from reachy_mini_openclaw.gradio_app import launch_gradio
        launch_gradio(
            gateway_url=args.gateway_url,
            robot_name=args.robot_name,
            enable_camera=not args.no_camera,
            enable_openclaw=not args.no_openclaw,
        )
    else:
        # Console mode
        app = ClawBodyCore(
            gateway_url=args.gateway_url,
            robot_name=args.robot_name,
            enable_camera=not args.no_camera,
            enable_openclaw=not args.no_openclaw,
        )
        
        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            app.stop()


if __name__ == "__main__":
    main()
