"""ClawBody - Give your OpenClaw AI agent a physical robot body.

This module provides the main application that connects:
- Local/OpenAI-compatible STT and provider-based TTS
- OpenClaw Gateway as the sole conversational brain
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
import time
import asyncio
import base64
import logging
import argparse
import threading
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from reachy_mini import ReachyMiniApp

# Load local development settings without overriding deployment-injected env.
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env", override=False)

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
    clawbody --gateway-url http://openclaw-host.local:18789
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
        "--robot-host",
        type=str,
        default=os.getenv("ROBOT_HOST", "reachy-mini.local"),
        help="Reachy daemon hostname or IP",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=int(os.getenv("ROBOT_PORT", "8000")),
        help="Reachy daemon TCP port",
    )
    parser.add_argument(
        "--robot-connection-mode",
        choices=["auto", "localhost_only", "network"],
        default=os.getenv("ROBOT_CONNECTION_MODE", "auto"),
        help="Reachy SDK connection mode",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=os.getenv("OPENCLAW_GATEWAY_URL", ""),
        help="OpenClaw gateway URL (required unless supplied by environment)"
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
        "--profile",
        type=str,
        help="Custom personality profile to use"
    )
    
    return parser.parse_args()


class ClawBodyCore:
    """ClawBody core application controller.
    
    This class orchestrates all components:
    - Reachy Mini robot connection and movement control
    - Provider-based speech recognition and synthesis
    - OpenClaw gateway bridge for conversational intelligence
    - Audio input/output loops
    """
    
    def __init__(
        self,
        gateway_url: str = "",
        robot_name: Optional[str] = None,
        robot_host: Optional[str] = None,
        robot_port: Optional[int] = None,
        robot_connection_mode: Optional[str] = None,
        enable_camera: bool = True,
        enable_openclaw: bool = True,
        robot: Optional["ReachyMini"] = None,
        external_stop_event: Optional[threading.Event] = None,
    ):
        """Initialize the application.
        
        Args:
            gateway_url: OpenClaw gateway URL
            robot_name: Optional robot name for connection
            robot_host: Reachy daemon hostname or IP
            robot_port: Reachy daemon TCP port
            robot_connection_mode: Reachy SDK connection mode
            enable_camera: Whether to enable camera functionality
            enable_openclaw: Whether to enable OpenClaw integration
            robot: Optional pre-initialized robot (for app framework)
            external_stop_event: Optional external stop event
        """
        from reachy_mini import ReachyMini
        from reachy_mini_openclaw.config import config
        from reachy_mini_openclaw.moves import MovementManager
        from reachy_mini_openclaw.audio.head_wobbler import HeadWobbler
        from reachy_mini_openclaw.openclaw_bridge import OpenClawBridge
        from reachy_mini_openclaw.motion_catalog import MotionCatalog
        from reachy_mini_openclaw.tools.core_tools import ToolDependencies
        from reachy_mini_openclaw.audio.conversation import ConversationHandler, EnergyVAD
        from reachy_mini_openclaw.audio.providers import (
            ChatterboxTTS,
            ElevenLabsTTS,
            FallbackTTS,
            OpenAICompatibleSTT,
        )
        
        self.gateway_url = gateway_url
        self.audio_output_gain = config.AUDIO_OUTPUT_GAIN
        self._external_stop_event = external_stop_event
        self._owns_robot = robot is None
        
        # Validate configuration
        errors = config.validate(gateway_url=self.gateway_url)
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            raise ValueError("Invalid ClawBody configuration")
        
        # Connect to robot
        if robot is not None:
            self.robot = robot
            logger.info("Using provided Reachy Mini instance")
        else:
            logger.info("Connecting to Reachy Mini...")
            robot_kwargs = {}
            if robot_name:
                robot_kwargs["robot_name"] = robot_name
            robot_kwargs["host"] = robot_host or config.ROBOT_HOST
            robot_kwargs["port"] = robot_port or config.ROBOT_PORT
            robot_kwargs["connection_mode"] = robot_connection_mode or config.ROBOT_CONNECTION_MODE
                
            try:
                self.robot = ReachyMini(**robot_kwargs)
            except TimeoutError as e:
                logger.error("Connection timeout: %s", e)
                logger.error("Check that the robot is powered on and reachable.")
                raise RuntimeError("Reachy Mini connection timed out") from e
            except Exception as e:
                logger.error("Robot connection failed: %s", e)
                raise RuntimeError("Reachy Mini connection failed") from e
                
            logger.info("Connected to robot: %s", self.robot.client.get_status())
        
        # Initialize movement system
        logger.info("Initializing movement system...")
        self.movement_manager = MovementManager(current_robot=self.robot)
        self.head_wobbler = HeadWobbler(
            set_speech_offsets=self.movement_manager.set_speech_offsets
        )
        logger.info("Loading Reachy emotion and dance catalogs...")
        self.motion_catalog = MotionCatalog()
        logger.info(
            "Loaded %d emotion moves and %d dance moves",
            len(self.motion_catalog.list("emotion")),
            len(self.motion_catalog.list("dance")),
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

            daemon_tracking = config.ENABLE_FACE_TRACKING and config.HEAD_TRACKER_TYPE == "daemon"
            
            # Initialize head tracker for local face tracking
            if config.ENABLE_FACE_TRACKING and not daemon_tracking:
                self.head_tracker = self._initialize_head_tracker(config.HEAD_TRACKER_TYPE)
            
            # Initialize camera worker with head tracker
            self.camera_worker = CameraWorker(
                reachy_mini=self.robot,
                head_tracker=self.head_tracker,
                daemon_tracking=daemon_tracking,
                daemon_url=(
                    "http://127.0.0.1:8000"
                    if robot is not None
                    else f"http://{robot_host or config.ROBOT_HOST}:{robot_port or config.ROBOT_PORT}"
                ),
                tracking_weight=config.FACE_TRACKING_WEIGHT,
                sound_tracking=config.ENABLE_SOUND_TRACKING,
            )
            
            # Enable/disable head tracking based on whether we have a tracker
            self.camera_worker.set_head_tracking_enabled(daemon_tracking or self.head_tracker is not None)
            
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
            motion_catalog=self.motion_catalog,
        )
        
        if self.openclaw_bridge is None:
            raise ValueError("The conversational pipeline requires OpenClaw")

        stt = OpenAICompatibleSTT(
            config.STT_BASE_URL,
            config.STT_MODEL,
            api_key=config.STT_API_KEY,
            language=config.STT_LANGUAGE,
            timeout=config.STT_TIMEOUT_SECONDS,
        )
        chatterbox = ChatterboxTTS(
            config.CHATTERBOX_URL,
            voice=config.CHATTERBOX_VOICE,
            api_key=config.CHATTERBOX_API_KEY,
            sample_rate=config.CHATTERBOX_SAMPLE_RATE,
            timeout=config.TTS_TIMEOUT_SECONDS,
        )
        elevenlabs = None
        if config.ELEVENLABS_API_KEY and config.ELEVENLABS_VOICE_ID:
            elevenlabs = ElevenLabsTTS(
                config.ELEVENLABS_API_KEY,
                config.ELEVENLABS_VOICE_ID,
                model_id=config.ELEVENLABS_MODEL_ID,
                base_url=config.ELEVENLABS_BASE_URL,
                timeout=config.TTS_TIMEOUT_SECONDS,
            )
        if config.TTS_PROVIDER == "elevenlabs":
            assert elevenlabs is not None
            tts = elevenlabs
        elif elevenlabs is not None:
            tts = FallbackTTS(chatterbox, elevenlabs)
        else:
            tts = chatterbox

        self.handler = ConversationHandler(
            stt=stt,
            tts=tts,
            deps=self.deps,
            openclaw_bridge=self.openclaw_bridge,
            vad=EnergyVAD(
                rms_threshold=config.VAD_RMS_THRESHOLD,
                activation_ms=config.VAD_ACTIVATION_MS,
                prefix_ms=config.VAD_PREFIX_MS,
                silence_ms=config.VAD_SILENCE_MS,
                min_speech_ms=config.VAD_MIN_SPEECH_MS,
                max_speech_seconds=config.VAD_MAX_SPEECH_SECONDS,
            ),
            attention_provider=self.camera_worker,
            require_hardware_speech=config.VAD_REQUIRE_HARDWARE_SPEECH,
            wake_phrase_enabled=config.ENABLE_WAKE_PHRASE,
            wake_phrase=config.WAKE_PHRASE,
        )

        self.attention_controller = None
        if config.ENABLE_ATTENTION_QUEUE:
            from reachy_mini_openclaw.attention import AttentionClient, AttentionController

            self.attention_controller = AttentionController(
                robot=self.robot,
                movement_manager=self.movement_manager,
                handler=self.handler,
                client=AttentionClient(
                    config.ATTENTION_API_URL,
                    config.OPENCLAW_TOKEN or "",
                ),
                poll_seconds=config.ATTENTION_POLL_SECONDS,
            )
        
        # State
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        
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
        failures = 0
        while not self._should_stop():
            try:
                audio_frame = self.robot.media.get_audio_sample()
                if audio_frame is not None:
                    await self.handler.receive((input_sr, audio_frame))
                failures = 0
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                if failures > 5:
                    raise RuntimeError("Reachy microphone loop failed repeatedly") from exc
                delay = min(0.25 * (2 ** (failures - 1)), 4.0)
                logger.warning("Reachy microphone unavailable; retrying in %.2fs", delay)
                await asyncio.sleep(delay)
            
    async def play_loop(self) -> None:
        """Play audio from handler through robot speakers."""
        from reachy_mini_openclaw.audio.playback import apply_output_gain

        output_sr = self.robot.media.get_output_audio_samplerate()
        logger.info("Playing at %d Hz with %.2fx software gain", output_sr, self.audio_output_gain)
        failures = 0
        while not self._should_stop():
            try:
                output = await self.handler.emit()
                if output is not None:
                    if isinstance(output, tuple):
                        input_sr, audio_data = output
                        # Feed the speech animator at playback time, not synthesis
                        # time, so movement stays synchronized with the speaker.
                        self.head_wobbler.feed(
                            base64.b64encode(audio_data.astype("int16").tobytes()).decode("ascii")
                        )

                        # Convert provider PCM16 to normalized float32.
                        audio_data = audio_data.flatten().astype("float32") / 32768.0

                        # Preserve synthesized level by default and permit a
                        # controlled boost without clipping.
                        audio_data = apply_output_gain(audio_data, self.audio_output_gain)

                        if input_sr != output_sr:
                            from scipy.signal import resample
                            num_samples = int(len(audio_data) * output_sr / input_sr)
                            audio_data = resample(audio_data, num_samples).astype("float32")

                        self.robot.media.push_audio_sample(audio_data)
                    # Otherwise it is a transcript event for a future UI consumer.
                failures = 0
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                if failures > 5:
                    raise RuntimeError("Reachy speaker loop failed repeatedly") from exc
                delay = min(0.25 * (2 ** (failures - 1)), 4.0)
                logger.warning("Reachy speaker unavailable; retrying in %.2fs", delay)
                await asyncio.sleep(delay)
            
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
            await asyncio.sleep(2)  # Keep native stop handling responsive.
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
        
        # Start audio
        logger.info("Starting audio...")
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        await asyncio.sleep(1)  # Let pipelines initialize without blocking stop handling.
        
        logger.info("Ready! Speak to me...")
        
        # Start the provider-based speech pipeline in the background.
        handler_task = asyncio.create_task(self.handler.start_up(), name="conversation-handler")
        
        # Start audio loops
        self._tasks = [
            handler_task,
            asyncio.create_task(self.record_loop(), name="record-loop"),
            asyncio.create_task(self.play_loop(), name="play-loop"),
        ]
        if self.attention_controller is not None:
            self._tasks.append(
                asyncio.create_task(self.attention_controller.run(), name="attention-loop")
            )
        
        stop_watcher = asyncio.create_task(self._watch_external_stop(), name="external-stop-watcher")
        self._tasks.append(stop_watcher)

        try:
            done, pending = await asyncio.wait(
                self._tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_watcher in done:
                logger.info("External stop requested")
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            # Do not hide an unexpected task failure from the Reachy app
            # manager; it must be able to record the failure and restart us.
            await asyncio.gather(*done)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")
            raise

    async def _watch_external_stop(self) -> None:
        """Bridge Reachy's threading stop event into the asyncio lifecycle."""
        if self._external_stop_event is None:
            await self._stop_event.wait()
            return
        # Poll instead of delegating Event.wait() to the default executor.
        # Cancelling asyncio.to_thread() cannot stop the underlying blocking
        # thread, which could survive app shutdown indefinitely.
        while not self._external_stop_event.is_set() and not self._stop_event.is_set():
            await asyncio.sleep(0.1)
        self._stop_event.set()

    async def stop(self) -> None:
        """Stop everything exactly once."""
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        logger.info("Stopping...")
        self._stop_event.set()
        
        # Cancel tasks
        if self.attention_controller is not None:
            await self.attention_controller.stop()
        current_task = asyncio.current_task()
        for task in self._tasks:
            if task is current_task:
                continue
            if not task.done():
                task.cancel()
                
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
                await self.openclaw_bridge.disconnect()
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


class ReachyMiniOpenclaw(ReachyMiniApp):
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
        
        gateway_url = os.getenv("OPENCLAW_GATEWAY_URL", "")
        
        app = None
        try:
            app = ClawBodyCore(
                gateway_url=gateway_url,
                robot=reachy_mini,
                external_stop_event=stop_event,
            )
            loop.run_until_complete(app.run())
        finally:
            if app is not None:
                loop.run_until_complete(app.stop())
            asyncio.set_event_loop(None)
            loop.close()


# Backwards-compatible import for older local launchers. The package entry
# point intentionally uses the official ReachyMiniOpenclaw name above.
ClawBodyApp = ReachyMiniOpenclaw


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.debug)
    
    # Set custom profile if specified
    if args.profile:
        from reachy_mini_openclaw.config import set_custom_profile
        set_custom_profile(args.profile)
    
    # Configure face tracking and local vision from args
    from reachy_mini_openclaw.config import (
        set_face_tracking_enabled, 
        set_local_vision_enabled,
    )
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
            robot_host=args.robot_host,
            robot_port=args.robot_port,
            robot_connection_mode=args.robot_connection_mode,
            enable_camera=not args.no_camera,
            enable_openclaw=not args.no_openclaw,
        )
    else:
        # Console mode
        app = ClawBodyCore(
            gateway_url=args.gateway_url,
            robot_name=args.robot_name,
            robot_host=args.robot_host,
            robot_port=args.robot_port,
            robot_connection_mode=args.robot_connection_mode,
            enable_camera=not args.no_camera,
            enable_openclaw=not args.no_openclaw,
        )
        
        async def run_and_stop() -> None:
            try:
                await app.run()
            finally:
                await app.stop()

        try:
            asyncio.run(run_and_stop())
        except KeyboardInterrupt:
            logger.info("Interrupted")


if __name__ == "__main__":
    # Reachy's wireless app manager launches the module from the
    # ``reachy_mini_apps`` entry point with ``python -m``.  That path must use
    # the SDK wrapper so the daemon supplies the already-connected robot and
    # owns the media session.  The ``clawbodylocal`` console script still calls
    # ``main()`` directly for standalone development.
    ReachyMiniOpenclaw().wrapped_run()
