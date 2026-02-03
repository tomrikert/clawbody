"""ClawBody - OpenAI Realtime API handler for voice I/O with OpenClaw intelligence.

This module implements ClawBody's hybrid voice conversation system:
- OpenAI Realtime API handles speech recognition and text-to-speech
- OpenClaw (Clawson) provides the AI intelligence and responses

Architecture:
    User speaks -> OpenAI Realtime (transcription) -> OpenClaw (AI response)
                                                   -> OpenAI Realtime (TTS) -> Robot speaks

This gives ClawBody the best of both worlds:
- Low-latency voice activity detection and speech recognition from OpenAI
- Full OpenClaw/Clawson capabilities (tools, memory, personality) for responses
"""

import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Literal, Optional, Tuple
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_openclaw.config import config
from reachy_mini_openclaw.prompts import get_session_voice
from reachy_mini_openclaw.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call
from reachy_mini_openclaw.openclaw_bridge import OpenClawBridge

logger = logging.getLogger(__name__)

# OpenAI Realtime API audio format
OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000

# System context for OpenClaw - tells it about its robot body
ROBOT_SYSTEM_CONTEXT = """You are Clawson, the OpenClaw AI assistant, speaking through a Reachy Mini robot body.

Your robot capabilities:
- You can see through a camera (when the user asks you to look at something)
- You can move your head expressively (look left/right/up/down, show emotions)
- You can dance to express joy
- You speak through a speaker

Guidelines:
- Keep responses concise and conversational (you're speaking, not typing)
- Be warm, helpful, and occasionally witty - you're a friendly space lobster 🦞
- Reference your robot body naturally ("let me look", "I can see...")
- When you want to do something physical, just describe it naturally and I'll make it happen

You ARE Clawson - not "an assistant" - speak as yourself."""


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """Handler for OpenAI Realtime API voice I/O with OpenClaw backend.
    
    This handler:
    - Receives audio from robot microphone
    - Sends to OpenAI for speech recognition
    - Routes transcripts to OpenClaw for AI response
    - Uses OpenAI TTS to speak OpenClaw's response
    - Handles robot movement tool calls locally
    """
    
    def __init__(
        self,
        deps: ToolDependencies,
        openclaw_bridge: Optional[OpenClawBridge] = None,
        gradio_mode: bool = False,
    ):
        """Initialize the handler.
        
        Args:
            deps: Tool dependencies for robot control
            openclaw_bridge: Bridge to OpenClaw gateway
            gradio_mode: Whether running with Gradio UI
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )
        
        self.deps = deps
        self.openclaw_bridge = openclaw_bridge
        self.gradio_mode = gradio_mode
        
        # OpenAI connection
        self.client: Optional[AsyncOpenAI] = None
        self.connection: Any = None
        
        # Output queue
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        
        # State tracking
        self.last_activity_time = 0.0
        self.start_time = 0.0
        self._processing_response = False
        
        # Pending transcript for OpenClaw
        self._pending_transcript: Optional[str] = None
        self._pending_image: Optional[str] = None
        
        # Lifecycle flags
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        
    def copy(self) -> "OpenAIRealtimeHandler":
        """Create a copy of the handler (required by fastrtc)."""
        return OpenAIRealtimeHandler(self.deps, self.openclaw_bridge, self.gradio_mode)
        
    async def start_up(self) -> None:
        """Start the handler and connect to OpenAI."""
        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.error("OPENAI_API_KEY not configured")
            raise ValueError("OPENAI_API_KEY required")
            
        self.client = AsyncOpenAI(api_key=api_key)
        self.start_time = asyncio.get_event_loop().time()
        self.last_activity_time = self.start_time
        
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_session()
                return
            except ConnectionClosedError as e:
                logger.warning("WebSocket closed unexpectedly (attempt %d/%d): %s", 
                             attempt, max_attempts, e)
                if attempt < max_attempts:
                    delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass
                    
    async def _run_session(self) -> None:
        """Run a single OpenAI Realtime session."""
        model = config.OPENAI_MODEL
        logger.info("Connecting to OpenAI Realtime API with model: %s", model)
        
        async with self.client.beta.realtime.connect(model=model) as conn:
            # Configure session for hybrid mode:
            # - OpenAI handles STT (transcription) and TTS (speaking)
            # - OpenClaw provides the AI intelligence
            # - We cancel OpenAI's auto-response and substitute OpenClaw's response
            await conn.session.update(
                session={
                    "modalities": ["text", "audio"],
                    "instructions": "Wait for instructions. Do not speak unless given specific text to read.",
                    "voice": get_session_voice(),
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        # Allow auto-response to keep connection healthy
                        # We'll cancel it and substitute OpenClaw's response
                    },
                    "tools": [],
                    "tool_choice": "none",
                },
            )
            logger.info("OpenAI Realtime session configured (hybrid mode)")
            
            self.connection = conn
            self._connected_event.set()
            
            # Process events
            async for event in conn:
                await self._handle_event(event)
                
    async def _handle_event(self, event: Any) -> None:
        """Handle an event from the OpenAI Realtime API."""
        event_type = event.type
        logger.debug("Event: %s", event_type)
        
        # Speech detection
        if event_type == "input_audio_buffer.speech_started":
            # User started speaking - clear any pending output
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            logger.info("User speech started")
            
        if event_type == "input_audio_buffer.speech_stopped":
            self.deps.movement_manager.set_listening(False)
            logger.info("User speech stopped")
            
        # Transcription completed - this is when we send to OpenClaw
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            if transcript and transcript.strip():
                logger.info("User said: %s", transcript)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": transcript})
                )
                # Cancel any OpenAI auto-response - we'll use OpenClaw instead
                try:
                    await self.connection.response.cancel()
                except Exception:
                    pass  # May fail if no response in progress
                # Process through OpenClaw
                await self._process_with_openclaw(transcript)
            
        # Audio output from TTS
        if event_type == "response.audio.delta":
            # Feed to head wobbler for expressive movement
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(event.delta)
            
            self.last_activity_time = asyncio.get_event_loop().time()
            
            # Queue audio for playback
            audio_data = np.frombuffer(
                base64.b64decode(event.delta), 
                dtype=np.int16
            ).reshape(1, -1)
            await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_data))
            
        # Response audio transcript (what was spoken)
        if event_type == "response.audio_transcript.done":
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": event.transcript})
            )
            
        # Response completed
        if event_type == "response.done":
            self._processing_response = False
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            
        # Tool calls (for robot movement)
        if event_type == "response.function_call_arguments.done":
            await self._handle_tool_call(event)
            
        # Errors
        if event_type == "error":
            err = getattr(event, "error", None)
            msg = getattr(err, "message", str(err))
            code = getattr(err, "code", "")
            logger.error("OpenAI error [%s]: %s", code, msg)
            
    async def _process_with_openclaw(self, transcript: str) -> None:
        """Send transcript to OpenClaw and speak the response."""
        if not self.openclaw_bridge or not self.openclaw_bridge.is_connected:
            logger.warning("OpenClaw not connected, using fallback")
            await self._speak_text("I'm sorry, I'm having trouble connecting to my brain right now.")
            return
            
        self._processing_response = True
        
        try:
            # Check if user is asking to look at something
            look_keywords = ["look", "see", "what do you see", "show me", "camera", "looking at"]
            should_capture = any(kw in transcript.lower() for kw in look_keywords)
            
            image_b64 = None
            if should_capture and self.deps.camera_worker:
                # Capture image from robot camera
                frame = self.deps.camera_worker.get_latest_frame()
                if frame is not None:
                    import cv2
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image_b64 = base64.b64encode(buffer).decode('utf-8')
                    logger.info("Captured camera image for OpenClaw")
            
            # Check if this might be a long-running query (weather, search, etc.)
            # If so, say something first to keep the connection alive
            long_keywords = ["weather", "search", "find", "look up", "check", "what's the"]
            might_be_long = any(kw in transcript.lower() for kw in long_keywords)
            
            if might_be_long:
                await self._speak_text("Let me check on that...")
            
            # Send to OpenClaw
            logger.info("Sending to OpenClaw: %s", transcript[:50])
            response = await self.openclaw_bridge.chat(
                transcript, 
                image_b64=image_b64,
                system_context=ROBOT_SYSTEM_CONTEXT,
            )
            
            if response.error:
                logger.error("OpenClaw error: %s", response.error)
                await self._speak_text("I had trouble thinking about that. Could you try again?")
            elif response.content:
                logger.info("OpenClaw response: %s", response.content[:100])
                # Parse for any robot commands in the response
                await self._execute_robot_actions(response.content)
                # Speak the response
                await self._speak_text(response.content)
            else:
                await self._speak_text("Hmm, I'm not sure what to say about that.")
                
        except Exception as e:
            logger.error("Error processing with OpenClaw: %s", e)
            await self._speak_text("Sorry, I encountered an error. Let me try again.")
            
    async def _speak_text(self, text: str) -> None:
        """Have OpenAI TTS speak the given text."""
        if not self.connection:
            logger.warning("No connection, cannot speak")
            return
            
        try:
            # Create a response with explicit instructions to speak the text
            await self.connection.response.create(
                response={
                    "modalities": ["text", "audio"],
                    "instructions": f"Read this text aloud naturally: {text}",
                }
            )
            logger.debug("Requested TTS for: %s", text[:50])
        except Exception as e:
            logger.error("Failed to speak text: %s", e)
            
    async def _execute_robot_actions(self, response_text: str) -> None:
        """Parse OpenClaw response for robot actions and execute them."""
        response_lower = response_text.lower()
        
        # Simple keyword-based action detection
        # (In the future, OpenClaw could return structured actions)
        
        if any(word in response_lower for word in ["look left", "looking left", "turn left"]):
            await dispatch_tool_call("look", '{"direction": "left"}', self.deps)
        elif any(word in response_lower for word in ["look right", "looking right", "turn right"]):
            await dispatch_tool_call("look", '{"direction": "right"}', self.deps)
        elif any(word in response_lower for word in ["look up", "looking up"]):
            await dispatch_tool_call("look", '{"direction": "up"}', self.deps)
        elif any(word in response_lower for word in ["look down", "looking down"]):
            await dispatch_tool_call("look", '{"direction": "down"}', self.deps)
            
        if any(word in response_lower for word in ["dance", "dancing", "celebrate"]):
            await dispatch_tool_call("dance", '{"dance_name": "happy"}', self.deps)
        elif any(word in response_lower for word in ["excited", "exciting"]):
            await dispatch_tool_call("emotion", '{"emotion_name": "excited"}', self.deps)
        elif any(word in response_lower for word in ["thinking", "let me think", "hmm"]):
            await dispatch_tool_call("emotion", '{"emotion_name": "thinking"}', self.deps)
        elif any(word in response_lower for word in ["curious", "interesting"]):
            await dispatch_tool_call("emotion", '{"emotion_name": "curious"}', self.deps)
            
    async def _handle_tool_call(self, event: Any) -> None:
        """Handle a tool call (for robot movement)."""
        tool_name = getattr(event, "name", None)
        args_json = getattr(event, "arguments", None)
        call_id = getattr(event, "call_id", None)
        
        if not isinstance(tool_name, str) or not isinstance(args_json, str):
            return
            
        try:
            result = await dispatch_tool_call(tool_name, args_json, self.deps)
            logger.debug("Tool '%s' result: %s", tool_name, result)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            result = {"error": str(e)}
            
        # Send result back
        if isinstance(call_id, str) and self.connection:
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                }
            )
            
    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        """Receive audio from the robot microphone."""
        if not self.connection:
            return
            
        input_sr, audio = frame
        
        # Handle stereo
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]
        
        audio = audio.flatten()
        
        # Convert to float for resampling
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
                
        # Resample to OpenAI sample rate
        if input_sr != OPENAI_SAMPLE_RATE:
            num_samples = int(len(audio) * OPENAI_SAMPLE_RATE / input_sr)
            audio = resample(audio, num_samples).astype(np.float32)
            
        # Convert to int16 for OpenAI
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Send to OpenAI
        try:
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug("Failed to send audio: %s", e)
            
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Get the next output (audio or transcript)."""
        return await wait_for_item(self.output_queue)
        
    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
            
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("Connection close: %s", e)
            self.connection = None
            
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
