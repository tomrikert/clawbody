"""Local voice handler for ClawBody.

This module replaces OpenAI Realtime with a local pipeline:
microphone audio -> local VAD -> faster-whisper STT -> Ollama chat -> local TTS.

It intentionally preserves the same fastrtc AsyncStreamHandler contract as the
OpenAI handler, so the robot audio and movement loops do not need to know which
provider is active.
"""

import asyncio
import base64
import json
import logging
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample

from reachy_mini_openclaw.config import config
from reachy_mini_openclaw.openai_realtime import FALLBACK_IDENTITY, ROBOT_BODY_INSTRUCTIONS
from reachy_mini_openclaw.tools.core_tools import ToolDependencies, dispatch_tool_call, get_tool_specs

logger = logging.getLogger(__name__)

LOCAL_SAMPLE_RATE: Final[Literal[24000]] = 24000
STT_SAMPLE_RATE: Final[Literal[16000]] = 16000


class LocalVoiceHandler(AsyncStreamHandler):
    """Local speech-to-speech handler backed by Ollama and local speech models."""

    def __init__(
        self,
        deps: ToolDependencies,
        openclaw_bridge: Any | None = None,
        gradio_mode: bool = False,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=LOCAL_SAMPLE_RATE,
            input_sample_rate=LOCAL_SAMPLE_RATE,
        )

        self.deps = deps
        self.openclaw_bridge = openclaw_bridge
        self.gradio_mode = gradio_mode
        self.output_queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        self._ready = asyncio.Event()
        self._shutdown_requested = False
        self._generation = 0
        self._processing_task: asyncio.Task | None = None

        self._pre_speech: deque[NDArray[np.float32]] = deque(maxlen=20)
        self._speech_chunks: list[NDArray[np.float32]] = []
        self._speech_started_at: float | None = None
        self._last_voice_at: float | None = None
        self._in_speech = False

        self._system_instructions: str | None = None
        self._history: list[dict[str, Any]] = []
        self._stt_model: Any = None
        self._tts_pipeline: Any = None

    def copy(self) -> "LocalVoiceHandler":
        """Create a copy of the handler for fastrtc."""
        return LocalVoiceHandler(self.deps, self.openclaw_bridge, self.gradio_mode)

    async def start_up(self) -> None:
        """Load local models and keep the handler alive."""
        self._system_instructions = await self._build_system_instructions()
        await asyncio.to_thread(self._initialize_stt)
        await asyncio.to_thread(self._initialize_tts)
        self._ready.set()

        logger.info(
            "Local voice ready: llm=%s stt=%s tts=%s",
            config.LOCAL_LLM_MODEL,
            config.LOCAL_STT_MODEL,
            config.LOCAL_TTS_ENGINE,
        )

        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

    async def _build_system_instructions(self) -> str:
        """Build system instructions from OpenClaw context when available."""
        agent_context = None
        if self.openclaw_bridge and self.openclaw_bridge.is_connected:
            logger.info("Fetching agent context from OpenClaw...")
            agent_context = await self.openclaw_bridge.get_agent_context()

        identity = agent_context or FALLBACK_IDENTITY
        return f"""{identity}

{ROBOT_BODY_INSTRUCTIONS}

You are running fully locally for voice I/O. Keep answers short because they will be spoken aloud.
Use tools naturally when they help, then continue the spoken answer."""

    def _initialize_stt(self) -> None:
        """Load faster-whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "Local STT requires faster-whisper. Install with: pip install faster-whisper"
            ) from e

        logger.info(
            "Loading local STT model %s on %s (%s)",
            config.LOCAL_STT_MODEL,
            config.LOCAL_STT_DEVICE,
            config.LOCAL_STT_COMPUTE_TYPE,
        )
        self._stt_model = WhisperModel(
            config.LOCAL_STT_MODEL,
            device=config.LOCAL_STT_DEVICE,
            compute_type=config.LOCAL_STT_COMPUTE_TYPE,
            download_root=config.HF_HOME,
        )

    def _initialize_tts(self) -> None:
        """Load local TTS."""
        if config.LOCAL_TTS_ENGINE != "kokoro":
            raise ValueError("Only LOCAL_TTS_ENGINE=kokoro is currently implemented")

        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise ImportError(
                "Local TTS requires kokoro. Install with: pip install kokoro soundfile"
            ) from e

        logger.info("Loading Kokoro TTS voice=%s lang=%s", config.LOCAL_TTS_VOICE, config.LOCAL_TTS_LANG)
        self._tts_pipeline = KPipeline(lang_code=config.LOCAL_TTS_LANG)

    async def receive(self, frame: tuple[int, NDArray]) -> None:
        """Receive microphone audio and run a simple local VAD."""
        if self._shutdown_requested:
            return

        input_sr, audio = frame
        audio_float = self._normalize_audio(audio)
        if input_sr != LOCAL_SAMPLE_RATE:
            audio_float = self._resample_float(audio_float, input_sr, LOCAL_SAMPLE_RATE)

        rms = float(np.sqrt(np.mean(audio_float ** 2))) if audio_float.size else 0.0
        now = time.monotonic()
        has_voice = rms >= config.LOCAL_VAD_THRESHOLD

        if not self._in_speech:
            self._pre_speech.append(audio_float)
            if has_voice:
                self._begin_speech(now)
                self._speech_chunks.extend(self._pre_speech)
                self._pre_speech.clear()
                self._speech_chunks.append(audio_float)
            return

        self._speech_chunks.append(audio_float)
        if has_voice:
            self._last_voice_at = now
            return

        silence_ms = ((now - (self._last_voice_at or now)) * 1000.0)
        speech_ms = ((now - (self._speech_started_at or now)) * 1000.0)
        if silence_ms >= config.LOCAL_VAD_SILENCE_MS:
            chunks = self._speech_chunks
            self._end_speech()
            if speech_ms >= config.LOCAL_MIN_SPEECH_MS and chunks:
                utterance = np.concatenate(chunks).astype(np.float32)
                gen = self._generation
                self._processing_task = asyncio.create_task(
                    self._process_utterance(utterance, LOCAL_SAMPLE_RATE, gen),
                    name="local-voice-utterance",
                )

    def _begin_speech(self, now: float) -> None:
        """Mark speech start and interrupt current output."""
        self._generation += 1
        self._in_speech = True
        self._speech_started_at = now
        self._last_voice_at = now
        self._speech_chunks = []
        self.deps.movement_manager.set_processing(False)
        self.deps.movement_manager.set_listening(True)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
        self._clear_output_queue()
        logger.info("User started speaking")

    def _end_speech(self) -> None:
        """Mark speech end."""
        self._in_speech = False
        self._speech_started_at = None
        self._last_voice_at = None
        self._speech_chunks = []
        self.deps.movement_manager.set_listening(False)
        logger.info("User stopped speaking")

    async def _process_utterance(self, audio: NDArray[np.float32], sample_rate: int, generation: int) -> None:
        """Transcribe, query the local LLM, synthesize, and enqueue audio."""
        await self._ready.wait()
        if generation != self._generation:
            return

        transcript = await asyncio.to_thread(self._transcribe, audio, sample_rate)
        if not transcript:
            return

        logger.info("User: %s", transcript)
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        self.deps.movement_manager.set_processing(True)
        try:
            answer = await self._chat_with_tools(transcript, generation)
        finally:
            self.deps.movement_manager.set_processing(False)

        if not answer or generation != self._generation:
            return

        logger.info("Assistant: %s", answer[:100] if len(answer) > 100 else answer)
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": answer}))
        await self._sync_to_openclaw(transcript, answer)

        audio_out = await asyncio.to_thread(self._synthesize, answer)
        if audio_out.size == 0 or generation != self._generation:
            return

        await self._queue_audio(audio_out, generation)

    def _transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> str:
        """Transcribe one utterance with faster-whisper."""
        if self._stt_model is None:
            return ""

        audio_16k = self._resample_float(audio, sample_rate, STT_SAMPLE_RATE)
        audio_int16 = np.clip(audio_16k * 32767.0, -32768, 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        try:
            wavfile.write(wav_path, STT_SAMPLE_RATE, audio_int16)
            segments, _info = self._stt_model.transcribe(
                str(wav_path),
                beam_size=1,
                vad_filter=True,
                without_timestamps=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def _chat_with_tools(self, user_text: str, generation: int) -> str:
        """Send a message to Ollama and execute requested tools."""
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        for _ in range(4):
            if generation != self._generation:
                return ""

            response = await asyncio.to_thread(self._ollama_chat, self._build_messages(), self._ollama_tools())
            message = response.get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                content = (message.get("content") or "").strip()
                if content:
                    self._history.append({"role": "assistant", "content": content})
                    self._trim_history()
                return content

            self._history.append({"role": "assistant", "content": message.get("content") or ""})
            for call in tool_calls:
                result_text = await self._execute_tool_call(call)
                self._history.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool result. Use this result to continue the spoken answer.\n"
                            f"{result_text}"
                        ),
                    }
                )

        return "I tried to use a tool, but I could not finish that request cleanly."

    def _ollama_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Call Ollama's local chat API."""
        payload = {
            "model": config.LOCAL_LLM_MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": config.LOCAL_LLM_THINK,
            "options": {
                "num_predict": 220,
                "temperature": 0.7,
            },
        }
        url = config.LOCAL_LLM_BASE_URL.rstrip("/") + "/api/chat"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(request, timeout=config.LOCAL_LLM_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error("Ollama chat failed: %s", e)
            return {"message": {"content": "My local language model is not reachable right now."}}

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build Ollama chat messages with system prompt and bounded history."""
        return [{"role": "system", "content": self._system_instructions or ""}] + self._history

    def _trim_history(self) -> None:
        """Keep local history bounded for latency."""
        max_messages = max(2, config.LOCAL_MAX_HISTORY_MESSAGES)
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _ollama_tools(self) -> list[dict[str, Any]]:
        """Convert local tool specs to Ollama/OpenAI-compatible tool objects."""
        tools = []
        specs = list(get_tool_specs())
        if self.openclaw_bridge is not None:
            specs.append(
                {
                    "type": "function",
                    "name": "ask_openclaw",
                    "description": "Query OpenClaw for external tools, memory, web, calendar, or smart-home requests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "include_image": {"type": "boolean", "default": False},
                        },
                        "required": ["query"],
                    },
                }
            )

        for spec in specs:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return tools

    async def _execute_tool_call(self, call: dict[str, Any]) -> str:
        """Execute a tool call returned by Ollama."""
        function = call.get("function") or call
        name = function.get("name")
        args = function.get("arguments") or {}
        if not isinstance(name, str):
            return json.dumps({"error": "Tool call did not include a function name"})
        if isinstance(args, str):
            args_json = args
        else:
            args_json = json.dumps(args)

        logger.info("Tool call: %s(%s)", name, args_json[:80])
        if name == "ask_openclaw":
            result = await self._handle_openclaw_query(args_json)
        else:
            result = await dispatch_tool_call(name, args_json, self.deps)
        return json.dumps({"tool": name, "result": result})

    async def _handle_openclaw_query(self, args_json: str) -> dict[str, Any]:
        """Handle local LLM requests to OpenClaw."""
        if self.openclaw_bridge is None:
            return {"error": "OpenClaw bridge is not initialized."}
        if not self.openclaw_bridge.is_connected:
            connected = await self.openclaw_bridge.connect()
            if not connected:
                return {"error": "OpenClaw gateway is unreachable."}

        try:
            args = json.loads(args_json)
            query = args.get("query", "")
            response = await self.openclaw_bridge.chat(
                query,
                system_context="User is asking through their Reachy Mini robot. Keep response concise for voice.",
            )
            if response.error:
                return {"error": response.error}
            return {"response": response.content}
        except Exception as e:
            logger.error("OpenClaw query failed: %s", e)
            return {"error": str(e)}

    def _synthesize(self, text: str) -> NDArray[np.float32]:
        """Synthesize speech with Kokoro."""
        if self._tts_pipeline is None:
            return np.array([], dtype=np.float32)

        pieces: list[NDArray[np.float32]] = []
        generator = self._tts_pipeline(
            text,
            voice=config.LOCAL_TTS_VOICE,
            speed=config.LOCAL_TTS_SPEED,
        )
        for _gs, _ps, audio in generator:
            pieces.append(np.asarray(audio, dtype=np.float32))

        if not pieces:
            return np.array([], dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)

    async def _queue_audio(self, audio: NDArray[np.float32], generation: int) -> None:
        """Queue synthesized audio in small chunks."""
        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        chunk_size = LOCAL_SAMPLE_RATE // 10
        for start in range(0, len(audio_int16), chunk_size):
            if generation != self._generation:
                return
            chunk = audio_int16[start:start + chunk_size].reshape(1, -1)
            await self.output_queue.put((LOCAL_SAMPLE_RATE, chunk))

    async def _sync_to_openclaw(self, user_text: str, assistant_text: str) -> None:
        """Sync a completed turn to OpenClaw for memory continuity."""
        if not self.openclaw_bridge or not self.openclaw_bridge.is_connected:
            return
        try:
            await self.openclaw_bridge.sync_conversation(user_text, assistant_text)
        except Exception as e:
            logger.debug("Failed to sync conversation: %s", e)

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Get the next output item."""
        item = await wait_for_item(self.output_queue)
        if isinstance(item, tuple) and self.deps.head_wobbler is not None:
            _sample_rate, audio_data = item
            audio_b64 = base64.b64encode(audio_data.astype(np.int16).tobytes()).decode("utf-8")
            self.deps.head_wobbler.feed(audio_b64)
        return item

    async def shutdown(self) -> None:
        """Shutdown the local handler."""
        self._shutdown_requested = True
        self._generation += 1
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        self._clear_output_queue()

    def _clear_output_queue(self) -> None:
        """Drop queued audio/transcript outputs."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @staticmethod
    def _normalize_audio(audio: NDArray) -> NDArray[np.float32]:
        """Convert arbitrary mono/stereo audio to mono float32 [-1, 1]."""
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            audio = audio[:, 0] if audio.shape[1] > 1 else audio[:, 0]
        audio = audio.flatten()
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    @staticmethod
    def _resample_float(audio: NDArray[np.float32], input_sr: int, output_sr: int) -> NDArray[np.float32]:
        """Resample float audio."""
        if input_sr == output_sr or audio.size == 0:
            return audio.astype(np.float32)
        num_samples = max(1, int(len(audio) * output_sr / input_sr))
        return resample(audio, num_samples).astype(np.float32)
