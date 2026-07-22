"""Provider-based voice pipeline with OpenClaw as the only conversational brain."""

from __future__ import annotations

import asyncio
import base64
from difflib import SequenceMatcher
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini_openclaw.audio.providers import SpeechToText, TextToSpeech

logger = logging.getLogger(__name__)

_VISION_REQUESTS = (
    re.compile(r"\b(?:can|do) you see\b", re.I),
    re.compile(r"\bwhat (?:do|can) you see\b", re.I),
    re.compile(r"\blook at\b", re.I),
    re.compile(r"\bwhat(?:'s| is) (?:this|that|in front of you|on my|on the)\b", re.I),
    re.compile(r"\bwho (?:is|are) (?:here|there|in front of you)\b", re.I),
)


def needs_vision(text: str) -> bool:
    return any(pattern.search(text) for pattern in _VISION_REQUESTS)


def command_after_wake_phrase(text: str, wake_phrase: str) -> str | None:
    """Return the command after a spoken wake phrase, or None if absent.

    Matching is case-insensitive, restricted to the start of the transcript,
    and tolerant of punctuation inserted by STT (for example, "Hey, Asmo").
    """
    words = re.findall(r"[^\W_]+", wake_phrase, flags=re.UNICODE)
    if not words:
        return None
    spoken = list(re.finditer(r"[^\W_]+", text, flags=re.UNICODE))
    if len(spoken) < len(words):
        return None

    for expected, observed_match in zip(words, spoken[: len(words)], strict=True):
        observed = observed_match.group(0)
        expected_folded = expected.casefold()
        observed_folded = observed.casefold()
        if observed_folded == expected_folded:
            continue
        # Short wake words are occasionally normalized by Whisper ("hey" ->
        # "hi"). Names are even more vulnerable ("Asmo" -> "asthma").
        # Keep the fuzzy allowance narrow: it only applies to the leading wake
        # phrase tokens and still requires the entire configured phrase.
        if expected_folded == "hey" and observed_folded == "hi":
            continue
        threshold = 0.60 if len(expected_folded) >= 4 else 0.75
        if (
            abs(len(expected_folded) - len(observed_folded)) > 2
            or SequenceMatcher(None, expected_folded, observed_folded).ratio() < threshold
        ):
            return None

    return re.sub(
        r"^[\s,;:!?.\-–—]+", "", text[spoken[len(words) - 1].end() :]
    ).strip()


@dataclass(frozen=True)
class Utterance:
    sample_rate: int
    samples: NDArray[np.float32]


class EnergyVAD:
    """Small deterministic utterance segmenter for the always-hot local STT path."""

    def __init__(
        self,
        *,
        rms_threshold: float = 0.022,
        activation_ms: int = 160,
        prefix_ms: int = 250,
        silence_ms: int = 650,
        min_speech_ms: int = 300,
        max_speech_seconds: float = 20.0,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.activation_ms = activation_ms
        self.prefix_ms = prefix_ms
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_seconds = max_speech_seconds
        self.speaking = False
        self._sample_rate = 0
        self._prefix: deque[NDArray[np.float32]] = deque()
        self._prefix_samples = 0
        self._frames: list[NDArray[np.float32]] = []
        self._activation_samples = 0
        self._voiced_samples = 0
        self._silence_samples = 0

    @staticmethod
    def normalize(audio: NDArray[Any]) -> NDArray[np.float32]:
        values = np.asarray(audio)
        if values.ndim == 2:
            channel_axis = 0 if values.shape[0] <= values.shape[1] else 1
            values = values.mean(axis=channel_axis)
        values = values.reshape(-1)
        if values.dtype == np.int16:
            return values.astype(np.float32) / 32768.0
        return values.astype(np.float32)

    def feed(
        self,
        sample_rate: int,
        audio: NDArray[Any],
        *,
        allow_start: bool = True,
    ) -> Utterance | None:
        frame = self.normalize(audio)
        if frame.size == 0:
            return None
        if self._sample_rate and sample_rate != self._sample_rate:
            self.reset()
        self._sample_rate = sample_rate
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
        voiced = rms >= self.rms_threshold and (self.speaking or allow_start)

        if not self.speaking:
            self._prefix.append(frame.copy())
            self._prefix_samples += frame.size
            prefix_limit = max(1, int(sample_rate * self.prefix_ms / 1000))
            while self._prefix and self._prefix_samples - self._prefix[0].size >= prefix_limit:
                self._prefix_samples -= self._prefix.popleft().size
            if not voiced:
                self._activation_samples = 0
                return None
            self._activation_samples += frame.size
            activation_limit = max(1, int(sample_rate * self.activation_ms / 1000))
            if self._activation_samples < activation_limit:
                return None
            self.speaking = True
            self._frames = list(self._prefix)
            self._voiced_samples = self._activation_samples
            self._silence_samples = 0
            self._prefix.clear()
            self._prefix_samples = 0
            return None

        self._frames.append(frame.copy())
        if voiced:
            self._voiced_samples += frame.size
            self._silence_samples = 0
        else:
            self._silence_samples += frame.size

        silence_limit = int(sample_rate * self.silence_ms / 1000)
        max_limit = int(sample_rate * self.max_speech_seconds)
        if self._silence_samples < silence_limit and sum(chunk.size for chunk in self._frames) < max_limit:
            return None

        min_voiced = int(sample_rate * self.min_speech_ms / 1000)
        samples = np.concatenate(self._frames) if self._voiced_samples >= min_voiced else None
        self.reset()
        return Utterance(sample_rate, samples) if samples is not None else None

    def reset(self) -> None:
        self.speaking = False
        self._sample_rate = 0
        self._prefix.clear()
        self._prefix_samples = 0
        self._frames = []
        self._activation_samples = 0
        self._voiced_samples = 0
        self._silence_samples = 0


class ConversationHandler:
    """Turn Reachy microphone frames into spoken responses from OpenClaw."""

    def __init__(
        self,
        *,
        stt: SpeechToText,
        tts: TextToSpeech,
        openclaw_bridge: Any,
        deps: Any,
        vad: EnergyVAD,
        output_chunk_samples: int = 2048,
        attention_provider: Any = None,
        require_hardware_speech: bool = True,
        wake_phrase_enabled: bool = False,
        wake_phrase: str = "Hey Asmo",
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.openclaw_bridge = openclaw_bridge
        self.deps = deps
        self.vad = vad
        self.output_chunk_samples = output_chunk_samples
        self.attention_provider = attention_provider
        self.require_hardware_speech = require_hardware_speech
        self.wake_phrase_enabled = wake_phrase_enabled
        self.wake_phrase = wake_phrase
        self.output_queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | dict[str, str]] = asyncio.Queue()
        self._utterances: asyncio.Queue[Utterance] = asyncio.Queue(maxsize=2)
        self._shutdown_requested = False
        self._ignore_input_until = 0.0
        self._turn_lock = asyncio.Lock()

    async def _retry_async(self, operation: Any, label: str, attempts: int = 3) -> Any:
        """Retry a transient network/provider operation with bounded backoff."""
        delay = 0.25
        for attempt in range(1, attempts + 1):
            try:
                return await operation()
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt == attempts:
                    raise
                logger.warning(
                    "%s unavailable; retrying in %.2fs (%d/%d)",
                    label,
                    delay,
                    attempt,
                    attempts - 1,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
        raise AssertionError("retry loop exhausted")

    async def _ensure_gateway(self) -> None:
        """Reconnect the gateway without making a failed turn fatal immediately."""
        if self.openclaw_bridge.is_connected:
            return

        async def connect() -> bool:
            if await self.openclaw_bridge.connect():
                return True
            raise RuntimeError("OpenClaw gateway is unavailable")

        await self._retry_async(connect, "OpenClaw gateway")

    async def start_up(self) -> None:
        while not self._shutdown_requested:
            utterance = await self._utterances.get()
            try:
                await self._process_utterance(utterance)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Voice turn failed: %s", exc, exc_info=True)
                self.deps.movement_manager.set_processing(False)

    async def receive(self, frame: tuple[int, NDArray[Any]]) -> None:
        sample_rate, audio = frame
        if time.monotonic() < self._ignore_input_until:
            if self.vad.speaking:
                self.vad.reset()
                self.deps.movement_manager.set_listening(False)
            return
        was_speaking = self.vad.speaking
        allow_start = True
        attention_state = None
        if not was_speaking and self.require_hardware_speech and self.attention_provider is not None:
            attention_state = self.attention_provider.get_attention_state()
            # Fail open if DoA is unavailable. When it is available, require
            # the microphone array's own speech classifier as a second vote.
            if attention_state.get("doa_angle") is not None:
                allow_start = bool(attention_state.get("speech_detected"))
        utterance = self.vad.feed(sample_rate, audio, allow_start=allow_start)
        if self.vad.speaking and not was_speaking:
            logger.info("Listening started (attention=%s)", attention_state)
            self.deps.movement_manager.set_listening(True)
            self.deps.movement_manager.set_processing(False)
            self._clear_audio_output()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
        if was_speaking and not self.vad.speaking:
            self.deps.movement_manager.set_listening(False)
        if utterance is not None:
            if self._utterances.full():
                logger.warning("Dropping utterance because the conversational pipeline is busy")
                return
            await self._utterances.put(utterance)

    async def _process_utterance(self, utterance: Utterance) -> None:
        async with self._turn_lock:
            await self._process_utterance_locked(utterance)

    async def _process_utterance_locked(self, utterance: Utterance) -> None:
        turn_started = time.monotonic()
        self.deps.movement_manager.set_processing(True)
        transcript = await self._retry_async(
            lambda: self.stt.transcribe(utterance.samples, utterance.sample_rate),
            "STT service",
        )
        stt_done = time.monotonic()
        if not transcript:
            self.deps.movement_manager.set_processing(False)
            return
        transcript = transcript.strip()
        if self.wake_phrase_enabled:
            command = command_after_wake_phrase(transcript, self.wake_phrase)
            if command is None:
                logger.info("Ignored utterance without configured wake phrase: %r", transcript)
                self.deps.movement_manager.set_processing(False)
                return
            # A wake phrase by itself remains a valid greeting. When a command
            # follows it, keep the conversational session free of boilerplate.
            transcript = command or transcript
        logger.info("User: %s", transcript)
        await self.output_queue.put({"role": "user", "content": transcript})

        await self._ensure_gateway()
        motion_catalog = getattr(self.deps, "motion_catalog", None)
        motion_hint = f" {motion_catalog.response_hint()}" if motion_catalog is not None else ""
        image_b64 = None
        if needs_vision(transcript) and self.attention_provider is not None:
            jpeg = self.attention_provider.get_latest_jpeg()
            if jpeg:
                image_b64 = base64.b64encode(jpeg).decode("ascii")
                logger.info("Attached current Reachy camera frame to visual question")
        response = await self.openclaw_bridge.chat(
            transcript,
            image_b64=image_b64,
            system_context=(
                "The user is speaking to you through your Reachy Mini body. "
                "Answer naturally and concisely for spoken delivery. "
                "When a camera image is attached, it is your current first-person view."
                + motion_hint
            ),
        )
        openclaw_done = time.monotonic()
        if response.error:
            raise RuntimeError(f"OpenClaw response failed: {response.error}")
        if motion_catalog is not None:
            text, directive = motion_catalog.parse_response(response.content)
            directive = (
                directive
                or motion_catalog.infer_request(transcript)
                or motion_catalog.infer(text)
            )
        else:
            text, directive = response.content.strip(), None
        if not text:
            raise RuntimeError("OpenClaw returned an empty response")

        logger.info("Assistant: %s", text)
        audio = await self._retry_async(lambda: self.tts.synthesize(text), "TTS service")
        tts_done = time.monotonic()
        # Reachy's microphone hears its own speaker over the remote WebRTC
        # stream. Suppress VAD for the synthesized duration plus a short room-
        # echo tail so the robot does not recursively answer itself.
        self._ignore_input_until = max(
            self._ignore_input_until,
            time.monotonic() + (audio.samples.size / audio.sample_rate) + 0.8,
        )
        self.vad.reset()
        self.deps.movement_manager.set_listening(False)
        self.deps.movement_manager.set_processing(False)
        # Start primary motion only after synthesis is ready. This synchronizes
        # the gesture with playback and prevents thinking offsets from being
        # blended into the opening of recorded choreography.
        if directive is not None:
            logger.info("Motion: %s:%s", directive.kind, directive.name)
            self.deps.movement_manager.queue_move(
                motion_catalog.get(directive.kind, directive.name)
            )
        await self.output_queue.put({"role": "assistant", "content": text})
        for offset in range(0, audio.samples.size, self.output_chunk_samples):
            chunk = audio.samples[offset : offset + self.output_chunk_samples]
            await self.output_queue.put((audio.sample_rate, chunk.reshape(1, -1)))
        logger.info(
            "Turn latency: stt=%.2fs openclaw=%.2fs tts=%.2fs ready=%.2fs",
            stt_done - turn_started,
            openclaw_done - stt_done,
            tts_done - openclaw_done,
            tts_done - turn_started,
        )

    async def speak_text(self, text: str) -> float:
        """Speak trusted local notification text without creating an agent turn."""
        async with self._turn_lock:
            self.deps.movement_manager.set_processing(False)
            audio = await self._retry_async(lambda: self.tts.synthesize(text), "TTS service")
            duration = audio.samples.size / audio.sample_rate
            self._ignore_input_until = max(
                self._ignore_input_until,
                time.monotonic() + duration + 0.8,
            )
            self.vad.reset()
            self.deps.movement_manager.set_listening(False)
            await self.output_queue.put({"role": "assistant", "content": text})
            for offset in range(0, audio.samples.size, self.output_chunk_samples):
                chunk = audio.samples[offset : offset + self.output_chunk_samples]
                await self.output_queue.put((audio.sample_rate, chunk.reshape(1, -1)))
            return duration

    def _clear_audio_output(self) -> None:
        retained: list[dict[str, str]] = []
        while not self.output_queue.empty():
            try:
                item = self.output_queue.get_nowait()
                if isinstance(item, dict):
                    retained.append(item)
            except asyncio.QueueEmpty:
                break
        for item in retained:
            self.output_queue.put_nowait(item)

    async def emit(self) -> tuple[int, NDArray[np.int16]] | dict[str, str] | None:
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except TimeoutError:
            return None

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        self.vad.reset()
        self._clear_audio_output()
        self.deps.movement_manager.set_listening(False)
        self.deps.movement_manager.set_processing(False)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
