"""Provider adapters for ClawBody speech recognition and synthesis."""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesizedAudio:
    """Mono signed 16-bit PCM returned by a TTS provider."""

    sample_rate: int
    samples: NDArray[np.int16]


class SpeechToText(Protocol):
    async def transcribe(self, samples: NDArray[np.float32], sample_rate: int) -> str: ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str) -> SynthesizedAudio: ...


def encode_wav(samples: NDArray[np.float32], sample_rate: int) -> bytes:
    """Encode normalized mono float audio as a PCM16 WAV."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def decode_wav(payload: bytes) -> SynthesizedAudio:
    """Decode a mono/stereo PCM WAV response to mono int16 samples."""
    with wave.open(io.BytesIO(payload), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"TTS WAV must be PCM16, got {width * 8}-bit audio")
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
    return SynthesizedAudio(sample_rate=sample_rate, samples=samples.copy())


class OpenAICompatibleSTT:
    """Client for a local/OpenAI-compatible audio transcription endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        language: str = "en",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/audio/transcriptions") else f"{base}/audio/transcriptions"
        self.model = model
        self.api_key = api_key
        self.language = language
        self.timeout = timeout
        self._client = client

    async def transcribe(self, samples: NDArray[np.float32], sample_rate: int) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        files = {"file": ("utterance.wav", encode_wav(samples, sample_rate), "audio/wav")}
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(self.url, headers=headers, data=data, files=files)
            response.raise_for_status()
            body = response.json()
            text = body.get("text", "") if isinstance(body, dict) else ""
            return str(text).strip()
        finally:
            if self._client is None:
                await client.aclose()


class ChatterboxTTS:
    """Client for the local Chatterbox OpenAI-compatible speech endpoint."""

    def __init__(
        self,
        url: str,
        *,
        voice: str = "asmo",
        api_key: str | None = None,
        sample_rate: int = 24000,
        timeout: float = 45.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url
        self.voice = voice
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._client = client

    async def synthesize(self, text: str) -> SynthesizedAudio:
        headers = {"Accept": "audio/wav"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "input": text,
            "voice": self.voice,
            "model": "chatterbox-turbo",
            "response_format": "wav",
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "wav" in content_type or response.content[:4] == b"RIFF":
                return decode_wav(response.content)
            return SynthesizedAudio(
                sample_rate=self.sample_rate,
                samples=np.frombuffer(response.content, dtype="<i2").copy(),
            )
        finally:
            if self._client is None:
                await client.aclose()


class ElevenLabsTTS:
    """ElevenLabs PCM streaming endpoint used as a quality fallback."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model_id: str = "eleven_flash_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        sample_rate: int = 24000,
        timeout: float = 45.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/v1/text-to-speech/{voice_id}/stream"
        self.api_key = api_key
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._client = client

    async def synthesize(self, text: str) -> SynthesizedAudio:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(
                self.url,
                params={"output_format": f"pcm_{self.sample_rate}"},
                headers={"xi-api-key": self.api_key, "Accept": "application/octet-stream"},
                json={"text": text, "model_id": self.model_id},
            )
            response.raise_for_status()
            return SynthesizedAudio(
                sample_rate=self.sample_rate,
                samples=np.frombuffer(response.content, dtype="<i2").copy(),
            )
        finally:
            if self._client is None:
                await client.aclose()


class FallbackTTS:
    """Try providers in order, surfacing the final error if all fail."""

    def __init__(self, *providers: TextToSpeech) -> None:
        if not providers:
            raise ValueError("at least one TTS provider is required")
        self.providers = providers

    async def synthesize(self, text: str) -> SynthesizedAudio:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return await provider.synthesize(text)
            except Exception as exc:
                last_error = exc
                logger.warning("TTS provider %s failed: %s", type(provider).__name__, exc)
        assert last_error is not None
        raise last_error
