import io
import wave

import httpx
import numpy as np
import pytest

from reachy_mini_openclaw.audio.providers import (
    ChatterboxTTS,
    ElevenLabsTTS,
    FallbackTTS,
    OpenAICompatibleSTT,
    SynthesizedAudio,
    decode_wav,
    encode_wav,
)


def test_wav_round_trip() -> None:
    source = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    decoded = decode_wav(encode_wav(source, 16000))
    assert decoded.sample_rate == 16000
    assert decoded.samples.dtype == np.int16
    assert decoded.samples.size == source.size


@pytest.mark.asyncio
async def test_openai_compatible_stt_posts_wav() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer local-secret"
        body = await request.aread()
        assert b'distil-large-v3' in body
        assert b'filename="utterance.wav"' in body
        return httpx.Response(200, json={"text": "  hello Reachy  "})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = OpenAICompatibleSTT(
        "http://stt.test/v1",
        "distil-large-v3",
        api_key="local-secret",
        client=client,
    )
    try:
        text = await provider.transcribe(np.ones(1600, dtype=np.float32) * 0.1, 16000)
    finally:
        await client.aclose()
    assert text == "hello Reachy"


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(np.arange(64, dtype=np.int16).tobytes())
    return output.getvalue()


@pytest.mark.asyncio
async def test_chatterbox_requests_named_voice_and_decodes_wav() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        payload = __import__("json").loads((await request.aread()).decode())
        assert payload == {
            "input": "hello",
            "voice": "asmo",
            "model": "chatterbox-turbo",
            "response_format": "wav",
        }
        return httpx.Response(200, content=_wav_bytes(), headers={"content-type": "audio/wav"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = ChatterboxTTS("http://tts.test/v1/audio/speech", voice="asmo", client=client)
    try:
        audio = await provider.synthesize("hello")
    finally:
        await client.aclose()
    assert audio.sample_rate == 24000
    assert audio.samples.size == 64


@pytest.mark.asyncio
async def test_elevenlabs_requests_raw_pcm() -> None:
    pcm = np.arange(32, dtype=np.int16).tobytes()

    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/text-to-speech/voice-123/stream"
        assert request.url.params["output_format"] == "pcm_24000"
        assert request.headers["xi-api-key"] == "eleven-secret"
        return httpx.Response(200, content=pcm)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = ElevenLabsTTS("eleven-secret", "voice-123", base_url="http://eleven.test", client=client)
    try:
        audio = await provider.synthesize("hello")
    finally:
        await client.aclose()
    assert audio.sample_rate == 24000
    assert audio.samples.tolist() == list(range(32))


@pytest.mark.asyncio
async def test_fallback_tts_uses_second_provider() -> None:
    class Broken:
        async def synthesize(self, text: str) -> SynthesizedAudio:
            raise RuntimeError("offline")

    class Working:
        async def synthesize(self, text: str) -> SynthesizedAudio:
            return SynthesizedAudio(24000, np.array([1, 2], dtype=np.int16))

    audio = await FallbackTTS(Broken(), Working()).synthesize("hello")
    assert audio.samples.tolist() == [1, 2]
