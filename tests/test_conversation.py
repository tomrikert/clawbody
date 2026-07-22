import asyncio
import numpy as np
import pytest
import threading

from reachy_mini_openclaw.audio.conversation import (
    ConversationHandler,
    EnergyVAD,
    Utterance,
    command_after_wake_phrase,
    needs_vision,
)
from reachy_mini_openclaw.audio.providers import SynthesizedAudio
from reachy_mini_openclaw.openclaw_bridge import OpenClawResponse


def test_vad_segments_one_utterance_with_prefix() -> None:
    vad = EnergyVAD(
        rms_threshold=0.05,
        activation_ms=50,
        prefix_ms=100,
        silence_ms=100,
        min_speech_ms=100,
        max_speech_seconds=2,
    )
    sample_rate = 1000
    assert vad.feed(sample_rate, np.zeros(50, dtype=np.float32)) is None
    assert vad.feed(sample_rate, np.ones(100, dtype=np.float32) * 0.2) is None
    utterance = vad.feed(sample_rate, np.zeros(100, dtype=np.float32))
    assert utterance is not None
    assert utterance.sample_rate == sample_rate
    assert utterance.samples.size == 200
    assert not vad.speaking


def test_vad_discards_short_noise() -> None:
    vad = EnergyVAD(rms_threshold=0.05, activation_ms=50, silence_ms=50, min_speech_ms=100)
    assert vad.feed(1000, np.ones(20, dtype=np.float32)) is None
    assert vad.feed(1000, np.zeros(50, dtype=np.float32)) is None
    assert not vad.speaking


def test_vad_requires_sustained_loud_onset() -> None:
    vad = EnergyVAD(
        rms_threshold=0.05,
        activation_ms=100,
        silence_ms=50,
        min_speech_ms=100,
    )
    assert vad.feed(1000, np.ones(40, dtype=np.float32) * 0.2) is None
    assert not vad.speaking
    assert vad.feed(1000, np.zeros(20, dtype=np.float32)) is None
    assert vad.feed(1000, np.ones(60, dtype=np.float32) * 0.2) is None
    assert not vad.speaking
    assert vad.feed(1000, np.ones(40, dtype=np.float32) * 0.2) is None
    assert vad.speaking


def test_vad_hardware_gate_blocks_onset_but_not_active_speech() -> None:
    vad = EnergyVAD(rms_threshold=0.05, activation_ms=50)
    loud = np.ones(50, dtype=np.float32) * 0.2
    assert vad.feed(1000, loud, allow_start=False) is None
    assert not vad.speaking
    assert vad.feed(1000, loud, allow_start=True) is None
    assert vad.speaking
    assert vad.feed(1000, loud, allow_start=False) is None
    assert vad.speaking


def test_visual_questions_request_a_camera_frame() -> None:
    assert needs_vision("Asmo, can you see me?")
    assert needs_vision("Look at what I'm holding")
    assert not needs_vision("I see what you mean")


def test_wake_phrase_matching_is_prefix_only_and_punctuation_tolerant() -> None:
    assert command_after_wake_phrase("Hey, Asmo! What time is it?", "Hey Asmo") == "What time is it?"
    assert command_after_wake_phrase("hey asmo", "Hey Asmo") == ""
    assert command_after_wake_phrase("Someone said hey Asmo", "Hey Asmo") is None
    assert command_after_wake_phrase("Hey Asmodeus", "Hey Asmo") is None


def test_wake_phrase_tolerates_common_stt_mangling() -> None:
    assert command_after_wake_phrase("Hey asthma, tell me a joke", "Hey Asmo") == "tell me a joke"
    assert command_after_wake_phrase("Hi Esmo what time is it", "Hey Asmo") == "what time is it"
    assert command_after_wake_phrase("They asked me a question", "Hey Asmo") is None


class Movement:
    def __init__(self) -> None:
        self.processing = []
        self.listening = []

    def set_processing(self, value: bool) -> None:
        self.processing.append(value)

    def set_listening(self, value: bool) -> None:
        self.listening.append(value)


class Wobbler:
    def __init__(self) -> None:
        self.feeds = 0

    def feed(self, audio: str) -> None:
        self.feeds += 1

    def reset(self) -> None:
        pass


class Deps:
    def __init__(self) -> None:
        self.movement_manager = Movement()
        self.head_wobbler = Wobbler()


class STT:
    async def transcribe(self, samples, sample_rate) -> str:
        return "what time is it"


class StaticSTT:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    async def transcribe(self, samples, sample_rate) -> str:
        return self.transcript


class TTS:
    async def synthesize(self, text: str) -> SynthesizedAudio:
        assert text == "It is robot o'clock."
        return SynthesizedAudio(24000, np.arange(5000, dtype=np.int16))


class FlakySTT(STT):
    def __init__(self) -> None:
        self.attempts = 0

    async def transcribe(self, samples, sample_rate) -> str:
        self.attempts += 1
        if self.attempts < 3:
            raise ConnectionError("temporary STT outage")
        return await super().transcribe(samples, sample_rate)


class FlakyTTS(TTS):
    def __init__(self) -> None:
        self.attempts = 0

    async def synthesize(self, text: str) -> SynthesizedAudio:
        self.attempts += 1
        if self.attempts < 3:
            raise ConnectionError("temporary TTS outage")
        return await super().synthesize(text)


class Bridge:
    is_connected = True

    async def connect(self) -> bool:
        return True

    async def chat(
        self, message: str, image_b64: str | None, system_context: str
    ) -> OpenClawResponse:
        assert message == "what time is it"
        assert image_b64 is None
        assert "Reachy Mini" in system_context
        return OpenClawResponse("It is robot o'clock.")


class FlakyBridge(Bridge):
    def __init__(self) -> None:
        self.is_connected = False
        self.attempts = 0

    async def connect(self) -> bool:
        self.attempts += 1
        if self.attempts < 3:
            return False
        self.is_connected = True
        return True


class RecordingBridge:
    is_connected = True

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def connect(self) -> bool:
        return True

    async def chat(
        self, message: str, image_b64: str | None, system_context: str
    ) -> OpenClawResponse:
        self.messages.append(message)
        return OpenClawResponse("It is robot o'clock.")


@pytest.mark.asyncio
async def test_pipeline_routes_openclaw_response_to_tts_and_audio_queue() -> None:
    deps = Deps()
    handler = ConversationHandler(
        stt=STT(),
        tts=TTS(),
        openclaw_bridge=Bridge(),
        deps=deps,
        vad=EnergyVAD(),
        output_chunk_samples=2048,
    )
    await handler._process_utterance(Utterance(16000, np.ones(1600, dtype=np.float32)))

    outputs = []
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())
    assert outputs[0] == {"role": "user", "content": "what time is it"}
    assert outputs[1] == {"role": "assistant", "content": "It is robot o'clock."}
    audio = [item for item in outputs if isinstance(item, tuple)]
    assert [chunk.shape[1] for _, chunk in audio] == [2048, 2048, 904]
    assert deps.head_wobbler.feeds == 0
    assert deps.movement_manager.processing == [True, False]


@pytest.mark.asyncio
async def test_wake_phrase_gate_strips_phrase_before_openclaw() -> None:
    deps = Deps()
    bridge = RecordingBridge()
    handler = ConversationHandler(
        stt=StaticSTT("Hey, Asmo! What time is it?"),
        tts=TTS(),
        openclaw_bridge=bridge,
        deps=deps,
        vad=EnergyVAD(),
        wake_phrase_enabled=True,
        wake_phrase="Hey Asmo",
    )

    await handler._process_utterance(Utterance(16000, np.ones(1600, dtype=np.float32)))

    assert bridge.messages == ["What time is it?"]
    assert handler.output_queue.get_nowait() == {"role": "user", "content": "What time is it?"}


@pytest.mark.asyncio
async def test_wake_phrase_gate_ignores_ambient_transcript() -> None:
    deps = Deps()
    bridge = RecordingBridge()
    handler = ConversationHandler(
        stt=StaticSTT("the television is still talking"),
        tts=TTS(),
        openclaw_bridge=bridge,
        deps=deps,
        vad=EnergyVAD(),
        wake_phrase_enabled=True,
        wake_phrase="Hey Asmo",
    )

    await handler._process_utterance(Utterance(16000, np.ones(1600, dtype=np.float32)))

    assert bridge.messages == []
    assert handler.output_queue.empty()
    assert deps.movement_manager.processing == [True, False]


@pytest.mark.asyncio
async def test_pipeline_retries_transient_stt_and_tts_failures() -> None:
    deps = Deps()
    stt = FlakySTT()
    tts = FlakyTTS()
    handler = ConversationHandler(
        stt=stt,
        tts=tts,
        openclaw_bridge=Bridge(),
        deps=deps,
        vad=EnergyVAD(),
    )

    await handler._process_utterance(Utterance(16000, np.ones(1600, dtype=np.float32)))

    assert stt.attempts == 3
    assert tts.attempts == 3


@pytest.mark.asyncio
async def test_pipeline_reconnects_gateway_with_bounded_backoff() -> None:
    deps = Deps()
    bridge = FlakyBridge()
    handler = ConversationHandler(
        stt=STT(),
        tts=TTS(),
        openclaw_bridge=bridge,
        deps=deps,
        vad=EnergyVAD(),
    )

    await handler._ensure_gateway()

    assert bridge.attempts == 3
    assert bridge.is_connected


@pytest.mark.asyncio
async def test_native_stop_watcher_does_not_spawn_a_blocking_executor_thread() -> None:
    import reachy_mini_openclaw.main as native_main

    core = native_main.ClawBodyCore.__new__(native_main.ClawBodyCore)
    core._external_stop_event = threading.Event()
    core._stop_event = asyncio.Event()

    watcher = asyncio.create_task(core._watch_external_stop())
    await asyncio.sleep(0)
    watcher.cancel()
    with pytest.raises(asyncio.CancelledError):
        await watcher


def test_native_wrapper_propagates_failure_after_cleanup(monkeypatch) -> None:
    import reachy_mini_openclaw.main as native_main

    class FailingCore:
        stopped = False

        def __init__(self, **kwargs) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("simulated daemon loss")

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(native_main, "ClawBodyCore", FailingCore)
    app = native_main.ReachyMiniOpenclaw.__new__(native_main.ReachyMiniOpenclaw)

    with pytest.raises(RuntimeError, match="simulated daemon loss"):
        app.run(object(), threading.Event())


def test_native_config_requires_explicit_service_endpoints() -> None:
    from reachy_mini_openclaw.config import Config

    config = Config(STT_BASE_URL="", OPENCLAW_GATEWAY_URL="", CHATTERBOX_URL="")
    errors = config.validate()

    assert "STT_BASE_URL is required" in errors
    assert "OPENCLAW_GATEWAY_URL is required" in errors
    assert "CHATTERBOX_URL is required for Chatterbox TTS" in errors


def test_native_config_rejects_empty_enabled_wake_phrase() -> None:
    from reachy_mini_openclaw.config import Config

    config = Config(ENABLE_WAKE_PHRASE=True, WAKE_PHRASE="   ")

    assert "WAKE_PHRASE must not be empty when wake phrase gating is enabled" in config.validate()


@pytest.mark.asyncio
async def test_microphone_is_suppressed_while_robot_audio_is_playing() -> None:
    deps = Deps()
    vad = EnergyVAD(rms_threshold=0.01)
    handler = ConversationHandler(
        stt=STT(),
        tts=TTS(),
        openclaw_bridge=Bridge(),
        deps=deps,
        vad=vad,
    )
    handler._ignore_input_until = float("inf")

    await handler.receive((16000, np.ones(1600, dtype=np.float32)))

    assert not vad.speaking
    assert handler._utterances.empty()
