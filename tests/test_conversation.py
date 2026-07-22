import numpy as np
import pytest

from reachy_mini_openclaw.audio.conversation import ConversationHandler, EnergyVAD, Utterance, needs_vision
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


class TTS:
    async def synthesize(self, text: str) -> SynthesizedAudio:
        assert text == "It is robot o'clock."
        return SynthesizedAudio(24000, np.arange(5000, dtype=np.int16))


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
