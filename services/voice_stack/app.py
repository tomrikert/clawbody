from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=1200)
    voice: str = "asmo"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.75, le=1.25)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)


class ModelRegistry:
    def __init__(self) -> None:
        self.device = os.getenv("VOICE_DEVICE", "cuda")
        self.stt_model_name = os.getenv("STT_MODEL", "distil-large-v3")
        self.stt_compute_type = os.getenv("STT_COMPUTE_TYPE", "int8_float16")
        self.voice_reference = Path(os.environ["ASMO_VOICE_REFERENCE"])
        self.stt = None
        self.tts = None
        self.stt_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.loaded_at: float | None = None

    def load(self) -> None:
        if not self.voice_reference.is_file():
            raise RuntimeError(f"Voice reference does not exist: {self.voice_reference}")

        from faster_whisper import WhisperModel
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self.stt = WhisperModel(
            self.stt_model_name,
            device=self.device,
            compute_type=self.stt_compute_type,
        )
        self.tts = ChatterboxTurboTTS.from_pretrained(self.device)
        self.loaded_at = time.time()

    @property
    def ready(self) -> bool:
        return self.stt is not None and self.tts is not None


models: ModelRegistry | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global models
    models = ModelRegistry()
    models.load()
    yield


app = FastAPI(title="ClawBody Voice Stack", version="0.1.0", lifespan=lifespan)


def registry() -> ModelRegistry:
    if models is None or not models.ready:
        raise HTTPException(status_code=503, detail="Models are not ready")
    return models


@app.get("/health")
def health() -> dict[str, object]:
    current = registry()
    return {
        "ok": True,
        "device": current.device,
        "stt_model": current.stt_model_name,
        "tts_model": "ResembleAI/chatterbox-turbo",
        "loaded_at": current.loaded_at,
    }


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "en",
    response_format: Annotated[str, Form()] = "json",
) -> dict[str, object]:
    current = registry()
    suffix = Path(file.filename or "utterance.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        while chunk := file.file.read(1024 * 1024):
            tmp.write(chunk)
        tmp.flush()
        started = time.perf_counter()
        with current.stt_lock:
            segments, info = current.stt.transcribe(
                tmp.name,
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segment_list = list(segments)

    text = " ".join(segment.text.strip() for segment in segment_list).strip()
    elapsed = time.perf_counter() - started
    result = {
        "text": text,
        "language": info.language,
        "duration": info.duration,
        "elapsed_seconds": elapsed,
    }
    if response_format == "verbose_json":
        result["segments"] = [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in segment_list
        ]
    return result


@app.post("/v1/audio/speech")
def synthesize(request: SpeechRequest) -> Response:
    if request.response_format != "wav":
        raise HTTPException(status_code=400, detail="Only wav output is supported locally")
    if request.voice != "asmo":
        raise HTTPException(status_code=400, detail="Unknown local voice")

    current = registry()
    started = time.perf_counter()
    with current.tts_lock:
        waveform = current.tts.generate(
            request.input,
            audio_prompt_path=str(current.voice_reference),
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            norm_loudness=True,
        )

    audio = waveform.squeeze(0).detach().cpu().numpy()
    output = io.BytesIO()
    sf.write(output, audio, current.tts.sr, format="WAV", subtype="PCM_16")
    return Response(
        content=output.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Generation-Seconds": f"{time.perf_counter() - started:.3f}",
            "X-Voice-Model": "chatterbox-turbo",
        },
    )
