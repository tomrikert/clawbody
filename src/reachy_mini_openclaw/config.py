"""Configuration management for Reachy Mini OpenClaw.

Handles environment variables and configuration settings for the application.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


def _gateway_token() -> Optional[str]:
    token = os.getenv("OPENCLAW_TOKEN") or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    if token:
        return token
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("gateway", {}).get("auth", {}).get("token")
    except (OSError, ValueError, TypeError):
        return None


@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    
    # Optional OpenAI configuration (currently used only by the legacy vision fallback)
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Speech-to-text. The endpoint follows OpenAI's multipart
    # /v1/audio/transcriptions contract, but can be entirely local.
    STT_BASE_URL: str = field(default_factory=lambda: os.getenv("STT_BASE_URL", ""))
    STT_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("STT_API_KEY"))
    STT_MODEL: str = field(default_factory=lambda: os.getenv("STT_MODEL", "distil-large-v3"))
    STT_LANGUAGE: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "en"))
    STT_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(os.getenv("STT_TIMEOUT_SECONDS", "30")))

    # Text-to-speech. Chatterbox is primary; ElevenLabs can be enabled as a
    # fallback without changing the conversational pipeline.
    TTS_PROVIDER: str = field(default_factory=lambda: os.getenv("TTS_PROVIDER", "chatterbox").lower())
    CHATTERBOX_URL: str = field(default_factory=lambda: os.getenv("CHATTERBOX_URL", ""))
    CHATTERBOX_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("CHATTERBOX_API_KEY"))
    CHATTERBOX_VOICE: str = field(default_factory=lambda: os.getenv("CHATTERBOX_VOICE", "asmo"))
    CHATTERBOX_SAMPLE_RATE: int = field(default_factory=lambda: int(os.getenv("CHATTERBOX_SAMPLE_RATE", "24000")))
    ELEVENLABS_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY"))
    ELEVENLABS_VOICE_ID: Optional[str] = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID"))
    ELEVENLABS_MODEL_ID: str = field(default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"))
    ELEVENLABS_BASE_URL: str = field(default_factory=lambda: os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io"))
    TTS_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(os.getenv("TTS_TIMEOUT_SECONDS", "45")))
    # Software gain applied immediately before Reachy playback. Reachy's app
    # volume remains the final hardware control; 1.0 preserves the TTS signal.
    AUDIO_OUTPUT_GAIN: float = field(default_factory=lambda: float(os.getenv("AUDIO_OUTPUT_GAIN", "1.0")))

    # Local voice activity detection.
    VAD_RMS_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("VAD_RMS_THRESHOLD", "0.022")))
    VAD_ACTIVATION_MS: int = field(default_factory=lambda: int(os.getenv("VAD_ACTIVATION_MS", "160")))
    VAD_PREFIX_MS: int = field(default_factory=lambda: int(os.getenv("VAD_PREFIX_MS", "250")))
    VAD_SILENCE_MS: int = field(default_factory=lambda: int(os.getenv("VAD_SILENCE_MS", "650")))
    VAD_MIN_SPEECH_MS: int = field(default_factory=lambda: int(os.getenv("VAD_MIN_SPEECH_MS", "300")))
    VAD_MAX_SPEECH_SECONDS: float = field(default_factory=lambda: float(os.getenv("VAD_MAX_SPEECH_SECONDS", "20")))
    VAD_REQUIRE_HARDWARE_SPEECH: bool = field(default_factory=lambda: os.getenv("VAD_REQUIRE_HARDWARE_SPEECH", "true").lower() == "true")

    # Optional transcript-level wake phrase gate. Audio is still transcribed
    # locally, but only utterances beginning with the phrase reach OpenClaw.
    ENABLE_WAKE_PHRASE: bool = field(default_factory=lambda: os.getenv("ENABLE_WAKE_PHRASE", "false").lower() == "true")
    WAKE_PHRASE: str = field(default_factory=lambda: os.getenv("WAKE_PHRASE", "Hey Asmo"))

    # Hourly mail attention queue. The broker is gateway-local and uses the
    # same bearer token already provisioned for OpenClaw.
    ENABLE_ATTENTION_QUEUE: bool = field(default_factory=lambda: os.getenv("ENABLE_ATTENTION_QUEUE", "true").lower() == "true")
    ATTENTION_API_URL: str = field(default_factory=lambda: os.getenv("ATTENTION_API_URL", "http://192.168.1.238:18790"))
    ATTENTION_POLL_SECONDS: float = field(default_factory=lambda: float(os.getenv("ATTENTION_POLL_SECONDS", "60")))
    
    # OpenClaw Gateway Configuration
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv("OPENCLAW_GATEWAY_URL", ""))
    OPENCLAW_TOKEN: Optional[str] = field(default_factory=_gateway_token)
    OPENCLAW_AGENT_ID: str = field(default_factory=lambda: os.getenv("OPENCLAW_AGENT_ID", "main"))
    # Spoken body turns should feel immediate. "off" is OpenClaw's lowest-
    # latency setting and applies only to the dedicated Reachy session.
    OPENCLAW_THINKING: str = field(default_factory=lambda: os.getenv("OPENCLAW_THINKING", "off").lower())
    # Session key for OpenClaw - uses "main" to share context with WhatsApp and other channels
    # Format: agent:<agent_id>:<session_key>, but we only need the session key part here
    OPENCLAW_SESSION_KEY: str = field(default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "main"))
    
    # Robot Configuration
    ROBOT_NAME: Optional[str] = field(default_factory=lambda: os.getenv("ROBOT_NAME"))
    ROBOT_HOST: str = field(default_factory=lambda: os.getenv("ROBOT_HOST", "reachy-mini.local"))
    ROBOT_PORT: int = field(default_factory=lambda: int(os.getenv("ROBOT_PORT", "8000")))
    ROBOT_CONNECTION_MODE: str = field(default_factory=lambda: os.getenv("ROBOT_CONNECTION_MODE", "auto"))
    
    # Feature Flags
    ENABLE_OPENCLAW_TOOLS: bool = field(default_factory=lambda: os.getenv("ENABLE_OPENCLAW_TOOLS", "true").lower() == "true")
    ENABLE_CAMERA: bool = field(default_factory=lambda: os.getenv("ENABLE_CAMERA", "true").lower() == "true")
    ENABLE_FACE_TRACKING: bool = field(default_factory=lambda: os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true")
    
    # Face Tracking Configuration
    # "daemon" uses Reachy 1.9's built-in detector and avoids a duplicate model.
    # "yolo" and "mediapipe" remain available for older daemon versions.
    HEAD_TRACKER_TYPE: Optional[str] = field(default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "daemon"))
    FACE_TRACKING_WEIGHT: float = field(default_factory=lambda: float(os.getenv("FACE_TRACKING_WEIGHT", "0.7")))
    ENABLE_SOUND_TRACKING: bool = field(default_factory=lambda: os.getenv("ENABLE_SOUND_TRACKING", "true").lower() == "true")
    
    # Local Vision Processing
    ENABLE_LOCAL_VISION: bool = field(default_factory=lambda: os.getenv("ENABLE_LOCAL_VISION", "false").lower() == "true")
    LOCAL_VISION_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"))
    VISION_DEVICE: str = field(default_factory=lambda: os.getenv("VISION_DEVICE", "auto"))  # "auto", "cuda", "mps", "cpu"
    HF_HOME: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    
    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: Optional[str] = field(default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))
    
    def validate(self, gateway_url: Optional[str] = None) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.STT_BASE_URL:
            errors.append("STT_BASE_URL is required")
        if not (gateway_url or self.OPENCLAW_GATEWAY_URL):
            errors.append("OPENCLAW_GATEWAY_URL is required")
        if self.TTS_PROVIDER not in {"chatterbox", "elevenlabs"}:
            errors.append("TTS_PROVIDER must be 'chatterbox' or 'elevenlabs'")
        if self.TTS_PROVIDER == "chatterbox" and not self.CHATTERBOX_URL:
            errors.append("CHATTERBOX_URL is required for Chatterbox TTS")
        if self.TTS_PROVIDER == "elevenlabs":
            if not self.ELEVENLABS_API_KEY:
                errors.append("ELEVENLABS_API_KEY is required for ElevenLabs TTS")
            if not self.ELEVENLABS_VOICE_ID:
                errors.append("ELEVENLABS_VOICE_ID is required for ElevenLabs TTS")
        if self.AUDIO_OUTPUT_GAIN <= 0:
            errors.append("AUDIO_OUTPUT_GAIN must be greater than zero")
        if self.ENABLE_WAKE_PHRASE and not self.WAKE_PHRASE.strip():
            errors.append("WAKE_PHRASE must not be empty when wake phrase gating is enabled")
        if self.ENABLE_ATTENTION_QUEUE and not self.ATTENTION_API_URL:
            errors.append("ATTENTION_API_URL is required when attention queue is enabled")
        if self.ENABLE_ATTENTION_QUEUE and not self.OPENCLAW_TOKEN:
            errors.append("OPENCLAW_TOKEN is required when attention queue is enabled")
        if self.ATTENTION_POLL_SECONDS < 5:
            errors.append("ATTENTION_POLL_SECONDS must be at least 5")
        if self.OPENCLAW_THINKING not in {
            "off", "minimal", "low", "medium", "high", "xhigh", "adaptive", "max"
        }:
            errors.append("OPENCLAW_THINKING is not a supported thinking level")
        return errors


# Global configuration instance
config = Config()


def set_custom_profile(profile: Optional[str]) -> None:
    """Update the custom profile at runtime."""
    global config
    config.CUSTOM_PROFILE = profile
    os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile or ""


def set_face_tracking_enabled(enabled: bool) -> None:
    """Enable or disable face tracking at runtime."""
    global config
    config.ENABLE_FACE_TRACKING = enabled


def set_local_vision_enabled(enabled: bool) -> None:
    """Enable or disable local vision processing at runtime."""
    global config
    config.ENABLE_LOCAL_VISION = enabled
