"""Configuration management for Reachy Mini OpenClaw.

Handles environment variables and configuration settings for the application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # AI Provider Configuration
    # Options: "openai" (OpenAI Realtime) or "local" (local STT + Ollama + local TTS)
    AI_PROVIDER: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "openai").lower())

    # OpenAI Configuration
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-realtime-1.5"))
    OPENAI_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))

    # Local AI Configuration
    LOCAL_LLM_BASE_URL: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434"))
    LOCAL_LLM_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "gemma4:latest"))
    LOCAL_LLM_THINK: bool = field(default_factory=lambda: os.getenv("LOCAL_LLM_THINK", "false").lower() == "true")
    LOCAL_LLM_TIMEOUT: float = field(default_factory=lambda: float(os.getenv("LOCAL_LLM_TIMEOUT", "120")))
    LOCAL_STT_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_STT_MODEL", "large-v3-turbo"))
    LOCAL_STT_DEVICE: str = field(default_factory=lambda: os.getenv("LOCAL_STT_DEVICE", "cpu"))
    LOCAL_STT_COMPUTE_TYPE: str = field(default_factory=lambda: os.getenv("LOCAL_STT_COMPUTE_TYPE", "int8"))
    LOCAL_TTS_ENGINE: str = field(default_factory=lambda: os.getenv("LOCAL_TTS_ENGINE", "kokoro").lower())
    LOCAL_TTS_LANG: str = field(default_factory=lambda: os.getenv("LOCAL_TTS_LANG", "a"))
    LOCAL_TTS_VOICE: str = field(default_factory=lambda: os.getenv("LOCAL_TTS_VOICE", "af_heart"))
    LOCAL_TTS_SPEED: float = field(default_factory=lambda: float(os.getenv("LOCAL_TTS_SPEED", "1.0")))
    LOCAL_VAD_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("LOCAL_VAD_THRESHOLD", "0.015")))
    LOCAL_VAD_SILENCE_MS: int = field(default_factory=lambda: int(os.getenv("LOCAL_VAD_SILENCE_MS", "700")))
    LOCAL_MIN_SPEECH_MS: int = field(default_factory=lambda: int(os.getenv("LOCAL_MIN_SPEECH_MS", "250")))
    LOCAL_MAX_HISTORY_MESSAGES: int = field(default_factory=lambda: int(os.getenv("LOCAL_MAX_HISTORY_MESSAGES", "12")))
    LOCAL_VISION_LLM_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_VISION_LLM_MODEL", os.getenv("LOCAL_LLM_MODEL", "gemma4:latest")))

    # OpenClaw Gateway Configuration
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789"))
    OPENCLAW_TOKEN: Optional[str] = field(default_factory=lambda: os.getenv("OPENCLAW_TOKEN"))
    OPENCLAW_AGENT_ID: str = field(default_factory=lambda: os.getenv("OPENCLAW_AGENT_ID", "main"))
    # Session key for OpenClaw - uses "main" to share context with WhatsApp and other channels
    # Format: agent:<agent_id>:<session_key>, but we only need the session key part here
    OPENCLAW_SESSION_KEY: str = field(default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "main"))
    
    # Robot Configuration
    ROBOT_NAME: Optional[str] = field(default_factory=lambda: os.getenv("ROBOT_NAME"))
    
    # Feature Flags
    ENABLE_OPENCLAW_TOOLS: bool = field(default_factory=lambda: os.getenv("ENABLE_OPENCLAW_TOOLS", "true").lower() == "true")
    ENABLE_CAMERA: bool = field(default_factory=lambda: os.getenv("ENABLE_CAMERA", "true").lower() == "true")
    ENABLE_FACE_TRACKING: bool = field(default_factory=lambda: os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true")
    
    # Face Tracking Configuration
    # Options: "yolo", "mediapipe", or None for auto-detect
    HEAD_TRACKER_TYPE: Optional[str] = field(default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "yolo"))
    
    # Local Vision Processing
    ENABLE_LOCAL_VISION: bool = field(default_factory=lambda: os.getenv("ENABLE_LOCAL_VISION", "false").lower() == "true")
    LOCAL_VISION_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"))
    VISION_DEVICE: str = field(default_factory=lambda: os.getenv("VISION_DEVICE", "auto"))  # "auto", "cuda", "mps", "cpu"
    HF_HOME: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    
    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: Optional[str] = field(default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if self.AI_PROVIDER not in {"openai", "local"}:
            errors.append("AI_PROVIDER must be 'openai' or 'local'")
        if self.AI_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
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
