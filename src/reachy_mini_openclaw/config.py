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
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17"))
    OPENAI_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))
    
    # OpenClaw Gateway Configuration
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789"))
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
    HEAD_TRACKER_TYPE: Optional[str] = field(default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE"))
    
    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: Optional[str] = field(default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        return errors


# Global configuration instance
config = Config()


def set_custom_profile(profile: Optional[str]) -> None:
    """Update the custom profile at runtime."""
    global config
    config.CUSTOM_PROFILE = profile
    os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile or ""
