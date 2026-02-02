"""Prompt management for the robot assistant.

Handles loading and customizing system prompts for the OpenAI Realtime session.
"""

import logging
from pathlib import Path
from typing import Optional

from reachy_mini_openclaw.config import config

logger = logging.getLogger(__name__)

# Default prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


def get_session_instructions() -> str:
    """Get the system instructions for the OpenAI Realtime session.
    
    Loads from custom profile if configured, otherwise uses default.
    
    Returns:
        System instructions string
    """
    # Check for custom profile
    custom_profile = config.CUSTOM_PROFILE
    if custom_profile:
        custom_path = PROMPTS_DIR / f"{custom_profile}.txt"
        if custom_path.exists():
            try:
                instructions = custom_path.read_text(encoding="utf-8")
                logger.info("Loaded custom profile: %s", custom_profile)
                return instructions
            except Exception as e:
                logger.warning("Failed to load custom profile %s: %s", custom_profile, e)
    
    # Load default
    default_path = PROMPTS_DIR / "default.txt"
    if default_path.exists():
        try:
            return default_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to load default prompt: %s", e)
    
    # Fallback inline prompt
    return """You are a friendly AI assistant with a robot body. You can see, hear, and move expressively. 
Be conversational and use your movement capabilities to be engaging. 
Use the camera tool when asked about your surroundings.
Express emotions through movement to enhance communication."""


def get_session_voice() -> str:
    """Get the voice to use for the OpenAI Realtime session.
    
    Returns:
        Voice name string
    """
    return config.OPENAI_VOICE


def get_available_profiles() -> list[str]:
    """Get list of available prompt profiles.
    
    Returns:
        List of profile names (without .txt extension)
    """
    profiles = []
    if PROMPTS_DIR.exists():
        for path in PROMPTS_DIR.glob("*.txt"):
            profiles.append(path.stem)
    return sorted(profiles)


def save_custom_profile(name: str, instructions: str) -> bool:
    """Save a custom prompt profile.
    
    Args:
        name: Profile name (alphanumeric and underscores only)
        instructions: The prompt instructions
        
    Returns:
        True if saved successfully
    """
    # Validate name
    if not name or not name.replace("_", "").isalnum():
        logger.error("Invalid profile name: %s", name)
        return False
    
    try:
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        profile_path = PROMPTS_DIR / f"{name}.txt"
        profile_path.write_text(instructions, encoding="utf-8")
        logger.info("Saved custom profile: %s", name)
        return True
    except Exception as e:
        logger.error("Failed to save profile %s: %s", name, e)
        return False
