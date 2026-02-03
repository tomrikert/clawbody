"""ClawBody - Bridge to OpenClaw Gateway for AI responses.

This module provides ClawBody's integration with the OpenClaw gateway
using the OpenAI-compatible Chat Completions HTTP API.

ClawBody uses OpenAI Realtime API for voice I/O (speech recognition + TTS)
but routes all responses through OpenClaw (Clawson) for intelligence.
"""

import json
import asyncio
import logging
from typing import Optional, Any, AsyncIterator
from dataclasses import dataclass

import httpx
from httpx_sse import aconnect_sse

from reachy_mini_openclaw.config import config

logger = logging.getLogger(__name__)


@dataclass
class OpenClawResponse:
    """Response from OpenClaw gateway."""
    content: str
    error: Optional[str] = None


class OpenClawBridge:
    """Bridge to OpenClaw Gateway using HTTP Chat Completions API.
    
    This class sends user messages to OpenClaw and receives AI responses.
    The robot maintains conversation context and can include images.
    
    Example:
        bridge = OpenClawBridge()
        await bridge.connect()
        
        # Simple query
        response = await bridge.chat("Hello!")
        print(response.content)
        
        # With image
        response = await bridge.chat("What do you see?", image_b64="...")
    """
    
    def __init__(
        self,
        gateway_url: Optional[str] = None,
        gateway_token: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """Initialize the OpenClaw bridge.
        
        Args:
            gateway_url: URL of the OpenClaw gateway (default: from env/config)
            gateway_token: Authentication token (default: from env/config)
            agent_id: OpenClaw agent ID to use (default: from env/config)
            timeout: Request timeout in seconds
        """
        import os
        # Read from env directly as fallback (config may have been loaded before .env)
        self.gateway_url = gateway_url or os.getenv("OPENCLAW_GATEWAY_URL") or config.OPENCLAW_GATEWAY_URL
        self.gateway_token = gateway_token or os.getenv("OPENCLAW_TOKEN") or config.OPENCLAW_TOKEN
        self.agent_id = agent_id or os.getenv("OPENCLAW_AGENT_ID") or config.OPENCLAW_AGENT_ID
        self.timeout = timeout
        
        # Session key - use "main" to share context with WhatsApp and other channels
        # The full session key is: agent:<agent_id>:<session_key>
        self.session_key = os.getenv("OPENCLAW_SESSION_KEY") or config.OPENCLAW_SESSION_KEY or "main"
        
        # Connection state
        self._connected = False
        
    async def connect(self) -> bool:
        """Test connection to the OpenClaw gateway.
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info("Attempting to connect to OpenClaw at %s (token: %s)", 
                    self.gateway_url, "set" if self.gateway_token else "not set")
        try:
            # Use longer timeout for first connection (OpenClaw may need to initialize)
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Test the chat completions endpoint with a simple request
                url = f"{self.gateway_url}/v1/chat/completions"
                logger.info("Testing endpoint: %s", url)
                response = await client.post(
                    url,
                    json={
                        "model": f"openclaw:{self.agent_id}",
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    headers=self._get_headers(),
                )
                logger.info("Response status: %d", response.status_code)
                if response.status_code == 200:
                    self._connected = True
                    logger.info("Connected to OpenClaw gateway at %s", self.gateway_url)
                    return True
                else:
                    logger.warning("OpenClaw gateway returned %d: %s", 
                                 response.status_code, response.text[:100])
                    self._connected = False
                    return False
        except Exception as e:
            logger.error("Failed to connect to OpenClaw gateway: %s (type: %s)", e, type(e).__name__)
            self._connected = False
            return False
    
    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenClaw API requests."""
        headers = {
            "Content-Type": "application/json",
            # Use session key header to share context with WhatsApp and other channels
            # Format: agent:<agent_id>:<session_key> - default "main" shares with all DMs
            "x-openclaw-session-key": f"agent:{self.agent_id}:{self.session_key}",
        }
        if self.gateway_token:
            headers["Authorization"] = f"Bearer {self.gateway_token}"
        return headers
    
    async def chat(
        self, 
        message: str, 
        image_b64: Optional[str] = None,
        system_context: Optional[str] = None,
    ) -> OpenClawResponse:
        """Send a message to OpenClaw and get a response.
        
        OpenClaw maintains conversation memory on its end, so it will be aware
        of conversations from other channels (WhatsApp, web, etc.). We only send
        the current message and let OpenClaw handle the context.
        
        Args:
            message: The user's message (transcribed speech)
            image_b64: Optional base64-encoded image from robot camera
            system_context: Optional additional system context
            
        Returns:
            OpenClawResponse with the AI's response
        """
        # Build user message content
        if image_b64:
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        else:
            content = message
        
        # Build request messages - just the current message
        # OpenClaw maintains conversation memory on its end
        request_messages = []
        
        # Add system context if provided (e.g., "User is speaking to you through the robot")
        if system_context:
            request_messages.append({"role": "system", "content": system_context})
        
        # Add the current user message
        request_messages.append({"role": "user", "content": content})
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.post(
                    f"{self.gateway_url}/v1/chat/completions",
                    json={
                        "model": f"openclaw:{self.agent_id}",
                        "messages": request_messages,
                        "stream": False,
                    },
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    assistant_content = choices[0].get("message", {}).get("content", "")
                    return OpenClawResponse(content=assistant_content)
                return OpenClawResponse(content="", error="No response from OpenClaw")
                
        except httpx.HTTPStatusError as e:
            logger.error("OpenClaw HTTP error: %d - %s", e.response.status_code, e.response.text[:200])
            return OpenClawResponse(content="", error=f"HTTP {e.response.status_code}")
        except Exception as e:
            logger.error("OpenClaw chat error: %s", e)
            return OpenClawResponse(content="", error=str(e))
    
    async def stream_chat(
        self, 
        message: str, 
        image_b64: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenClaw.
        
        OpenClaw maintains conversation memory on its end, so it will be aware
        of conversations from other channels (WhatsApp, web, etc.).
        
        Args:
            message: The user's message
            image_b64: Optional base64-encoded image
            
        Yields:
            String chunks of the response as they arrive
        """
        # Build user message content
        if image_b64:
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        else:
            content = message
        
        # Only send current message - OpenClaw handles memory
        request_messages = [{"role": "user", "content": content}]
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            try:
                async with aconnect_sse(
                    client,
                    "POST",
                    f"{self.gateway_url}/v1/chat/completions",
                    json={
                        "model": f"openclaw:{self.agent_id}",
                        "messages": request_messages,
                        "stream": True,
                    },
                    headers=self._get_headers(),
                ) as event_source:
                    event_source.response.raise_for_status()
                    
                    async for sse in event_source.aiter_sse():
                        if sse.data == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(sse.data)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
                            
            except httpx.HTTPStatusError as e:
                logger.error("OpenClaw streaming error: %d", e.response.status_code)
                yield f"[Error: HTTP {e.response.status_code}]"
            except Exception as e:
                logger.error("OpenClaw streaming error: %s", e)
                yield f"[Error: {e}]"
    
    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to gateway."""
        return self._connected


# Global bridge instance (lazy initialization)
_bridge: Optional[OpenClawBridge] = None


def get_bridge() -> OpenClawBridge:
    """Get the global OpenClaw bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = OpenClawBridge()
    return _bridge
