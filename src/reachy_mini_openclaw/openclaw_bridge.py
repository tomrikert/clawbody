"""ClawBody - Bridge to OpenClaw Gateway for AI responses.

This module provides ClawBody's integration with the OpenClaw gateway
using the WebSocket protocol (the gateway's native transport).

ClawBody uses OpenAI Realtime API for voice I/O (speech recognition + TTS)
but routes all responses through OpenClaw (Clawson) for intelligence.
"""

import json
import asyncio
import base64
import logging
import uuid
import time
from pathlib import Path
from typing import Optional, Any, AsyncIterator
from dataclasses import dataclass

import websockets

from reachy_mini_openclaw.config import config

logger = logging.getLogger(__name__)

# Protocol version supported by this client
PROTOCOL_VERSION = 3
PLACEHOLDER_TOKENS = {"", "your-gateway-token", "sk-your-openai-key"}
ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


@dataclass
class OpenClawResponse:
    """Response from OpenClaw gateway."""
    content: str
    error: Optional[str] = None


class OpenClawBridge:
    """Bridge to OpenClaw Gateway using WebSocket protocol.

    The OpenClaw gateway speaks WebSocket with a JSON frame protocol.
    This class handles the connect handshake, authentication, and
    chat operations.

    Example:
        bridge = OpenClawBridge()
        await bridge.connect()

        # Simple query
        response = await bridge.chat("Hello!")
        print(response.content)
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        gateway_token: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """Initialize the OpenClaw bridge.

        Args:
            gateway_url: URL of the OpenClaw gateway (default: from env/config).
                         Accepts http:// or ws:// schemes; http is converted to ws.
            gateway_token: Authentication token (default: from env/config)
            agent_id: OpenClaw agent ID to use (default: from env/config)
            timeout: Request timeout in seconds
        """
        import os

        raw_url = (
            gateway_url
            or os.getenv("OPENCLAW_GATEWAY_URL")
            or config.OPENCLAW_GATEWAY_URL
        )
        # Normalise to ws:// (the gateway listens on the same port for both)
        self.gateway_url = self._normalise_ws_url(raw_url)

        self.gateway_token = (
            gateway_token
            or os.getenv("OPENCLAW_TOKEN")
            or config.OPENCLAW_TOKEN
        )
        if (self.gateway_token or "") in PLACEHOLDER_TOKENS:
            self.gateway_token = self._read_local_gateway_token()
        self.agent_id = (
            agent_id
            or os.getenv("OPENCLAW_AGENT_ID")
            or config.OPENCLAW_AGENT_ID
        )
        self.timeout = timeout

        # Session key – "main" shares context with WhatsApp and other channels.
        # Full key format: agent:<agent_id>:<session_key>
        self.session_key = (
            os.getenv("OPENCLAW_SESSION_KEY")
            or config.OPENCLAW_SESSION_KEY
            or "main"
        )

        # Persistent WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._conn_id: Optional[str] = None

        # Background listener task & pending request futures
        self._listener_task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}
        # Events keyed by runId -> list of event payloads
        self._run_events: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ws_url(url: str) -> str:
        """Convert http(s) URL to ws(s)."""
        if url.startswith("http://"):
            return "ws://" + url[7:]
        if url.startswith("https://"):
            return "wss://" + url[8:]
        if not url.startswith("ws://") and not url.startswith("wss://"):
            return "ws://" + url
        return url

    @staticmethod
    def _read_local_gateway_token() -> Optional[str]:
        """Read the local OpenClaw gateway token when using a local install."""
        try:
            path = Path.home() / ".openclaw" / "openclaw.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            token = data.get("gateway", {}).get("auth", {}).get("token")
            return token if isinstance(token, str) and token else None
        except Exception:
            return None

    @staticmethod
    def _base64url(data: bytes) -> str:
        """Encode bytes using unpadded base64url."""
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    @staticmethod
    def _raw_public_key_from_pem(public_key_pem: str) -> bytes:
        """Extract raw Ed25519 public-key bytes from an SPKI PEM."""
        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        der = public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        if der.startswith(ED25519_SPKI_PREFIX):
            return der[len(ED25519_SPKI_PREFIX):]
        return der

    def _build_local_device_auth(self, nonce: str, client: dict, role: str) -> tuple[dict, list[str]] | None:
        """Build signed device auth for the local OpenClaw gateway."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

            identity_path = Path.home() / ".openclaw" / "identity" / "device.json"
            auth_path = Path.home() / ".openclaw" / "identity" / "device-auth.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            auth_store = json.loads(auth_path.read_text(encoding="utf-8"))
            token_record = (auth_store.get("tokens") or {}).get(role)
            if not isinstance(token_record, dict):
                return None

            device_token = token_record.get("token")
            scopes = token_record.get("scopes")
            if not isinstance(device_token, str) or not isinstance(scopes, list):
                return None

            signed_at = int(time.time() * 1000)
            platform = (client.get("platform") or "").strip().lower()
            device_family = (client.get("deviceFamily") or "").strip().lower()
            payload = "|".join(
                [
                    "v3",
                    identity["deviceId"],
                    client["id"],
                    client["mode"],
                    role,
                    ",".join(scopes),
                    str(signed_at),
                    device_token,
                    nonce,
                    platform,
                    device_family,
                ]
            )

            private_key = serialization.load_pem_private_key(
                identity["privateKeyPem"].encode("utf-8"),
                password=None,
            )
            if not isinstance(private_key, Ed25519PrivateKey):
                return None

            signature = self._base64url(private_key.sign(payload.encode("utf-8")))
            public_key = self._base64url(self._raw_public_key_from_pem(identity["publicKeyPem"]))
            return (
                {
                    "auth": {"deviceToken": device_token},
                    "device": {
                        "id": identity["deviceId"],
                        "publicKey": public_key,
                        "signature": signature,
                        "signedAt": signed_at,
                        "nonce": nonce,
                    },
                },
                scopes,
            )
        except Exception as e:
            logger.debug("Local OpenClaw device auth unavailable: %s", e)
            return None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the OpenClaw gateway and authenticate.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(
            "Connecting to OpenClaw at %s (token: %s)",
            self.gateway_url,
            "set" if self.gateway_token else "not set",
        )
        try:
            # Build origin header from the gateway URL so the control-UI
            # origin check accepts programmatic WebSocket clients.
            origin = self.gateway_url.replace("ws://", "http://").replace("wss://", "https://")
            self._ws = await websockets.connect(
                self.gateway_url,
                origin=origin,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=5,
            )

            # 1. Receive challenge
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            challenge = json.loads(raw)
            if challenge.get("event") != "connect.challenge":
                logger.warning("Unexpected first frame: %s", challenge.get("event"))
            nonce = (challenge.get("payload") or {}).get("nonce")

            # 2. Send connect request
            req_id = str(uuid.uuid4())
            client_info = {
                "id": "cli",
                "version": "1.0.0",
                "platform": "linux",
                "mode": "cli",
            }
            role = "operator"
            scopes = ["operator.write", "operator.read"]
            auth: dict[str, str] = {"token": self.gateway_token} if self.gateway_token else {}
            device: Optional[dict] = None
            if isinstance(nonce, str) and nonce:
                device_auth = self._build_local_device_auth(nonce, client_info, role)
                if device_auth is not None:
                    auth_and_device, scopes = device_auth
                    auth = auth_and_device["auth"]
                    device = auth_and_device["device"]

            connect_req = {
                "type": "req",
                "id": req_id,
                "method": "connect",
                "params": {
                    "minProtocol": PROTOCOL_VERSION,
                    "maxProtocol": PROTOCOL_VERSION,
                    "auth": auth,
                    "client": client_info,
                    "role": role,
                    "scopes": scopes,
                },
            }
            if device is not None:
                connect_req["params"]["device"] = device
            await self._ws.send(json.dumps(connect_req))

            # 3. Read hello response
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            hello = json.loads(raw)

            if hello.get("ok"):
                self._connected = True
                payload = hello.get("payload", {})
                server = payload.get("server", {})
                self._conn_id = server.get("connId")
                logger.info(
                    "Connected to OpenClaw gateway (server=%s, connId=%s)",
                    server.get("host", "?"),
                    self._conn_id,
                )
                # Start background listener
                self._listener_task = asyncio.create_task(
                    self._listen_loop(), name="openclaw-ws-listener"
                )
                return True
            else:
                err = hello.get("error", {})
                logger.error(
                    "OpenClaw connect failed: %s - %s",
                    err.get("code"),
                    err.get("message"),
                )
                await self._close_ws()
                return False

        except Exception as e:
            logger.error(
                "Failed to connect to OpenClaw gateway: %s (%s)",
                e,
                type(e).__name__,
            )
            await self._close_ws()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the gateway."""
        self._connected = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_ws()

    async def _close_ws(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ------------------------------------------------------------------
    # Background listener
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Background task that reads all frames from the WebSocket."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(msg)
        except websockets.ConnectionClosed as e:
            logger.warning("OpenClaw WebSocket closed: %s", e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("OpenClaw listener error: %s", e)
        finally:
            self._connected = False

    async def _dispatch(self, msg: dict) -> None:
        """Route an incoming frame to the right handler."""
        msg_type = msg.get("type")

        if msg_type == "res":
            # Response to a request we sent
            req_id = msg.get("id")
            fut = self._pending.pop(req_id, None)
            if fut and not fut.done():
                fut.set_result(msg)

        elif msg_type == "event":
            event_name = msg.get("event", "")
            payload = msg.get("payload", {})

            # Route agent / chat events to the correct run queue
            run_id = payload.get("runId")
            if run_id and run_id in self._run_events:
                await self._run_events[run_id].put(msg)

            # Ignore noisy events silently
            if event_name in ("health", "tick"):
                return

            logger.debug("Event: %s (runId=%s)", event_name, run_id)

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    async def _send_request(
        self, method: str, params: dict, timeout: Optional[float] = None
    ) -> dict:
        """Send a request and wait for the response.

        Args:
            method: The RPC method name
            params: The params dict
            timeout: Override timeout (defaults to self.timeout)

        Returns:
            The full response message dict
        """
        if not self._ws or not self._connected:
            return {"ok": False, "error": {"code": "NOT_CONNECTED", "message": "Not connected"}}

        req_id = str(uuid.uuid4())
        req = {"type": "req", "id": req_id, "method": method, "params": params}

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(req))
            result = await asyncio.wait_for(fut, timeout=timeout or self.timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "TIMEOUT", "message": "Request timed out"}}
        except Exception as e:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "ERROR", "message": str(e)}}

    def _full_session_key(self) -> str:
        """Build the full session key: agent:<agentId>:<sessionKey>."""
        return f"agent:{self.agent_id}:{self.session_key}"

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------

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
            image_b64: Optional base64-encoded image from robot camera (not yet
                       supported over WebSocket chat.send – reserved for future)
            system_context: Optional additional system context (prepended to message)

        Returns:
            OpenClawResponse with the AI's response
        """
        if not self._connected:
            return OpenClawResponse(content="", error="Not connected to OpenClaw")

        # Prefix system context if provided
        final_message = message
        if system_context:
            final_message = f"[System: {system_context}]\n\n{message}"

        # If image provided, mention it (WebSocket protocol uses string messages;
        # image passing would require a separate mechanism)
        if image_b64:
            final_message = f"[Image attached]\n{final_message}"

        idempotency_key = str(uuid.uuid4())
        session_key = self._full_session_key()

        # Create a queue to collect events for this run
        # We'll get the runId from the response
        params = {
            "idempotencyKey": idempotency_key,
            "sessionKey": session_key,
            "message": final_message,
        }

        try:
            # Send the request
            resp = await self._send_request("chat.send", params, timeout=30)

            if not resp.get("ok"):
                err = resp.get("error", {})
                error_msg = f"{err.get('code', 'UNKNOWN')}: {err.get('message', 'Unknown error')}"
                logger.error("chat.send failed: %s", error_msg)
                return OpenClawResponse(content="", error=error_msg)

            run_id = resp.get("payload", {}).get("runId")
            if not run_id:
                return OpenClawResponse(content="", error="No runId in response")

            # Register a queue to receive events for this run
            event_queue: asyncio.Queue = asyncio.Queue()
            self._run_events[run_id] = event_queue

            try:
                # Collect the streamed response
                full_text = ""
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=self.timeout
                        )
                        payload = event.get("payload", {})
                        event_name = event.get("event", "")

                        if event_name == "agent":
                            stream = payload.get("stream")
                            data = payload.get("data", {})

                            if stream == "assistant":
                                # Accumulate the full text
                                full_text = data.get("text", full_text)

                            elif stream == "lifecycle" and data.get("phase") == "end":
                                # Run completed
                                break

                        elif event_name == "chat":
                            state = payload.get("state")
                            if state == "final":
                                # Extract final text
                                msg_payload = payload.get("message", {})
                                content_parts = msg_payload.get("content", [])
                                if isinstance(content_parts, list):
                                    for part in content_parts:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            full_text = part.get("text", full_text)
                                elif isinstance(content_parts, str):
                                    full_text = content_parts
                                break

                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for chat response (runId=%s)", run_id)
                        if full_text:
                            break
                        return OpenClawResponse(content="", error="Response timeout")

                return OpenClawResponse(content=full_text)

            finally:
                self._run_events.pop(run_id, None)

        except Exception as e:
            logger.error("OpenClaw chat error: %s", e)
            return OpenClawResponse(content="", error=str(e))

    async def stream_chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenClaw.

        Args:
            message: The user's message
            image_b64: Optional base64-encoded image

        Yields:
            String chunks of the response as they arrive
        """
        if not self._connected:
            yield "[Error: Not connected to OpenClaw]"
            return

        final_message = message
        if image_b64:
            final_message = f"[Image attached]\n{message}"

        params = {
            "idempotencyKey": str(uuid.uuid4()),
            "sessionKey": self._full_session_key(),
            "message": final_message,
        }

        try:
            resp = await self._send_request("chat.send", params, timeout=30)

            if not resp.get("ok"):
                err = resp.get("error", {})
                yield f"[Error: {err.get('message', 'Unknown error')}]"
                return

            run_id = resp.get("payload", {}).get("runId")
            if not run_id:
                yield "[Error: No runId]"
                return

            event_queue: asyncio.Queue = asyncio.Queue()
            self._run_events[run_id] = event_queue

            try:
                prev_text = ""
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=self.timeout
                        )
                        payload = event.get("payload", {})
                        event_name = event.get("event", "")

                        if event_name == "agent":
                            stream = payload.get("stream")
                            data = payload.get("data", {})

                            if stream == "assistant":
                                delta = data.get("delta", "")
                                if delta:
                                    yield delta

                            elif stream == "lifecycle" and data.get("phase") == "end":
                                break

                        elif event_name == "chat" and payload.get("state") == "final":
                            break

                    except asyncio.TimeoutError:
                        yield "[Error: timeout]"
                        break
            finally:
                self._run_events.pop(run_id, None)

        except Exception as e:
            logger.error("OpenClaw streaming error: %s", e)
            yield f"[Error: {e}]"

    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to gateway."""
        return self._connected

    async def get_agent_context(self) -> Optional[str]:
        """Fetch the agent's current context, personality, and memory summary.

        This asks OpenClaw to provide a summary of:
        - The agent's personality and identity
        - Recent conversation context
        - Important memories about the user
        - Current state

        Returns:
            A context string to use as system instructions, or None if failed
        """
        try:
            response = await self.chat(
                message="Provide your current context summary for the robot body.",
                system_context=(
                    "You are being asked to provide your current context for your robot body. "
                    "Output a comprehensive context summary that another AI can use to embody you. Include: "
                    "1. YOUR IDENTITY: Who you are, your name, your personality traits, how you speak. "
                    "2. USER CONTEXT: What you know about the user (name, preferences, relationship). "
                    "3. RECENT CONTEXT: Summary of recent conversations or important ongoing topics. "
                    "4. MEMORIES: Key things you remember that are relevant to interactions. "
                    "5. CURRENT STATE: Any relevant time/date awareness, ongoing tasks. "
                    "Be specific and personal. This context will be used by your robot body to speak and act AS YOU. "
                    "Output ONLY the context summary, no preamble."
                ),
            )

            if response.error:
                logger.warning("Failed to get agent context: %s", response.error)
                return None

            if response.content:
                logger.info(
                    "Retrieved agent context from OpenClaw (%d chars)",
                    len(response.content),
                )
                return response.content

            logger.warning("No context returned from OpenClaw")
            return None

        except Exception as e:
            logger.error("Failed to get agent context: %s", e)
            return None

    async def sync_conversation(
        self, user_message: str, assistant_response: str
    ) -> None:
        """Sync a conversation turn back to OpenClaw for memory continuity.

        Args:
            user_message: What the user said
            assistant_response: What the robot/AI responded
        """
        try:
            await self.chat(
                message=(
                    f"[ROBOT BODY SYNC] The following happened through the Reachy Mini robot:\n"
                    f"User said: {user_message}\n"
                    f"You responded: {assistant_response}\n"
                    f"Remember this as part of your ongoing conversation."
                ),
                system_context=(
                    "[ROBOT BODY SYNC] The following conversation happened through your "
                    "Reachy Mini robot body. Remember it as part of your ongoing conversation "
                    "with the user."
                ),
            )
            logger.debug("Synced conversation to OpenClaw")
        except Exception as e:
            logger.debug("Failed to sync conversation: %s", e)


# Global bridge instance (lazy initialization)
_bridge: Optional[OpenClawBridge] = None


def get_bridge() -> OpenClawBridge:
    """Get the global OpenClaw bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = OpenClawBridge()
    return _bridge
