"""Device identity for the OpenClaw gateway (6.x device-auth).

OpenClaw 6.x requires WebSocket clients to present a device identity (an
Ed25519 keypair) to receive operator scopes; device-less connections are
granted an empty scope set and cannot send chat. This module persists an
Ed25519 keypair, derives the device id (sha256 fingerprint of the raw public
key) and signs the gateway connect challenge.

References (openclaw/openclaw):
- src/infra/device-identity.ts (keypair, fingerprint = sha256(raw pubkey).hex,
  signature = Ed25519 over UTF-8 payload, base64url)
- packages/gateway-client/src/device-auth.ts (signed payload format)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

logger = logging.getLogger(__name__)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class DeviceIdentity:
    """Persistent Ed25519 device identity + cached gateway device token."""

    def __init__(self, private_key: Ed25519PrivateKey, path: Path):
        self._private_key = private_key
        self._path = path
        self.public_raw = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        # device id = sha256(raw public key) hex (matches fingerprintPublicKey)
        self.device_id = hashlib.sha256(self.public_raw).hexdigest()
        self.public_key_b64url = _b64url_encode(self.public_raw)
        self.device_token: Optional[str] = None

    @classmethod
    def load_or_create(cls, path: Path) -> "DeviceIdentity":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                pk = Ed25519PrivateKey.from_private_bytes(
                    _b64url_decode(data["privateKey"])
                )
                ident = cls(pk, path)
                ident.device_token = data.get("deviceToken")
                logger.info("Loaded device identity: %s", ident.device_id)
                return ident
            except Exception as e:  # noqa: BLE001
                logger.warning("Device identity unreadable (%s); regenerating", e)
        ident = cls(Ed25519PrivateKey.generate(), path)
        ident._save()
        logger.info("Generated new device identity: %s", ident.device_id)
        return ident

    def _save(self) -> None:
        priv_raw = self._private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        payload = {
            "deviceId": self.device_id,
            "privateKey": _b64url_encode(priv_raw),
            "publicKey": self.public_key_b64url,
            "deviceToken": self.device_token,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.chmod(tmp, 0o600)
        tmp.replace(self._path)

    def set_device_token(self, token: Optional[str]) -> None:
        if token and token != self.device_token:
            self.device_token = token
            self._save()

    def sign_payload(self, payload: str) -> str:
        """Ed25519-sign a UTF-8 payload, return base64url (no padding)."""
        return _b64url_encode(self._private_key.sign(payload.encode("utf-8")))


def build_device_auth_payload_v2(
    *,
    device_id: str,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    signed_at_ms: int,
    token: str,
    nonce: str,
) -> str:
    """Build the V2 device-auth payload (pipe-joined).

    The gateway verifies the signature against both the V3 and V2 payloads, so
    signing V2 is sufficient and avoids the V3-only platform/deviceFamily fields.
    """
    return "|".join(
        [
            "v2",
            device_id,
            client_id,
            client_mode,
            role,
            ",".join(scopes),
            str(signed_at_ms),
            token or "",
            nonce,
        ]
    )
