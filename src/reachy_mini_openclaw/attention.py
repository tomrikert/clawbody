"""Quiet, consent-driven attention notifications for ClawBody Local."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TapDetector:
    """Detect a deliberate double tap from Reachy's head IMU."""

    acceleration_delta: float = 2.4
    gyroscope_threshold: float = 0.55
    min_gap: float = 0.12
    max_gap: float = 0.85
    refractory: float = 1.5
    baseline_alpha: float = 0.04
    _baseline: float | None = None
    _first_impact_at: float | None = None
    _last_impact_at: float = -100.0
    _refractory_until: float = 0.0

    def feed(self, imu: dict[str, Any] | None, now: float | None = None) -> bool:
        if not imu:
            return False
        now = time.monotonic() if now is None else now
        acceleration = imu.get("accelerometer") or []
        gyroscope = imu.get("gyroscope") or []
        if len(acceleration) != 3 or len(gyroscope) != 3:
            return False
        accel_norm = math.sqrt(sum(float(value) ** 2 for value in acceleration))
        gyro_norm = math.sqrt(sum(float(value) ** 2 for value in gyroscope))
        if self._baseline is None:
            self._baseline = accel_norm
            return False
        delta = abs(accel_norm - self._baseline)
        if delta < self.acceleration_delta:
            self._baseline = (
                (1.0 - self.baseline_alpha) * self._baseline
                + self.baseline_alpha * accel_norm
            )
        impact = delta >= self.acceleration_delta or gyro_norm >= self.gyroscope_threshold
        if not impact or now < self._refractory_until:
            if self._first_impact_at is not None and now - self._first_impact_at > self.max_gap:
                self._first_impact_at = None
            return False
        if now - self._last_impact_at < self.min_gap:
            return False
        self._last_impact_at = now
        if self._first_impact_at is None:
            self._first_impact_at = now
            return False
        gap = now - self._first_impact_at
        self._first_impact_at = None
        if self.min_gap <= gap <= self.max_gap:
            self._refractory_until = now + self.refractory
            return True
        self._first_impact_at = now
        return False


class AttentionClient:
    def __init__(self, base_url: str, token: str, timeout: float = 8.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    async def pending(self) -> dict[str, Any]:
        response = await self._http.get("/v1/pending")
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async def acknowledge(self, event_ids: list[str]) -> None:
        response = await self._http.post("/v1/ack", json={"eventIds": event_ids})
        response.raise_for_status()

    async def close(self) -> None:
        await self._http.aclose()


def format_attention(snapshot: dict[str, Any]) -> str:
    needs_attention = snapshot.get("needsAttention") or []
    actions_taken = snapshot.get("actionsTaken") or []
    parts: list[str] = []
    if needs_attention:
        parts.append("Here's what needs your attention.")
        for item in needs_attention[:6]:
            source = "Gmail" if item.get("source") == "gmail" else "Annexus"
            sender = str(item.get("sender") or "unknown sender")
            subject = str(item.get("subject") or "no subject")
            parts.append(f"{source}, from {sender}: {subject}.")
        remaining = len(needs_attention) - 6
        if remaining > 0:
            parts.append(f"There {'is' if remaining == 1 else 'are'} {remaining} more item{'s' if remaining != 1 else ''}.")
    if actions_taken:
        parts.append("Actions taken.")
        parts.extend(str(action) for action in actions_taken[:6])
        remaining = len(actions_taken) - 6
        if remaining > 0:
            parts.append(f"There {'is' if remaining == 1 else 'are'} {remaining} more action{'s' if remaining != 1 else ''}.")
    return " ".join(parts) or "There is nothing waiting for your attention."


class AttentionController:
    def __init__(
        self,
        *,
        robot: Any,
        movement_manager: Any,
        handler: Any,
        client: AttentionClient,
        poll_seconds: float = 60.0,
        tap_detector: TapDetector | None = None,
    ) -> None:
        self.robot = robot
        self.movement_manager = movement_manager
        self.handler = handler
        self.client = client
        self.poll_seconds = max(5.0, poll_seconds)
        self.tap_detector = tap_detector or TapDetector()
        self._snapshot: dict[str, Any] = {}
        self._last_poll = -self.poll_seconds
        self._stopping = False

    async def run(self) -> None:
        try:
            while not self._stopping:
                now = time.monotonic()
                if now - self._last_poll >= self.poll_seconds:
                    self._last_poll = now
                    try:
                        self._snapshot = await self.client.pending()
                        self.movement_manager.set_attention_pending(
                            bool(self._snapshot.get("hasPending"))
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("Attention queue poll failed: %s", exc)

                if self._snapshot.get("hasPending"):
                    status = self.movement_manager.get_status()
                    can_accept_tap = not status.get("is_listening") and not status.get("processing")
                    if can_accept_tap and self.tap_detector.feed(self.robot.imu, now):
                        await self._read_pending()
                await asyncio.sleep(0.05 if self._snapshot.get("hasPending") else 0.5)
        finally:
            self.movement_manager.set_attention_pending(False)
            await self.client.close()

    async def _read_pending(self) -> None:
        event_ids = [str(value) for value in self._snapshot.get("eventIds") or []]
        if not event_ids:
            return
        logger.info("Attention double-tap detected; reading %d queued event(s)", len(event_ids))
        text = format_attention(self._snapshot)
        duration = await self.handler.speak_text(text)
        await asyncio.sleep(duration + 0.4)
        await self.client.acknowledge(event_ids)
        self._snapshot = {}
        self.movement_manager.set_attention_pending(False)

    async def stop(self) -> None:
        self._stopping = True
