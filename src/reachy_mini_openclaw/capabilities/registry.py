"""Runtime capability registry for dances/emotions.

ClawBody can run in environments where optional Reachy Mini packages
are not installed (e.g. dev laptop/devbox). This module provides a small,
robust way to detect what's available at runtime.

Design goals:
- Never crash when optional deps are missing.
- Provide a uniform interface for listing and instantiating dances.
- Provide a place to expand emotion support beyond built-in macros.

Notes:
- Reachy Mini Desktop/Daemon also exposes recorded move datasets (emotions/dances)
  via localhost:8000. We treat those as additional capabilities when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional

# Recorded move datasets exposed by Reachy Mini daemon (localhost:8000)
DEFAULT_RECORDED_EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DEFAULT_RECORDED_DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"


@dataclass(frozen=True)
class CapabilityReport:
    dances_available: bool
    dance_names: list[str]
    emotions_available: bool
    emotion_names: list[str]
    notes: list[str]


def _safe_import(module: str):
    try:
        return import_module(module)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dances library (Python package) detection
# ---------------------------------------------------------------------------

def _get_dances_available_moves() -> dict[str, Any] | None:
    """Return AVAILABLE_MOVES mapping from reachy_mini_dances_library, if present.

    The dances library has had multiple public layouts:
    - reachy_mini_dances_library.collection.dance: AVAILABLE_MOVES (current)
    - reachy_mini_dances_library.dances: callable factories (older)

    We normalize to a dict-like mapping when possible.
    """
    mod = _safe_import("reachy_mini_dances_library.collection.dance")
    if mod is not None and hasattr(mod, "AVAILABLE_MOVES"):
        moves = getattr(mod, "AVAILABLE_MOVES")
        if isinstance(moves, dict):
            return moves
    return None


def list_dances() -> list[str]:
    """List dance names from reachy_mini_dances_library if installed."""
    moves = _get_dances_available_moves()
    if moves is not None:
        return sorted(moves.keys())

    # Fallback: older layout with callable symbols under reachy_mini_dances_library.dances
    mod = _safe_import("reachy_mini_dances_library.dances")
    if mod is None:
        return []

    names: list[str] = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if callable(obj):
            names.append(name)
    return sorted(set(names))


def get_dance_factory(name: str) -> Optional[Callable[[], Any]]:
    """Return a 0-arg factory that creates a dance Move instance, if available."""
    moves = _get_dances_available_moves()
    if moves is not None:
        # Preferred API: DanceMove(move_name)
        dance_move_mod = _safe_import("reachy_mini_dances_library.dance_move")
        if dance_move_mod is None or not hasattr(dance_move_mod, "DanceMove"):
            return None
        if name not in moves:
            return None
        DanceMove = getattr(dance_move_mod, "DanceMove")

        def _factory():
            return DanceMove(name)

        return _factory

    # Fallback older layout
    mod = _safe_import("reachy_mini_dances_library.dances")
    if mod is None or not hasattr(mod, name):
        return None

    obj = getattr(mod, name)
    if not callable(obj):
        return None

    def _factory():
        return obj()

    return _factory


# ---------------------------------------------------------------------------
# Recorded move datasets (Reachy Mini daemon)
# ---------------------------------------------------------------------------

def _http_get_json(url: str):
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(url, timeout=2.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def list_recorded_emotions(dataset_name: str = DEFAULT_RECORDED_EMOTIONS_DATASET) -> list[str]:
    """List recorded emotion names from the Reachy Mini daemon (if available)."""
    data = _http_get_json(f"http://localhost:8000/api/move/recorded-move-datasets/list/{dataset_name}")
    if isinstance(data, list):
        return sorted([str(x) for x in data])
    return []


def list_recorded_dances(dataset_name: str = DEFAULT_RECORDED_DANCES_DATASET) -> list[str]:
    """List recorded dances from the Reachy Mini daemon (if available)."""
    data = _http_get_json(f"http://localhost:8000/api/move/recorded-move-datasets/list/{dataset_name}")
    if isinstance(data, list):
        return sorted([str(x) for x in data])
    return []


def play_recorded_move(dataset_name: str, move_name: str) -> bool:
    """Ask the Reachy Mini daemon to play a recorded move."""
    try:
        import urllib.request

        req = urllib.request.Request(
            f"http://localhost:8000/api/move/play/recorded-move-dataset/{dataset_name}/{move_name}",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            status = int(getattr(r, "status", 200))
            return 200 <= status < 300
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Emotions (SDK-native; rare) detection
# ---------------------------------------------------------------------------

def list_emotions() -> list[str]:
    """List emotion names, if a Reachy Mini emotion module is installed.

    The Reachy Mini SDK may expose emotions in different ways depending on
    version. We try a couple of conventional locations.

    If nothing is detected, return an empty list.
    """
    candidates = [
        "reachy_mini.emotions",  # hypothetical
        "reachy_mini.emotion",  # hypothetical
    ]

    for module_name in candidates:
        mod = _safe_import(module_name)
        if mod is None:
            continue
        names: list[str] = []
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name, None)
            if callable(obj):
                names.append(name)
        return sorted(set(names))

    return []


def capabilities_report(
    *,
    macro_emotions: Optional[list[str]] = None,
    macro_dances: Optional[list[str]] = None,
) -> CapabilityReport:
    dance_names = list_dances()
    emotion_names = list_emotions()

    # Merge daemon-recorded datasets when available
    rec_emotions = list_recorded_emotions()
    if rec_emotions:
        emotion_names = sorted(set(emotion_names + rec_emotions))

    rec_dances = list_recorded_dances()
    if rec_dances:
        dance_names = sorted(set(dance_names + rec_dances))

    notes: list[str] = []
    if not dance_names:
        notes.append("No dances detected; using macro fallback")
    if not emotion_names:
        notes.append("No emotions detected; using macro fallback")

    # Include macro lists as a convenience for UIs / debugging
    if macro_dances:
        dance_names = sorted(set(dance_names + macro_dances))
    if macro_emotions:
        emotion_names = sorted(set(emotion_names + macro_emotions))

    return CapabilityReport(
        dances_available=bool(dance_names),
        dance_names=dance_names,
        emotions_available=bool(emotion_names),
        emotion_names=emotion_names,
        notes=notes,
    )
