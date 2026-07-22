"""Dynamic access to Reachy's official recorded emotion and dance libraries."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from reachy_mini_dances_library.dance_move import DanceMove

EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DIRECTIVE_RE = re.compile(r"\[\[(emotion|dance):([a-zA-Z0-9_-]+)\]\]")
DANCE_REQUEST_RE = re.compile(
    r"\b(?:do|give|show|perform|try|bust out|let(?:'s| us))\b.{0,32}\b(?:dance|moves?)\b"
    r"|\b(?:dance|boogie)\b.{0,24}\b(?:for me|please|a little|now)\b",
    re.I,
)
DANCE_NEGATION_RE = re.compile(r"\b(?:do not|don't|dont|stop|no|not)\b.{0,16}\bdanc", re.I)
DEFAULT_REQUEST_DANCES = (
    "side_to_side_sway",
    "groovy_sway_and_roll",
    "chicken_peck",
)

# The upstream dataset currently contains 85 recordings. Reachy's console calls
# 81 of them emotions; these four are lifecycle/attention utilities rather than
# expressive reactions and are deliberately kept out of the emotion picker.
UTILITY_RECORDED_MOVES = frozenset(
    {"mini-deep-sleep", "toc-toc-toc", "waiting", "wake-mini-up"}
)

# Every expressive recording is assigned to one semantic family. The first
# move is the restrained/default rendering; numbered alternatives remain
# addressable by exact name. This is intentionally explicit so a dataset update
# cannot silently turn an unknown recording into an arbitrary emotion.
EMOTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "amazed": ("amazed1",),
    "anxiety": ("anxiety1",),
    "attentive": ("attentive1", "attentive2"),
    "boredom": ("boredom1", "boredom2"),
    "calming": ("calming1",),
    "cheerful": ("cheerful1",),
    "come": ("come1",),
    "confused": ("confused1",),
    "contempt": ("contempt1",),
    "curious": ("curious1",),
    "dance": ("dance1", "dance2", "dance3"),
    "disgusted": ("disgusted1",),
    "displeased": ("displeased1", "displeased2"),
    "downcast": ("downcast1",),
    "dying": ("dying1",),
    "electric": ("electric1",),
    "enthusiastic": ("enthusiastic1", "enthusiastic2"),
    "exhausted": ("exhausted1",),
    "fear": ("fear1",),
    "frustrated": ("frustrated1",),
    "furious": ("furious1",),
    "go_away": ("go_away1",),
    "grateful": ("grateful1",),
    "helpful": ("helpful1", "helpful2"),
    "impatient": ("impatient1", "impatient2"),
    "incomprehensible": ("incomprehensible2",),
    "indifferent": ("indifferent1",),
    "inquiring": ("inquiring1", "inquiring2", "inquiring3"),
    "irritated": ("irritated1", "irritated2"),
    "laughing": ("laughing1", "laughing2"),
    "lonely": ("lonely1",),
    "lost": ("lost1",),
    "loving": ("loving1",),
    "no": ("no1", "no_excited1", "no_sad1"),
    "oops": ("oops1", "oops2"),
    "proud": ("proud1", "proud2", "proud3"),
    "rage": ("rage1",),
    "relief": ("relief1", "relief2"),
    "reprimand": ("reprimand1", "reprimand2", "reprimand3"),
    "resigned": ("resigned1",),
    "sad": ("sad1", "sad2"),
    "scared": ("scared1",),
    "serenity": ("serenity1",),
    "shy": ("shy1",),
    "sleep": ("sleep1",),
    "success": ("success1", "success2"),
    "surprised": ("surprised1", "surprised2"),
    "thoughtful": ("thoughtful1", "thoughtful2"),
    "tired": ("tired1",),
    "uncertain": ("uncertain1",),
    "uncomfortable": ("uncomfortable1",),
    "understanding": ("understanding1", "understanding2"),
    "welcoming": ("welcoming1", "welcoming2"),
    "yes": ("yes1", "yes_sad1"),
}

EMOTION_ALIASES: dict[str, str] = {
    "astonished": "amazed",
    "impressed": "amazed",
    "wow": "amazed",
    "anxious": "anxiety",
    "nervous": "anxiety",
    "worried": "anxiety",
    "alert": "attentive",
    "focused": "attentive",
    "listening": "attentive",
    "bored": "boredom",
    "soothing": "calming",
    "reassuring": "calming",
    "happy": "cheerful",
    "upbeat": "cheerful",
    "come_here": "come",
    "beckon": "come",
    "invite": "come",
    "puzzled": "confused",
    "disdain": "contempt",
    "dismissive": "contempt",
    "grossed_out": "disgusted",
    "dissatisfied": "displeased",
    "dejected": "downcast",
    "dramatic_death": "dying",
    "energized": "electric",
    "excited": "enthusiastic",
    "eager": "enthusiastic",
    "wiped_out": "exhausted",
    "afraid": "fear",
    "livid": "furious",
    "shoo": "go_away",
    "leave": "go_away",
    "thankful": "grateful",
    "assist": "helpful",
    "baffled": "incomprehensible",
    "dont_understand": "incomprehensible",
    "meh": "indifferent",
    "apathetic": "indifferent",
    "questioning": "inquiring",
    "question": "inquiring",
    "annoyed": "irritated",
    "laugh": "laughing",
    "amused": "laughing",
    "love": "loving",
    "affectionate": "loving",
    "negative": "no",
    "decline": "no",
    "refuse": "no",
    "mistake": "oops",
    "enraged": "rage",
    "relieved": "relief",
    "scold": "reprimand",
    "admonish": "reprimand",
    "give_up": "resigned",
    "accept_defeat": "resigned",
    "unhappy": "sad",
    "frightened": "scared",
    "peaceful": "serenity",
    "bashful": "shy",
    "sleepy": "sleep",
    "victory": "success",
    "celebrating": "success",
    "startled": "surprised",
    "thinking": "thoughtful",
    "contemplative": "thoughtful",
    "unsure": "uncertain",
    "doubtful": "uncertain",
    "awkward": "uncomfortable",
    "uneasy": "uncomfortable",
    "got_it": "understanding",
    "comprehend": "understanding",
    "welcome": "welcoming",
    "greet": "welcoming",
    "agree": "yes",
    "affirmative": "yes",
}

SPECIFIC_EMOTION_ALIASES: dict[str, str] = {
    "excited_no": "no_excited1",
    "no_excited": "no_excited1",
    "sad_no": "no_sad1",
    "no_sad": "no_sad1",
    "sad_yes": "yes_sad1",
    "yes_sad": "yes_sad1",
}


def _normalize_intent(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    for word, digit in (("one", "1"), ("two", "2"), ("three", "3")):
        normalized = re.sub(rf"(?:number_)?{word}(?=_|$)", digit, normalized)
    return re.sub(r"(?<=[a-z])_(?=[123](?:_|$))", "", normalized)


class EmotionMove(Move):
    """Recorded emotion marked as exclusive from autonomous tracking."""

    exclusive_tracking = True

    def __init__(self, name: str, move: Move) -> None:
        self.name = name
        self.move = move

    @property
    def duration(self) -> float:
        return float(self.move.duration)

    def evaluate(
        self, t: float
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        return self.move.evaluate(t)


class DanceCatalogMove(Move):
    """Official procedural dance marked as exclusive from tracking."""

    exclusive_tracking = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.move = DanceMove(name)

    @property
    def duration(self) -> float:
        return float(self.move.duration)

    def evaluate(
        self, t: float
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        return self.move.evaluate(t)


@dataclass(frozen=True)
class MotionDirective:
    kind: str
    name: str


class MotionCatalog:
    """Load, validate, and resolve every move currently published by Reachy."""

    def __init__(
        self,
        emotion_dataset: str = EMOTION_DATASET,
    ) -> None:
        self.emotions = RecordedMoves(emotion_dataset)

    def list(self, kind: str) -> list[str]:
        if kind == "emotion":
            return sorted(set(self.emotions.list_moves()) - UTILITY_RECORDED_MOVES)
        if kind == "utility":
            return sorted(set(self.emotions.list_moves()) & UTILITY_RECORDED_MOVES)
        if kind == "dance":
            return sorted(AVAILABLE_MOVES)
        raise ValueError(f"Unknown motion kind: {kind}")

    def get(self, kind: str, name: str):
        if kind == "emotion":
            resolved = self.resolve_emotion(name)
            if resolved is not None:
                return EmotionMove(resolved, self.emotions.get(resolved))
        if kind == "utility" and name in self.list("utility"):
            return EmotionMove(name, self.emotions.get(name))
        if kind == "dance" and name in AVAILABLE_MOVES:
            return DanceCatalogMove(name)
        raise ValueError(f"Unknown {kind} move: {name}")

    def has(self, kind: str, name: str) -> bool:
        return name in self.list(kind)

    def search(self, query: str, kind: str | None = None) -> dict[str, list[str]]:
        needle = query.casefold()
        kinds = [kind] if kind else ["emotion", "dance"]
        return {
            current: [name for name in self.list(current) if needle in name.casefold()]
            for current in kinds
        }

    def emotion_map(self) -> dict[str, list[str]]:
        """Return semantic families filtered to moves installed at runtime."""
        installed = set(self.list("emotion"))
        return {
            family: [name for name in variants if name in installed]
            for family, variants in EMOTION_FAMILIES.items()
            if any(name in installed for name in variants)
        }

    def unmapped_emotions(self) -> list[str]:
        """Expose upstream additions that need an intentional semantic mapping."""
        mapped = {name for variants in EMOTION_FAMILIES.values() for name in variants}
        return sorted(set(self.list("emotion")) - mapped)

    def resolve_emotion(self, intent: str) -> str | None:
        """Resolve an exact move, family, or human-friendly alias to a move."""
        normalized = _normalize_intent(intent)
        installed = set(self.list("emotion"))
        if normalized in installed:
            return normalized
        specific = SPECIFIC_EMOTION_ALIASES.get(normalized)
        if specific in installed:
            return specific
        family = EMOTION_ALIASES.get(normalized, normalized)
        for name in EMOTION_FAMILIES.get(family, ()):
            if name in installed:
                return name
        return None

    def parse_response(self, text: str) -> tuple[str, MotionDirective | None]:
        matches = list(DIRECTIVE_RE.finditer(text))
        if not matches:
            return text.strip(), None
        match = matches[-1]
        directive = MotionDirective(match.group(1), match.group(2))
        clean = DIRECTIVE_RE.sub("", text).strip()
        if directive.kind == "emotion":
            resolved = self.resolve_emotion(directive.name)
            return clean, MotionDirective("emotion", resolved) if resolved else None
        if not self.has(directive.kind, directive.name):
            return clean, None
        return clean, directive

    def infer(self, text: str) -> MotionDirective | None:
        """Conservative fallback when OpenClaw does not emit a motion directive."""
        lower = text.casefold()
        rules = (
            (("[laugh]", "haha", "lol", "funny"), "emotion", "laughing2"),
            (("sorry", "my fault", "oops"), "emotion", "oops1"),
            (("congrat", "hell yes", "fuck yeah", "we did it"), "emotion", "success2"),
            (("thank you", "thanks"), "emotion", "grateful1"),
            (("not sure", "let me think", "hmm"), "emotion", "thoughtful2"),
            (("what the fuck", "annoying", "irritat"), "emotion", "irritated1"),
        )
        for needles, kind, name in rules:
            if any(needle in lower for needle in needles) and self.has(kind, name):
                return MotionDirective(kind, name)
        if "?" in text and self.has("emotion", "inquiring2"):
            return MotionDirective("emotion", "inquiring2")
        return None

    def infer_request(self, text: str) -> MotionDirective | None:
        """Resolve explicit user motion requests without relying on model formatting."""
        lower = text.casefold()
        normalized = _normalize_intent(lower)

        # Exact catalog names spoken naturally ("try headbanger combo").
        for kind in ("dance", "emotion"):
            for name in self.list(kind):
                if name in normalized:
                    return MotionDirective(kind, name)

        # Resolve all 81 emotion families and their conversational aliases
        # independently of model formatting. Longer phrases win over words.
        if re.search(r"\b(?:act|be|express|show|try|perform|do|look)\b", lower):
            intents = {
                **{name: name for name in EMOTION_FAMILIES},
                **EMOTION_ALIASES,
                **SPECIFIC_EMOTION_ALIASES,
            }
            for intent in sorted(intents, key=len, reverse=True):
                spoken = intent.replace("_", r"[\s_-]+")
                if re.search(rf"\b{spoken}\b", lower):
                    resolved = self.resolve_emotion(intent)
                    if resolved:
                        return MotionDirective("emotion", resolved)

        if DANCE_NEGATION_RE.search(text) or not DANCE_REQUEST_RE.search(text):
            return None
        for name in DEFAULT_REQUEST_DANCES:
            if self.has("dance", name):
                return MotionDirective("dance", name)
        dances = self.list("dance")
        return MotionDirective("dance", dances[0]) if dances else None

    @staticmethod
    def response_hint() -> str:
        families = ", ".join(EMOTION_FAMILIES)
        return (
            "When movement genuinely improves the response, append one hidden directive as the final token: "
            "[[emotion:family_or_exact_name]] or [[dance:exact_name]]. Available emotion families are: "
            f"{families}. Family names resolve to a restrained default; exact numbered variants are also valid. "
            "Use dances sparingly for actual celebration, music, or a requested dance. Do not mention the directive."
        )
