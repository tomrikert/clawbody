from reachy_mini_openclaw.motion_catalog import (
    EMOTION_FAMILIES,
    UTILITY_RECORDED_MOVES,
    MotionCatalog,
)


class FakeMoves:
    def __init__(self, names):
        self.names = names

    def list_moves(self):
        return self.names

    def get(self, name):
        if name not in self.names:
            raise ValueError(name)
        return name


def catalog():
    item = object.__new__(MotionCatalog)
    item.emotions = FakeMoves(
        ["thoughtful1", "thoughtful2", "laughing2", "inquiring2", "success2", "anxiety1"]
    )
    return item


def full_catalog():
    item = object.__new__(MotionCatalog)
    expressive = [name for variants in EMOTION_FAMILIES.values() for name in variants]
    item.emotions = FakeMoves(expressive + list(UTILITY_RECORDED_MOVES))
    return item


def test_parse_and_strip_valid_directive():
    clean, directive = catalog().parse_response("Let me think. [[emotion:thoughtful2]]")
    assert clean == "Let me think."
    assert directive.name == "thoughtful2"


def test_invalid_directive_is_removed_but_not_played():
    clean, directive = catalog().parse_response("Nope. [[dance:table_flip]]")
    assert clean == "Nope."
    assert directive is None


def test_conservative_question_fallback():
    directive = catalog().infer("What fresh hell is this?")
    assert directive.name == "inquiring2"


def test_explicit_dance_request_has_deterministic_fallback():
    directive = catalog().infer_request("Hey Asmo, do a little dance for me")
    assert directive.kind == "dance"
    assert directive.name == "side_to_side_sway"


def test_negated_dance_request_does_not_move():
    assert catalog().infer_request("Please don't dance right now") is None


def test_spoken_official_dance_name_resolves_exactly():
    directive = catalog().infer_request("Can you try headbanger combo?")
    assert directive.kind == "dance"
    assert directive.name == "headbanger_combo"


def test_spoken_emotion_intent_resolves_without_model_directive():
    directive = catalog().infer_request("Can you express anxiety as an emotion?")
    assert directive.kind == "emotion"
    assert directive.name == "anxiety1"


def test_console_emotion_map_covers_exactly_81_unique_moves():
    names = [name for variants in EMOTION_FAMILIES.values() for name in variants]
    assert len(names) == 81
    assert len(set(names)) == 81


def test_utility_recordings_are_not_presented_as_emotions():
    item = full_catalog()
    assert len(item.list("emotion")) == 81
    assert item.list("utility") == sorted(UTILITY_RECORDED_MOVES)
    assert not (set(item.list("emotion")) & UTILITY_RECORDED_MOVES)


def test_all_installed_emotions_have_semantic_family():
    item = full_catalog()
    assert item.unmapped_emotions() == []
    assert set(item.emotion_map()) == set(EMOTION_FAMILIES)


def test_family_and_alias_resolve_to_restrained_default():
    item = full_catalog()
    assert item.resolve_emotion("proud") == "proud1"
    assert item.resolve_emotion("victory") == "success1"
    assert item.resolve_emotion("grossed out") == "disgusted1"


def test_exact_numbered_variant_remains_addressable():
    assert full_catalog().resolve_emotion("proud three") == "proud3"
    assert full_catalog().resolve_emotion("proud3") == "proud3"


def test_specific_yes_no_moods_resolve():
    item = full_catalog()
    assert item.resolve_emotion("sad no") == "no_sad1"
    assert item.resolve_emotion("excited no") == "no_excited1"
    assert item.resolve_emotion("sad yes") == "yes_sad1"


def test_response_directive_accepts_semantic_family():
    clean, directive = full_catalog().parse_response("Indeed. [[emotion:proud]]")
    assert clean == "Indeed."
    assert directive.name == "proud1"


def test_explicit_request_resolves_broad_emotion_aliases():
    directive = full_catalog().infer_request("Show me how grossed out you are")
    assert directive == type(directive)("emotion", "disgusted1")
