"""Tests for the CorrectionsEngine and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sitecoach.corrections import (
    CorrectionsEngine,
    HINDI_FUNCTION_WORDS,
    dedupe_loops,
    mark_low_confidence,
)


@pytest.fixture(scope="module")
def expert_seed(tmp_path_factory):
    """Create a tiny in-memory seed with only the expert-curated pairs."""
    entries = [
        {"wrongWord": "जगह रहा", "correctWord": "अजगरहा",
         "category": "PROPER_NOUN", "confidence": 1.0},
        {"wrongWord": "PhD विभाग", "correctWord": "PHE विभाग",
         "category": "ABBREVIATION", "confidence": 1.0},
        {"wrongWord": "सीमेंट भी लेना", "correctWord": "पेमेंट भी देना",
         "category": "ENGLISH_HINDI_MIX", "confidence": 1.0},
        {"wrongWord": "पंचनामा तो सब", "correctWord": "पेंडिंग तो सब",
         "category": "HOMOPHONE", "confidence": 1.0},
        {"wrongWord": "meter", "correctWord": "मीटर",
         "category": "ENGLISH_HINDI_MIX", "confidence": 0.9},
        # A pair that should be ignored because of function-word filter
        {"wrongWord": "है", "correctWord": "ही",
         "category": "ACCENT", "confidence": 0.9},
        # A low-confidence pair that should be dropped
        {"wrongWord": "अबोलक", "correctWord": "अनोखा",
         "category": "HOMOPHONE", "confidence": 0.3},
    ]
    path = tmp_path_factory.mktemp("seed") / "stt_corrections_seed.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoopDedup:
    def test_collapses_3plus_repeats(self):
        text = "A. A. A. A. B."
        out = dedupe_loops(text)
        assert out.count("A.") == 1
        assert "B." in out

    def test_preserves_normal_text(self):
        text = "मैं आ रहा हूं। ठीक है। फिर मिलेंगे।"
        assert dedupe_loops(text) == text

    def test_word_level_repetition(self):
        text = "hello hello hello world"
        out = dedupe_loops(text)
        assert out.count("hello") == 1


class TestCorrectionsEngine:
    def test_load_filters_function_words_and_low_conf(self, expert_seed):
        engine = CorrectionsEngine(seed_path=expert_seed)
        # The "है -> ही" entry must NOT be loaded
        assert "है" not in engine.single_word_map
        # The 0.3-confidence entry must NOT be loaded (default min=0.5)
        assert "अबोलक" not in engine.single_word_map
        # The good single-word transliteration must be loaded
        assert engine.single_word_map.get("meter") == "मीटर"

    def test_apply_phrase_substitution(self, expert_seed):
        engine = CorrectionsEngine(seed_path=expert_seed)
        out, edits = engine.apply("मेरा PhD विभाग और जगह रहा 26")
        assert "PHE विभाग" in out
        assert "अजगरहा" in out
        assert any(e["correct"] == "अजगरहा" for e in edits)

    def test_apply_english_to_hindi(self, expert_seed):
        engine = CorrectionsEngine(seed_path=expert_seed)
        out, edits = engine.apply("meter ka bill")
        assert "मीटर" in out
        assert any(e["type"] == "word" for e in edits)

    def test_apply_with_low_confidence_marker(self, expert_seed):
        engine = CorrectionsEngine(seed_path=expert_seed)
        out, _ = engine.apply("मैं [?meter] देख रहा हूं")
        # The marker is removed and the word is replaced
        assert "[?" not in out
        assert "मीटर" in out

    def test_apply_no_match_returns_input(self, expert_seed):
        engine = CorrectionsEngine(seed_path=expert_seed)
        out, edits = engine.apply("कोई गलती नहीं है यहाँ")
        assert out == "कोई गलती नहीं है यहाँ"
        assert edits == []


class TestLowConfidenceMarker:
    def test_extract_marked_tokens(self):
        text = "[?abc] और [?xyz] हैं"
        assert mark_low_confidence(text) == ["abc", "xyz"]

    def test_empty_when_no_markers(self):
        assert mark_low_confidence("कोई गलती नहीं") == []


class TestFunctionWordsConstant:
    def test_contains_common_particles(self):
        for w in ("है", "की", "तो", "में", "पर", "क्या"):
            assert w in HINDI_FUNCTION_WORDS
