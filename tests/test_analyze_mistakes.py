"""Tests for python_extractor/analyze_mistakes.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_extractor"))

from analyze_mistakes import (  # noqa: E402
    HINDI_FUNCTION_WORDS,
    classify_pair,
    discover_diff_pairs,
    _is_quality_csv_pair,
    split_bullets,
    normalize,
    looks_like_proper_noun,
    looks_like_abbreviation,
)


class TestClassifier:
    def test_proper_noun_known(self):
        assert classify_pair("जगह रहा", "अजगरहा") == "PROPER_NOUN"
        assert classify_pair("दिल्ली बेरी", "डेल्हीवेरी") == "PROPER_NOUN"

    def test_abbreviation_known(self):
        assert classify_pair("PhD विभाग", "PHE विभाग") == "ABBREVIATION"
        assert classify_pair("एसएससी", "ईएसआईसी") == "ABBREVIATION"

    def test_english_hindi_mix(self):
        assert classify_pair("meter", "मीटर") == "ENGLISH_HINDI_MIX"
        assert classify_pair("phone", "फोन") == "ENGLISH_HINDI_MIX"

    def test_number(self):
        assert classify_pair("पांच बज", "5 लाख") == "NUMBER"

    def test_homophone_default(self):
        # Unknown but plausible STT error → defaults to HOMOPHONE
        cat = classify_pair("कौरी", "कीटाणु")
        assert cat in {"HOMOPHONE", "ACCENT"}


class TestQualityFilter:
    def test_rejects_function_words(self):
        for w in ["है", "की", "तो", "में"]:
            assert _is_quality_csv_pair(w, "ही", count=99) is False
            assert _is_quality_csv_pair("कुछ", w, count=99) is False

    def test_cross_script_accepted(self):
        assert _is_quality_csv_pair("meter", "मीटर", count=3) is True

    def test_same_script_single_token_rejects_low_count(self):
        assert _is_quality_csv_pair("कौरी", "कीटाणु", count=2) is False

    def test_same_script_single_token_accepts_high_count(self):
        # 5+ recurrences and very similar — accepted
        assert _is_quality_csv_pair("लड़की", "लड़कि", count=5) is True

    def test_multi_word_shared_token(self):
        assert _is_quality_csv_pair("पाइप लाइन", "पाइप लेन", count=2) is True


class TestDiscoverDiff:
    def test_finds_simple_substitution(self):
        raw = "मेरे पापा से शायद कट गया"
        proc = "मैंने वापस से शायद कट गया"
        pairs = discover_diff_pairs(raw, proc)
        # Should detect that the first two tokens differ
        assert any("पापा" in w or "मेरे" in w for w, _r in pairs)

    def test_returns_empty_for_identical(self):
        assert discover_diff_pairs("ठीक है", "ठीक है") == []

    def test_ignores_speaker_prefix(self):
        proc = "Speaker 1: ठीक है"
        pairs = discover_diff_pairs("ठीक है", proc)
        assert pairs == []


class TestUtilities:
    def test_normalize_collapses_whitespace(self):
        assert normalize("a   b\t\nc") == "a b c"

    def test_split_bullets(self):
        items = split_bullets("• A• B•C")
        assert items == ["A", "B", "C"]

    def test_function_words_set_present(self):
        for w in ("है", "की", "तो", "क्या"):
            assert w in HINDI_FUNCTION_WORDS

    def test_proper_noun_heuristic(self):
        assert looks_like_proper_noun("अजगरहा") is True
        assert looks_like_proper_noun("Delhivery") is True
        assert looks_like_proper_noun("paani") is False

    def test_abbreviation_heuristic(self):
        assert looks_like_abbreviation("PHE") is True
        assert looks_like_abbreviation("hello") is False
