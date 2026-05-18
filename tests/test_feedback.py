"""Tests for the FeedbackStore JSONL persistence."""

from __future__ import annotations

import json

import pytest

from sitecoach.feedback import FeedbackStore


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(path=tmp_path / "feedback.jsonl")


class TestFeedbackStore:
    def test_record_round_trip(self, store):
        entry = store.record(
            wrong_word="पापा", correct_word="वापस",
            audio_id="aud-1", user_id="alok",
        )
        assert entry["wrongWord"] == "पापा"
        assert entry["correctWord"] == "वापस"

        all_entries = store.all()
        assert len(all_entries) == 1
        assert all_entries[0]["audioId"] == "aud-1"

    def test_rejects_empty(self, store):
        with pytest.raises(ValueError):
            store.record(wrong_word="", correct_word="वापस")
        with pytest.raises(ValueError):
            store.record(wrong_word="कुछ", correct_word="")

    def test_stats(self, store):
        store.record("a", "b", category="USER_REPORT")
        store.record("c", "d", category="USER_REPORT")
        store.record("e", "f", category="ADMIN")
        s = store.stats()
        assert s["total"] == 3
        assert s["by_category"]["USER_REPORT"] == 2
        assert s["by_category"]["ADMIN"] == 1

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "feedback.jsonl"
        FeedbackStore(path=path).record("a", "b")
        FeedbackStore(path=path).record("c", "d")
        # Read raw file to confirm appends, not overwrites
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["wrongWord"] == "a"
        assert json.loads(lines[1])["wrongWord"] == "c"
