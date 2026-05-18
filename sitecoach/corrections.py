"""Correction engine — TASK 2f, 2g, 2h.

Three layers of post-STT processing, applied in order:
  1. dedupe_loops()          – collapse 3+ consecutive repetitions
  2. dictionary substitution – replace known wrong→right phrases
  3. fuzzy match              – for any token still not in the known
                                vocabulary, pick the closest match in
                                the dictionary if similarity >= threshold
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Trailing punctuation we strip before vocabulary lookup
_PUNCT = re.compile(r"[।.,!?\(\)\[\]\"'`]+")
_LOW_CONF_MARK = re.compile(r"\[\?(.+?)\]")

# Hindi grammatical particles — never substitute these (context dependent)
HINDI_FUNCTION_WORDS = {
    "है", "हैं", "हो", "हूं", "हूँ", "ही", "और", "या", "तो", "भी",
    "का", "की", "के", "को", "ने", "से", "में", "पर", "तक",
    "यह", "वह", "ये", "वे", "इस", "उस", "इन", "उन",
    "मैं", "तू", "तुम", "हम", "आप", "मुझे", "तुझे", "हमें",
    "नहीं", "नही", "ना", "न", "जो", "जब", "कब", "कहाँ", "कहां",
    "क्या", "कौन", "क्यों", "कैसे", "अब", "तब", "फिर",
    "एक", "दो", "बहुत", "थोड़ा", "कुछ", "सब", "सभी",
    "ठीक", "अच्छा", "बुरा", "बड़ा", "छोटा",
}


def _strip_marker(token: str) -> tuple[str, bool]:
    """Remove [?word] wrapper added by TASK 2e and return (clean, was_marked)."""
    m = _LOW_CONF_MARK.match(token)
    if m:
        return m.group(1), True
    return token, False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Loop dedup (TASK 2h)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def dedupe_loops(text: str, min_repeat: int = 3) -> str:
    """Collapse N+ consecutive identical sentences into one.

    Handles both \n-separated and ।-separated Hindi sentences, plus
    consecutive repeated phrases produced by Deepgram glitches.
    """
    if not text:
        return text

    # Split on sentence-ish boundaries while preserving the separators
    parts = re.split(r"(\s*[।\n.!?]\s*)", text)
    sentences = []
    seps = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            sentences.append(p.strip())
        else:
            seps.append(p)
    seps.append("")  # pad

    out_sentences: list[str] = []
    out_seps: list[str] = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        run = 1
        while (i + run < len(sentences)
               and sentences[i + run].strip() == s.strip()
               and s.strip()):
            run += 1
        out_sentences.append(s)
        out_seps.append(seps[i] if i < len(seps) else "")
        i += run if run >= min_repeat else 1
        if run >= min_repeat:
            logger.info("dedupe_loops: collapsed %d repetitions of %r",
                        run, s[:40])

    rebuilt = "".join(
        s + sep for s, sep in zip(out_sentences, out_seps)
    ).strip()

    # Also collapse word-level repetition like "hello hello hello"
    rebuilt = re.sub(r"\b(\S+?)(?:\s+\1\b){2,}", r"\1", rebuilt)
    return rebuilt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2 & 3. Dictionary + fuzzy correction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CorrectionsEngine:
    """Loads stt_corrections_seed.json once and applies it efficiently.

    Strategy:
      - Multi-word entries are matched as longest-first phrases.
      - Single-word entries go into a hash map for O(1) replacement.
      - Unknown tokens are fuzzy-matched against a known-vocab list
        when rapidfuzz is available (similarity threshold default 88).
    """

    def __init__(
        self,
        seed_path: Optional[str | Path] = None,
        extra_vocab: Optional[Iterable[str]] = None,
        min_confidence: float = 0.5,
        fuzzy_threshold: int = 88,
    ) -> None:
        self.min_confidence = min_confidence
        self.fuzzy_threshold = fuzzy_threshold
        self.single_word_map: dict[str, str] = {}
        self.multi_word_pairs: list[tuple[str, str]] = []
        self.known_vocab: set[str] = set()

        if seed_path:
            self._load(Path(seed_path))

        for v in (extra_vocab or []):
            self.known_vocab.add(v.lower())

        try:
            from rapidfuzz import process, fuzz  # noqa: F401
            self._fuzzy_available = True
        except ImportError:
            logger.info("rapidfuzz not installed; fuzzy correction disabled")
            self._fuzzy_available = False

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Corrections seed not found at %s", path)
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            logger.error("Failed to read corrections seed: %s", e)
            return

        for e in entries:
            conf = float(e.get("confidence", 1.0))
            if conf < self.min_confidence:
                continue
            wrong = e["wrongWord"].strip()
            right = e["correctWord"].strip()
            if not wrong or not right:
                continue

            # Defensive: never substitute grammatical particles
            if wrong in HINDI_FUNCTION_WORDS or right in HINDI_FUNCTION_WORDS:
                continue

            if " " in wrong:
                self.multi_word_pairs.append((wrong, right))
            else:
                self.single_word_map[wrong.lower()] = right

            self.known_vocab.add(right.lower())
            for w in right.split():
                self.known_vocab.add(w.lower())

        # Longest first so phrase matches don't get clipped by shorter ones
        self.multi_word_pairs.sort(key=lambda p: -len(p[0]))
        logger.info(
            "CorrectionsEngine loaded: %d single, %d multi-word, %d known vocab",
            len(self.single_word_map),
            len(self.multi_word_pairs),
            len(self.known_vocab),
        )

    # ── Public API ─────────────────────────────────────────

    def apply(self, transcript: str) -> tuple[str, list[dict]]:
        """Return (corrected_transcript, list_of_corrections_applied)."""
        if not transcript:
            return transcript, []

        applied: list[dict] = []
        text = transcript

        # 1. Multi-word phrase substitution (case-insensitive)
        for wrong, right in self.multi_word_pairs:
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            new_text, n = pattern.subn(right, text)
            if n > 0:
                applied.append({
                    "wrong": wrong, "correct": right,
                    "type": "phrase", "count": n,
                })
                text = new_text

        # 2. Single-word substitution
        out_tokens: list[str] = []
        for raw in text.split():
            tok, was_marked = _strip_marker(raw)
            clean = _PUNCT.sub("", tok).lower()
            replacement = self.single_word_map.get(clean)
            if replacement:
                applied.append({
                    "wrong": tok, "correct": replacement, "type": "word",
                })
                # Preserve trailing punctuation
                trailing = tok[len(clean):]
                out_tokens.append(replacement + trailing)
            elif was_marked and self._fuzzy_available:
                fuzzy = self._fuzzy_lookup(clean)
                if fuzzy:
                    applied.append({
                        "wrong": tok, "correct": fuzzy,
                        "type": "fuzzy",
                    })
                    out_tokens.append(fuzzy)
                else:
                    out_tokens.append(tok)
            else:
                out_tokens.append(tok)

        return " ".join(out_tokens), applied

    # ── Internal helpers ───────────────────────────────────

    def _fuzzy_lookup(self, token: str) -> Optional[str]:
        if not self._fuzzy_available or not token:
            return None
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            token,
            self.known_vocab,
            scorer=fuzz.ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        if result is None:
            return None
        match, _score, _idx = result
        return match


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mark_low_confidence(text: str) -> list[str]:
    """Extract the list of words still wrapped as [?word] after correction."""
    return _LOW_CONF_MARK.findall(text)
