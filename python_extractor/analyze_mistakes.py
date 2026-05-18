#!/usr/bin/env python3
"""
STT Mistake Analyzer
======================
Reads two data sources:
  1. AI mistake.xlsx — 14 manually documented STT mistakes with
     wrong/correct/reason columns (Hindi).
  2. dialogue_batch_latest (4).csv — 323 rows of (Raw ASR | Processed AI)
     pairs from real call recordings.

Classifies every discovered mistake into one of 10 categories:
  PROPER_NOUN, ABBREVIATION, ENGLISH_HINDI_MIX, HOMOPHONE,
  ACCENT, DIARIZATION, LOOP, CONTEXT_BLIND, NOISE, NUMBER.

Outputs:
  stt_corrections_seed.json — schema:
    [{"wrongWord": "...", "correctWord": "...",
      "category": "...", "confidence": 0.0..1.0}]

Usage:
  python analyze_mistakes.py
  python analyze_mistakes.py --stats
  python analyze_mistakes.py --xlsx "AI mistake.xlsx" \
                             --csv "dialogue_batch_latest (4).csv"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ─── Categories ───
CATEGORIES = [
    "PROPER_NOUN",
    "ABBREVIATION",
    "ENGLISH_HINDI_MIX",
    "HOMOPHONE",
    "ACCENT",
    "DIARIZATION",
    "LOOP",
    "CONTEXT_BLIND",
    "NOISE",
    "NUMBER",
]

# ─── Classification cues ───
# These are seeds taken from AI mistake.xlsx to make classification
# deterministic for well-known cases; everything else falls back to
# heuristic rules.

KNOWN_PROPER_NOUNS = {
    "अजगरहा", "सरकिनी", "अमिरिती", "मडुवा", "गोविंदगढ़",
    "प्रेम नगर", "रीवा", "Delhivery", "डेल्हीवेरी",
}

KNOWN_ABBREVIATIONS = {
    "PHE", "JE", "GIS", "ESIC", "EPF", "MB", "WC", "CM",
    "PWD", "CPWD", "NHAI", "DBL", "JCB", "PCC", "RCC",
    "STP", "DPC", "RTI", "NOC", "GST", "TDS", "ITR", "USB",
    "PhD", "AC", "PG", "SSC",
}

ENGLISH_TECH_WORDS = {
    "payment", "pipeline", "brief", "forward", "duty",
    "pending", "policy", "report", "transfer", "format",
    "block", "good", "support", "passport", "sports",
}

# Known mistake → correction pairs harvested from xlsx with explicit
# category labels. These get confidence 1.0 because they are
# domain-expert verified.
EXPERT_CORRECTIONS: list[tuple[str, str, str]] = [
    # PROPER NOUN
    ("जगह रहा", "अजगरहा", "PROPER_NOUN"),
    ("सर किन्हीं", "सरकिनी", "PROPER_NOUN"),
    ("सेट के नहीं", "सरकिनी", "PROPER_NOUN"),
    ("अमेरिकी", "अमिरिती", "PROPER_NOUN"),
    ("कोलाहा", "अमिरिती", "PROPER_NOUN"),
    ("दिल्ली बेरी", "डेल्हीवेरी", "PROPER_NOUN"),
    # ABBREVIATION
    ("पीएचडी विभाग", "पीएचई विभाग", "ABBREVIATION"),
    ("पीएचडी", "पीएचई", "ABBREVIATION"),
    ("PhD विभाग", "PHE विभाग", "ABBREVIATION"),
    ("PhD", "PHE", "ABBREVIATION"),
    ("पीएसई", "पीएचई", "ABBREVIATION"),
    ("एसी वाले", "पीएचई वाले", "ABBREVIATION"),
    ("एसएससी", "ईएसआईसी", "ABBREVIATION"),
    ("SSC", "ESIC", "ABBREVIATION"),
    ("यूएसबी", "जीआईएस", "ABBREVIATION"),
    ("USB", "GIS", "ABBREVIATION"),
    ("सीए मेल्टलाइन", "सीएम हेल्पलाइन", "ABBREVIATION"),
    ("डब्ल्यूसी पोकलेन", "डब्ल्यूसी पॉलिसी", "ABBREVIATION"),
    ("जीने में कॉन्ट्रैक्टर", "जेई या कॉन्ट्रैक्टर", "ABBREVIATION"),
    ("जीने", "जेई", "ABBREVIATION"),
    # ENGLISH-HINDI MIX
    ("सीमेंट भी लेना", "पेमेंट भी देना", "ENGLISH_HINDI_MIX"),
    ("मेरा सीमेंट", "मेरा पेमेंट", "ENGLISH_HINDI_MIX"),
    ("सीमेंट देखिए", "पेमेंट देखिए", "ENGLISH_HINDI_MIX"),
    ("एक करीब देंगे", "एक ब्रीफ देंगे", "ENGLISH_HINDI_MIX"),
    ("पैसे लाइन", "पाइप लाइन", "ENGLISH_HINDI_MIX"),
    ("पेशेंट लाइन", "पाइप लाइन", "ENGLISH_HINDI_MIX"),
    ("पाइप लेनी", "पाइप लाइन", "ENGLISH_HINDI_MIX"),
    ("फॉर्मेट करिए", "फॉरवर्ड करिए", "ENGLISH_HINDI_MIX"),
    ("रेलवे दिन कर रहा", "रेलवे में ड्यूटी कर रहा", "ENGLISH_HINDI_MIX"),
    ("दिन कर रहा हूं", "ड्यूटी कर रहा हूँ", "ENGLISH_HINDI_MIX"),
    ("थोड़ा सपोर्ट", "थोड़ा पासपोर्ट", "ENGLISH_HINDI_MIX"),
    # HOMOPHONE
    ("मेरे पापा", "मैंने वापस", "HOMOPHONE"),
    ("पापा से शायद", "वापस से शायद", "HOMOPHONE"),
    ("पापा", "वापस", "HOMOPHONE"),
    ("दाई", "भाई", "HOMOPHONE"),
    ("गुड हो जाएगा", "ठीक हो जाएगा", "HOMOPHONE"),
    ("थोड़ा सा गुड", "थोड़ा सा ठीक", "HOMOPHONE"),
    ("नाहर", "नहर", "HOMOPHONE"),
    ("पंचनामा तो सब", "पेंडिंग तो सब", "HOMOPHONE"),
    ("पंचनामा", "पेंडिंग", "HOMOPHONE"),
    ("लिखवा देता हूं", "दिखवा देता हूँ", "HOMOPHONE"),
    ("ब्लॉक पिट जाएगा", "ब्लॉक कट जाएगा", "HOMOPHONE"),
    ("पिट जाएगा", "कट जाएगा", "HOMOPHONE"),
    ("यही आदमी लिखा", "यही आदेश लिखा", "HOMOPHONE"),
    ("कॉफी नहीं है", "हो भी नहीं पा रहा है", "HOMOPHONE"),
    ("संपर्क कॉफी", "संपर्क हो भी", "HOMOPHONE"),
    ("शुभम कौन", "शुभम बोल रहा हूँ", "HOMOPHONE"),
    # ACCENT (regional)
    ("तालमाओ का ही दिया है", "टालमटोल कर ही दिया है", "ACCENT"),
    ("ना चाखथे", "ना चाहते हुए भी", "ACCENT"),
    # NUMBER
    ("पांच बज तक का", "5 लाख तक का", "NUMBER"),
    ("पांच बज", "5 लाख", "NUMBER"),
    ("तीन दिन से, दस दिन से बारिश", "जिस दिन से बारिश", "NUMBER"),
    # CONTEXT-BLIND
    ("कि यह सब वाले हैं", "कि यह सीवेज वाले हैं", "CONTEXT_BLIND"),
    ("यह सब वाले", "यह सीवेज वाले", "CONTEXT_BLIND"),
    ("कल आज छुट्टी में वह", "आज छुट्टी पर हूँ", "CONTEXT_BLIND"),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
DIGITS = re.compile(r"\d")


def normalize(text: str) -> str:
    """Normalize whitespace/Unicode and strip bullet markers."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2022", "•").replace("•", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_bullets(cell: str) -> list[str]:
    """Split a bullet-separated cell into items."""
    if not cell:
        return []
    parts = re.split(r"[\n•]", cell)
    return [p.strip() for p in parts if p.strip()]


def looks_like_proper_noun(word: str) -> bool:
    """Heuristic: capitalised English, or matches known Indian place patterns."""
    if word in KNOWN_PROPER_NOUNS:
        return True
    # Hindi place names often end in -हा, -की, -गढ़, -पुर, -नगर, -ती
    suffixes = ("हा", "की", "नी", "गढ़", "पुर", "नगर", "वाड़ा", "ती")
    if any(word.endswith(s) for s in suffixes) and DEVANAGARI.search(word):
        return True
    if LATIN.match(word) and word[0].isupper() and len(word) > 3:
        return True
    return False


def looks_like_abbreviation(word: str) -> bool:
    if word.upper() in KNOWN_ABBREVIATIONS:
        return True
    if LATIN.match(word) and word.isupper() and 2 <= len(word) <= 6:
        return True
    # Devanagari letter-by-letter abbreviations like पीएचई, ईएसआईसी
    if re.fullmatch(r"(?:[पबजकगलमनरसतथदहयभीएच़ौंोआआोई]{1,3}\s*){2,}",
                    word.replace(" ", "")):
        return False  # too broad — keep manual list instead
    return False


def looks_like_english_word(text: str) -> bool:
    return bool(LATIN.search(text)) or any(
        eng in text.lower() for eng in ENGLISH_TECH_WORDS
    )


def looks_like_number(text: str) -> bool:
    return bool(DIGITS.search(text)) or any(
        n in text for n in ("लाख", "करोड़", "हज़ार", "हजार", "एक", "दो",
                            "तीन", "चार", "पाँच", "पांच", "छह", "सात",
                            "आठ", "नौ", "दस", "बीस", "तीस")
    )


def classify_pair(wrong: str, right: str) -> str:
    """Assign one of the 10 categories to a wrong→right correction pair."""
    wrong_lower = wrong.lower()
    right_lower = right.lower()

    # Quick lookups via expert rules first
    for w_rule, _r_rule, cat in EXPERT_CORRECTIONS:
        if w_rule.lower() == wrong_lower or w_rule.lower() in wrong_lower:
            return cat

    wrong_is_latin = bool(LATIN.search(wrong)) and not DEVANAGARI.search(wrong)
    right_is_devanagari = bool(DEVANAGARI.search(right)) and not LATIN.search(right)

    # English-Hindi mix: Latin source becoming Devanagari (transliteration)
    if wrong_is_latin and right_is_devanagari:
        return "ENGLISH_HINDI_MIX"
    # Or the reverse: Devanagari being rewritten to keep English
    if DEVANAGARI.search(wrong) and LATIN.search(right) and not DEVANAGARI.search(right):
        return "ENGLISH_HINDI_MIX"

    # Proper noun
    if any(looks_like_proper_noun(tok) for tok in right.split()):
        return "PROPER_NOUN"

    # Abbreviation
    if any(looks_like_abbreviation(tok) for tok in right.split()):
        return "ABBREVIATION"
    if any(k in right for k in ("विभाग", "हेल्पलाइन", "पॉलिसी")):
        return "ABBREVIATION"

    # Number
    if looks_like_number(wrong) and looks_like_number(right):
        return "NUMBER"
    if "लाख" in right and "लाख" not in wrong:
        return "NUMBER"

    # English-Hindi mix via well-known tech words
    if looks_like_english_word(right) or any(
        eng in right.lower() for eng in ENGLISH_TECH_WORDS
    ):
        return "ENGLISH_HINDI_MIX"

    # Accent: very long Hindi gibberish becoming clean Hindi
    if (len(wrong.split()) >= 3
            and len(right.split()) <= 4
            and DEVANAGARI.search(wrong)
            and DEVANAGARI.search(right)):
        ratio = SequenceMatcher(None, wrong, right).ratio()
        if ratio < 0.3:
            return "ACCENT"

    # Spelling-variant within Devanagari (e.g. कहाँ <-> कहां, हज़ार <-> हजार)
    if (DEVANAGARI.search(wrong) and DEVANAGARI.search(right)
            and SequenceMatcher(None, wrong, right).ratio() > 0.75):
        return "ACCENT"

    # Default to HOMOPHONE for close-but-wrong words
    return "HOMOPHONE"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data source 1: AI mistake.xlsx
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_xlsx_pairs(xlsx_path: Path) -> list[dict]:
    """Extract wrong→correct pairs from the manual mistake spreadsheet."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas/openpyxl required: pip install pandas openpyxl",
              file=sys.stderr)
        return []

    df = pd.read_excel(xlsx_path)
    wrong_col = "मूल शब्द/वाक्य (जो गलत था)"
    right_col = "सुधारा गया शब्द/वाक्य (जो सही होगा)"
    if wrong_col not in df.columns or right_col not in df.columns:
        print(f"Warning: expected columns not found in {xlsx_path}",
              file=sys.stderr)
        return []

    pairs: list[dict] = []
    for _, row in df.iterrows():
        wrongs = split_bullets(normalize(row[wrong_col]))
        rights = split_bullets(normalize(row[right_col]))
        for w, r in zip(wrongs, rights):
            if not w or not r or w == r:
                continue
            pairs.append({
                "wrongWord": w,
                "correctWord": r,
                "category": classify_pair(w, r),
                "confidence": 1.0,
                "source": "expert_xlsx",
            })
    return pairs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data source 2: dialogue_batch_latest (4).csv — diff discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPEAKER_PREFIX = re.compile(r"^Speaker\s*\d+:\s*", re.IGNORECASE)


def strip_speaker(text: str) -> str:
    return SPEAKER_PREFIX.sub("", text.strip())


def tokenize_hi(text: str) -> list[str]:
    """Token split that keeps Devanagari/Latin words intact."""
    text = normalize(text)
    text = re.sub(r"[।.,!?\-\\(\\)\[\]\"']", " ", text)
    return [t for t in text.split() if t]


def discover_diff_pairs(raw: str, processed: str,
                        min_len: int = 2) -> list[tuple[str, str]]:
    """Use SequenceMatcher to find substituted spans between raw and processed."""
    if not raw or not processed:
        return []

    raw_tokens = tokenize_hi(raw)
    proc_tokens = tokenize_hi(strip_speaker(
        re.sub(r"Speaker\s*\d+:", " ", processed)
    ))

    matcher = SequenceMatcher(None, raw_tokens, proc_tokens, autojunk=False)
    pairs: list[tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        wrong_span = " ".join(raw_tokens[i1:i2]).strip()
        right_span = " ".join(proc_tokens[j1:j2]).strip()
        if not wrong_span or not right_span or wrong_span == right_span:
            continue
        if len(wrong_span) < min_len or len(right_span) < min_len:
            continue
        # Drop spans that are too long — likely two unrelated chunks
        if len(wrong_span.split()) > 6 or len(right_span.split()) > 6:
            continue
        # Drop pure punctuation diffs
        if not (DEVANAGARI.search(wrong_span) or LATIN.search(wrong_span)):
            continue
        pairs.append((wrong_span, right_span))

    return pairs


# Common Hindi grammatical particles / function words.
# We MUST NOT generate one-way corrections for these, because the
# correct form depends entirely on local context.
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


def _is_quality_csv_pair(wrong: str, right: str, count: int) -> bool:
    """Only keep CSV-diff pairs that look like genuine STT mistakes,
    not paraphrases the LLM introduced.

    Heuristics:
      - Reject any pair where either side is a single Hindi function word.
      - Cross-script pairs (Latin → Devanagari) are gold — keep at count>=2.
      - Single Devanagari token → Devanagari token requires count>=3 AND
        phonetic closeness (>=0.5 ratio).
      - Multi-word phrases need shared tokens or phonetic closeness >=0.45.
    """
    wrong = wrong.strip()
    right = right.strip()
    if not wrong or not right or wrong == right:
        return False

    # Reject grammatical-particle substitutions entirely
    if wrong in HINDI_FUNCTION_WORDS or right in HINDI_FUNCTION_WORDS:
        return False

    wrong_has_latin = bool(LATIN.search(wrong)) and not DEVANAGARI.search(wrong)
    right_has_dev = bool(DEVANAGARI.search(right)) and not LATIN.search(right)
    cross_script = wrong_has_latin and right_has_dev

    w_tokens = set(wrong.lower().split())
    r_tokens = set(right.lower().split())

    if cross_script:
        return count >= 2

    # Same-script single-token swap — strictest filter to avoid spurious
    # gender/tense variations like "जाएगा/जाएगी" or "हुआ/हुई".
    if len(w_tokens) == 1 and len(r_tokens) == 1:
        if count < 5:
            return False
        ratio = SequenceMatcher(None, wrong, right).ratio()
        # Either phonetically very close, or completely different
        # (genuine misrecognition like जलज → कमल).
        return ratio >= 0.7 or ratio <= 0.3

    # Multi-word phrases need either shared tokens or phonetic closeness
    if w_tokens & r_tokens:
        return count >= 2
    if SequenceMatcher(None, wrong, right).ratio() >= 0.45:
        return count >= 2

    return False


def load_csv_pairs(csv_path: Path, max_rows: int | None = None) -> list[dict]:
    """Diff-mine wrong→correct candidates from the dialogue CSV.

    Applies a quality filter so paraphrase noise doesn't pollute the seed.
    """
    try:
        import pandas as pd
    except ImportError:
        return []

    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)

    raw_col = "Raw Input (ASR)"
    proc_col = "Processed Output (AI)"
    if raw_col not in df.columns or proc_col not in df.columns:
        return []

    pair_counter: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        for wrong, right in discover_diff_pairs(
            str(row.get(raw_col, "")), str(row.get(proc_col, ""))
        ):
            pair_counter[(wrong, right)] += 1

    entries: list[dict] = []
    for (wrong, right), count in pair_counter.most_common():
        if not _is_quality_csv_pair(wrong, right, count):
            continue
        # Confidence scales with how often the same correction recurred
        confidence = min(0.5 + 0.05 * count, 0.9)
        entries.append({
            "wrongWord": wrong,
            "correctWord": right,
            "category": classify_pair(wrong, right),
            "confidence": round(confidence, 2),
            "source": "csv_diff",
            "occurrences": count,
        })
    return entries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregation + output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def merge_with_expert(base: list[dict]) -> list[dict]:
    """Always prefer expert-curated mappings; append new diff-discoveries."""
    expert = [
        {"wrongWord": w, "correctWord": r, "category": c,
         "confidence": 1.0, "source": "expert_seed"}
        for (w, r, c) in EXPERT_CORRECTIONS
    ]
    seen_keys: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for entry in expert + base:
        key = (entry["wrongWord"].strip().lower(),
               entry["correctWord"].strip().lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(entry)
    return merged


def category_examples(entries: list[dict], n: int = 3) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if len(by_cat[e["category"]]) < n:
            by_cat[e["category"]].append(e)
    return by_cat


def print_stats(entries: list[dict]) -> None:
    counts = Counter(e["category"] for e in entries)
    print(f"\nTotal corrections: {len(entries)}")
    print("Per-category counts:")
    for cat in CATEGORIES:
        print(f"  {cat:20s} {counts.get(cat, 0):>5}")
    other = len(entries) - sum(counts.get(c, 0) for c in CATEGORIES)
    if other:
        print(f"  {'OTHER':20s} {other:>5}")

    print("\nExamples per category (up to 3):")
    examples = category_examples(entries, n=3)
    for cat in CATEGORIES:
        for ex in examples.get(cat, []):
            print(f"  [{cat}] {ex['wrongWord']!r} -> {ex['correctWord']!r}"
                  f" (conf={ex['confidence']})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default="AI mistake.xlsx")
    parser.add_argument("--csv", default="dialogue_batch_latest (4).csv")
    parser.add_argument(
        "--out", default="python_extractor/stt_corrections_seed.json"
    )
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    csv_path = Path(args.csv)

    print(f"Reading expert mistakes from {xlsx_path}...")
    xlsx_pairs = load_xlsx_pairs(xlsx_path) if xlsx_path.exists() else []
    print(f"  -> {len(xlsx_pairs)} pairs")

    print(f"Mining diff candidates from {csv_path}...")
    csv_pairs = load_csv_pairs(csv_path) if csv_path.exists() else []
    print(f"  -> {len(csv_pairs)} pairs")

    merged = merge_with_expert(xlsx_pairs + csv_pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nWritten {len(merged)} corrections to {out_path}")

    if args.stats:
        print_stats(merged)

    return 0


if __name__ == "__main__":
    sys.exit(main())
