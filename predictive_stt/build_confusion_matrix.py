#!/usr/bin/env python3
"""Build a real confusion matrix from corpus.csv (6,379 STT-mistake pairs)
and STT_Error_Dictionary.txt (sentence-level pairs).

Outputs (all in predictive_stt/artifacts/):
  - confusion_matrix_char.json    — char-level P(heard|actual)
  - confusion_matrix_syll.json    — syllable-level P(heard|actual)
  - confusion_phrases.json        — full multi-word phrase swaps with counts
  - top_patterns.md               — human-readable Top-20 patterns
  - per_domain_patterns.json      — domain-specific confusion stats
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

# ─── Devanagari helpers ───
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[\u0900-\u097Fa-zA-Z]+")

# Independent vowels, consonants, signs etc.
DEV_CONSONANT_BASE = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक़ख़ग़ज़ड़ढ़फ़य़"
DEV_VOWEL_SIGN = "\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0946\u0947\u0948\u094a\u094b\u094c\u094d"


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।,!?\"'`\(\)\[\]:;]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Syllabification (Akshara segmentation) for Devanagari
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def syllabify_devanagari(word: str) -> list[str]:
    """Split a Devanagari word into akshara (orthographic syllables).

    Rule of thumb: a syllable = consonant cluster + (optional vowel sign).
    A halant (\u094d) keeps the next consonant attached to the current cluster.
    Standalone vowels (अ, आ, इ, ई, ...) are their own syllable.
    """
    if not word:
        return []

    sylls: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in "\u0905\u0906\u0907\u0908\u0909\u090a\u090b\u090f\u0910\u0913\u0914":
            # Independent vowel
            sylls.append(ch)
            i += 1
            continue
        if not DEVANAGARI_RANGE.match(ch):
            # Non-Devanagari char (digit, punctuation, space)
            sylls.append(ch)
            i += 1
            continue
        # Start a consonant cluster
        cluster = ch
        i += 1
        # Pull in halant + next consonant pairs
        while i < n and word[i] == "\u094d" and i + 1 < n:
            cluster += word[i] + word[i + 1]
            i += 2
        # Pull in optional vowel sign / anusvara / chandra-bindu / visarga
        while i < n and word[i] in "\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0946\u0947\u0948\u094a\u094b\u094c\u0902\u0903\u0901":
            cluster += word[i]
            i += 1
        sylls.append(cluster)
    return sylls


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Word/phrase alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def align_words(heard: str, correct: str) -> list[tuple[str, str, str]]:
    """Return list of (tag, heard_span, correct_span) opcodes."""
    h = WORD.findall(normalize(heard))
    c = WORD.findall(normalize(correct))
    sm = SequenceMatcher(None, h, c, autojunk=False)
    out: list[tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append((tag, " ".join(h[i1:i2]), " ".join(c[j1:j2])))
    return out


def char_pairs_from_aligned_words(h: str, c: str) -> list[tuple[str, str]]:
    """Char-level confusion pairs between two word strings of any length.

    Uses opcodes from SequenceMatcher on chars.
    """
    h = h.replace(" ", "")
    c = c.replace(" ", "")
    if not h or not c:
        return []
    sm = SequenceMatcher(None, h, c, autojunk=False)
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # Emit char-against-char pair for the longest aligned span
        L = max(i2 - i1, j2 - j1)
        for k in range(L):
            hc = h[i1 + k] if i1 + k < i2 else ""
            cc = c[j1 + k] if j1 + k < j2 else ""
            if hc or cc:
                pairs.append((hc, cc))
    return pairs


def syllable_pairs_from_aligned_words(h: str, c: str) -> list[tuple[str, str]]:
    """Syllable-level confusion pairs, Devanagari-aware."""
    if not (DEVANAGARI_RANGE.search(h) and DEVANAGARI_RANGE.search(c)):
        return []
    h_sylls = syllabify_devanagari(h.replace(" ", ""))
    c_sylls = syllabify_devanagari(c.replace(" ", ""))
    if not h_sylls or not c_sylls:
        return []
    sm = SequenceMatcher(None, h_sylls, c_sylls, autojunk=False)
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        L = max(i2 - i1, j2 - j1)
        for k in range(L):
            hs = h_sylls[i1 + k] if i1 + k < i2 else ""
            cs = c_sylls[j1 + k] if j1 + k < j2 else ""
            if hs or cs:
                pairs.append((hs, cs))
    return pairs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Corpus loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_csv(csv_path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows: list[dict] = []
    for _, r in df.iterrows():
        h = str(r.get("heard", "")).strip()
        c = str(r.get("correct", "")).strip()
        d = str(r.get("domain", "General")).strip().lower() or "general"
        if not h or not c or h == c:
            continue
        rows.append({"heard": h, "correct": c, "domain": d})
    return rows


SENTENCE_LINE = re.compile(
    r"^Correct:\s*\[(?P<correct>.+?)\]\s*->\s*STT Misheard as:\s*\[(?P<heard>.+?)\]\s*$"
)


def load_dictionary_txt(txt_path: Path) -> list[dict]:
    rows: list[dict] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = SENTENCE_LINE.match(line.strip())
            if not m:
                continue
            rows.append({
                "heard": m.group("heard"),
                "correct": m.group("correct"),
                "domain": "sentence",
            })
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_matrices(rows: list[dict]) -> dict:
    char_pairs: Counter = Counter()
    syll_pairs: Counter = Counter()
    phrase_pairs: Counter = Counter()
    domain_phrases: dict[str, Counter] = defaultdict(Counter)

    for r in rows:
        opcodes = align_words(r["heard"], r["correct"])
        for tag, h, c in opcodes:
            if not h or not c:
                continue
            phrase_pairs[(h, c)] += 1
            domain_phrases[r["domain"]][(h, c)] += 1

            # char-level confusion
            for cp in char_pairs_from_aligned_words(h, c):
                char_pairs[cp] += 1

            # syllable-level (Devanagari only)
            for sp in syllable_pairs_from_aligned_words(h, c):
                syll_pairs[sp] += 1

    # Convert (actual_char → {heard_char: prob}) format
    char_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    actual_char_totals: Counter = Counter()
    for (hc, cc), n in char_pairs.items():
        if not cc:
            continue
        actual_char_totals[cc] += n
    for (hc, cc), n in char_pairs.items():
        if not cc:
            continue
        char_matrix[cc][hc] = round(n / actual_char_totals[cc], 4)

    syll_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    actual_syll_totals: Counter = Counter()
    for (hs, cs), n in syll_pairs.items():
        if not cs:
            continue
        actual_syll_totals[cs] += n
    for (hs, cs), n in syll_pairs.items():
        if not cs:
            continue
        syll_matrix[cs][hs] = round(n / actual_syll_totals[cs], 4)

    return {
        "char_pairs": char_pairs,
        "syll_pairs": syll_pairs,
        "phrase_pairs": phrase_pairs,
        "char_matrix": char_matrix,
        "syll_matrix": syll_matrix,
        "domain_phrases": domain_phrases,
        "n_rows_processed": len(rows),
    }


def top_n_per_actual(matrix: dict[str, dict[str, float]],
                    n: int = 5,
                    min_prob: float = 0.02) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for actual, dist in matrix.items():
        ranked = sorted(dist.items(), key=lambda x: -x[1])
        filt = [(h, p) for h, p in ranked if p >= min_prob and h != actual]
        if filt:
            out[actual] = filt[:n]
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report writer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def write_top_patterns_md(data: dict, out_path: Path,
                          top_n: int = 20) -> None:
    md: list[str] = []
    md.append("# Top STT Confusion Patterns")
    md.append("")
    md.append(f"Source: **{data['n_rows_processed']}** pair rows from "
              f"`corpus.csv` + `STT_Error_Dictionary.txt`.\n")

    # Top 20 multi-word phrase swaps
    md.append("## Top 20 phrase-level mishearings (Deepgram heard → should be)\n")
    md.append("| # | Heard | Should be | Count |")
    md.append("|---|---|---|---|")
    for i, ((h, c), n) in enumerate(
        sorted(data["phrase_pairs"].items(), key=lambda kv: -kv[1])[:top_n],
        start=1,
    ):
        md.append(f"| {i} | `{h}` | `{c}` | {n} |")
    md.append("")

    # Top 20 syllable confusions
    md.append("## Top 20 Devanagari syllable confusions (actual → heard)\n")
    md.append("| # | Actual | Most-likely heard | P |")
    md.append("|---|---|---|---|")
    rows = []
    for actual, dist in data["syll_matrix"].items():
        for heard, p in dist.items():
            if heard != actual:
                rows.append((actual, heard, p))
    rows.sort(key=lambda r: -r[2])
    for i, (actual, heard, p) in enumerate(rows[:top_n], start=1):
        md.append(f"| {i} | `{actual}` | `{heard}` | {p:.2%} |")
    md.append("")

    # Top 20 character confusions
    md.append("## Top 20 character-level confusions (actual → heard)\n")
    md.append("| # | Actual | Most-likely heard | P |")
    md.append("|---|---|---|---|")
    rows = []
    for actual, dist in data["char_matrix"].items():
        for heard, p in dist.items():
            if heard != actual:
                rows.append((actual, heard, p))
    rows.sort(key=lambda r: -r[2])
    for i, (actual, heard, p) in enumerate(rows[:top_n], start=1):
        md.append(f"| {i} | `{actual}` | `{heard}` | {p:.2%} |")
    md.append("")

    # Per-domain top 3
    md.append("## Top 3 phrase confusions per domain\n")
    md.append("| Domain | # | Heard | Should be | Count |")
    md.append("|---|---|---|---|---|")
    domain_phrases = data["domain_phrases"]
    for dom in sorted(domain_phrases.keys()):
        top3 = sorted(domain_phrases[dom].items(), key=lambda kv: -kv[1])[:3]
        for i, ((h, c), n) in enumerate(top3, start=1):
            md.append(f"| {dom} | {i} | `{h[:35]}` | `{c[:35]}` | {n} |")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="corpus.csv")
    ap.add_argument("--txt", default="STT_Error_Dictionary.txt")
    ap.add_argument("--out", default="predictive_stt/artifacts")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    csv_path = Path(args.csv)
    txt_path = Path(args.txt)

    if csv_path.exists():
        csv_rows = load_csv(csv_path)
        print(f"Loaded {len(csv_rows)} word/phrase pairs from {csv_path}")
        rows.extend(csv_rows)
    if txt_path.exists():
        txt_rows = load_dictionary_txt(txt_path)
        print(f"Loaded {len(txt_rows)} sentence-level pairs from {txt_path}")
        rows.extend(txt_rows)

    if not rows:
        print("No data loaded; aborting.", file=sys.stderr)
        return 1

    data = build_matrices(rows)

    # Compact JSON dumps
    top_syll = top_n_per_actual(data["syll_matrix"], n=5)
    top_char = top_n_per_actual(data["char_matrix"], n=5)

    (out_dir / "confusion_matrix_char.json").write_text(
        json.dumps(top_char, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "confusion_matrix_syll.json").write_text(
        json.dumps(top_syll, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Phrase pairs as list, sorted by count
    phrase_list = [
        {"heard": h, "correct": c, "count": n}
        for (h, c), n in sorted(data["phrase_pairs"].items(),
                                key=lambda kv: -kv[1])
        if n >= 2
    ]
    (out_dir / "confusion_phrases.json").write_text(
        json.dumps(phrase_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    domain_phrases_serial = {
        dom: [{"heard": h, "correct": c, "count": n}
              for (h, c), n in sorted(d.items(), key=lambda kv: -kv[1])
              if n >= 2]
        for dom, d in data["domain_phrases"].items()
    }
    (out_dir / "per_domain_patterns.json").write_text(
        json.dumps(domain_phrases_serial, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_top_patterns_md(data, out_dir / "top_patterns.md")

    print(f"\nUnique phrase confusions: {len(data['phrase_pairs'])}")
    print(f"Unique char confusions   : {len(data['char_pairs'])}")
    print(f"Unique syllable confusions: {len(data['syll_pairs'])}")
    print(f"Phrase pairs with count>=2: {len(phrase_list)}")
    print(f"\nWrote artifacts to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
