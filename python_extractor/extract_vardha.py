#!/usr/bin/env python3
"""
Vardha Hindi Dictionary PDF Extractor
=====================================
Extracts Hindi words, meanings, and synonyms from the Vardha Hindi Dictionary PDF.
Outputs valid_words.json ready for Android Room DB seeding.

Usage:
    pip install pdfplumber
    python extract_vardha.py "vardha hindi dictionary.pdf"

The script tries two strategies:
  1. pdfplumber (preferred — better table/layout parsing)
  2. PyMuPDF/fitz fallback (if pdfplumber fails)
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
DEVANAGARI_WORD = re.compile(r"[\u0900-\u097F\u0964\u0965]+")
JUNK_PATTERN = re.compile(
    r"[A-Za-z0-9@#$%^&*(){}\[\]<>|\\~`!+=_\u2022\u2023\u25CF\u25CB\ufffd]"
)
WHITESPACE_COLLAPSE = re.compile(r"\s+")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

BROKEN_LIGATURE_MAP = {
    "\u0928\u094d\u200d": "\u0928\u094d",
    "\u0915\u094d\u200d": "\u0915\u094d",
    "\u0930\u094d\u200d": "\u0930\u094d",
    "\u0924\u094d\u200d": "\u0924\u094d",
}

COMMON_SEPARATORS = ["—", "–", "-", ":", "=", "।"]
SYNONYM_MARKERS = [
    "पर्यायवाची", "पर्या.", "समानार्थी", "समान अर्थ",
    "syn", "पर्याय", "अन्य नाम",
]
MEANING_MARKERS = [
    "अर्थ", "भावार्थ", "तात्पर्य", "meaning", "defn",
]


def clean_devanagari(text: str) -> str:
    """Aggressively clean text to retain only valid Devanagari content."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    for broken, fixed in BROKEN_LIGATURE_MAP.items():
        text = text.replace(broken, fixed)

    text = CONTROL_CHARS.sub("", text)

    text = re.sub(r"[A-Za-z]{3,}", " ", text)

    text = re.sub(r"\d{3,}", " ", text)

    text = re.sub(r"[(){}\[\]<>|\\~`@#$%^&*+=]", " ", text)

    text = WHITESPACE_COLLAPSE.sub(" ", text).strip()
    return text


def is_valid_hindi_word(word: str) -> bool:
    """Check if a string is a plausible Hindi word."""
    if not word or len(word) < 2:
        return False
    devanagari_chars = len(DEVANAGARI_RANGE.findall(word))
    total_alpha = sum(1 for c in word if c.isalpha() or DEVANAGARI_RANGE.match(c))
    if total_alpha == 0:
        return False
    return (devanagari_chars / total_alpha) > 0.7


def extract_fields_from_line(line: str) -> dict | None:
    """Parse a single dictionary line into word/meaning/synonyms."""
    line = clean_devanagari(line)
    if not line or not DEVANAGARI_RANGE.search(line):
        return None

    word = ""
    meaning = ""
    synonyms = ""

    for sep in COMMON_SEPARATORS:
        if sep in line:
            parts = line.split(sep, 2)
            candidate_word = parts[0].strip()
            if is_valid_hindi_word(candidate_word):
                word = candidate_word
                rest = sep.join(parts[1:]).strip() if len(parts) > 1 else ""
                break

    if not word:
        hindi_words = DEVANAGARI_WORD.findall(line)
        hindi_words = [w for w in hindi_words if len(w) >= 2]
        if not hindi_words:
            return None
        word = hindi_words[0]
        rest = line[line.index(word) + len(word):].strip()
    else:
        if "rest" not in dir():
            rest = ""

    rest_lower = rest.lower()
    syn_start = -1
    for marker in SYNONYM_MARKERS:
        idx = rest_lower.find(marker.lower())
        if idx != -1:
            syn_start = idx
            synonyms = rest[idx + len(marker):].strip().strip(":").strip()
            meaning = rest[:idx].strip()
            break

    if syn_start == -1:
        meaning = rest

    for sep in COMMON_SEPARATORS:
        meaning = meaning.lstrip(sep)
        synonyms = synonyms.lstrip(sep)

    meaning = meaning.strip()
    synonyms = synonyms.strip()

    if not is_valid_hindi_word(word):
        return None

    return {
        "word": word,
        "meaning": meaning if meaning else "",
        "synonyms": synonyms if synonyms else "",
        "category": "General",
    }


def extract_with_pdfplumber(pdf_path: str) -> list[dict]:
    """Extract dictionary entries using pdfplumber."""
    import pdfplumber

    entries = []
    seen_words = set()

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            print(f"  [pdfplumber] Processing page {i + 1}/{total}...", end="\r")
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                entry = extract_fields_from_line(line)
                if entry and entry["word"] not in seen_words:
                    seen_words.add(entry["word"])
                    entries.append(entry)

    print()
    return entries


def extract_with_fitz(pdf_path: str) -> list[dict]:
    """Fallback extraction using PyMuPDF (fitz)."""
    import fitz

    entries = []
    seen_words = set()

    doc = fitz.open(pdf_path)
    total = len(doc)
    for i in range(total):
        print(f"  [PyMuPDF] Processing page {i + 1}/{total}...", end="\r")
        page = doc[i]
        text = page.get_text("text")
        if not text:
            continue

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            entry = extract_fields_from_line(line)
            if entry and entry["word"] not in seen_words:
                seen_words.add(entry["word"])
                entries.append(entry)

    doc.close()
    print()
    return entries


def post_process(entries: list[dict]) -> list[dict]:
    """Final cleanup pass on extracted entries."""
    cleaned = []
    for entry in entries:
        w = entry["word"].strip()
        if len(w) < 2:
            continue
        if not is_valid_hindi_word(w):
            continue
        if JUNK_PATTERN.search(w):
            continue

        entry["word"] = w
        entry["meaning"] = clean_devanagari(entry["meaning"])
        entry["synonyms"] = clean_devanagari(entry["synonyms"])
        cleaned.append(entry)

    cleaned.sort(key=lambda e: e["word"])
    return cleaned


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_vardha.py <path_to_pdf>")
        print('Example: python extract_vardha.py "vardha hindi dictionary.pdf"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting from: {pdf_path}")

    entries = []
    try:
        print("Trying pdfplumber...")
        entries = extract_with_pdfplumber(pdf_path)
        print(f"  pdfplumber extracted {len(entries)} raw entries.")
    except ImportError:
        print("  pdfplumber not installed, trying PyMuPDF...")
    except Exception as e:
        print(f"  pdfplumber failed: {e}. Trying PyMuPDF fallback...")

    if not entries:
        try:
            entries = extract_with_fitz(pdf_path)
            print(f"  PyMuPDF extracted {len(entries)} raw entries.")
        except ImportError:
            print("Error: Neither pdfplumber nor PyMuPDF is installed.")
            print("  pip install pdfplumber   OR   pip install PyMuPDF")
            sys.exit(1)
        except Exception as e:
            print(f"Error: PyMuPDF also failed: {e}")
            sys.exit(1)

    entries = post_process(entries)
    print(f"After cleanup: {len(entries)} valid entries.")

    output_path = Path(pdf_path).parent / "valid_words.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Output written to: {output_path}")
    print(f"Sample entries:")
    for entry in entries[:5]:
        print(f"  {entry['word']}: {entry['meaning'][:60]}...")


if __name__ == "__main__":
    main()
