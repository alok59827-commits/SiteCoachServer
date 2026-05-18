#!/usr/bin/env python3
"""Extract questions + figures from MP ESB Civil Engineering solved
papers PDF and emit:

    output/
        questions.json           — list of question objects
        figures/                  — PNG files referenced from questions.json
        manifest.json             — counts + processing metadata

JSON schema:
    {
      "question_id": int,
      "paper": str,              # which paper the question came from
      "subject": str,
      "sub_topic": str,
      "question_english": str,
      "question_hindi": str,
      "options": {"A": str, "B": str, "C": str, "D": str},
      "correct_answer": "A|B|C|D",
      "correct_option_text": str,
      "detailed_explanation_english": str,
      "detailed_explanation_hindi": str,
      "figure": str | null         # relative path to PNG, e.g. "figures/q42_1.png"
    }
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Tesseract uses OpenMP internally; with multiple threads it actually
# slows down inside container CPU caps. Force single-thread per process
# and parallelize at the Python level instead.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

PDF_PATH = "MP_ESB_GROUP_3_SUB_ENGINEER_CIVIL_ENGINEERING_SOLVED_PAPERS_HINDI.pdf"
OUTPUT_DIR = Path("pdf_to_json/output")
DPI = 200

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1 — render pages and OCR them
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ocr_one_page(pdf_path: str, page_no: int, lang: str = "hin+eng") -> tuple[int, str]:
    """Render page to image, OCR with hin+eng, return (page_no, text)."""
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()

    img = Image.open(io.BytesIO(img_bytes))
    try:
        text = pytesseract.image_to_string(img, lang=lang)
    except Exception as e:
        logger.warning("OCR failed for page %d: %s", page_no, e)
        text = ""
    return page_no, text


def ocr_range(pdf_path: str, pages: list[int],
              workers: int = 4, lang: str = "hin+eng") -> dict[int, str]:
    """Threaded OCR — each tesseract subprocess is pinned to 1 OMP
    thread via OMP_THREAD_LIMIT=1 (set at module level), so we can
    safely parallelize across the available CPU cores at the Python
    level.
    """
    results: dict[int, str] = {}
    start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(ocr_one_page, pdf_path, p, lang): p
            for p in pages
        }
        for fut in as_completed(futures):
            pno, text = fut.result()
            results[pno] = text
            completed += 1
            if completed % 10 == 0 or completed == len(pages):
                elapsed = time.time() - start
                rate = completed / max(elapsed, 0.1)
                eta = (len(pages) - completed) / max(rate, 0.001)
                print(f"  OCR progress: {completed}/{len(pages)} "
                      f"({rate:.2f} pages/s, ETA {eta / 60:.1f} min)",
                      flush=True)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2 — parse OCR text into questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Header that appears at the top of each paper section
PAPER_HEADER = re.compile(
    r"(?:MP\s*(?:ESB|VYAPAM)\s*[^\n]*?(?:Sub\s*Engineer|Draftsman|Surveyor)[^\n]*?)\s*-?\s*\d{4}",
    re.IGNORECASE,
)
# Only treat "Subject : <Name>" as a header when it's near the top of the
# page (line index < 8) and the value is short and looks like a subject name
SUBJECT_LINE = re.compile(r"^\s*Subject\s*[:\-]\s*([^\n]{2,60})\s*$", re.MULTILINE)
KNOWN_SUBJECTS = {
    "General Knowledge", "General English", "General Reasoning",
    "General Math", "General Maths", "General Hindi", "General Science",
    "General Computer", "Civil Engineering", "Civil Engineering (Diploma)",
    "Draftman Civil", "Drafting (Civil)", "Surveyor", "Survey",
    "Mechanics", "Strength of Materials", "RCC", "Hydraulics",
    "Environmental Engineering", "Transportation Engineering",
    "Building Construction", "Estimation and Costing",
    "Soil Mechanics", "Concrete Technology", "PWD Manual",
}


def _is_plausible_subject(value: str) -> bool:
    value = value.strip()
    if not value or len(value) > 60:
        return False
    if value in KNOWN_SUBJECTS:
        return True
    # Heuristic: short Latin-only string with title-cased words
    if (
        re.match(r"^[A-Z][A-Za-z &/()]+$", value)
        and len(value.split()) <= 6
    ):
        return True
    return False

# Question number can be misread by OCR: '1' → 'l' or 'I', '0' → 'O' or 'o'.
# We use a more lenient anchor and require the digit (or look-alike) at the
# very start of a line followed by '.' or ')'.
QUESTION_NUM = re.compile(r"(?m)^\s*([0-9lIoO]{1,3})\s*[.\)]\s+(?=\S)")

# Options must look like "(a) text" at line start (or multiple on one line).
OPTION_MARKER = re.compile(
    r"\(\s*([a-dA-Dअबसद१२३४])\s*\)",
    re.IGNORECASE,
)
OPTION_LINE_START = re.compile(
    r"^\s*\(\s*([a-dA-Dअबसद१२३४])\s*\)\s*(.*)$",
    re.IGNORECASE,
)
ANSWER_LINE = re.compile(
    r"(?:Ans\.?|उत्तर|उतर|उतत्तर)\s*\.?\s*[\:\-\(]?\s*\(?\s*"
    r"([a-dA-Dअबसद१२३४0Oo०-९])\s*\)?",
    re.IGNORECASE,
)
NOISE_LINE = re.compile(
    r"^(?:@\s*|%\s*|Join\s+TG\s+@|YCT\s*$|‘?Yo,?\s*$)",
    re.IGNORECASE,
)

OPTION_LETTER_MAP = {
    "a": "A", "b": "B", "c": "C", "d": "D",
    # OCR confusions — '0' looks like 'd', 'O'/'o' often = 'a' or 'c'
    "0": "D",
    "O": "C", "o": "C",
    # Devanagari labels
    "अ": "A", "ब": "B", "स": "C", "द": "D",
    "१": "A", "२": "B", "३": "C", "४": "D",
}


def _ocr_digit_fix(tok: str) -> str:
    """Convert OCR-look-alikes back to digits at the question-number anchor."""
    return tok.replace("l", "1").replace("I", "1").replace("O", "0").replace("o", "0")


def normalize_letter(s: str) -> str:
    s = s.strip()
    if s in OPTION_LETTER_MAP:
        return OPTION_LETTER_MAP[s]
    return OPTION_LETTER_MAP.get(s.lower(), s.upper())


DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
OCR_GARBAGE_RE = re.compile(
    r"[@%™©]|Join\s+TG\s+@|YCT\b|n—[_\-]+|»\]|\[60\s*a\b",
    re.IGNORECASE,
)


def clean_ocr_text(text: str) -> str:
    """Remove repeated headers/watermarks that confuse the parser."""
    if not text:
        return ""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or NOISE_LINE.match(line):
            continue
        if re.match(r"^25\s+MP\s+ESB", line, re.I):
            continue
        lines.append(raw)
    return "\n".join(lines)


def clean_option_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    # Drop accidental trailing option markers, e.g. "70 (b) 60" → "70"
    text = re.sub(r"\s*\([a-dA-D]\)\s*.*$", "", text, flags=re.I).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text[:240]


def clean_explanation_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"n[—\-_]+[^\s।]{0,30}", " ", text)
    text = re.sub(r"\]\s*5\s*\[60", "115 160", text)
    if OCR_GARBAGE_RE.search(text) and not DEVA_RE.search(text):
        # Pure-Latin OCR junk from diagrams — prefer empty English explanation
        if len(re.findall(r"[A-Za-z]{3,}", text)) < 3:
            return ""
    return text[:4000]


def _option_looks_invalid(text: str) -> bool:
    if not text:
        return False
    if len(text) > 180:
        return True
    if re.search(r"\bAns\.?\b|Which of the following|निम्नलिखित में से कौन", text, re.I):
        return True
    if text.lstrip().startswith(":") and DEVA_RE.search(text):
        return True
    return False


def extract_hindi_only(text: str) -> str:
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not DEVA_RE.search(line):
            continue
        if "/" in line and LATIN_RE.search(line):
            _left, _sep, right = line.partition("/")
            if DEVA_RE.search(right):
                parts.append(right.strip())
                continue
        # Strip leading Latin before '/'
        if "/" in line:
            _left, _sep, right = line.partition("/")
            if DEVA_RE.search(right):
                parts.append(right.strip())
                continue
        parts.append(line)
    return " ".join(parts).strip()


def extract_english_only(text: str) -> str:
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "/" in line and DEVA_RE.search(line) and LATIN_RE.search(line):
            left, _sep, _right = line.partition("/")
            if LATIN_RE.search(left):
                parts.append(left.strip())
            continue
        if DEVA_RE.search(line) and not LATIN_RE.search(line):
            continue
        if LATIN_RE.search(line):
            parts.append(line)
    return " ".join(parts).strip()


def split_bilingual(text: str) -> tuple[str, str]:
    """Split a chunk of text (possibly multi-line) into english + hindi parts.

    Strategy:
      1. If a `/` separator is on a single line, split there.
      2. Else walk line-by-line: collect Latin-dominant lines as English,
         Devanagari-dominant lines as Hindi.
    """
    if not text:
        return "", ""
    text = text.strip()

    # Try the inline-slash form first (works well for short option labels)
    if "/" in text and "\n" not in text:
        parts = text.split("/", 1)
        eng = parts[0].strip()
        hin = parts[1].strip() if len(parts) > 1 else ""
        if DEVA_RE.search(hin) or not DEVA_RE.search(eng):
            return eng, hin

    # Multi-line: bucket each line by script majority
    eng_lines: list[str] = []
    hin_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Some lines have BOTH English and Hindi separated by '/'
        if "/" in line and DEVA_RE.search(line) and LATIN_RE.search(line):
            left, _sep, right = line.partition("/")
            if LATIN_RE.search(left):
                eng_lines.append(left.strip())
            if DEVA_RE.search(right):
                hin_lines.append(right.strip())
            continue
        if DEVA_RE.search(line) and not LATIN_RE.search(line):
            hin_lines.append(line)
        elif LATIN_RE.search(line) and not DEVA_RE.search(line):
            eng_lines.append(line)
        elif DEVA_RE.search(line):
            hin_lines.append(line)
        else:
            eng_lines.append(line)

    eng = " ".join(eng_lines).strip()
    hin = " ".join(hin_lines).strip()
    if not eng and not hin:
        eng = extract_english_only(text)
        hin = extract_hindi_only(text)
    if eng and hin and eng == hin:
        eng = extract_english_only(text)
        hin = extract_hindi_only(text)
    if not hin and DEVA_RE.search(text):
        hin = extract_hindi_only(text)
    if not eng and LATIN_RE.search(text):
        eng = extract_english_only(text)
    return eng, hin


def _first_option_line_index(lines: list[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if OPTION_LINE_START.match(stripped):
            return i
        # Multiple options on one line: "(a) 70 (b) 60"
        markers = OPTION_MARKER.findall(stripped)
        if len(markers) >= 2:
            return i
    return None


def extract_options(pre_ans: str) -> tuple[str, dict[str, str]]:
    """Split question stem from (a)(b)(c)(d) options without false positives."""
    lines = [ln for ln in pre_ans.splitlines() if ln.strip()]
    opt_idx = _first_option_line_index(lines)
    if opt_idx is None:
        return pre_ans.strip(), {}

    question_text = "\n".join(lines[:opt_idx]).strip()
    opt_blob = "\n".join(lines[opt_idx:]).strip()

    options: dict[str, str] = {}
    # Split on option markers while keeping the letter
    parts = re.split(r"(?=\(\s*[a-dA-Dअबसद१२३४]\s*\))", opt_blob, flags=re.I)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(
            r"^\(\s*([a-dA-Dअबसद१२३४])\s*\)\s*(.*)",
            part,
            re.DOTALL | re.I,
        )
        if not m:
            continue
        letter = normalize_letter(m.group(1))
        if letter not in {"A", "B", "C", "D"}:
            continue
        body = m.group(2).strip()
        # Trim at embedded next-option marker
        next_m = OPTION_MARKER.search(body)
        if next_m and next_m.start() > 0:
            body = body[: next_m.start()].strip()
        options[letter] = clean_option_text(body)
    return question_text, options


def parse_paper_text(pages_text: dict[int, str]) -> list[dict]:
    """Parse OCR'd pages into question dicts."""
    questions: list[dict] = []

    current_paper: str = "MP ESB Sub Engineer"
    current_subject: str = "General"
    current_question_id_global = 0

    sorted_pages = sorted(pages_text.keys())
    for page_no in sorted_pages:
        text = clean_ocr_text(pages_text[page_no])
        if not text:
            continue
        if len(text) < 80 and not ANSWER_LINE.search(text):
            continue

        # Update paper / subject context if header on this page
        old_paper = current_paper
        for m in PAPER_HEADER.finditer(text):
            current_paper = m.group(0).strip()
            break
        # Reset subject whenever the paper context changes
        if current_paper != old_paper:
            current_subject = "General"

        # Look ANYWHERE on the page for a clean "Subject : <Name>" header.
        # Newer subject overrides whatever was carried from the prior page.
        for cand_match in SUBJECT_LINE.finditer(text):
            cand = cand_match.group(1).strip()
            if _is_plausible_subject(cand):
                current_subject = cand
                break

        # Split into question blocks by question-number anchors
        anchors = list(QUESTION_NUM.finditer(text))
        if not anchors:
            continue

        for idx, anchor in enumerate(anchors):
            raw_num = _ocr_digit_fix(anchor.group(1))
            try:
                local_num = int(raw_num)
            except ValueError:
                continue
            # Skip obvious noise: question numbers > 200 or < 1
            if local_num < 1 or local_num > 200:
                continue

            block_start = anchor.end()
            block_end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(text)
            block = text[block_start:block_end].strip()
            if not block:
                continue

            parsed = parse_question_block(block)
            if parsed is None:
                continue

            current_question_id_global += 1
            parsed.update({
                "question_id": current_question_id_global,
                "paper": current_paper[:120],
                "subject": current_subject[:80],
                "sub_topic": guess_sub_topic(
                    parsed.get("question_english", ""),
                    parsed.get("question_hindi", ""),
                    parsed.get("detailed_explanation_hindi", ""),
                ),
                "_source_page": page_no,
                "_source_local_num": local_num,
                "figure": None,
            })
            questions.append(parsed)

    return questions


def parse_question_block(block: str) -> Optional[dict]:
    """Pull english + hindi text + 4 options + answer + explanation out of one block."""
    block = clean_ocr_text(block)
    ans_match = ANSWER_LINE.search(block)
    if not ans_match:
        return None
    correct = normalize_letter(ans_match.group(1))
    if correct not in {"A", "B", "C", "D"}:
        return None

    pre_ans = block[: ans_match.start()].strip()
    post_ans = block[ans_match.end():].strip()
    # Explanation often starts after ':' on same line
    if post_ans.startswith(":"):
        post_ans = post_ans[1:].strip()

    question_text, options = extract_options(pre_ans)
    if not options or "A" not in options or "B" not in options:
        return None
    if not question_text or len(question_text) < 8:
        return None
    question_text = re.sub(
        r"^Subject\s*[:\-]\s*[^\n]+\s*",
        "",
        question_text,
        count=1,
        flags=re.I,
    ).strip()
    for letter, opt in list(options.items()):
        if _option_looks_invalid(opt):
            options[letter] = ""

    q_eng, q_hin = split_bilingual(question_text)
    if q_eng == q_hin:
        q_eng = extract_english_only(question_text)
        q_hin = extract_hindi_only(question_text)
    if not q_eng:
        q_eng = q_hin
    if not q_hin:
        q_hin = ""

    options_out: dict[str, str] = {}
    for letter in ("A", "B", "C", "D"):
        raw = options.get(letter, "")
        if not raw:
            options_out[letter] = ""
            continue
        e, h = split_bilingual(raw)
        if e == h:
            e = extract_english_only(raw)
            h = extract_hindi_only(raw)
        options_out[letter] = e or h or raw

    exp_eng, exp_hin = split_bilingual(post_ans)
    if exp_eng == exp_hin:
        exp_eng = extract_english_only(post_ans)
        exp_hin = extract_hindi_only(post_ans)
    exp_eng = clean_explanation_text(exp_eng)
    exp_hin = clean_explanation_text(exp_hin)
    # Diagram OCR noise in English slot when Hindi has the real explanation
    if exp_hin and (
        not exp_eng
        or (len(exp_eng) < 40 and DEVA_RE.search(exp_hin))
    ):
        exp_eng = ""

    return {
        "question_english": q_eng,
        "question_hindi": q_hin,
        "options": options_out,
        "correct_answer": correct,
        "correct_option_text": options_out.get(correct, ""),
        "detailed_explanation_english": exp_eng,
        "detailed_explanation_hindi": exp_hin,
    }


# ─── Lightweight sub-topic guesser ───
SUBTOPIC_HINTS = {
    "Indian Geography": ["district", "state", "river", "mountain", "border", "क्षेत्रफल", "नदी", "पर्वत", "जिला"],
    "Indian History": ["century", "ancient", "british", "freedom", "1857", "स्वतंत्रता", "आज़ादी", "इतिहास"],
    "Indian Polity": ["president", "parliament", "constitution", "minister", "राष्ट्रपति", "संविधान"],
    "Indian Economy": ["GDP", "rupee", "bank", "RBI", "मुद्रा", "अर्थव्यवस्था"],
    "Science": ["physics", "chemistry", "biology", "Newton", "atom", "रसायन", "भौतिक"],
    "Mathematics": [
        "equation", "triangle", "circle", "percent", "series", "गुणनफल", "वर्गमूल",
        "श्रृंखला", "संख्या", "प्रतिशत", "समीकरण",
    ],
    "Logical Reasoning": [
        "venn", "diagram", "analogy", "odd one", "coding", "decoding",
        "वेन", "आकृति", "विषम", "कोडित", "क्रम",
    ],
    "Civil Engineering": ["concrete", "RCC", "shuttering", "rebar", "कंक्रीट", "सरिया", "स्लैब", "मृदा", "सिंचाई"],
    "English Grammar": ["synonym", "antonym", "idiom", "tense", "preposition", "passive", "indirect"],
    "Computer": ["software", "RAM", "CPU", "internet", "कंप्यूटर", "इंटरनेट"],
}


def guess_sub_topic(*texts: str) -> str:
    combined = " ".join(t for t in texts if t)
    if not combined:
        return ""
    low = combined.lower()
    for topic, hints in SUBTOPIC_HINTS.items():
        for h in hints:
            if h.lower() in low:
                return topic
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3 — figure extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_figures(pdf_path: str,
                    questions: list[dict],
                    figures_dir: Path,
                    min_size_px: int = 90) -> int:
    """Per-page, find placed images and assign them to the question whose
    text block lies immediately above the image bbox.

    Returns the number of figures saved.
    """
    import fitz

    figures_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    # Build a lookup: page → list of (local_num, question_id, y_top)
    page_to_questions: dict[int, list[tuple[int, int, float]]] = {}
    for q in questions:
        p = q.get("_source_page")
        if p:
            page_to_questions.setdefault(p, []).append(
                (q["_source_local_num"], q["question_id"], 0.0)
            )

    # Only inspect pages that have at least one question (huge speedup)
    relevant_pages = sorted(page_to_questions.keys())
    figs_saved = 0
    for page_no in relevant_pages:
        page = doc[page_no - 1]
        infos = page.get_image_info(xrefs=True)
        if not infos:
            continue

        # Skip tiny images (icons / footer marks / repeated branding)
        big = [info for info in infos
               if info["width"] >= min_size_px
               and info["height"] >= min_size_px
               and info["height"] < page.rect.height * 0.6]
        if not big:
            continue

        # Get text positions for question numbers on this page
        text_dict = page.get_text("dict")
        q_positions: list[tuple[int, float]] = []
        for blk in text_dict.get("blocks", []):
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    m = re.match(r"^(\d{1,3})\s*[.\)]", txt)
                    if m:
                        local_num = int(m.group(1))
                        if 1 <= local_num <= 200:
                            y = span["bbox"][1]
                            q_positions.append((local_num, y))

        candidate_questions = page_to_questions.get(page_no, [])
        # Match every "big" image to the nearest preceding question number on page
        for img_idx, info in enumerate(big, start=1):
            img_y_top = info["bbox"][1]
            # Pick highest local_num whose y is <= img_y_top (most recent above)
            best_local = None
            best_y = -1.0
            for ln, y in q_positions:
                if y <= img_y_top and y > best_y:
                    best_local = ln
                    best_y = y
            if best_local is None:
                continue
            # Map to global question_id
            matched_qid: Optional[int] = None
            for ln, qid, _ in candidate_questions:
                if ln == best_local:
                    matched_qid = qid
                    break
            if matched_qid is None:
                continue

            # Extract image bytes (use clip rather than xref to handle masks)
            try:
                clip = fitz.Rect(info["bbox"])
                mat = fitz.Matrix(DPI / 72, DPI / 72)
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                fig_name = f"q{matched_qid}_{img_idx}.png"
                pix.save(str(figures_dir / fig_name))
                figs_saved += 1
                # Attach to question
                for q in questions:
                    if q["question_id"] == matched_qid and not q.get("figure"):
                        q["figure"] = f"figures/{fig_name}"
                        break
            except Exception as e:
                logger.warning("Could not extract figure on page %d: %s", page_no, e)

    doc.close()
    return figs_saved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF_PATH)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first-page", type=int, default=3,
                    help="Skip cover + TOC (default 3)")
    ap.add_argument("--last-page", type=int, default=None)
    ap.add_argument("--page-cap", type=int, default=None,
                    help="Hard cap on pages to OCR (for quick test runs)")
    ap.add_argument("--skip-ocr", action="store_true",
                    help="Skip OCR; reuse pages_text.json if it exists")
    ap.add_argument("--batch-pages", type=int, default=None,
                    help="OCR this many pages per batch (writes cache incrementally)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    import fitz
    doc = fitz.open(args.pdf)
    total_pages = len(doc)
    doc.close()
    last = args.last_page or total_pages
    pages = list(range(args.first_page, last + 1))
    if args.page_cap:
        pages = pages[: args.page_cap]
    print(f"Processing {len(pages)} pages "
          f"({pages[0]}..{pages[-1]}) with {args.workers} OCR workers")

    cache_path = out_dir / "pages_text.json"

    if args.skip_ocr and cache_path.exists():
        pages_text = {int(k): v for k, v in json.loads(
            cache_path.read_text(encoding="utf-8")).items()}
        print(f"Loaded {len(pages_text)} cached OCR pages from {cache_path}")
    else:
        pages_text: dict[int, str] = {}
        if cache_path.exists():
            pages_text = {int(k): v for k, v in json.loads(
                cache_path.read_text(encoding="utf-8")).items()}
        batch = args.batch_pages or len(pages)
        for start in range(0, len(pages), batch):
            chunk = pages[start: start + batch]
            print(f"  OCR batch pages {chunk[0]}..{chunk[-1]} "
                  f"({start + 1}-{start + len(chunk)} of {len(pages)})")
            pages_text.update(ocr_range(args.pdf, chunk, workers=args.workers))
            cache_path.write_text(
                json.dumps({str(k): v for k, v in pages_text.items()},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"\nParsing {len(pages_text)} OCR'd pages into questions...")
    questions = parse_paper_text(pages_text)
    print(f"  Got {len(questions)} questions")

    print("\nExtracting figures...")
    figs = extract_figures(args.pdf, questions, figures_dir)
    print(f"  Saved {figs} figures")

    # Strip internal keys before writing the public JSON
    public: list[dict] = []
    for q in questions:
        clean = {k: v for k, v in q.items() if not k.startswith("_")}
        public.append(clean)

    (out_dir / "questions.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "pdf": Path(args.pdf).name,
        "total_questions": len(public),
        "total_figures": figs,
        "pages_processed": len(pages_text),
        "first_page": pages[0] if pages else None,
        "last_page": pages[-1] if pages else None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    zip_path = out_dir.parent / "questions_package.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "questions.json", "questions.json")
        zf.write(out_dir / "manifest.json", "manifest.json")
        for fig in sorted(figures_dir.glob("*.png")):
            zf.write(fig, f"figures/{fig.name}")
    print(f"\nWrote {out_dir}/questions.json + {figs} figures")
    print(f"ZIP: {zip_path} ({zip_path.stat().st_size // 1024} KB)")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
