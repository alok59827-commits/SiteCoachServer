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

OPTION_LINE = re.compile(
    r"^\s*\(?\s*([a-dA-Dअबसद१२३४0Oo])\s*\)?\s*[\.\:\)]?\s*(.+?)\s*$",
    re.MULTILINE,
)
ANSWER_LINE = re.compile(
    r"(?:Ans\.?|उत्तर|उतर|उतत्तर)\s*\.?\s*[\:\-\(]?\s*\(?\s*([a-dA-Dअबसद१२३४0Oo])\s*\)?",
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
    return OPTION_LETTER_MAP.get(s.strip().lower(), s.upper())


DEVA_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


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
        return text, text
    return eng, hin


def parse_paper_text(pages_text: dict[int, str]) -> list[dict]:
    """Parse OCR'd pages into question dicts."""
    questions: list[dict] = []

    current_paper: str = "MP ESB Sub Engineer"
    current_subject: str = "General"
    current_question_id_global = 0

    sorted_pages = sorted(pages_text.keys())
    for page_no in sorted_pages:
        text = pages_text[page_no]
        if not text or len(text) < 200:
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
                "sub_topic": guess_sub_topic(parsed.get("question_english", "")),
                "_source_page": page_no,
                "_source_local_num": local_num,
                "figure": None,
            })
            questions.append(parsed)

    return questions


def parse_question_block(block: str) -> Optional[dict]:
    """Pull english + hindi text + 4 options + answer + explanation out of one block."""
    # Find the answer marker so we can split question/options/explanation
    ans_match = ANSWER_LINE.search(block)
    if not ans_match:
        return None
    correct = normalize_letter(ans_match.group(1))

    pre_ans = block[: ans_match.start()].strip()
    post_ans = block[ans_match.end():].strip()

    # In pre_ans, separate question text from options
    options: dict[str, str] = {}
    option_matches = list(OPTION_LINE.finditer(pre_ans))
    # Keep only matches that look like real options (not random lines)
    plausible: list[re.Match] = []
    for m in option_matches:
        letter = normalize_letter(m.group(1))
        if letter in {"A", "B", "C", "D"} and len(m.group(2).strip()) >= 1:
            plausible.append(m)

    question_text = pre_ans
    if plausible:
        question_text = pre_ans[: plausible[0].start()].strip()
        # Build options dict
        for i, m in enumerate(plausible):
            letter = normalize_letter(m.group(1))
            opt_text = m.group(2).strip()
            # Continue until next option starts
            if i + 1 < len(plausible):
                next_start = plausible[i + 1].start()
                opt_text = pre_ans[m.start(2): next_start].strip()
            options[letter] = opt_text.replace("\n", " ").strip()

    if not options or "A" not in options or "B" not in options:
        return None

    q_eng, q_hin = split_bilingual(question_text)

    opts_eng: dict[str, str] = {}
    opts_hin: dict[str, str] = {}
    for letter in ("A", "B", "C", "D"):
        e, h = split_bilingual(options.get(letter, ""))
        opts_eng[letter] = e
        opts_hin[letter] = h

    # Explanation handling — try to split bilingual but typically all-Hindi
    exp_eng, exp_hin = split_bilingual(post_ans)

    options_out = opts_eng if any(opts_eng.values()) else opts_hin
    # Truncate any noisy multi-line spillover
    for k, v in options_out.items():
        options_out[k] = v.split("\n")[0].strip()[:160]
    return {
        "question_english": q_eng or q_hin,
        "question_hindi": q_hin or q_eng,
        "options": options_out,
        "correct_answer": correct,
        "correct_option_text": options_out.get(correct, ""),
        "detailed_explanation_english": exp_eng if exp_eng != exp_hin else "",
        "detailed_explanation_hindi": exp_hin or exp_eng,
    }


# ─── Lightweight sub-topic guesser ───
SUBTOPIC_HINTS = {
    "Indian Geography": ["district", "state", "river", "mountain", "border", "क्षेत्रफल", "नदी", "पर्वत"],
    "Indian History": ["century", "ancient", "british", "freedom", "1857", "स्वतंत्रता", "आज़ादी"],
    "Indian Polity": ["president", "parliament", "constitution", "minister", "राष्ट्रपति", "संविधान"],
    "Indian Economy": ["GDP", "rupee", "bank", "RBI", "मुद्रा", "अर्थव्यवस्था"],
    "Science": ["physics", "chemistry", "biology", "Newton", "atom", "रसायन", "भौतिक"],
    "Mathematics": ["equation", "triangle", "circle", "percent", "गुणनफल", "वर्गमूल"],
    "Civil Engineering": ["concrete", "RCC", "shuttering", "rebar", "कंक्रीट", "सरिया", "स्लैब"],
    "English Grammar": ["synonym", "antonym", "idiom", "tense", "preposition"],
    "Computer": ["software", "RAM", "CPU", "internet", "कंप्यूटर", "इंटरनेट"],
}


def guess_sub_topic(text: str) -> str:
    if not text:
        return ""
    low = text.lower()
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
        pages_text = ocr_range(args.pdf, pages, workers=args.workers)
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

    print(f"\nWrote {out_dir}/questions.json + {figs} figures")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
