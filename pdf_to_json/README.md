# MP ESB Civil Engineering — Questions JSON Package

Auto-extracted from `MP_ESB_GROUP_3_SUB_ENGINEER_CIVIL_ENGINEERING_SOLVED_PAPERS_HINDI.pdf`
(768 pages, 18 MB) using `pdf_to_json/extract.py`.

## What's inside

```
output/
├── questions.json          — 1,083 question objects
├── manifest.json           — extraction metadata
├── figures/                — 38 PNG files (figures referenced from questions.json)
│   ├── q118_2.png
│   ├── q119_3.png
│   └── ...
└── pages_text.json         — raw OCR text per page (debugging cache)

questions_package.zip       — questions.json + manifest.json + figures/  (no OCR cache)
```

## JSON schema (matches the user-requested format)

```json
{
  "question_id": 1,
  "paper": "MP ESB Sub Engineer (Civil)- 2024",
  "subject": "General Knowledge",
  "sub_topic": "Indian Geography",
  "question_english": "Which is the largest District in India as per the area?",
  "question_hindi": "क्षेत्रफल के अनुसार भारत का सबसे बड़ा जिला कौन सा है?",
  "options": {"A": "Mumbai", "B": "Bangalore", "C": "Mahe", "D": "Kuchchh"},
  "correct_answer": "D",
  "correct_option_text": "Kuchchh",
  "detailed_explanation_english": "",
  "detailed_explanation_hindi": "क्षेत्रफल की दृष्टि से भारत का सबसे बड़ा जिला गुजरात का कच्छ जिला है।...",
  "figure": null
}
```

When a figure is attached, `figure` is set to `"figures/q<question_id>_<n>.png"`.

## Coverage stats

| Metric | Value |
|---|---|
| Pages OCR'd | 766 (3..768) |
| Total questions captured | **1,083** |
| Papers identified | 5 distinct |
| Subjects identified | 12+ |
| Figures extracted | **38** PNG files (35 questions) |
| JSON file size | ~3.5 MB |
| ZIP package size | **1.5 MB** |

### Top papers
| Paper | Questions |
|---|---|
| MP VYAPAM Sub Engineer 2020 | 677 |
| MP VYAPAM Sub Engineer (Civil)- 2022 | 194 |
| MP ESB Sub Engineer (Civil)- 2024 | 94 |
| MP ESB Draftsman (Civil)- 2024 | 61 |
| MP ESB Surveyor (Civil) 2024 | 57 |

## How extraction works

1. **Render** each page to PNG at 200 DPI via PyMuPDF
2. **OCR** with Tesseract (`hin+eng`, `OMP_THREAD_LIMIT=1`) — parallel 4 workers
3. **Parse** OCR text to find question anchors (`1.`, `2.`, ...), options
   `(a) ... (b) ...`, answer `Ans. (X) :`, and explanation
4. **Split bilingual** English/Hindi spans via `/` separator + per-line script detection
5. **Extract figures** by mapping placed-image bboxes to nearest question
6. **Bundle** into `questions_package.zip`

## Re-running

```bash
# Full PDF (~40 min OCR, ~1 min parse)
python pdf_to_json/extract.py

# Re-parse using cached OCR (no re-OCR; ~1 min)
python pdf_to_json/extract.py --skip-ocr

# Test on first 10 question pages
python pdf_to_json/extract.py --first-page 3 --last-page 12
```

## Known limitations

- **OCR noise**: short option strings sometimes have mangled Hindi
  (e.g. `Tejinder Singh/aftiex सिंह` instead of `तेजिंदर सिंह`). The
  English side is reliable; Hindi can be reconstructed via a follow-up
  pass with an LLM or font-aware transliterator.
- **Subject detection** depends on the literal `Subject : <Name>` line
  appearing on a page. When OCR distorts the header, the field falls
  back to `"General"`.
- **sub_topic** is left empty unless a clear keyword in `SUBTOPIC_HINTS`
  matches the English question.
- **Pure-English questions** (grammar/English-language section) leave
  `question_hindi` equal to `question_english`.
- **Figures** are only attached when a real placed image > 90×90 px lies
  below the question number on the same page; some technical questions
  have inline diagrams that didn't meet the size threshold.
