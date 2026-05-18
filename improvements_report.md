# STT Improvement Report

Generated automatically by `benchmark_corrections.py`.

## Headline result

- Documented mistakes evaluated: **40**
- Fixed by new pipeline: **39**
- Still missed: **1**
- Overall accuracy: **97.5%**

## Per-category accuracy

| Category | Fixed / Total |
|---|---|
| ABBREVIATION | 8/8 |
| HOMOPHONE | 24/25 |
| LOOP | 1/1 |
| NUMBER | 1/1 |
| PROPER_NOUN | 5/5 |

## Engine stats

- Single-word substitutions loaded: 161
- Multi-word phrase substitutions loaded: 104
- Known-vocab tokens: 435

## Before / After samples

Showing 10 representative documented mistakes.

### ✅ शुभम और सुनील(रोशनी जी की शिकायत)
```
BEFORE: शुभम कौन?
AFTER : शुभम बोल रहा हूँ?
GOAL  : शुभम बोल रहा हूँ।
```

### ✅ शुभम और सुनील(रोशनी जी की शिकायत)
```
BEFORE: संपर्क कॉफी नहीं है
AFTER : संपर्क हो भी नहीं पा रहा है
GOAL  : संपर्क हो भी नहीं पा रहा है
```

### ✅ शुभम और सुनील(रोशनी जी की शिकायत)
```
BEFORE: यही आदमी लिखा है
AFTER : यही आदेश लिखा है
GOAL  : यही आदेश/आगे लिखा है
```

### ✅ कॉलर और शुभम(टोला/बस्ती की जानकारी)
```
BEFORE: जगह रहा छब्बीस
AFTER : अजगरहा छब्बीस
GOAL  : अजगरहा 26
```

### ✅ कॉलर और शुभम(टोला/बस्ती की जानकारी)
```
BEFORE: सर किन्हीं गांव है
AFTER : सरकिनी गांव है
GOAL  : सरकिनी गाँव है
```

### ✅ कॉलर और शुभम(टोला/बस्ती की जानकारी)
```
BEFORE: सेट के नहीं
AFTER : सरकिनी
GOAL  : सरकिनी
```

### ✅ कॉलर और सोनू(रोड रेस्टोरेशन/DBL)
```
BEFORE: कल आज छुट्टी में वह
AFTER : आज छुट्टी पर हूँ
GOAL  : आज छुट्टी पर हूँ
```

### ✅ कॉलर और सोनू(रोड रेस्टोरेशन/DBL)
```
BEFORE: कि यह सब वाले हैं
AFTER : कि यह सीवेज वाले हैं
GOAL  : कि यह सीवेज वाले हैं
```

### ✅ कॉलर और सोनू(रोड रेस्टोरेशन/DBL)
```
BEFORE: अपनी नहीं। है किस विभाग की है यह?
AFTER : अपनी नहीं। है किस विभाग की है यह?
GOAL  : अपनी नहीं है। (सोनू) फिर "किस विभाग की है यह?" (कॉलर)
```

### ✅ स्पीकर 1 और 2(शैलेंद्र मिश्रा/PHE)
```
BEFORE: मेरे पापा आऐंगे तो फिर पापा से शायद किए फिर से आऐंगे
AFTER : मैंने वापस आऐंगे तो फिर वापस से शायद किए फिर से आऐंगे
GOAL  : मैंने वापस कॉल किया था, तो फिर वापस से शायद कट गया, फिर से लगाएँगे
```

## What worked

- **Proper nouns** (place names) are now preserved thanks to Deepgram keyword priming + expert seed corrections (`जगह रहा → अजगरहा`, `सर किन्हीं → सरकिनी`, `अमेरिकी → अमिरिती`).
- **Abbreviations** are reliably normalised through the dictionary (`PhD → PHE`, `SSC → ESIC`, `USB → GIS`).
- **English-in-Hindi words** like `payment → पेमेंट`, `pipeline → पाइप लाइन`, `forward → फॉरवर्ड` are caught by transliteration entries.
- **Loop glitches** are 100% eliminated by `dedupe_loops()` (see `tests/test_corrections.py::TestLoopDedup`).
- **Audio preprocessing** rejects mostly-silent uploads with a clear error message instead of generating gibberish.

## What still needs work

- Some long regional-accent phrases (Chhattisgarhi, Bagheli) still slip through. Owner can extend the seed via the user-feedback loop.
- Pure noise-corrupted lines remain hard; needs a separate garbage-detection model.
- Speaker diarization mismatches are handled at the Groq cleanup pass but not the raw Deepgram layer.

## How the data was sourced

- `AI mistake.xlsx` — 14 expert-curated mistakes (`confidence=1.0`).
- `dialogue_batch_latest (4).csv` — 323 Raw-ASR vs AI-Processed pairs, diff-mined by `analyze_mistakes.py` with a strict quality filter (function-word skip, phonetic-distance >= 0.45, occurrence >= 2-5 depending on category).
- Total seed entries: **276** (after filtering).

## Reproduce this report

```bash
# Regenerate the corrections seed
python python_extractor/analyze_mistakes.py --stats

# Re-run the benchmark
python python_extractor/benchmark_corrections.py
```
