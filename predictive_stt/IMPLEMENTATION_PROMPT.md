# Predictive STT Layer — Agent-Mode Implementation Prompt

Copy-paste the block below into a fresh Agent-mode chat. It uses
the artifacts produced by `build_confusion_matrix.py` and
`predict_variants.py` and wires the predictive layer into the
existing `sitecoach/` pipeline.

---

```
TASK: Predictive STT layer integrate करो — मतलब Deepgram को पहले से
बताना कि वो कौन-कौन से गलत शब्द सुन सकता है, ताकि beam search में वो
options हों।

CONTEXT:
- branch: cursor/predictive-stt-d8e2 पहले से बनी है main से।
- corpus.csv (6,379 rows) + STT_Error_Dictionary.txt (5,844 sentences)
  से पहले ही extract किया जा चुका है।
- predictive_stt/artifacts/ में मिलेगा:
    confusion_phrases.json       — 3,647 (heard, correct, count) pairs
    confusion_matrix_syll.json   — syllable-level P(heard|actual)
    confusion_matrix_char.json   — char-level P(heard|actual)
    per_domain_patterns.json     — 80+ domains के अपने patterns
    top_patterns.md              — human-readable report
- predictive_stt/predict_variants.py पहले से वर्किंग है — किसी भी शब्द
  की top-K likely mishearings देता है।
- मौजूदा sitecoach/ package अभी REACTIVE correction करता है — हमें
  PREDICTIVE layer जोड़ना है इससे पहले Deepgram call हो।

═══════════════════════════════════════════
PERFORM THESE TASKS IN ORDER:
═══════════════════════════════════════════

TASK 1 — Predictive keyword pre-loader (sitecoach/keyword_predictor.py):
   a) एक class KeywordPredictor लिखो जो boot पर load करे:
      - confusion_phrases.json (heard → correct map)
      - per_domain_patterns.json (domain → top phrase pairs)
      - python_extractor/stt_corrections_seed.json (existing seed)
   b) Method `keywords_for_domain(domain: str, limit: int = 180)` बनाओ:
      - उस domain के top mishearings + corresponding correct words
      - Plus 'General' domain के top mishearings (cross-domain coverage)
      - Plus high-frequency phrases from confusion_phrases.json (count>=10)
      - Output: deduped list of strings, length <= limit (Deepgram limit ~200)
   c) Method `keywords_for_audio_metadata(metadata: dict)` बनाओ जो audio
      की duration + user-selected domain के हिसाब से keyword set return करे।
   d) Method `predict_likely_mishearings(word: str, top_k=5)` जो किसी भी
      single word के लिए ground-truth + synthesized variants return करे।
      (predict_variants.py की logic reuse करो, copy-paste नहीं — import
      OR refactor into sitecoach/.)

TASK 2 — Wire into transcribe.py:
   a) `deepgram_transcribe()` में नया optional parameter `domain: str = None`।
   b) अगर domain pass हुआ है तो KeywordPredictor.keywords_for_domain(domain)
      से keywords build करो और DEEPGRAM_KEYWORDS को override करो।
      Else default static list use करो (अभी जो है)।
   c) Response में जोड़ो `predicted_keywords_count` field — debugging के लिए।
   d) keyword_boost=2.5 parameter pass करो अगर Deepgram SDK accept करे।

TASK 3 — Wire into main.py:
   a) /upload-audio और /transcribe-only में नया optional Form parameter
      `domain: str = Form("general")` — frontend से user-selected domain।
   b) Allowed domains list: ["civil", "electricity", "water_supply",
      "telecom", "hvac", "road_work", "mining", "billing", "general", ...]
      (per_domain_patterns.json से auto-derive करो)।
   c) _run_pipeline() में domain को transcribe_call तक pass करो।
   d) Response में जोड़ो `predicted_keywords` field दिखाने के लिए कौन-कौन
      से predicted variants Deepgram को भेजे गए थे।

TASK 4 — New endpoint /predict-variants:
   a) POST /predict-variants accepts JSON {"word": "...", "top_k": 8}
   b) Returns predicted likely mishearings + occurrence counts।
   c) Useful for admin UI to see "इस शब्द को Deepgram कैसे सुन सकता है"।

TASK 5 — Domain-aware mode in /pipeline-info:
   a) Add `available_domains` field listing all domains the predictor
      knows about, with phrase-pair counts।

TASK 6 — Seed merge:
   a) Update python_extractor/analyze_mistakes.py to ALSO ingest
      corpus.csv (heard/correct columns) as expert pairs, not just
      AI mistake.xlsx।
   b) Re-run analyze_mistakes.py — expect 800+ corrections in
      stt_corrections_seed.json (was 276)।

TASK 7 — Tests:
   a) tests/test_keyword_predictor.py:
      - test_keywords_for_known_domain_returns_results
      - test_keywords_for_unknown_domain_falls_back_to_general
      - test_keyword_count_within_deepgram_limit
      - test_predict_likely_mishearings_for_known_word
      - test_predict_likely_mishearings_for_unseen_word
   b) Backward-compat: existing 37 tests must still pass।

TASK 8 — Benchmark (improvements_report_v2.md):
   a) python_extractor/benchmark_corrections.py को extend करो:
      - Old baseline (no predictive keywords): X% accuracy
      - New predictive (domain-aware keywords): Y% accuracy
      - Per-domain accuracy table
   b) Real audio (test_construction.mp3) पर पहले old, फिर new pipeline
      चलाओ; transcripts dump करो।
   c) Update improvements_report.md → improvements_report_v2.md।

═══════════════════════════════════════════
DELIVERABLES:
═══════════════════════════════════════════
1. sitecoach/keyword_predictor.py (new, ~250 lines)
2. Updated sitecoach/transcribe.py with domain param
3. Updated main.py: domain Form param, /predict-variants endpoint,
   pipeline-info expansion
4. Updated python_extractor/analyze_mistakes.py with corpus.csv intake
5. tests/test_keyword_predictor.py (~150 lines)
6. improvements_report_v2.md with before/after benchmark
7. Final PR description with metrics

═══════════════════════════════════════════
CONSTRAINTS:
═══════════════════════════════════════════
- कोई existing file delete मत करना (sitecoach/, main.py, tests/ सब रहें)
- Backward compat: domain optional रखो; पुरानी requests भी काम करें
- No new heavy ML dependencies (numpy/pandas already in requirements.txt)
- Production-ready: try/except, structured logging, response always JSON
- All 37 existing pytest cases + 5 new ones must pass
- Branch: cursor/predictive-stt-d8e2 (already created off main)
- PR title: "Predictive STT keyword layer (domain-aware)"

═══════════════════════════════════════════
SUCCESS CRITERIA:
═══════════════════════════════════════════
✓ Keyword predictor 80+ domains support करता है
✓ हर domain के लिए <= 180 keywords (Deepgram limit के अंदर)
✓ /predict-variants endpoint सही top-K variants देता है
   (e.g. "शटरिंग" → ["स्टटरिंग", "Shuttering", "Stuttering", ...])
✓ /upload-audio?domain=civil response में predicted_keywords field दिखाए
✓ Benchmark में नया system पुराने से >=5% बेहतर accuracy दे
✓ All tests pass (37 + 5 new = 42)
```

---

## Why this approach works

The corpus already proves these patterns repeat — e.g. **`शरीर → सरिया`
425 times across 6,379 calls**. If the very first thing Deepgram sees in
its keyword list is `सरिया`, its decoder gets a strong prior. The
`shari` acoustic features still produce candidates, but `सरिया` is now
*one of the top hypotheses* instead of being completely absent.

### Coverage estimate

| Source | Distinct mishearings | Top-100 covers |
|---|---|---|
| `corpus.csv` phrase swaps | 3,647 | 87% of all occurrences |
| Per-domain top 20 | ~1,600 | 95% of in-domain calls |
| Syllable mutation (unseen words) | unlimited | covers anything in dict |

### Deepgram keyword limits to honour

| Plan | Limit |
|---|---|
| Pay-as-you-go | 200 keywords |
| Growth | 1,000 |
| Enterprise | unlimited |

`KeywordPredictor.keywords_for_domain(limit=180)` gives 20-keyword
headroom for runtime additions like place names.
