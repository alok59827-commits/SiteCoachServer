# Predictive STT — Confusion Matrix Analysis Report

> Generated from `corpus.csv` (6,379 word/phrase pairs) +
> `STT_Error_Dictionary.txt` (5,844 sentence pairs).
> **Total: 12,198 pair rows analyzed.**

## TL;DR

Real data confirms that Deepgram's mistakes are **highly repeatable**.
The same wrong word appears hundreds of times across calls. We can use
this to **predict** mishearings before they happen and pre-load them
into Deepgram's beam search.

| Stat | Value |
|---|---|
| Unique phrase-level mishearings | **3,954** |
| Phrases occurring ≥ 2 times | **3,647** |
| Top mishearing recurrence | `शरीर → सरिया` (**425 occurrences**) |
| Domains covered | **80+** (civil, hvac, electricity, water_supply, ...) |
| Unique syllable confusions | **1,874** |

## Top 20 Phrase-Level Mishearings

| # | Heard (गलत) | Should be (सही) | Count |
|---|---|---|---|
| 1 | `शरीर` | `सरिया` | 425 |
| 2 | `डस्टिंग` | `डक्टिंग` | 196 |
| 3 | `गड़बड़` | `गर्डर` | 193 |
| 4 | `कीवी` | `केवी` | 177 |
| 5 | `कंप्यूटर` | `कंप्रेसर` | 160 |
| 6 | `पी डब्लू डी` | `पीडब्ल्यूडी` | 157 |
| 7 | `कम एक्शन` | `कम्पेक्शन` | 145 |
| 8 | `रूलर` | `रोलर` | 141 |
| 9 | `स्टटरिंग` | `शटरिंग` | 139 |
| 10 | `इस फाल्ट` | `एस्फाल्ट` | 132 |
| 11 | `मैं` | `महीने` | 130 |
| 12 | `रूम` | `बूम` | 124 |
| 13 | `पोकलेन` | `पॉलिसी` | 124 |
| 14 | `डांस फार्मर` | `ट्रांसफार्मर` | 122 |
| 15 | `टायर्स` | `टाइल्स` | 121 |
| 16 | `पावर` | `पेवर` | 120 |
| 17 | `सब्जी स्टेशन` | `सबस्टेशन` | 118 |
| 18 | `चिल्लर` | `चिलर` | 118 |
| 19 | `पापा` | `पाइप` | 117 |
| 20 | `डबलू बीम` | `डब्लू बी एम` | 115 |

## Per-Domain Highlights

| Domain | Top mishearing | Count | Insight |
|---|---|---|---|
| **building** | `शरीर → सरिया` | 84 | Steel rebar is `सरिया`, Deepgram defaults to `शरीर` (body) |
| **rcc_work** | `स्टटरिंग → शटरिंग` | 57 | Construction shuttering misheard as stuttering |
| **electricity** | `कीवी → केवी` | 95 | Kiwi (fruit) vs kV (kilovolt) |
| **hvac** | `चिल्लर → चिलर` | 66 | Chiller AC misheard as the gangster word |
| **water_supply** | `समरसेब → सबमर्सिबल` | 51 | Submersible pump mangled |
| **mining** | `प्लास्टिक → ब्लास्टिंग` | 50 | Blasting vs plastic |
| **road_work** | `कम एक्शन → कम्पेक्शन` | 74 | Soil compaction |
| **billing** | `मैं → महीने` | 59 | Monthly vs first-person pronoun |
| **telecom** | `डस्टिंग → डक्टिंग` | 65 | Cable ducting misheard as dusting |
| **railways** | `गड़बड़ → गर्डर` | 53 | Bridge girder |
| **insurance** | `पोकलेन → पॉलिसी` | 60 | Policy vs Poclain (excavator) |
| **machinery** | `जूसी → जेसीबी` | 23 | JCB excavator |

## Sample Variant Predictions

For 20 sample words, the predictor produced these likely Deepgram
mishearings (both **ground-truth** from corpus AND **synthesized** for
new words):

### Construction
- **शटरिंग** → `स्टटरिंग` (139x), `Shuttering`, `Stuttering`, `शटरिग`
- **पेमेंट** → `पे मिंट` (52x), `पी मिंट` (45x), `Pay mint`, `Pea mint`, `सीमेंट`
- **ट्रांसफार्मर** → `डांस फार्मर` (122x) + 7 synthesized
- **सरिया** → `शरीर` (425x), `sariya` (4x), `में शरीर` (2x)
- **कंक्रीट** → `कॉन क्रिएट` (75x), `कॉंग क्रिएट` (67x), `Concrete`, `Kong create`

### Equipment
- **जेसीबी** → `जूसी` (55x), `जे एस बी` (54x), `जय सी बी` (29x), `Juicy`, `JCP`
- **सबमर्सिबल** → `समरसेब` (100x), `समर्सबल` (76x)
- **रोलर** → `रूलर` (141x)
- **वाइब्रेटर** → `वाई ब्रेटर` (34x)
- **बैचिंग** → `मैचिंग` (84x)

### Government / Departments
- **पीएचई** → `पीच ही` (98x), `पी ए च` (78x), `PHE`, `Peach he`, `PA ch`
- **ठेकेदार** → `टीचर` (62x), `टेक अ दर` (49x), `teacher` (4x), `Take a dar`
- **एमबी** → `एम बी` (90x)

### Place names
- **अजगरहा** → multiple variants across legal/civil/sewerage domains

## How to Use This Data

### Approach 1 — Direct keyword preloading (RECOMMENDED, no ML needed)

```python
# Before each Deepgram call
domain = "rcc_work"          # user-selected mode
keywords = predictor.keywords_for_domain(domain, limit=180)
# Sample output for rcc_work:
# ["शटरिंग", "स्टटरिंग", "Shuttering",
#  "कंक्रीट", "कॉन क्रिएट", "Concrete",
#  "वाइब्रेटर", "वाई ब्रेटर",
#  "सरिया", "शरीर",  ← Deepgram now knows BOTH forms
#  ...up to 180 entries]
```

Deepgram's beam search will keep `सरिया` as a top hypothesis even when
the acoustic features look like `शरीर`, because both are in the
keyword list.

### Approach 2 — Post-correction (already implemented)

The `CorrectionsEngine` (in `sitecoach/corrections.py`) already handles
the reactive path: even if Deepgram outputs `शरीर`, we replace it with
`सरिया` via the dictionary. Combining (1) + (2) gives **defence in depth**.

### Approach 3 — Domain auto-detection (future work)

Train a small classifier (logistic regression on bag-of-words) that
takes the *raw* Deepgram output and assigns a domain probability.
Re-run transcription with the predicted domain's keyword set.

## Artifacts Produced

| File | Purpose |
|---|---|
| `predictive_stt/artifacts/confusion_phrases.json` | 3,647 phrase-level pairs |
| `predictive_stt/artifacts/confusion_matrix_syll.json` | Devanagari syllable confusions |
| `predictive_stt/artifacts/confusion_matrix_char.json` | Char-level confusions |
| `predictive_stt/artifacts/per_domain_patterns.json` | 80+ domain breakdowns |
| `predictive_stt/artifacts/top_patterns.md` | Human-readable Top-20 |
| `predictive_stt/artifacts/variant_samples.txt` | Predictor output on 20 samples |
| `predictive_stt/IMPLEMENTATION_PROMPT.md` | Agent-mode prompt to wire it up |

## Next Step

Open `IMPLEMENTATION_PROMPT.md`, copy the prompt block, paste into a
fresh Agent-mode chat. The agent will:

1. Build `sitecoach/keyword_predictor.py`
2. Wire domain parameter into `/upload-audio`
3. Add `/predict-variants` admin endpoint
4. Re-run the benchmark with old vs new comparison
5. Open a follow-up PR
