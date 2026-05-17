#!/usr/bin/env python3
"""Predict likely STT mishearings for any input word using the confusion
matrix built from corpus.csv + STT_Error_Dictionary.txt.

Two prediction modes (cascaded):
  1. REVERSE_LOOKUP — if the input word appears as `correct` in our phrase
     map, we have ground-truth mishearings. Highest confidence.
  2. SYLL_MUTATION  — for unseen words, apply per-syllable confusion
     probabilities to generate variants.

The generated variants are exactly what should be PRELOADED into Deepgram
as `keywords` so the ASR's beam search considers them. This is the
predictive layer that complements the existing reactive layer.

Usage:
    python predict_variants.py --word शटरिंग
    python predict_variants.py --word पेमेंट --top-k 8
    python predict_variants.py --batch       # samples + dictionary
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def syllabify_devanagari(word: str) -> list[str]:
    """Reused from build_confusion_matrix.py."""
    sylls: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in "\u0905\u0906\u0907\u0908\u0909\u090a\u090b\u090f\u0910\u0913\u0914":
            sylls.append(ch)
            i += 1
            continue
        if not re.match(r"[\u0900-\u097F]", ch):
            sylls.append(ch)
            i += 1
            continue
        cluster = ch
        i += 1
        while i < n and word[i] == "\u094d" and i + 1 < n:
            cluster += word[i] + word[i + 1]
            i += 2
        while i < n and word[i] in "\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0946\u0947\u0948\u094a\u094b\u094c\u0902\u0903\u0901":
            cluster += word[i]
            i += 1
        sylls.append(cluster)
    return sylls


class VariantPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        self.artifacts = artifacts_dir
        self.phrase_pairs: dict[str, list[dict]] = defaultdict(list)
        self.reverse_pairs: dict[str, list[dict]] = defaultdict(list)
        self.syll_confusion: dict[str, list[tuple[str, float]]] = {}
        self.char_confusion: dict[str, list[tuple[str, float]]] = {}
        self._load_all()

    def _load_all(self) -> None:
        # 1. Phrase-level pairs (heard → correct)
        phrases_file = self.artifacts / "confusion_phrases.json"
        if phrases_file.exists():
            data = json.loads(phrases_file.read_text(encoding="utf-8"))
            for p in data:
                self.phrase_pairs[p["heard"]].append(p)
                self.reverse_pairs[p["correct"]].append(p)

        # 2. Syllable-level confusion (actual → [(heard, prob), ...])
        syll_file = self.artifacts / "confusion_matrix_syll.json"
        if syll_file.exists():
            data = json.loads(syll_file.read_text(encoding="utf-8"))
            for actual, lst in data.items():
                self.syll_confusion[actual] = [
                    (h, float(p)) for h, p in lst
                ]

        # 3. Character-level confusion
        char_file = self.artifacts / "confusion_matrix_char.json"
        if char_file.exists():
            data = json.loads(char_file.read_text(encoding="utf-8"))
            for actual, lst in data.items():
                self.char_confusion[actual] = [
                    (h, float(p)) for h, p in lst
                ]

    # ── Mode 1: Ground-truth reverse lookup ──

    def lookup_ground_truth(self, correct: str) -> list[dict]:
        """Return all known mishearings of this `correct` word/phrase."""
        return sorted(
            self.reverse_pairs.get(correct, []),
            key=lambda p: -p.get("count", 0),
        )

    # ── Mode 2: Syllable mutation ──

    def generate_syllable_variants(self,
                                   word: str,
                                   top_k: int = 8,
                                   min_prob: float = 0.05,
                                   max_substitutions: int = 2) -> list[dict]:
        """Generate variants by applying syllable-level confusion mutations."""
        sylls = syllabify_devanagari(word)
        if not sylls:
            return []

        # For each syllable, find candidate substitutions
        candidates: list[list[tuple[str, float]]] = []
        for s in sylls:
            alts = self.syll_confusion.get(s, [])
            # Always include original at p=1.0
            choices = [(s, 1.0)]
            for h, p in alts:
                if p >= min_prob and h and h != s:
                    choices.append((h, p))
            candidates.append(choices)

        # Build variants — cartesian product but cap substitution count
        variants: dict[str, float] = {}
        for combo in itertools.product(*candidates):
            n_subs = sum(1 for (h, p), s in zip(combo, sylls) if h != s)
            if n_subs == 0 or n_subs > max_substitutions:
                continue
            var = "".join(h for h, _ in combo)
            score = 1.0
            for (h, p), s in zip(combo, sylls):
                score *= p if h != s else 1.0
            if var not in variants or variants[var] < score:
                variants[var] = score

        ranked = sorted(variants.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            {"variant": v, "score": round(score, 4),
             "method": "syllable_mutation"}
            for v, score in ranked
        ]

    # ── Combined predict ──

    def predict(self, word: str, top_k: int = 8) -> dict:
        ground_truth = self.lookup_ground_truth(word)
        synthesized = self.generate_syllable_variants(word, top_k=top_k)

        # Merge: ground-truth first, synthesized appended for novel coverage
        seen_variants: set[str] = set()
        merged: list[dict] = []

        for gt in ground_truth[:top_k]:
            v = gt["heard"]
            if v in seen_variants:
                continue
            seen_variants.add(v)
            merged.append({
                "variant": v,
                "score": min(1.0, 0.6 + 0.04 * gt.get("count", 1)),
                "method": "ground_truth",
                "occurrences": gt.get("count", 1),
            })

        for syn in synthesized:
            if syn["variant"] in seen_variants:
                continue
            if syn["variant"] == word:
                continue
            seen_variants.add(syn["variant"])
            merged.append(syn)

        merged = merged[:top_k]
        return {
            "word": word,
            "variants": merged,
            "summary": {
                "ground_truth_hits": len(ground_truth),
                "synthesized": len(synthesized),
                "returned": len(merged),
            },
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--batch", action="store_true",
                    help="Show samples for shittering, payment, PHE...")
    args = ap.parse_args()

    predictor = VariantPredictor()

    if args.batch:
        samples = [
            "शटरिंग", "पेमेंट", "ट्रांसफार्मर", "पीएचई",
            "सरिया", "केवी", "कम्पेक्शन", "जेसीबी", "सबमर्सिबल",
            "कंक्रीट", "एमबी", "ठेकेदार", "बैचिंग", "अलाइनमेंट",
            "रोलर", "वाइब्रेटर", "टाइल्स", "गर्डर",
            "अजगरहा", "पाइप लाइन",
        ]
        for s in samples:
            r = predictor.predict(s, top_k=args.top_k)
            print(f"\n━━━ {s} ━━━")
            for v in r["variants"]:
                method = v["method"]
                if method == "ground_truth":
                    print(f"  [GT  ] {v['variant']:30} "
                          f"score={v['score']:.2f}  ×{v.get('occurrences', 0)}")
                else:
                    print(f"  [SYN ] {v['variant']:30} score={v['score']:.2f}")
        return 0

    if not args.word:
        print("Pass --word <word> or --batch", file=sys.stderr)
        return 1

    result = predictor.predict(args.word, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
