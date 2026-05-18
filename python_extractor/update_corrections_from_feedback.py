#!/usr/bin/env python3
"""
Merge user-reported corrections into stt_corrections_seed.json.

Reads feedback/corrections.jsonl (produced by the /report-correction
endpoint), aggregates by (wrongWord, correctWord) pair, and folds them
into the seed file with a confidence score driven by report count and
agreement.

Usage:
    python update_corrections_from_feedback.py
    python update_corrections_from_feedback.py \
        --feedback feedback/corrections.jsonl \
        --seed python_extractor/stt_corrections_seed.json \
        --min-reports 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_seed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_seed(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", default="feedback/corrections.jsonl")
    parser.add_argument(
        "--seed", default="python_extractor/stt_corrections_seed.json"
    )
    parser.add_argument("--min-reports", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    feedback_path = Path(args.feedback)
    seed_path = Path(args.seed)

    feedback = load_jsonl(feedback_path)
    seed = load_seed(seed_path)
    print(f"Loaded {len(feedback)} feedback entries, {len(seed)} seed entries.")

    # Aggregate user reports by (wrong, correct) pair
    counts: Counter[tuple[str, str]] = Counter()
    categories: dict[tuple[str, str], str] = {}
    for fb in feedback:
        key = (fb["wrongWord"].strip(), fb["correctWord"].strip())
        if not key[0] or not key[1]:
            continue
        counts[key] += 1
        categories.setdefault(key, fb.get("category", "USER_REPORT"))

    print(f"Distinct user-reported pairs: {len(counts)}")

    # Build an index of existing seed entries for quick lookup
    seed_index: dict[tuple[str, str], int] = {}
    for i, entry in enumerate(seed):
        seed_index[(entry["wrongWord"], entry["correctWord"])] = i

    added = updated = skipped = 0
    for (wrong, right), n in counts.items():
        if n < args.min_reports:
            skipped += 1
            continue
        confidence = min(0.6 + 0.05 * n, 0.95)
        category = categories.get((wrong, right), "USER_REPORT")

        if (wrong, right) in seed_index:
            idx = seed_index[(wrong, right)]
            existing_conf = seed[idx].get("confidence", 0.5)
            seed[idx]["confidence"] = round(
                max(float(existing_conf), confidence), 2
            )
            seed[idx]["source"] = "expert_xlsx" if seed[idx].get("source") == "expert_xlsx" \
                else "user_feedback"
            seed[idx]["user_reports"] = n
            updated += 1
        else:
            seed.append({
                "wrongWord": wrong,
                "correctWord": right,
                "category": category,
                "confidence": round(confidence, 2),
                "source": "user_feedback",
                "user_reports": n,
            })
            added += 1

    print(f"Added: {added}, Updated: {updated}, Skipped (below threshold): {skipped}")
    if args.dry_run:
        print("--dry-run: not writing back to seed file.")
        return 0

    save_seed(seed_path, seed)
    print(f"Wrote {len(seed)} entries to {seed_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
