#!/usr/bin/env python3
"""
Mega Hindi Dictionary Builder
==============================
Uses pyiwn (IIT Bombay IndoWordNet) to extract 1,00,000+ Hindi words
with meanings, synonyms, and domain-specific categories.

Reads the raw ontology data shipped with pyiwn to classify each synset
into professional domains: Medical, Legal, Civil/Construction,
Agriculture, Science, Technology, Education, Arts, Sports, Religion,
Military, Finance, and General.

Usage:
    pip install pyiwn
    python build_megadict.py            # outputs master_hindi_dict.json
    python build_megadict.py --stats    # also prints category breakdown

Output schema:
    [{"word": "...", "meaning": "...", "synonyms": "...", "category": "..."}]
"""

import json
import os
import sys
import time
from pathlib import Path

# ─── Ontology node ID → custom domain mapping ───
# Built from ~/iwn_data/ontology/nodes (223 nodes).
# Each key is an ontology node ID; the value is our app category.
# Nodes not listed here fall through to "General".

ONTOLOGY_TO_DOMAIN: dict[int, str] = {
    # ── Medical / Health ──
    30:  "Medical",       # शारीरिक वस्तु (Anatomical)
    81:  "Medical",       # शारीरिक अवस्था (Physiological State)
    82:  "Medical",       # Disease
    86:  "Medical",       # रोग (Disease)
    87:  "Medical",       # जैविक अवस्था (Biological State)
    18:  "Medical",       # सूक्ष्म-जीव (Micro organism)
    31:  "Medical",       # रासायनिक वस्तु (Chemical)
    62:  "Medical",       # शारीरिक कार्य (Physical action — medical context)
    169: "Medical",       # शारीरिक कार्यसूचक (bodily action)
    144: "Medical",       # भौतिक अवस्थासूचक (Physical State verb)

    # ── Legal ──
    65:  "Legal",         # समाज शास्त्र (Social Sciences — includes कानून)
    59:  "Legal",         # असामाजिक कार्य (Anti-social)

    # ── Civil / Construction / Engineering ──
    72:  "Civil/Construction",  # व्यवहार विज्ञान (Applied Sciences — engineering, agriculture)
    27:  "Civil/Construction",  # मानवकृति (Artifact)
    37:  "Civil/Construction",  # भौतिक स्थान (Physical Place)
    34:  "Civil/Construction",  # स्थान (Place)
    83:  "Civil/Construction",  # भौतिक अवस्था (Physical State)
    174: "Civil/Construction",  # रूप (Form) — ठोस, द्रव, गैस
    175: "Civil/Construction",  # ठोस (Solid)
    50:  "Civil/Construction",  # माप (Measurement)
    191: "Civil/Construction",  # मापसूचक (Measurement adj)

    # ── Agriculture ──
    7:   "Agriculture",   # वनस्पति (Flora)
    8:   "Agriculture",   # झाड़ी (Shrub)
    9:   "Agriculture",   # जलीय वनस्पति (Aquatic Plant)
    22:  "Agriculture",   # लता (Climber)
    23:  "Agriculture",   # वृक्ष (Tree)
    29:  "Agriculture",   # खाद्य (Edible)
    32:  "Agriculture",   # पेय (Drinkable)
    75:  "Agriculture",   # गृह विज्ञान (Home Science — food/nutrition)

    # ── Science ──
    70:  "Science",       # प्राकृतिक विज्ञान (Natural Sciences)
    71:  "Science",       # गणित (Mathematics)
    80:  "Science",       # प्रक्रिया (Process)
    89:  "Science",       # भौतिक प्रक्रिया (Physical Process)
    158: "Science",       # प्राकृतिक प्रक्रिया (Natural Process)
    177: "Science",       # गैस (Gas)
    176: "Science",       # द्रव (Liquid)
    48:  "Science",       # रंग (Colour)

    # ── Education ──
    54:  "Education",     # विषय ज्ञान (Logos)
    47:  "Education",     # ज्ञान (Cognition)
    66:  "Education",     # भाषा (Language)
    125: "Education",     # ज्ञानसूचक (Cognition verb)

    # ── Arts / Culture ──
    67:  "Arts/Culture",  # कला (The Arts)
    153: "Arts/Culture",  # कला (Art — node 153)
    77:  "Arts/Culture",  # फैशन डिज़ाइनिंग (Fashion Designing)
    140: "Arts/Culture",  # प्रदर्शनसूचक (Performance)

    # ── Sports ──
    73:  "Sports",        # खेल (Sports)
    128: "Sports",        # प्रतिस्पर्धासूचक (Competition verb)

    # ── Religion / Philosophy ──
    63:  "Religion",      # धर्म (Religion)
    64:  "Religion",      # दर्शन (Philosophy)
    166: "Religion",      # संकल्पना (Concept — ईश्वर, स्वर्ग)
    36:  "Religion",      # काल्पनिक स्थान (Imaginary Place — स्वर्ग, नरक)
    19:  "Religion",      # काल्पनिक प्राणी (Imaginary Creatures)
    33:  "Religion",      # काल्पनिक वस्तु (Imaginary Object)
    164: "Religion",      # पौराणिक काल (Mythological Period)
    173: "Religion",      # पौराणिक जीव (Mythological Character)
    194: "Religion",      # पौराणिक वस्तु (Mythological Object)
    214: "Religion",      # पौराणिक स्थान (Mythological Place)

    # ── Military / Defence ──
    43:  "Military",      # घातक घटना (Fatal Event)
    133: "Military",      # विनाशसूचक (Destruction verb)
    183: "Military",      # कार्यसूचक Act (attack)

    # ── Finance / Commerce ──
    141: "Finance",       # अधिकारसूचक (Possession verb)
    156: "Finance",       # स्वामित्व (Possession noun)

    # ── Media / Communication ──
    76:  "Media",         # जनसंपर्क साधन (Mass media)
    61:  "Media",         # संप्रेषण (Communication)
    126: "Media",         # संप्रेषणसूचक (Communication verb)
    195: "Media",         # संज्ञापन (Communication noun)

    # ── Transport ──
    74:  "Transport",     # वाहन (Transport)
    138: "Transport",     # गतिसूचक (Motion verb)

    # ── History / Geography ──
    68:  "History/Geography",  # भूगोल (Geography)
    69:  "History/Geography",  # इतिहास (History)
    39:  "History/Geography",  # ऐतिहासिक घटना (Historical Event)
    57:  "History/Geography",  # ऐतिहासिक युग (Historical Ages)

    # ── Fauna / Zoology ──
    10:  "Fauna/Zoology", # जन्तु (Fauna)
    11:  "Fauna/Zoology", # स्तनपायी (Mammal)
    12:  "Fauna/Zoology", # सरीसृप (Reptile)
    13:  "Fauna/Zoology", # उभयचर (Amphibian)
    14:  "Fauna/Zoology", # जलीय-जन्तु (Aquatic Animal)
    15:  "Fauna/Zoology", # पक्षी (Birds)
    16:  "Fauna/Zoology", # Fish
    17:  "Fauna/Zoology", # कीट (Insects)
    21:  "Fauna/Zoology", # वानर (Ape)
    24:  "Fauna/Zoology", # लघु स्तनपायी (Lesser Mammals)
    212: "Fauna/Zoology", # मछली (Fish)
    213: "Fauna/Zoology", # जलीय-स्तनपायी (Aquatic mammal)

    # ── Weather / Environment ──
    38:  "Weather/Environment",  # प्राकृतिक घटना (Natural Event)
    56:  "Weather/Environment",  # ऋतु (Season)
    198: "Weather/Environment",  # Natural State
    199: "Weather/Environment",  # Season
    200: "Weather/Environment",  # Weather
}

# ── Keyword-based fallback for synsets without ontology mapping ──
KEYWORD_DOMAIN_RULES: list[tuple[list[str], str]] = [
    # Medical
    (["रोग", "चिकित्सा", "औषधि", "दवा", "शल्य", "अस्पताल", "रक्त",
      "शरीर", "हड्डी", "मांसपेशी", "तंत्रिका", "हृदय", "फेफड़", "गुर्द",
      "यकृत", "मस्तिष्क", "ज्वर", "संक्रमण", "विटामिन", "प्रोटीन",
      "जीवाणु", "विषाणु", "टीका", "इंजेक्शन", "सर्जरी"], "Medical"),

    # Legal
    (["कानून", "न्याय", "अदालत", "वकील", "मुकदमा", "अपराध", "दंड",
      "संविधान", "अधिकार", "धारा", "जमानत", "गवाह", "फैसला",
      "अभियोग", "याचिका", "पुलिस"], "Legal"),

    # Civil / Construction
    (["निर्माण", "भवन", "सीमेंट", "ईंट", "लोहा", "कंक्रीट", "शटरिंग",
      "प्लिंथ", "नींव", "खंभा", "दीवार", "छत", "पुल", "सड़क",
      "पाइपलाइन", "नक्शा", "ठेकेदार", "मिस्त्री", "राजगीर",
      "बिजली", "प्लंबिंग", "JCB"], "Civil/Construction"),

    # Agriculture
    (["कृषि", "खेती", "फसल", "बीज", "सिंचाई", "उर्वरक", "कीटनाशक",
      "मिट्टी", "हल", "ट्रैक्टर", "किसान", "बुआई", "कटाई", "खाद",
      "पौधा", "अनाज", "गेहूं", "चावल", "धान"], "Agriculture"),

    # Finance
    (["बैंक", "ऋण", "ब्याज", "मुद्रा", "शेयर", "निवेश", "कर", "बजट",
      "लेखा", "वित्त", "बीमा", "पूंजी", "लाभ", "हानि"], "Finance"),
]


def load_ontology_map(iwn_data_dir: str) -> dict[int, list[int]]:
    """Load synset_id → [ontology_node_ids] from the raw IWN data."""
    map_path = os.path.join(iwn_data_dir, "ontology", "map")
    synset_to_nodes: dict[int, list[int]] = {}

    if not os.path.exists(map_path):
        print(f"  Warning: ontology map not found at {map_path}")
        return synset_to_nodes

    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                synset_id = int(parts[0])
                node_ids = [int(n.strip()) for n in parts[1].split(",") if n.strip().isdigit()]
                synset_to_nodes[synset_id] = node_ids
            except ValueError:
                continue

    return synset_to_nodes


def classify_by_ontology(
    node_ids: list[int],
) -> str | None:
    """Map ontology node IDs to a domain using the curated mapping."""
    for nid in node_ids:
        if nid in ONTOLOGY_TO_DOMAIN:
            return ONTOLOGY_TO_DOMAIN[nid]
    return None


def classify_by_keywords(gloss: str, head_word: str) -> str:
    """Fallback: scan the gloss + head_word for domain keywords."""
    combined = f"{gloss} {head_word}".lower()
    for keywords, domain in KEYWORD_DOMAIN_RULES:
        for kw in keywords:
            if kw in combined:
                return domain
    return "General"


def find_iwn_data_dir() -> str:
    """Locate the IndoWordNet data directory."""
    candidates = [
        os.path.expanduser("~/iwn_data"),
        os.path.join(os.path.dirname(__file__), "iwn_data"),
        "iwn_data",
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "ontology", "map")):
            return path
    return candidates[0]


def build_dictionary(show_stats: bool = False) -> list[dict]:
    """Main extraction pipeline."""
    import pyiwn

    print("Step 1/4: Loading IndoWordNet Hindi data...")
    iwn = pyiwn.IndoWordNet(pyiwn.Language.HINDI)

    print("Step 2/4: Loading ontology map for domain classification...")
    iwn_data_dir = find_iwn_data_dir()
    synset_to_nodes = load_ontology_map(iwn_data_dir)
    print(f"  Loaded ontology mappings for {len(synset_to_nodes)} synsets.")

    print("Step 3/4: Extracting words from all synsets...")
    all_synsets = iwn.all_synsets()
    print(f"  Total synsets: {len(all_synsets)}")

    entries: list[dict] = []
    seen_words: set[str] = set()
    category_counts: dict[str, int] = {}

    for i, synset in enumerate(all_synsets):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(all_synsets)} synsets...")

        try:
            lemmas = synset.lemma_names()
            gloss = synset.gloss() or ""
            sid = synset.synset_id()
        except Exception:
            continue

        if not lemmas:
            continue

        # Determine category
        node_ids = synset_to_nodes.get(sid, [])
        category = classify_by_ontology(node_ids)
        if category is None:
            category = classify_by_keywords(gloss, lemmas[0])

        for lemma in lemmas:
            word = lemma.strip()
            if not word or word in seen_words:
                continue
            seen_words.add(word)

            other_lemmas = [l.strip() for l in lemmas if l.strip() != word]
            synonyms = ", ".join(other_lemmas) if other_lemmas else ""

            entries.append({
                "word": word,
                "meaning": gloss,
                "synonyms": synonyms,
                "category": category,
            })

            category_counts[category] = category_counts.get(category, 0) + 1

    entries.sort(key=lambda e: e["word"])

    print(f"\nStep 4/4: Extraction complete!")
    print(f"  Total unique words: {len(entries)}")
    print(f"  Categories: {len(category_counts)}")

    if show_stats:
        print("\n  ── Category Breakdown ──")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:25s} → {count:>6,} words")

    return entries


def main():
    show_stats = "--stats" in sys.argv

    start_time = time.time()

    try:
        import pyiwn  # noqa: F401
    except ImportError:
        print("Error: pyiwn is not installed.")
        print("  pip install pyiwn")
        sys.exit(1)

    # Ensure IWN data is downloaded
    iwn_data_dir = find_iwn_data_dir()
    if not os.path.isdir(iwn_data_dir):
        print("Downloading IndoWordNet data (~31 MB)...")
        pyiwn.download()

    entries = build_dictionary(show_stats=show_stats)

    output_path = Path(__file__).parent / "master_hindi_dict.json"
    print(f"\nWriting {len(entries)} entries to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"Done! File size: {file_size_mb:.1f} MB ({elapsed:.1f}s)")
    print(f"\nSample entries:")
    for entry in entries[:5]:
        print(f"  {entry['word']}: {entry['meaning'][:60]}... [{entry['category']}]")


if __name__ == "__main__":
    main()
