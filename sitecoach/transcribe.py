"""STT layer — TASK 2 + TASK 4.

Wraps Deepgram (primary) and Groq Whisper-large-v3 (fallback).
Designed so the FastAPI route never has to care about which engine
produced the transcript; both return a normalised dict.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq

logger = logging.getLogger(__name__)

# ─── Tuned keyword vocabulary (TASK 2a) ───
# Covers technical abbreviations, English-Hindi mix terms, place names,
# and the trade/construction domain pulled from AI mistake.xlsx.
DEEPGRAM_KEYWORDS: list[str] = [
    # Govt departments / abbreviations
    "PHE", "PWD", "CPWD", "NHAI", "IRC", "NBC", "RERA", "ESIC", "EPF",
    "GST", "MB", "JE", "GIS", "USB", "CM Helpline", "DBL", "JCB",
    "WC Policy", "NOC", "RTI", "PMFBY", "KCC", "MGNREGA", "FASTag",
    # Construction terms
    "PCC", "RCC", "TMT", "DPC", "STP", "BOQ", "RMC", "M20", "M25",
    "Shuttering", "Plinth", "Rebar", "Stirrup", "Mortar", "Concrete",
    "Cement", "Beam", "Column", "Slab", "Lintel", "Chajja", "Parapet",
    "Curing", "Foundation", "Footing", "Aggregate", "Reinforcement",
    "Pipeline", "Bore Well", "Submersible", "Manhole", "Septic Tank",
    "Excavator", "Bulldozer", "Mixer", "Vibrator", "Roller", "Crane",
    "Rising main", "Backfill", "Layout", "Levelling", "Survey",
    # English words STT often mishears
    "Payment", "Forward", "Pending", "Duty", "Brief", "Pipeline",
    "Policy", "Delhivery", "Photo", "Phone", "Meter", "Kilometer",
    "Sir", "Madam", "Sahab", "Number", "Clear", "Bill", "Block",
    "Sewage", "Drainage", "Tanker", "Boring", "Hand pump",
    # Hindi proper nouns / villages from xlsx
    "अजगरहा", "सरकिनी", "अमिरिती", "मडुवा", "गोविंदगढ़",
    "प्रेम नगर", "रीवा", "जल निगम", "नगर निगम", "ग्राम पंचायत",
    # Roles
    "ठेकेदार", "मिस्त्री", "राजगीर", "बेलदार", "इंजीनियर",
    "सुपरवाइज़र", "साइट इंजीनियर", "जूनियर इंजीनियर",
    # Hindi-script abbreviations Deepgram should recognise
    "पीएचई", "पीडब्ल्यूडी", "जीआईएस", "ईएसआईसी", "ईपीएफ",
    "सीएम हेल्पलाइन", "जेई", "एमबी", "डब्ल्यूसी पॉलिसी",
    "पाइप लाइन", "पाइपलाइन", "बोरवेल", "सीवेज",
    # Misc work vocabulary
    "Shuttering", "Plinth", "PCC", "RCC", "Panchayat", "Contractor",
]

# Dedupe while preserving order
_seen: set[str] = set()
DEEPGRAM_KEYWORDS = [k for k in DEEPGRAM_KEYWORDS
                     if not (k.lower() in _seen or _seen.add(k.lower()))]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deepgram primary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_options(num_speakers: int, model: str) -> PrerecordedOptions:
    """Build Deepgram options with TASK 2a-d tuning."""
    kwargs = dict(
        model=model,
        language="hi",                  # TASK 2c: force Hindi
        detect_language=False,
        smart_format=True,
        diarize=True,
        punctuate=True,
        utterances=True,
        keywords=DEEPGRAM_KEYWORDS[:200],
    )
    # Older deepgram-sdk may not accept num_speakers; set if supported
    try:
        return PrerecordedOptions(**kwargs, num_speakers=num_speakers)
    except TypeError:
        return PrerecordedOptions(**kwargs)


def deepgram_transcribe(audio_bytes: bytes,
                        api_key: Optional[str] = None,
                        num_speakers: int = 2,
                        prefer_model: str = "nova-3") -> dict:
    """Call Deepgram and return a normalised payload.

    Output:
        {
          "engine":    "deepgram",
          "model":     "nova-3" | "nova-2",
          "transcript": str,            # full text with low-confidence
                                        # words wrapped as [?word] (TASK 2e)
          "raw_transcript": str,        # without low-confidence marking
          "confidence": float,          # overall confidence
          "low_confidence_words": [str],
          "chat_dialogue": [{speaker, text, time}],
        }
    """
    api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY env var is not set")

    client = DeepgramClient(api_key)

    # TASK 2b: prefer nova-3, fall back to nova-2 on error
    used_model = prefer_model
    try:
        opts = _build_options(num_speakers, prefer_model)
        response = client.listen.rest.v("1").transcribe_file(
            {"buffer": audio_bytes}, opts
        )
    except Exception as e:
        logger.warning("Deepgram %s failed (%s); falling back to nova-2",
                       prefer_model, e)
        used_model = "nova-2"
        opts = _build_options(num_speakers, "nova-2")
        response = client.listen.rest.v("1").transcribe_file(
            {"buffer": audio_bytes}, opts
        )

    alt = response.results.channels[0].alternatives[0]
    raw_transcript = alt.transcript or ""
    overall_conf = getattr(alt, "confidence", 0.0) or 0.0

    words = getattr(alt, "words", None) or []

    # TASK 2e: tag low-confidence words inline for the LLM
    marked_tokens: list[str] = []
    low_conf_words: list[str] = []
    for w in words:
        wtxt = getattr(w, "punctuated_word", None) or getattr(w, "word", "")
        if not wtxt:
            continue
        c = getattr(w, "confidence", 1.0)
        if c is not None and c < 0.6:
            marked_tokens.append(f"[?{wtxt}]")
            low_conf_words.append(wtxt)
        else:
            marked_tokens.append(wtxt)
    marked_transcript = " ".join(marked_tokens) if marked_tokens else raw_transcript

    chat_dialogue = _build_chat_dialogue(words, raw_transcript)

    return {
        "engine": "deepgram",
        "model": used_model,
        "transcript": marked_transcript,
        "raw_transcript": raw_transcript,
        "confidence": float(overall_conf),
        "low_confidence_words": low_conf_words,
        "chat_dialogue": chat_dialogue,
    }


def _build_chat_dialogue(words, fallback_transcript: str) -> list[dict]:
    """Reconstruct speaker-segmented dialogue from word-level diarization."""
    if not words:
        return [{"speaker": "S1", "text": fallback_transcript, "time": "00:00"}]

    dialogue: list[dict] = []
    current_speaker = getattr(words[0], "speaker", 0) or 0
    current_sentence: list[str] = []
    start_time = getattr(words[0], "start", 0.0) or 0.0

    for w in words:
        speaker = getattr(w, "speaker", 0) or 0
        token = getattr(w, "punctuated_word", None) or getattr(w, "word", "")
        if speaker == current_speaker:
            current_sentence.append(token)
        else:
            mins, secs = int(start_time // 60), int(start_time % 60)
            dialogue.append({
                "speaker": f"S{current_speaker + 1}",
                "text": " ".join(current_sentence),
                "time": f"{mins:02d}:{secs:02d}",
            })
            current_speaker = speaker
            current_sentence = [token]
            start_time = getattr(w, "start", start_time) or start_time

    if current_sentence:
        mins, secs = int(start_time // 60), int(start_time % 60)
        dialogue.append({
            "speaker": f"S{current_speaker + 1}",
            "text": " ".join(current_sentence),
            "time": f"{mins:02d}:{secs:02d}",
        })
    return dialogue


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Groq Whisper fallback (TASK 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")


def whisper_fallback_groq(audio_bytes: bytes,
                          api_key: Optional[str] = None) -> dict:
    """Transcribe with Groq's whisper-large-v3 endpoint.

    Returns a payload shaped like `deepgram_transcribe` for symmetry.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY env var is not set")

    client = Groq(api_key=api_key)

    # Groq audio endpoint expects a file-like object with a name attribute
    bio = io.BytesIO(audio_bytes)
    bio.name = "audio.wav"

    response = client.audio.transcriptions.create(
        file=("audio.wav", bio.getvalue()),
        model=WHISPER_MODEL,
        language="hi",
        response_format="verbose_json",
        temperature=0.0,
    )

    text = getattr(response, "text", "") or ""
    return {
        "engine": "groq-whisper",
        "model": WHISPER_MODEL,
        "transcript": text,
        "raw_transcript": text,
        "confidence": 0.75,
        "low_confidence_words": [],
        "chat_dialogue": [{"speaker": "S1", "text": text, "time": "00:00"}],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hybrid selection (TASK 4b)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pick_best(primary: dict, fallback: dict,
              known_vocab: set[str]) -> dict:
    """Pick whichever transcript shares more tokens with the known vocab."""
    def hits(t: str) -> int:
        return sum(1 for tok in t.split() if tok.lower() in known_vocab)

    p_hits = hits(primary["transcript"])
    f_hits = hits(fallback["transcript"])
    winner = primary if p_hits >= f_hits else fallback
    winner = dict(winner)
    winner["selection"] = {
        "primary_engine": primary["engine"],
        "primary_hits": p_hits,
        "fallback_engine": fallback["engine"],
        "fallback_hits": f_hits,
        "winner": winner["engine"],
    }
    return winner
