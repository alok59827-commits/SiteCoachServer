"""Site Coach backend package.

Modular implementation of the STT + coaching pipeline:

    audio bytes
        │
        ▼
    preprocess.reduce_noise        ◄── TASK 3: audio cleanup, silence check
        │
        ▼
    transcribe.deepgram_transcribe ◄── TASK 2: tuned keywords, nova-3,
        │                              hi forced, num_speakers,
        │                              confidence flags, loop dedup
        ▼  (low confidence?)
    transcribe.whisper_fallback    ◄── TASK 4: Groq whisper-large-v3
        │
        ▼
    corrections.apply              ◄── TASK 2g+f+h: dictionary +
        │                              fuzzy match + loop dedup
        ▼
    coaching.cleanup_pass          ◄── TASK 2k.1: Groq cleanup
        │
        ▼
    coaching.coach_pass            ◄── TASK 2k.2: Groq coaching
"""

from .preprocess import reduce_noise, silence_ratio
from .transcribe import deepgram_transcribe, whisper_fallback_groq
from .corrections import CorrectionsEngine, dedupe_loops, mark_low_confidence
from .coaching import cleanup_transcript, generate_coaching
from .feedback import FeedbackStore

__all__ = [
    "reduce_noise",
    "silence_ratio",
    "deepgram_transcribe",
    "whisper_fallback_groq",
    "CorrectionsEngine",
    "dedupe_loops",
    "mark_low_confidence",
    "cleanup_transcript",
    "generate_coaching",
    "FeedbackStore",
]
