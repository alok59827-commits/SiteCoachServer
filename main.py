"""Site Coach API — improved STT + coaching pipeline.

Endpoints:
  GET  /                  health check
  POST /upload-audio      backward-compatible audio coaching endpoint
                          (now uses the full improved pipeline)
  POST /transcribe-only   STT-only without coaching (for debug / UI preview)
  POST /report-correction user-reported wrong→correct pairs (TASK 5a)
  GET  /report-stats      view aggregated reports (TASK 5)
  GET  /pipeline-info     show what's wired up (debugging aid)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from sitecoach import (
    CorrectionsEngine,
    FeedbackStore,
    cleanup_transcript,
    dedupe_loops,
    deepgram_transcribe,
    generate_coaching,
    reduce_noise,
    silence_ratio,
    whisper_fallback_groq,
)
from sitecoach.transcribe import pick_best as _pick_best_transcript

logging.basicConfig(
    level=os.environ.get("SITECOACH_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Site Coach API",
    version="2.0.0",
    description=(
        "Hindi/Hinglish STT + coaching for construction & govt-dept calls. "
        "Includes noise reduction, dictionary correction, fuzzy matching, "
        "loop dedup, two-pass Groq cleanup + coaching, and Whisper fallback."
    ),
)

# ─── Singletons (built once at boot) ───
_CORRECTIONS_SEED = os.environ.get(
    "SITECOACH_CORRECTIONS_SEED",
    "python_extractor/stt_corrections_seed.json",
)
_FEEDBACK_PATH = os.environ.get(
    "SITECOACH_FEEDBACK_PATH",
    "feedback/corrections.jsonl",
)
_SILENCE_LIMIT = float(os.environ.get("SITECOACH_MAX_SILENCE", "0.6"))

corrections_engine = CorrectionsEngine(seed_path=_CORRECTIONS_SEED)
feedback_store = FeedbackStore(path=_FEEDBACK_PATH)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Backward-compatible health endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/")
def read_root():
    return {
        "status": "Site Coach Server is Running with Fixed JSON! 🌍🚀",
        "version": "2.0.0",
        "pipeline": "preprocess → deepgram → whisper-fallback → corrections "
                    "→ cleanup → coaching",
    }


@app.get("/pipeline-info")
def pipeline_info():
    return {
        "corrections_loaded": {
            "single_word": len(corrections_engine.single_word_map),
            "multi_word": len(corrections_engine.multi_word_pairs),
            "known_vocab": len(corrections_engine.known_vocab),
        },
        "feedback": feedback_store.stats(),
        "config": {
            "max_silence_ratio": _SILENCE_LIMIT,
            "corrections_seed": _CORRECTIONS_SEED,
            "feedback_path": _FEEDBACK_PATH,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main /upload-audio (backward compatible — TASK 2..4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_pipeline(audio_bytes: bytes,
                  num_speakers: int = 2,
                  audience: Optional[str] = None,
                  output_language: Optional[str] = None,
                  run_coaching: bool = True) -> dict:
    """End-to-end pipeline used by /upload-audio and /transcribe-only."""
    # TASK 3: audio preprocessing
    cleaned_bytes, audio_meta = reduce_noise(audio_bytes)
    sr_silence = audio_meta.get("silence_ratio")
    if sr_silence is not None and sr_silence > _SILENCE_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Audio is mostly silent/noise ({sr_silence:.0%} silent). "
                "कृपया साफ़ audio upload करें।"
            ),
        )

    # TASK 2: Deepgram primary
    try:
        primary = deepgram_transcribe(cleaned_bytes, num_speakers=num_speakers)
    except Exception as e:
        logger.exception("Deepgram failed; trying Whisper fallback first")
        primary = None

    # TASK 4: Whisper fallback when primary is weak/missing
    fallback = None
    if primary is None or primary["confidence"] < 0.5:
        try:
            fallback = whisper_fallback_groq(cleaned_bytes)
        except Exception as e:
            logger.warning("Whisper fallback failed: %s", e)

    if primary is None and fallback is None:
        raise HTTPException(
            status_code=502,
            detail="Both Deepgram and Whisper transcription failed.",
        )
    elif primary is None:
        chosen = fallback
        chosen["selection"] = {"reason": "deepgram_unavailable"}
    elif fallback is None:
        chosen = primary
        chosen["selection"] = {"reason": "deepgram_only"}
    else:
        chosen = _pick_best_transcript(
            primary, fallback, corrections_engine.known_vocab
        )

    # TASK 2h: loop dedup BEFORE corrections so duplicates don't bloat counts
    deduped = dedupe_loops(chosen["transcript"])

    # TASK 2f + 2g: dictionary + fuzzy correction
    corrected, applied = corrections_engine.apply(deduped)

    # TASK 2k.1: Groq cleanup pass
    cleanup = cleanup_transcript(
        corrected,
        chat_dialogue=chosen.get("chat_dialogue"),
    )

    response: dict = {
        "success": True,
        "transcript": cleanup.get("cleaned_transcript", corrected),
        "raw_transcript": chosen.get("raw_transcript", ""),
        "chat_dialogue": cleanup.get("speakers", chosen.get("chat_dialogue", [])),
        "stt": {
            "engine": chosen.get("engine"),
            "model": chosen.get("model"),
            "confidence": chosen.get("confidence"),
            "selection": chosen.get("selection"),
        },
        "audio_preprocessing": audio_meta,
        "corrections_applied": applied,
        "low_confidence_words": chosen.get("low_confidence_words", []),
        "llm_edits": cleanup.get("edits", []),
    }

    # TASK 2k.2: Groq coaching pass
    if run_coaching:
        coaching = generate_coaching(
            response["transcript"],
            audience=audience or "Peer/Colleague",
            output_language=output_language or "Hindi",
        )
        response["coaching_feedback"] = coaching

    return response


@app.post("/upload-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    audience: str = Form("Peer/Colleague"),
    output_language: str = Form("Hindi"),
    num_speakers: int = Form(2),
):
    try:
        audio_bytes = await file.read()
        return _run_pipeline(
            audio_bytes,
            num_speakers=num_speakers,
            audience=audience,
            output_language=output_language,
            run_coaching=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/upload-audio failed")
        return {"success": False, "error": str(e)}


@app.post("/transcribe-only")
async def transcribe_only(
    file: UploadFile = File(...),
    num_speakers: int = Form(2),
):
    """STT pipeline without the coaching pass — useful for live UI preview."""
    try:
        audio_bytes = await file.read()
        return _run_pipeline(
            audio_bytes,
            num_speakers=num_speakers,
            run_coaching=False,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/transcribe-only failed")
        return {"success": False, "error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# User feedback endpoints (TASK 5a + 5b)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CorrectionReport(BaseModel):
    wrongWord: str = Field(..., min_length=1)
    correctWord: str = Field(..., min_length=1)
    audioId: Optional[str] = None
    userId: Optional[str] = None
    category: str = "USER_REPORT"
    notes: str = ""


@app.post("/report-correction")
def report_correction(report: CorrectionReport):
    """Accept a user-reported correction and persist it for later seed updates."""
    try:
        entry = feedback_store.record(
            wrong_word=report.wrongWord,
            correct_word=report.correctWord,
            audio_id=report.audioId,
            user_id=report.userId,
            category=report.category,
            notes=report.notes,
        )
        return {"success": True, "entry": entry}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/report-stats")
def report_stats():
    """Return aggregate counts of reported corrections."""
    return feedback_store.stats()
