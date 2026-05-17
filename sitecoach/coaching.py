"""Groq LLM passes — TASK 2i + 2k.

Two distinct Groq calls:
  cleanup_transcript() — domain-aware transcript correction
  generate_coaching()   — practical Site Coach feedback JSON

Both are configured to produce JSON via Groq's `response_format="json_object"`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")

# ─── Cleanup-pass system prompt (TASK 2i) ───
_CLEANUP_PROMPT = """तुम एक Hindi/Hinglish STT cleanup expert हो।
यह बातचीत भारत में सरकारी विभागों (PHE, PWD, जल निगम, नगर निगम) के
निर्माण/शिकायत कार्य की है।

DOMAIN KNOWLEDGE (इन्हें जरूर सही रखो):
- Abbreviations: PHE (Public Health Engineering), JE (Junior Engineer),
  GIS (Geographic Information System), ESIC, EPF, MB (Measurement Book),
  CM Helpline (181), DBL, JCB, PCC, RCC, WC Policy (Workmen Compensation),
  GST, NOC, RERA.
- Common STT mistakes to fix:
    "पापा" अगर office context में हो → "वापस"
    "पंचनामा" अगर शिकायत context में हो → "पेंडिंग"
    "सीमेंट" अगर पैसे context में हो → "पेमेंट"
    "पोकलेन" अगर बीमा context में हो → "पॉलिसी"
    "गुड" → "ठीक"
    "दाई" → "भाई"
    "पैसे लाइन"/"पेशेंट लाइन" → "पाइप लाइन"
    "जीने" अगर इंजीनियर context में हो → "जेई" (JE)
    "USB" अगर डेटा/map context में हो → "GIS"
    "SSC" अगर बीमा context में हो → "ESIC"
    "PhD" अगर पानी विभाग context में हो → "PHE"
- Place names को preserve करो: अजगरहा, सरकिनी, अमिरिती, मडुवा,
  गोविंदगढ़, प्रेम नगर, रीवा.
- Numeric values verify: "5 लाख", "20 फीट", "26 गाँव" जैसे numbers सही
  format में रखो; "5 लाख → पांच बज" जैसी गलती न हो।
- Diarization fix: अगर एक speaker line में दो अलग tones/contexts हों तो
  Speaker 1 / Speaker 2 में बाँटो।
- Loop dedup: एक ही sentence 3+ बार repeat हो तो सिर्फ एक बार रखो।
- Low-confidence tokens `[?word]` के रूप में marked होंगे — उन्हें context
  से सही करके `[?]` brackets हटा दो।

OUTPUT FORMAT (strict JSON):
{
  "cleaned_transcript": "साफ़ किया हुआ पूरा transcript",
  "speakers": [{"speaker": "S1", "text": "...", "time": "00:00"}],
  "edits": [{"from": "...", "to": "...", "reason": "..."}]
}"""

# ─── Coaching-pass system prompt (TASK 2i — pass 2) ───
_COACHING_PROMPT_TEMPLATE = """तुम एक highly practical "Site Communication Coach" हो।
Engineer audio में किस से बात कर रहा है: '{audience}'.

CRITICAL RULES:
1. Grammar पर मत focus करो — construction site पर broken Hindi normal है।
2. Negotiation strategy, tone, और conflict warnings पर focus करो।
3. सारा output STRICTLY इस भाषा में दो: {output_language}.
4. "score" must be a single integer 0..100.
5. JSON must be valid (no trailing commas).

EXPECTED JSON STRUCTURE:
{{
  "score": 80,
  "mistakes": ["..."],
  "improvements": ["..."],
  "action_items": ["..."],
  "summary": "...",
  "learn_points": ["..."]
}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _client(api_key: Optional[str] = None) -> Groq:
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY env var is not set")
    return Groq(api_key=api_key)


def _trim(text: str, max_words: int = 600) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def cleanup_transcript(transcript: str,
                       chat_dialogue: Optional[list[dict]] = None,
                       api_key: Optional[str] = None) -> dict:
    """Pass 1 — domain-aware STT cleanup.

    Returns:
        {
          "cleaned_transcript": str,
          "speakers": [{speaker, text, time}],
          "edits": [{from, to, reason}],
        }
    """
    if not transcript.strip():
        return {
            "cleaned_transcript": "",
            "speakers": chat_dialogue or [],
            "edits": [],
        }

    user_payload = {
        "raw_transcript": _trim(transcript),
        "speaker_segments": chat_dialogue or [],
    }

    client = _client(api_key)
    completion = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": _CLEANUP_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=4000,
    )

    try:
        data = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        logger.warning("cleanup_transcript: JSON parse failed, returning raw")
        return {
            "cleaned_transcript": transcript,
            "speakers": chat_dialogue or [],
            "edits": [],
        }

    data.setdefault("cleaned_transcript", transcript)
    data.setdefault("speakers", chat_dialogue or [])
    data.setdefault("edits", [])
    return data


def generate_coaching(cleaned_transcript: str,
                      audience: str = "Peer/Colleague",
                      output_language: str = "Hindi",
                      api_key: Optional[str] = None) -> dict:
    """Pass 2 — coaching feedback JSON."""
    if not cleaned_transcript.strip():
        return {
            "score": 0,
            "mistakes": ["कोई transcript नहीं मिला"],
            "improvements": [],
            "action_items": [],
            "summary": "",
            "learn_points": [],
        }

    system = _COACHING_PROMPT_TEMPLATE.format(
        audience=audience,
        output_language=output_language,
    )
    client = _client(api_key)
    completion = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _trim(cleaned_transcript, 800)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=3000,
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError as e:
        logger.warning("generate_coaching: JSON parse failed: %s", e)
        return {
            "score": 0,
            "mistakes": ["LLM did not return valid JSON"],
            "improvements": [],
            "action_items": [],
            "summary": cleaned_transcript[:200],
            "learn_points": [],
        }
