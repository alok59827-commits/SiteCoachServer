"""Audio preprocessing — TASK 3.

Applies stationary-noise reduction via `noisereduce` and computes a
silence ratio so the API can early-return on dead/noisy uploads.

All routines accept and return raw audio bytes so callers don't have to
deal with numpy directly.
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def _decode(audio_bytes: bytes) -> Tuple["np.ndarray", int]:  # noqa: F821
    """Decode any common audio container into mono float32 numpy + samplerate.

    Uses soundfile (libsndfile) which handles wav/flac/ogg/opus natively
    and falls back to pydub+ffmpeg for mp3/m4a/webm.
    """
    import numpy as np

    try:
        import soundfile as sf

        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, always_2d=False, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)
    except Exception as e:
        logger.info("soundfile decode failed (%s); trying pydub fallback", e)

    from pydub import AudioSegment

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * seg.sample_width - 1))
    return samples, sr


def _encode_wav(samples, sr: int) -> bytes:
    """Encode a mono float32 array back to a 16-bit PCM WAV buffer."""
    import numpy as np
    import soundfile as sf

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def silence_ratio(audio_bytes: bytes, rms_threshold: float = 0.005) -> float:
    """Return the fraction of audio frames whose RMS energy is below threshold.

    A return value of 0.6 means 60% of the audio is effectively silent/noise.
    """
    import numpy as np

    try:
        samples, sr = _decode(audio_bytes)
    except Exception as e:
        logger.warning("silence_ratio: decode failed: %s", e)
        return 0.0

    if samples.size == 0:
        return 1.0

    frame_size = max(int(sr * 0.025), 1)
    n_frames = max(samples.size // frame_size, 1)
    trimmed = samples[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    silent = int(np.sum(rms < rms_threshold))
    return float(silent) / float(n_frames)


def reduce_noise(audio_bytes: bytes,
                 prop_decrease: float = 0.85) -> Tuple[bytes, dict]:
    """Run stationary noise reduction on the input audio.

    Returns the cleaned WAV bytes plus metadata (sr, duration, silence_ratio).
    On any failure the original bytes are returned unchanged so the
    pipeline degrades gracefully.
    """
    metadata: dict = {
        "preprocessed": False,
        "duration_sec": None,
        "samplerate": None,
        "silence_ratio": None,
    }

    try:
        import noisereduce as nr  # type: ignore

        samples, sr = _decode(audio_bytes)
        metadata["duration_sec"] = round(len(samples) / max(sr, 1), 2)
        metadata["samplerate"] = sr

        reduced = nr.reduce_noise(
            y=samples,
            sr=sr,
            stationary=True,
            prop_decrease=prop_decrease,
        )
        out = _encode_wav(reduced, sr)
        metadata["preprocessed"] = True
        # Recompute silence on cleaned signal
        metadata["silence_ratio"] = silence_ratio(out)
        return out, metadata
    except Exception as e:
        logger.warning("reduce_noise: falling back to raw audio: %s", e)
        try:
            metadata["silence_ratio"] = silence_ratio(audio_bytes)
        except Exception:
            metadata["silence_ratio"] = None
        return audio_bytes, metadata
