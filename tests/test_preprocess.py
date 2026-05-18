"""Tests for the audio preprocessing layer."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from sitecoach.preprocess import reduce_noise, silence_ratio


def _make_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def silent_audio():
    return _make_wav(np.zeros(16000, dtype=np.float32))


@pytest.fixture
def speechlike_audio():
    rng = np.random.default_rng(42)
    sr = 16000
    t = np.arange(sr * 2) / sr
    # Mix of low-frequency tones to mimic speech energy
    speech = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * rng.normal(size=sr * 2)
    ).astype(np.float32)
    return _make_wav(speech, sr=sr)


class TestSilenceRatio:
    def test_all_silent_returns_high(self, silent_audio):
        assert silence_ratio(silent_audio) > 0.95

    def test_speech_returns_low(self, speechlike_audio):
        ratio = silence_ratio(speechlike_audio)
        assert ratio < 0.4


class TestReduceNoise:
    def test_returns_bytes_and_metadata(self, speechlike_audio):
        out, meta = reduce_noise(speechlike_audio)
        assert isinstance(out, (bytes, bytearray))
        assert meta["samplerate"] == 16000
        assert meta["duration_sec"] is not None
        # On well-formed audio it should have succeeded
        assert meta["preprocessed"] is True

    def test_graceful_on_invalid_bytes(self):
        out, meta = reduce_noise(b"not audio at all")
        # Should return the input unchanged when decode fails
        assert out == b"not audio at all"
        assert meta["preprocessed"] is False
