#!/usr/bin/env python3
"""
Tests to verify that feature extractors in atdataset.feature produce the same
output as their original external implementations.

Comparisons:
  1. Fbank        vs  torchaudio.transforms.MelSpectrogram
  2. KaldiFbank   vs  lhotse.features.kaldi.layers.Wav2LogFilterBank
  3. WhisperFbank vs  whisper.audio.log_mel_spectrogram

Run:
    cd /star-kw/kangwei/code/ATdataset
    python tests/test_features.py          # standalone
    python -m pytest tests/test_features.py -v -p no:hydra   # pytest
"""

import math
import sys
import os

import numpy as np
import torch
import torchaudio

# Ensure the project root is on sys.path so we can import atdataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atdataset.feature import Fbank, KaldiFbank, WhisperFbank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000  # 1 second at 16 kHz

# Fixed seed for reproducibility
rng = np.random.RandomState(42)
AUDIO_NP = rng.randn(NUM_SAMPLES).astype(np.float32)
AUDIO_TORCH = torch.from_numpy(AUDIO_NP)


# ---------------------------------------------------------------------------
# 1. Fbank  vs  torchaudio
# ---------------------------------------------------------------------------

def test_fbank_shape():
    """Fbank output shape should be (1, n_mels, n_frames)."""
    ext = Fbank(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80)
    out = ext(AUDIO_TORCH, sample_rate=SAMPLE_RATE)
    # center=True -> n_frames = floor(16000/160) + 1 = 101
    assert out.shape == (1, 80, 101), f"Unexpected shape: {out.shape}"


def test_fbank_matches_torchaudio():
    """Fbank wraps torchaudio MelSpectrogram; output must be identical."""
    ext = Fbank(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80,
        center=True, power=1,
    )
    ref = mel_spec(AUDIO_TORCH.unsqueeze(0)).clamp(min=1e-7).log()
    # Pass tensor (not numpy) — Fbank.forward references self.device which is
    # only available via nn.Module parameter tracking, not as an attribute.
    out = ext(AUDIO_TORCH, sample_rate=SAMPLE_RATE)
    max_diff = float((out - ref).abs().max())
    assert max_diff < 1e-6, (
        f"Fbank diverges from torchaudio MelSpectrogram (max diff={max_diff:.2e})"
    )


# ---------------------------------------------------------------------------
# 2. KaldiFbank  vs  lhotse
# ---------------------------------------------------------------------------

def test_kaldifbank_shape():
    """KaldiFbank output shape should be (1, n_frames, num_filters)."""
    ext = KaldiFbank(
        sampling_rate=SAMPLE_RATE, frame_length=0.025, frame_shift=0.01,
        window_type="povey", snip_edges=False, energy_floor=1e-10,
        low_freq=20.0, high_freq=-400.0, num_filters=80, dither=0.0,
        torchaudio_compatible_mel_scale=True,
    )
    out = ext(AUDIO_TORCH.unsqueeze(0), sample_rate=SAMPLE_RATE)
    # snip_edges=False, 16000 samples -> 100 frames, 80 filters
    assert out.shape == (1, 100, 80), f"Unexpected shape: {out.shape}"


def test_kaldifbank_matches_lhotse():
    """KaldiFbank reimplements lhotse Wav2LogFilterBank; outputs should match."""
    from lhotse.features.kaldi.layers import Wav2LogFilterBank

    params = dict(
        sampling_rate=SAMPLE_RATE, frame_length=0.025, frame_shift=0.01,
        window_type="povey", snip_edges=False, energy_floor=1e-10,
        low_freq=20.0, high_freq=-400.0, num_filters=80, dither=0.0,
        torchaudio_compatible_mel_scale=True,
    )
    ours = KaldiFbank(**params)
    ref = Wav2LogFilterBank(**params)

    wav = AUDIO_TORCH.unsqueeze(0)
    ref_out = ref(wav)
    out = ours(wav, sample_rate=SAMPLE_RATE)
    max_diff = float((out - ref_out).abs().max())
    assert max_diff < 1e-4, (
        f"KaldiFbank diverges from lhotse Wav2LogFilterBank (max diff={max_diff:.2e})"
    )


# ---------------------------------------------------------------------------
# 3. WhisperFbank  vs  openai-whisper
# ---------------------------------------------------------------------------

def test_whisperfbank_shape():
    """WhisperFbank output shape should be (1, n_mels, n_frames)."""
    ext = WhisperFbank(
        sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80,
    )
    out = ext(AUDIO_NP, sample_rate=SAMPLE_RATE)
    # n_frames = floor(16000/160) = 100
    assert out.shape == (1, 80, 100), f"Unexpected shape: {out.shape}"


def test_whisperfbank_output_range():
    """Whisper normalisation should map output to [-1, 0]."""
    ext = WhisperFbank(
        sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80,
    )
    out = ext(AUDIO_NP, sample_rate=SAMPLE_RATE)
    assert float(out.min()) >= -1.0 - 1e-6, f"Min {float(out.min()):.4f} below -1"
    assert float(out.max()) <= 0.0 + 1e-6, f"Max {float(out.max()):.4f} above 0"


def test_whisperfbank_mel_filters():
    """WhisperFbank mel filters should match librosa.filters.mel (used by whisper).

    whisper.audio loads mel filters from a .npz originally generated by
    librosa.filters.mel(sr=16000, n_fft=400, n_mels=80).  The custom
    _whisper_mel_filters must reproduce the same filters.
    """
    import librosa
    import whisper.audio as waudio

    ref_filters = waudio.mel_filters("cpu", 80)  # from whisper's bundled npz
    librosa_filters = torch.from_numpy(
        librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    )
    our_filters = WhisperFbank._whisper_mel_filters(
        n_mels=80, n_fft=400, sample_rate=16000,
    ) if hasattr(WhisperFbank, "_whisper_mel_filters") else None

    # Also import the standalone function
    from atdataset.feature import _whisper_mel_filters
    our_filters = _whisper_mel_filters(n_mels=80, n_fft=400, sample_rate=16000)

    # Verify whisper's npz matches librosa (sanity check)
    assert torch.allclose(ref_filters, librosa_filters, atol=1e-7), (
        "whisper npz and librosa.filters.mel diverge — whisper may have been updated"
    )

    # Our filters should match the librosa/whisper reference
    max_diff = float((our_filters - ref_filters).abs().max())
    assert max_diff < 1e-4, (
        f"_whisper_mel_filters diverges from librosa.filters.mel "
        f"(max diff={max_diff:.2e}).  Likely missing Slaney normalization: "
        f"librosa normalizes each filter triangle so its area equals 1, "
        f"but _whisper_mel_filters uses raw unnormalized weights."
    )


def test_whisperfbank_matches_whisper():
    """WhisperFbank output should match whisper.audio.log_mel_spectrogram."""
    import whisper.audio as waudio

    ext = WhisperFbank(
        sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80,
    )
    ref = waudio.log_mel_spectrogram(AUDIO_TORCH, n_mels=80)
    out = ext(AUDIO_NP, sample_rate=SAMPLE_RATE)
    max_diff = float(
        torch.from_numpy(out).squeeze(0).sub(ref).abs().max()
    )
    assert max_diff < 1e-4, (
        f"WhisperFbank diverges from whisper.audio.log_mel_spectrogram "
        f"(max diff={max_diff:.2e})"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_fbank_shape,
    test_fbank_matches_torchaudio,
    test_kaldifbank_shape,
    test_kaldifbank_matches_lhotse,
    test_whisperfbank_shape,
    test_whisperfbank_output_range,
    test_whisperfbank_mel_filters,
    test_whisperfbank_matches_whisper,
]


def main():
    """Minimal test runner (no pytest dependency)."""
    passed = 0
    failed = 0
    errors = []

    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            errors.append((name, str(e)))
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            errors.append((name, str(e)))
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {len(ALL_TESTS)} total")
    if errors:
        print()
        for name, msg in errors:
            print(f"  FAILED: {name}")
            print(f"    {msg}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Try pytest first; fall back to built-in runner
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "-p", "no:hydra"]))
    except Exception:
        sys.exit(main())
