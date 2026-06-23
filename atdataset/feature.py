#!/usr/bin/env python3
# Copyright  2025 Wei Kang (wkang@pku.edu.cn)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torchaudio

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class FeatureExtractor(ABC, torch.nn.Module):
    """Base class for all feature extractors.

    Subclasses must implement:
      - ``frame_shift`` (property): frame shift in seconds.
      - ``feature_dim()``: number of feature dimensions per frame.
      - ``forward()``: feature extraction.
    """

    @property
    @abstractmethod
    def frame_shift(self) -> float: ...

    @abstractmethod
    def feature_dim(self) -> int: ...

    @abstractmethod
    def forward(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        average_channels: bool = True,
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Fbank (torchaudio-based)
# ---------------------------------------------------------------------------


class Fbank(FeatureExtractor):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = 160,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 1.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
            norm=norm,
            mel_scale=mel_scale,
        )
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.window_fn = window_fn
        self.power = power
        self.normalized = normalized
        self.wkwargs = wkwargs
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.norm = norm
        self.mel_scale = mel_scale

    def __repr__(self):
        return (
            f"Fbank(sample_rate={self.sample_rate}, "
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"f_min={self.f_min}, "
            f"f_max={self.f_max}, "
            f"pad={self.pad}, "
            f"n_mels={self.n_mels}, "
            f"window_fn={self.window_fn.__name__}, "
            f"power={self.power}, "
            f"normalized={self.normalized}, "
            f"wkwargs={self.wkwargs}, "
            f"center={self.center}, "
            f"pad_mode={self.pad_mode}, "
            f"onesided={self.onesided}, "
            f"norm={self.norm}, "
            f"mel_scale={self.mel_scale})"
        )

    def __str__(self):
        return self.__repr__()

    @property
    def frame_shift(self) -> float:
        return self.hop_length / self.sample_rate

    def feature_dim(self) -> int:
        return self.n_mels

    def forward(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        average_channels: bool = True,
    ) -> torch.Tensor:
        expected_sr = self.sample_rate
        assert sample_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sample_rate}"
        )
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples).to(self.device)

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape
            if samples.shape[0] > 1:
                if average_channels:
                    samples = samples.mean(dim=0, keepdim=True)
                else:
                    samples = samples[0:1]
        mel = self.fbank(samples)
        mel = mel.clamp(min=1e-7).log()
        return mel


class KaldiFbank(FeatureExtractor):
    """Kaldi-style log Mel filter bank feature extractor.

    Thin wrapper around :func:`torchaudio.compliance.kaldi.fbank` that
    implements the :class:`FeatureExtractor` interface.  Produces log-mel
    filter bank features identical to Kaldi's ``compute-fbank-feats``.

    The constructor accepts the same parameter names as the original
    ``KaldiFbank`` class for backward compatibility; parameters are mapped
    to the ``torchaudio`` API internally (e.g. frame lengths are converted
    from seconds to milliseconds).
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: float = 0.025,
        frame_shift: float = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = 1e-10,
        raw_energy: bool = True,
        use_energy: bool = False,
        use_fft_mag: bool = False,
        low_freq: float = 20.0,
        high_freq: float = -400.0,
        num_filters: int = 80,
        norm_filters: bool = False,
        torchaudio_compatible_mel_scale: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_filters = num_filters
        self._frame_length = frame_length
        self._frame_shift = frame_shift
        self._device = device

        # Store parameters for torchaudio.compliance.kaldi.fbank
        self._fbank_kwargs = dict(
            blackman_coeff=0.42,
            dither=dither,
            energy_floor=energy_floor,
            frame_length=frame_length * 1000,  # seconds → ms
            frame_shift=frame_shift * 1000,    # seconds → ms
            high_freq=high_freq,
            low_freq=low_freq,
            num_mel_bins=num_filters,
            preemphasis_coefficient=preemph_coeff,
            raw_energy=raw_energy,
            remove_dc_offset=remove_dc_offset,
            round_to_power_of_two=round_to_power_of_two,
            sample_frequency=float(sampling_rate),
            snip_edges=snip_edges,
            use_energy=use_energy,
            use_log_fbank=True,
            use_power=True,
            window_type=window_type,
        )

    # ------------------------------------------------------------------
    # FeatureExtractor interface
    # ------------------------------------------------------------------

    @property
    def frame_shift(self) -> float:
        return self._frame_shift

    def feature_dim(self) -> int:
        return self.num_filters

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

    def to(self, device: str):
        self._device = device
        super().to(device)

    def __repr__(self):
        return (
            f"KaldiFbank(sampling_rate={self.sampling_rate}, "
            f"num_filters={self.num_filters}, "
            f"frame_length={self._frame_length}, "
            f"frame_shift={self._frame_shift})"
        )

    def __str__(self):
        return self.__repr__()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        average_channels: bool = True,
    ) -> torch.Tensor:
        assert sample_rate == self.sampling_rate, (
            f"KaldiFbank was instantiated for sampling_rate "
            f"{self.sampling_rate}, but "
            f"sample_rate={sample_rate} was passed to forward()."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape
            if samples.shape[0] > 1:
                if average_channels:
                    samples = samples.mean(dim=0, keepdim=True)
                else:
                    samples = samples[0:1]

        samples = samples.to(self._device)

        feats = torch.stack(
            [
                torchaudio.compliance.kaldi.fbank(
                    wav.unsqueeze(0), **self._fbank_kwargs
                )
                for wav in samples
            ]
        )

        if is_numpy:
            return feats.cpu().numpy()
        return feats


# ---------------------------------------------------------------------------
# WhisperFbank  –  Whisper-style log-Mel spectrogram
# ---------------------------------------------------------------------------
#
# Faithfully reproduces ``whisper.audio.log_mel_spectrogram`` with identical
# STFT parameters (n_fft=400, hop_length=160, hann window) and the
# librosa-compatible mel filter bank (80 or 128 bins).  The output
# normalisation follows Whisper: (log10 + 4) / 4 with an 8-dB dynamic range.
#
# The mel filters are computed on-the-fly using the standard triangular
# filter bank algorithm (identical to ``librosa.filters.mel``), so the
# module has zero dependency on whisper's assets/mel_filters.npz.


def _hz_to_mel(frequencies):
    """Convert Hz to Mels (Slaney Auditory Toolbox formula, matching librosa)."""
    frequencies = np.asanyarray(frequencies)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels):
    """Convert Mels to Hz (Slaney Auditory Toolbox formula, matching librosa)."""
    mels = np.asanyarray(mels)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def _whisper_mel_filters(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
) -> torch.Tensor:
    """Compute triangular mel filter bank (same output as librosa.filters.mel).

    Returns a tensor of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    n_freqs = n_fft // 2 + 1
    fmax = float(sample_rate) / 2

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)

    fftfreqs = np.linspace(0, fmax, n_freqs)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0), _hz_to_mel(fmax), n_mels + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return torch.from_numpy(weights)


class WhisperFbank(FeatureExtractor):
    """Whisper-style log-Mel spectrogram extractor.

    Produces the same features as ``whisper.audio.log_mel_spectrogram``:

    * STFT: n_fft=400, hop_length=160, Hann window, ``return_complex=True``
    * Mel: librosa-compatible triangular filter bank (80 or 128 bins)
    * Log: ``log10``, clamped at 1e-10
    * Dynamic range: maximum ``max - 8.0`` dB
    * Normalisation: ``(log_spec + 4.0) / 4.0``

    The output range is approximately ``[-1, 0]``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.register_buffer(
            "_mel_filters",
            _whisper_mel_filters(n_mels, n_fft, sample_rate),
            persistent=False,
        )
        self.register_buffer(
            "_window",
            torch.hann_window(n_fft),
            persistent=False,
        )

    @property
    def frame_shift(self) -> float:
        return self.hop_length / self.sample_rate

    def feature_dim(self) -> int:
        return self.n_mels

    def __repr__(self):
        return (
            f"WhisperFbank(sample_rate={self.sample_rate}, "
            f"n_fft={self.n_fft}, "
            f"hop_length={self.hop_length}, "
            f"n_mels={self.n_mels})"
        )

    def __str__(self):
        return self.__repr__()

    def _log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Core computation for a single 1-D waveform tensor.

        Parameters
        ----------
        audio : torch.Tensor, shape ``(num_samples,)``

        Returns
        -------
        torch.Tensor, shape ``(n_mels, n_frames)``
        """
        stft = torch.stft(
            audio,
            self.n_fft,
            self.hop_length,
            window=self._window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self._mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def forward(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        average_channels: bool = True,
    ) -> torch.Tensor:
        assert sample_rate == self.sample_rate, (
            f"WhisperFbank was instantiated for sampling_rate "
            f"{self.sample_rate}, but "
            f"sample_rate={sample_rate} was passed to forward()."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape
            if samples.shape[0] > 1:
                if average_channels:
                    samples = samples.mean(dim=0, keepdim=True)
                else:
                    samples = samples[0:1]

        device = self._mel_filters.device
        feats = torch.stack([self._log_mel(wav.to(device)) for wav in samples])

        if is_numpy:
            return feats.cpu().numpy()
        return feats
