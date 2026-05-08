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

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from typing import Optional, Union


# ---------------------------------------------------------------------------
# Fbank (torchaudio-based)
# ---------------------------------------------------------------------------

class Fbank(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length

    def __repr__(self):
        return (
            f"Fbank(sample_rate={self.sample_rate}, "
            f"n_fft={self.fbank.n_fft}, "
            f"hop_length={self.hop_length}, "
            f"n_mels={self.n_mels})"
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-10

HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"


# ---------------------------------------------------------------------------
# Helper functions (extracted from lhotse/features/kaldi/layers.py)
# ---------------------------------------------------------------------------


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _lin2mel(x):
    return 1127.0 * np.log(1 + x / 700)


def _mel2lin(x):
    return 700 * (np.exp(x / 1127.0) - 1)


def _create_frame_window(window_size, window_type: str = "povey", blackman_coeff=0.42):
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        )
    else:
        raise Exception(f"Invalid window type: {window_type}")


def _get_strided_batch(
    waveform: torch.Tensor, window_length: int, window_shift: int, snip_edges: bool
) -> torch.Tensor:
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    num_samples = waveform.size(-1)

    if snip_edges:
        if num_samples < window_length:
            return torch.empty((0, 0, 0))
        else:
            num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        num_frames = (num_samples + (window_shift // 2)) // window_shift
        new_num_samples = (num_frames - 1) * window_shift + window_length
        npad = new_num_samples - num_samples
        npad_left = int((window_length - window_shift) // 2)
        npad_right = npad - npad_left
        pad_left = torch.flip(waveform[:, :npad_left], (1,))
        if npad_right >= 0:
            pad_right = torch.flip(waveform[:, -npad_right:], (1,))
        else:
            pad_right = torch.zeros(0, dtype=waveform.dtype, device=waveform.device)
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)

    strides = (
        waveform.stride(0),
        window_shift * waveform.stride(1),
        waveform.stride(1),
    )
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides)


def _get_log_energy(x: torch.Tensor, energy_floor: float) -> torch.Tensor:
    log_energy = (x.pow(2).sum(-1) + 1e-15).log()
    if energy_floor > 0.0:
        log_energy = torch.max(
            log_energy,
            torch.tensor(math.log(energy_floor), dtype=log_energy.dtype),
        )
    return log_energy


def _rfft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft(x, dim=-1)


def _pow_spectrogram(x: torch.Tensor) -> torch.Tensor:
    return x.abs() ** 2


def _get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
):
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (
        (0.0 <= low_freq < nyquist)
        and (0.0 < high_freq <= nyquist)
        and (low_freq < high_freq)
    ), "Bad values in options: low-freq {} and high-freq {} vs. nyquist {}".format(
        low_freq, high_freq, nyquist
    )

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = _lin2mel(low_freq)
    mel_high_freq = _lin2mel(high_freq)

    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta

    center_freqs = _mel2lin(center_mel)
    mel = _lin2mel(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))

    return bins, center_freqs


def _create_mel_scale(
    num_filters: int,
    fft_length: int,
    sampling_rate: int,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    norm_filters: bool = True,
) -> torch.Tensor:
    if high_freq is None or high_freq == 0:
        high_freq = sampling_rate / 2
    if high_freq < 0:
        high_freq = sampling_rate / 2 + high_freq

    mel_low_freq = _lin2mel(low_freq)
    mel_high_freq = _lin2mel(high_freq)
    melfc = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
    mels = _lin2mel(np.linspace(0, sampling_rate, fft_length))

    B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=np.float32)
    for k in range(num_filters):
        left_mel = melfc[k]
        center_mel = melfc[k + 1]
        right_mel = melfc[k + 2]
        for j in range(int(fft_length / 2)):
            mel_j = mels[j]
            if left_mel < mel_j < right_mel:
                if mel_j <= center_mel:
                    B[j, k] = (mel_j - left_mel) / (center_mel - left_mel)
                else:
                    B[j, k] = (right_mel - mel_j) / (right_mel - center_mel)

    if norm_filters:
        B = B / np.sum(B, axis=0, keepdims=True)

    return torch.from_numpy(B)




# ---------------------------------------------------------------------------
# KaldiFbank  –  self-contained Kaldi-style log-Mel filter bank extractor
# ---------------------------------------------------------------------------
#
# Flattens the Wav2Win → Wav2FFT → Wav2LogFilterBank inheritance chain from
# lhotse/features/kaldi/layers.py into a single torch.nn.Module so that the
# file has zero dependency on lhotse.


class KaldiFbank(torch.nn.Module):
    """Kaldi-style log Mel filter bank feature extractor.

    The implementation is identical to lhotse's ``Wav2LogFilterBank`` but is
    fully self-contained (no lhotse imports).
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
        energy_floor: float = EPSILON,
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
        if snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in "
                "duration to num-frames conversion."
            )

        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.round_to_power_of_two = round_to_power_of_two
        self.remove_dc_offset = remove_dc_offset
        self.preemph_coeff = preemph_coeff
        self.window_type = window_type
        self.dither = dither
        self.snip_edges = snip_edges
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.use_energy = use_energy
        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self.torchaudio_compatible_mel_scale = torchaudio_compatible_mel_scale
        self._device = device

        N = int(math.floor(frame_length * sampling_rate))
        self._win_length = N
        self._shift = int(math.floor(frame_shift * sampling_rate))
        self.fft_length = _next_power_of_2(N) if round_to_power_of_two else N

        # Window (from Wav2Win)
        self._window = nn.Parameter(
            _create_frame_window(N, window_type=window_type), requires_grad=False
        )

        # Epsilon for log stability (from Wav2LogFilterBank)
        self._eps = nn.Parameter(
            torch.tensor(torch.finfo(torch.float).eps), requires_grad=False
        )

        # Mel filter bank (from Wav2LogFilterBank)
        if torchaudio_compatible_mel_scale:
            fb, _ = _get_mel_banks(
                num_bins=num_filters,
                window_length_padded=self.fft_length,
                sample_freq=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
            )
            fb = torch.nn.functional.pad(fb, (0, 1), mode="constant", value=0).T
        else:
            fb = _create_mel_scale(
                num_filters=num_filters,
                fft_length=self.fft_length,
                sampling_rate=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                norm_filters=norm_filters,
            )
        self._fb = nn.Parameter(fb, requires_grad=False)

        # Spectrogram function (from Wav2LogFilterBank)
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

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
            f"frame_length={self.frame_length}, "
            f"frame_shift={self.frame_shift})"
        )

    def __str__(self):
        return self.__repr__()

    def feature_dim(self) -> int:
        return self.num_filters

    # ------------------------------------------------------------------
    # Core processing (Wav2Win._forward_strided)
    # ------------------------------------------------------------------

    def _wav2win_forward_strided(self, x_strided: torch.Tensor):
        if self.remove_dc_offset:
            mu = torch.mean(x_strided, dim=2, keepdim=True)
            x_strided = x_strided - mu

        log_energy = None
        if self.use_energy and self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)

        if self.preemph_coeff != 0.0:
            x_offset = torch.nn.functional.pad(x_strided, (1, 0), mode="replicate")
            x_strided = x_strided - self.preemph_coeff * x_offset[:, :, :-1]

        x_strided = x_strided * self._window

        pad_length = self.fft_length
        if pad_length != self._win_length:
            pad = pad_length - self._win_length
            x_strided = torch.nn.functional.pad(
                x_strided.unsqueeze(1), [0, pad], mode="constant", value=0.0
            ).squeeze(1)

        if self.use_energy and not self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)

        return x_strided, log_energy

    # ------------------------------------------------------------------
    # Core processing (Wav2LogFilterBank._forward_strided)
    # ------------------------------------------------------------------

    def _log_filter_bank_forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()

        if self.use_energy and log_e is not None:
            pow_spec = torch.cat((log_e.unsqueeze(-1), pow_spec), dim=-1)

        return pow_spec

    # ------------------------------------------------------------------
    # Core waveform → features (no input validation / conversion)
    # ------------------------------------------------------------------

    def _extract_core(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_samples) on self._device"""
        # Dither
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n

        # Frame
        x_strided = _get_strided_batch(
            x, self._win_length, self._shift, self.snip_edges
        )

        # Pre-emphasis, windowing, padding
        x_strided, log_energy = self._wav2win_forward_strided(x_strided)

        # FFT → mel filterbank → log
        return self._log_filter_bank_forward_strided(x_strided, log_energy)

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

        feats = self._extract_core(samples.to(self._device))

        if is_numpy:
            return feats.cpu().numpy()
        else:
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


def _whisper_mel_filters(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
) -> torch.Tensor:
    """Compute triangular mel filter bank (same output as librosa.filters.mel).

    Returns a tensor of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    n_freqs = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mels = np.linspace(low_mel, high_mel, n_mels + 2)
    hz = mel_to_hz(mels)

    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        f_left, f_center, f_right = hz[i], hz[i + 1], hz[i + 2]
        up = (fft_freqs - f_left) / (f_center - f_left + 1e-10)
        down = (f_right - fft_freqs) / (f_right - f_center + 1e-10)
        weights[i] = np.maximum(0, np.minimum(up, down))

    return torch.from_numpy(weights)


class WhisperFbank(torch.nn.Module):
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
        feats = torch.stack(
            [self._log_mel(wav.to(device)) for wav in samples]
        )

        if is_numpy:
            return feats.cpu().numpy()
        return feats
