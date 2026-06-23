#!/usr/bin/env python3
# Copyright  2025 Wei Kang (wkang@pku.edu.cn)
#
# See ../LICENSE for clarification regarding multiple authors
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
import random

import numpy as np
import torch
import torchaudio

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Dict, Any


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

# SpecAugmentation and time warping implementation is copied and modified from
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/signal_transforms.py
class SpecAugment(torch.nn.Module):
    """
    SpecAugment performs three augmentations:
    - time warping of the feature matrix
    - masking of ranges of features (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(
        self,
        time_warp_factor: Optional[int] = 80,
        num_feature_masks: int = 2,
        features_mask_size: int = 27,
        num_frame_masks: int = 10,
        frames_mask_size: int = 100,
        max_frames_mask_fraction: float = 0.15,
        p=0.9,
    ):
        """
        SpecAugment's constructor.

        :param time_warp_factor: parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
        :param num_feature_masks: how many feature masks should be applied. Set to ``0`` to disable.
        :param features_mask_size: the width of the feature mask (expressed in the number of masked feature bins).
            This is the ``F`` parameter from the SpecAugment paper.
        :param num_frame_masks: the number of masking regions for utterances. Set to ``0`` to disable.
        :param frames_mask_size: the width of the frame (temporal) masks (expressed in the number of masked frames).
            This is the ``T`` parameter from the SpecAugment paper.
        :param max_frames_mask_fraction: limits the size of the frame (temporal) mask to this value times the length
            of the utterance (or supervision segment).
            This is the parameter denoted by ``p`` in the SpecAugment paper.
        :param p: the probability of applying this transform.
            It is different from ``p`` in the SpecAugment paper!
        """
        super().__init__()
        assert 0 <= p <= 1
        assert num_feature_masks >= 0
        assert num_frame_masks >= 0
        assert features_mask_size > 0
        assert frames_mask_size > 0
        self.time_warp_factor = time_warp_factor
        self.num_feature_masks = num_feature_masks
        self.features_mask_size = features_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.max_frames_mask_fraction = max_frames_mask_fraction
        self.p = p

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: Optional[torch.IntTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes SpecAugment for a batch of feature matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``feature_lens`` tensor that will be used to apply SpecAugment
        only to the valid (non-padded) areas of the input.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param feature_lens: an int tensor of shape ``(B,)`` or ``(B, 1)``.
            Each element is the number of valid frames for the corresponding
            sequence in ``features``. The valid region is assumed to start
            from frame 0.
        :return: an augmented tensor of shape ``(B, T, F)``.
        """
        assert len(features.shape) == 3, (
            "SpecAugment only supports batches of " "single-channel feature matrices."
        )
        features = features.clone()
        if feature_lens is None:
            # No feature_lens - apply spec augment to full feature matrices.
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(features[sequence_idx])
        else:
            # feature_lens provided - we will apply time warping only on the valid areas.
            for sequence_idx, num_frames in enumerate(feature_lens):
                num_frames = int(num_frames)
                features[sequence_idx, :num_frames] = self._forward_single(
                    features[sequence_idx, :num_frames], warp=True, mask=False
                )
            # ... and then time-mask the full feature matrices. Note that in this mode,
            # it might happen that masks are applied to different sequences/examples
            # than the time warping.
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(
                    features[sequence_idx], warp=False, mask=True
                )
        return features

    def _forward_single(
        self, features: torch.Tensor, warp: bool = True, mask: bool = True
    ) -> torch.Tensor:
        """
        Apply SpecAugment to a single feature matrix of shape (T, F).
        """
        if random.random() > self.p:
            # Randomly choose whether this transform is applied
            return features
        if warp:
            if self.time_warp_factor is not None and self.time_warp_factor >= 1:
                features = time_warp_single(features, factor=self.time_warp_factor)
        if mask:
            mean = features.mean()
            # Frequency masking
            features = mask_along_axis_optimized(
                features,
                mask_size=self.features_mask_size,
                mask_times=self.num_feature_masks,
                mask_value=mean,
                axis=2,
            )
            # Time masking
            max_tot_mask_frames = self.max_frames_mask_fraction * features.size(0)
            num_frame_masks = min(
                self.num_frame_masks,
                math.ceil(max_tot_mask_frames / self.frames_mask_size),
            )
            max_mask_frames = min(
                self.frames_mask_size, max_tot_mask_frames // num_frame_masks
            )
            features = mask_along_axis_optimized(
                features,
                mask_size=max_mask_frames,
                mask_times=num_frame_masks,
                mask_value=mean,
                axis=1,
            )

        return features

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        return dict(
            time_warp_factor=self.time_warp_factor,
            num_feature_masks=self.num_feature_masks,
            features_mask_size=self.features_mask_size,
            num_frame_masks=self.num_frame_masks,
            frames_mask_size=self.frames_mask_size,
            max_frames_mask_fraction=self.max_frames_mask_fraction,
            p=self.p,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.time_warp_factor = state_dict.get(
            "time_warp_factor", self.time_warp_factor
        )
        self.num_feature_masks = state_dict.get(
            "num_feature_masks", self.num_feature_masks
        )
        self.features_mask_size = state_dict.get(
            "features_mask_size", self.features_mask_size
        )
        self.num_frame_masks = state_dict.get("num_frame_masks", self.num_frame_masks)
        self.frames_mask_size = state_dict.get(
            "frames_mask_size", self.frames_mask_size
        )
        self.max_frames_mask_fraction = state_dict.get(
            "max_frames_mask_fraction", self.max_frames_mask_fraction
        )
        self.p = state_dict.get("p", self.p)


def mask_along_axis_optimized(
    features: torch.Tensor,
    mask_size: int,
    mask_times: int,
    mask_value: float,
    axis: int,
) -> torch.Tensor:
    """
    Apply Frequency and Time masking along axis.
    Frequency and Time masking as described in the SpecAugment paper.

    :param features: input tensor of shape ``(T, F)``
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    :mask_value: Value to assign to the masked regions.
    :axis: Axis to apply masking on (1 -> time, 2 -> frequency)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported!")

    features = features.unsqueeze(0)
    features = features.reshape([-1] + list(features.size()[-2:]))

    values = torch.randint(int(0), int(mask_size), (1, mask_times))
    min_values = torch.rand(1, mask_times) * (features.size(axis) - values)
    mask_starts = (min_values.long()).squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()

    if axis == 1:
        if mask_times == 1:
            features[:, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for (mask_start, mask_end) in zip(mask_starts, mask_ends):
            features[:, mask_start:mask_end] = mask_value
    else:
        if mask_times == 1:
            features[:, :, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for (mask_start, mask_end) in zip(mask_starts, mask_ends):
            features[:, :, mask_start:mask_end] = mask_value

    features = features.squeeze(0)
    return features


def time_warp_single(features: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(T, F)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(T, F)``
    """
    t = features.size(0)
    if t - factor <= factor + 1:
        return features
    center = np.random.randint(factor + 1, t - factor)
    warped = np.random.randint(center - factor, center + factor + 1)
    if warped == center:
        return features
    features = features.unsqueeze(0).unsqueeze(0)
    left = torch.nn.functional.interpolate(
        features[:, :, :center, :],
        size=(warped, features.size(3)),
        mode="bicubic",
        align_corners=False,
    )
    right = torch.nn.functional.interpolate(
        features[:, :, center:, :],
        size=(t - warped, features.size(3)),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0)


def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    feature_lens: Optional[torch.Tensor] = None,
):
    """
    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(B, T, F)``
    :param p: the probability of applying this transform.
    :param time_warp_factor: time warping parameter.
    :param feature_lens: an int tensor of shape ``(B,)`` or ``(B, 1)``.
        Each element is the number of valid frames for the corresponding
        sequence in ``features``. The valid region is assumed to start
        from frame 0.
    :return: a warped tensor of shape ``(B, T, F)``
    """
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert len(features.shape) == 3, (
        f"SpecAugment only supports 3D tensors: {features.shape}"
    )
    features = features.clone()
    if feature_lens is None:
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                continue
            features[sequence_idx] = time_warp_single(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        for sequence_idx, num_frames in enumerate(feature_lens):
            if random.random() > p:
                continue
            num_frames = int(num_frames)
            features[sequence_idx, :num_frames] = time_warp_single(
                features[sequence_idx, :num_frames], factor=time_warp_factor
            )
    return features