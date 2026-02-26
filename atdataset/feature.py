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


import torch
import torchaudio

import numpy as np

from typing import Union


class FbankExtractor(torch.nn.Module):
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

    @property
    def frame_shift(self) -> float:
        return self.hop_length / self.sample_rate

    def feature_dim(self) -> int:
        return self.n_mels

    def forward(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        batch_mode: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        # Check for sampling rate compatibility.
        expected_sr = self.sample_rate
        assert sample_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sample_rate}"
        )
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples).to(self.device)

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape
            if samples.shape[0] > 1 and not batch_mode:
                samples = torch.mean(samples, dim=0, keepdim=True)

        mel = self.fbank(samples)
        mel = mel.clamp(min=1e-7).log()
        return mel
