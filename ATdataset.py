#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Wei Kang,
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


import collections
import copy
import glob
import io
import json
import logging
import math
import os
import random
import time

from functools import partial
from typing import Any, List, Tuple, Dict, Callable, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torchaudio
import webdataset as wds

from torch.utils.data import get_worker_info


def fix_sample_key(sample):
    """
    If the sample file name in tar files contains multiple dots, webdataset
    splits them with the first dot, which is not correct. This function fix it.
    For example, if the sample contains "abc.def.wav",
    webdataset will create a sample with key "def.wav"(value is audio) and
    __key__ equal to "abc".
    This function will rename the key to "wav" and __key__ to "abc.def".
    """
    new_sample = copy.copy(sample)
    for key in sample:
        if "." in key:
            base, ext = ".".join([sample["__key__"], key]).rsplit(".", 1)
            new_sample[ext] = new_sample.pop(key)
            new_sample["__key__"] = base
    return new_sample


def load_audio(data, sample_rate: int = 16000, device="cpu"):
    """
    Load audio from bytes data and resample to the target sample rate if needed.
    Return a tensor of shape (1, num_samples)
    """
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = torch.tensor(audio, device=device)
    if audio.size(1) > 1:
        audio = torch.mean(audio, dim=1)
    audio = audio.permute(1, 0)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


class FbankExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        device: str = "cpu",
    ):
        super().__init__()
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        ).to(device)
        self._device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

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
            samples = torch.from_numpy(samples).to(self._device)

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape
            if samples.shape[0] > 1 and not batch_mode:
                samples = torch.mean(samples, dim=0, keepdim=True)

        mel = self.fbank(samples)
        mel = mel.clamp(min=1e-7).log()
        return mel


def _simple_filter_keys(sample, audio_formats=("wav", "flac", "mp3")):
    return any(k in sample for k in audio_formats)


def _simple_decode_audio(
    sample, audio_formats=("wav", "flac", "mp3"), sample_rate=16000
):
    for ext in audio_formats:
        if ext in sample:
            return load_audio(sample[ext], sample_rate=sample_rate)
    raise RuntimeError("No noise audio found in sample")


def create_simple_audio_dataset(
    audio_tars: List[str],
    sample_rate: int = 16000,
    buffer_size: int = 1000,
    nodesplitter: Optional[Any] = wds.split_by_node,
    workersplitter: Optional[Any] = wds.split_by_worker,
    audio_formats: Tuple[str] = ("wav", "flac", "mp3"),
):
    """
    Create a simple audio dataset from webdataset tar files.
    Args:
      audio_tars:
        List of audio tar files.
      sample_rate:
        Target sample rate for audio.
      buffer_size:
        Buffer size for shuffling.
      nodesplitter:
        Node splitter for webdataset.
      workersplitter:
        Worker splitter for webdataset.
      audio_formats:
        Tuple of audio file extensions to look for in the sample.
    """

    simple_filter_keys = partial(_simple_filter_keys, audio_formats=audio_formats)
    simple_decode_audio = partial(
        _simple_decode_audio,
        audio_formats=audio_formats,
        sample_rate=sample_rate,
    )
    audio_ds = (
        wds.WebDataset(
            audio_tars,
            shardshuffle=len(audio_tars),
            nodesplitter=nodesplitter,
            workersplitter=workersplitter,
        )
        .decode()
        .select(simple_filter_keys)
        .map(simple_decode_audio)
        .shuffle(buffer_size)
        .repeat()
    )
    return audio_ds


class NoiseSampler:
    """
    Sample random noise segments from a noise dataset.
    """

    def __init__(self, noise_ds):
        self.noise_ds = noise_ds
        self.iterator = None

    def random_noise(self, target_length):
        if self.iterator is None:
            self.iterator = iter(self.noise_ds)

        try:
            noise = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.noise_ds)
            noise = next(self.iterator)

        if noise.size(1) < target_length:
            repeats = (target_length // noise.size(1)) + 1
            noise = noise.repeat(1, repeats)[:, :target_length]
        elif noise.size(1) > target_length:
            start = random.randint(0, noise.size(1) - target_length)
            noise = noise[:, start : start + target_length]
        return noise


class LabelDataset:
    def __init__(self, manifest_path: str):
        """
        Load labels from a manifest (jsonl) file.
        Args:
          manifest_path:
            Path to the manifest file containing labels.
            Each line in the manifest file is in the format of:
            {"audio_filepath": "filepath.{wav,mp3,flac}", "text": "transcription text"}
        """
        self._labels = {}

        # if the manifest file does not exist, return empty labels
        # for some non speech audios.
        if not os.path.exists(manifest_path):
            logging.warning(f"Label manifest file {manifest_path} does not exist.")
            return

        self.path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "audio_filepath" in item and "text" in item:
                    key = item["audio_filepath"].rsplit(".", 1)[0]
                    self._labels[key] = item["text"]

    def __getitem__(self, key):
        if key not in self._labels or not self._labels[key].strip():
            return "<|EMPTY|>"
        return self._labels[key]


class SampleDecoder:
    """
    Decode a sample from webdataset, including loading audio and fetching label.
    The returned audio is a tensor of shape (1, num_samples) on CPU.
    """

    def __init__(
        self,
        labels_to_audios: Dict,
        sample_rate: int = 16000,
        audio_format: Tuple[str] = ("flac", "wav", "mp3"),
    ):
        """
        Args:
          labels_to_audios:
            A dict mapping from audio tar file to label tar file.
          sample_rate:
            Target sample rate for audio.
          audio_format:
            Tuple of audio file extensions to look for in the sample.
        """
        self.labels = labels_to_audios
        self.sample_rate = sample_rate
        self.label_dataset = None
        self.audio_format = audio_format

    def __call__(self, sample):
        sample = fix_sample_key(sample)
        src = sample["__url__"]
        key = sample["__key__"]
        if self.label_dataset is None or self.label_dataset.path != self.labels[src]:
            logging.info(f"Loading labels from {self.labels[src]}")
            self.label_dataset = LabelDataset(self.labels[src])

        audio = torch.empty(0)
        for ext in self.audio_format:
            if ext in sample:
                # load audio (1, num_samples)
                audio = load_audio(sample[ext], sample_rate=self.sample_rate)
                break

        label = self.label_dataset[key]
        return {
            "audio": audio,
            "label": label,
        }


def audio_augmentation(
    audio,
    sample_rate: int = 16000,
    speed_perturb: Optional[Tuple] = (0.9, 1.0, 1.1),  # speeds
    volume_perturb: Optional[Tuple] = (0.5, -10, 6),  # prob, lower_db, upper_db
):
    """
    Apply speed and volume perturbation to the audio tensor.
    Args:
      audio:
        Audio tensor of shape (1, num_samples).
      sample_rate:
        Sample rate of the audio.
      speed_perturb:
        Tuple of speeds for speed perturbation.
      volume_perturb:
        Tuple of (probability, lower_db, upper_db) for volume perturbation.
    Returns:
      The augmented audio tensor.
    """
    # apply speed perturbation
    if speed_perturb is not None and audio.numel() > 0:
        assert isinstance(speed_perturb, (list, tuple))
        speed = random.choice(speed_perturb)
        if speed != 1:
            audio = torchaudio.functional.resample(
                audio, sample_rate, int(sample_rate * speed)
            )

    # apply volume perturbation
    if volume_perturb is not None and audio.numel() > 0:
        prob, lower_db, upper_db = volume_perturb
        if random.random() <= prob:
            gain_db = random.uniform(lower_db, upper_db)
            audio = audio * (10 ** (gain_db / 20))
    return audio


def augment_with_noise(audio, noise_sampler, lower_snr_db, upper_snr_db, is_train=True):
    if noise_sampler is None or not is_train:
        return audio
    snr_db = random.uniform(lower_snr_db, upper_snr_db)
    noise = noise_sampler.random_noise(audio.size(0)).squeeze(0)
    audio_rms = audio.pow(2).mean().sqrt()
    noise_rms = noise.pow(2).mean().sqrt()
    snr = 10 ** (snr_db / 20)
    scaled_noise = noise * (audio_rms / (snr * noise_rms + 1e-8))
    audio = audio + scaled_noise
    return audio


class StreamingBucketBatcher:
    """
    Streaming bucketing batcher using multiple fixed-length buckets.
    Each bucket holds samples with similar durations.
    """

    def __init__(
        self,
        max_duration: float = 500.0,  # in seconds
        min_length: float = 1,  # in seconds
        max_length: float = 30,  # in seconds
        num_buckets: int = 30,
        sample_rate: int = 16000,
        is_train: bool = True,
        length_key="audio",
    ):
        """
        Args:
          max_duration:
            Maximum duration (in seconds) for each batch.
          min_length:
            Minimum length (in seconds) of samples to consider.
          max_length:
            Maximum length (in seconds) of samples to consider.
          num_buckets:
            Number of buckets to use.
          sample_rate:
            Sample rate of the audio samples.
          is_train:
            Whether the batcher is for training or not.
          length_key:
            Key in the sample dict to use for length calculation.
        """
        self.max_duration = max_duration
        self.num_buckets = num_buckets
        self.min_length = min_length
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.length_key = length_key
        self.is_train = is_train

        # max number of samples per bucket, calculated based on max_duration and min_length
        self.buffer_per_bucket = math.ceil(max_duration / max(1, min_length) * 2)
        self.buckets = collections.defaultdict(collections.deque)
        self.bucket_item_lengths = [
            math.ceil((max_length - min_length) / num_buckets) * (i + 1)
            for i in range(num_buckets)
        ]

    def bucket_id(self, length):
        length = max(self.min_length, min(length, self.max_length))
        return int(
            (length - self.min_length)
            / (self.max_length - self.min_length)
            * (self.num_buckets - 1)
        )

    def __call__(
        self, data_streams: List[wds.WebDataset], weights: Optional[List[float]] = None
    ):
        if weights is None:
            weights = [1.0 / len(data_streams)] * len(data_streams)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        streams = [iter(data_stream) for data_stream in data_streams]
        stream_idx = 0

        logging.info("Starting StreamingBucketBatching with weights:", weights)

        while True:
            # Fill buckets
            full_buckets = []
            try:
                while True:
                    stream_idx = np.random.choice(len(streams), p=weights)
                    sample = next(streams[stream_idx])
                    length = sample[self.length_key].size(1) / self.sample_rate
                    b_id = self.bucket_id(length)
                    self.buckets[b_id].append(sample)

                    full_buckets = [
                        i
                        for i in range(self.num_buckets)
                        if self.bucket_item_lengths[i] * len(self.buckets[i])
                        > self.max_duration * 1.5
                    ]
                    if full_buckets:
                        break

            except StopIteration:
                if self.is_train:
                    # repeat the data stream, the StreamingWebDataset will handle epoch ending
                    streams[stream_idx] = iter(data_streams[stream_idx])
                    continue

            batch = []
            batch_duration = 0
            bucket_range = []

            if full_buckets:
                bucket_range.append(random.choice(full_buckets))
            else:
                # Normally, if self.is_train is True, will not run into this branch
                if not self.is_train:
                    # all non-empty buckets
                    bucket_range = [
                        i for i in range(self.num_buckets) if self.buckets[i]
                    ]
                    bucket_range.reverse()

            last_b_id = bucket_range[0] if bucket_range else None
            for b_id in bucket_range:
                while self.buckets[b_id]:
                    sample = self.buckets[b_id][0]
                    length = sample[self.length_key].size(1) / self.sample_rate
                    if batch_duration + length > self.max_duration:
                        if not batch:
                            last_b_id = b_id
                            batch_duration += length  # for break the for loop
                        break
                    else:
                        batch.append(self.buckets[b_id].popleft())
                        batch_duration += length
                if batch_duration >= self.max_duration:
                    break

            if not batch:
                # Has full buckets but could not form a batch within max_duration
                # If a single sample exceeds batch_frames, yield it alone
                if last_b_id and self.buckets[last_b_id]:
                    batch.append(self.buckets[last_b_id].popleft())
                else:
                    if not self.is_train:
                        return
            yield batch


def get_manifest_duration(manifest_path: str) -> float:
    """
    Calculate total duration of audio files in a manifest file.
    Args:
      manifest_path:
        Path to the manifest file containing audio file paths and durations.
        Each line in the manifest file is in the format of:
        {"audio_filepath": path, "text": text, "duration": duration_in_seconds}
    """
    total_duration = 0.0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "duration" in item:
                total_duration += float(item["duration"])
    return total_duration


class StreamingWebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        manifests: Union[str, List[str]],
        sample_rate: int = 16000,
        max_duration: float = 1000.0,
        epoch_hours: Optional[float] = None,
        mux_weights: Optional[List[float]] = None,
        feature_extractor: Optional[Callable] = None,
        noise_tars: Optional[List[str]] = None,
        noise_augment: Tuple = (0.5, 10, 20),  # probs lower_snr_db, upper_snr_db
        speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
        volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
        buffer_size: int = 1000,
        is_train: bool = True,
        device=torch.device("cpu"),
    ):
        """
        Streaming webdataset for ASR training with dynamic bucketing batching.
        Args:
          manifests:
            A list of manifest files containing audio tar files and label files.
          sample_rate:
            Target sample rate for audio.
          max_duration:
            Maximum duration (in seconds) for each batch.
          epoch_hours:
            Number of hours per epoch. If None, will calculate based on manifest durations.
          mux_weights:
            A list of weights for each manifest for muxing.
          feature_extractor:
            Feature extractor to extract features from raw audio.
          noise_tars:
            List of noise tar files for noise augmentation.
          noise_augment:
            Tuple of (probability, lower_snr_db, upper_snr_db) for noise augmentation.
          speed_perturb:
            Tuple of speeds for speed perturbation.
          volume_perturb:
            Tuple of (probability, lower_db, upper_db) for volume perturbation.
          buffer_size:
            Buffer size for shuffling.
          is_train:
            Whether the dataset is for training or not.
          device:
            Device to calculate features.
        """
        super().__init__()

        self.device = device
        self.is_train = is_train
        self.buffer_size = buffer_size
        self.max_duration = max_duration

        if isinstance(manifests, str):
            assert os.path.exists(
                manifests
            ), f"Manifest file {manifests} does not exist."
            self.manifests = [manifests]
        else:
            assert isinstance(
                manifests, list
            ), "manifests should be a string or a list of strings."
            for manifest in manifests:
                assert os.path.exists(
                    manifest
                ), f"Manifest file {manifest} does not exist."
            self.manifests = manifests

        nodes = 1 if not dist.is_initialized() else dist.get_world_size()
        workers = 1 if get_worker_info() is None else get_worker_info().num_workers
        self.num_workers = workers * nodes

        # we share all labels_to_audios among all manifests for simplicity
        labels_to_audios: Dict[str, str] = {}
        audio_tars_lists: List[List[str]] = []
        manifest_durations: List[float] = []
        for manifest in self.manifests:
            manifest_audio_tars: List[str] = []
            manifest_duration = 0.0
            log_duration_warning = True  # to avoid too many warnings
            with open(manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items = line.split()
                    if len(items) < 2:
                        logging.warning(
                            f"Each line in manifest file {manifest} should contain "
                            f"at least 2 fields: audio_tar label_json \n"
                            f"skipping line: {line}"
                        )
                        continue
                    if (not os.path.exists(items[0])) or (not os.path.exists(items[1])):
                        logging.warning(
                            f"Either audio tar file {items[0]} or label file "
                            f"{items[1]} does not exist, skipping line : {line}."
                        )
                        continue
                    if len(items) < 3:
                        if log_duration_warning:
                            logging.warning(
                                f"Manifest file {manifest} does not contain duration "
                                f"field, calculating duration from manifests, might be slow."
                                f" Please consider adding duration field to the manifest file."
                            )
                            log_duration_warning = False
                        manifest_duration += get_manifest_duration(items[1]) / 3600.0
                    else:
                        manifest_duration += float(items[2])
                    manifest_audio_tars.append(items[0])
                    labels_to_audios[items[0]] = items[1]

            if manifest_duration <= 0.001:
                logging.warning(
                    f"Manifest {manifest} has very small duration : {manifest_duration}, "
                    f"please check if the manifest files are correct, especially the duration field."
                )
            logging.info(f"Manifest {manifest} duration: {manifest_duration:.2f} hours")
            audio_tars_lists.append(manifest_audio_tars)
            manifest_durations.append(manifest_duration)

        self.audio_tars_lists = []
        # pad audio_tars to be multiple of num_workers (for proper sharding)
        for audio_tars in audio_tars_lists:
            tar_indexs = list(range(len(audio_tars)))
            pad_num = self.num_workers - (len(audio_tars) % self.num_workers)
            for i in range(pad_num):
                audio_tars.append(audio_tars[random.choice(tar_indexs)])
            self.audio_tars_lists.append(audio_tars)

        calculated_hours = sum(manifest_durations)
        if epoch_hours is None:
            logging.info(f"Using calculated epoch hours: {calculated_hours:.2f}")
            epoch_hours = calculated_hours
        else:
            if abs(calculated_hours - epoch_hours) / epoch_hours > 0.05:
                logging.warning(
                    f"Given epoch hours {epoch_hours} differ from calculated "
                    f"hours {calculated_hours:.2f} by more than 5%. "
                    f"Using given epoch hours, but you may want to double check."
                )

        calculated_mux_weights = [
            round(dur / calculated_hours, 2) for dur in manifest_durations
        ]
        if mux_weights is None and len(self.manifests) > 1:
            logging.info(f"Using calculated mux weights: {calculated_mux_weights}")
            mux_weights = calculated_mux_weights
        else:
            if mux_weights is not None:
                mux_weights_sum = sum(mux_weights)
                mux_weights = [round(w / mux_weights_sum, 2) for w in mux_weights]
                if any(
                    abs(mux_weights[i] - calculated_mux_weights[i])
                    / calculated_mux_weights[i]
                    > 0.05
                    for i in range(len(mux_weights))
                ):
                    logging.warning(
                        f"Given mux weights {mux_weights} differ from calculated "
                        f"weights {calculated_mux_weights} by more than 5%. "
                        f"Using given mux weights, but you may want to double check."
                    )
            else:
                logging.info("Only one manifest, no mux weights needed.")
                mux_weights = [1]

        self.epoch_batches = math.ceil(
            epoch_hours * 3600.0 / (self.num_workers) / max_duration
        )
        self.mux_weights = mux_weights

        # sample_decoder is to decode audio and assign label
        self.sample_decoder = SampleDecoder(
            labels_to_audios=labels_to_audios,
            sample_rate=sample_rate,
        )

        # noise_sampler for noise augmentation
        noise_sampler = None
        if noise_tars is not None and noise_augment is not None and is_train:
            noise_tars = list(noise_tars)
            tar_indexs = list(range(len(noise_tars)))
            pad_num = self.num_workers - (len(noise_tars) % self.num_workers)
            for i in range(pad_num):
                noise_tars.append(noise_tars[random.choice(tar_indexs)])
            noise_ds = create_simple_audio_dataset(
                noise_tars,
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            noise_sampler = NoiseSampler(noise_ds)

        self.num_copies = 1
        self.noise_prob = 0.0
        lower_snr_db = 10.0
        upper_snr_db = 20.0
        if is_train and noise_augment is not None and noise_sampler is not None:
            assert (
                isinstance(noise_augment, (list, tuple)) and len(noise_augment) == 3
            ), (
                "noise_augment should be a tuple of "
                "(probability, lower_snr_db, upper_snr_db)"
            )
            self.num_copies = 3
            self.noise_prob, lower_snr_db, upper_snr_db = noise_augment

        self.add_noise = partial(
            augment_with_noise,
            noise_sampler=noise_sampler,
            lower_snr_db=lower_snr_db,
            upper_snr_db=upper_snr_db,
            is_train=is_train,
        )
        self.augment_audio = partial(
            audio_augmentation,
            sample_rate=sample_rate,
            speed_perturb=speed_perturb,
            volume_perturb=volume_perturb,
        )

        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor.to(device)

    def __iter__(self):
        # read raw audio data and label
        datasets = [
            wds.WebDataset(
                audio_tars,
                shardshuffle=len(audio_tars) if self.is_train else False,
                nodesplitter=wds.split_by_node,
            )
            .decode()
            .map(self.sample_decoder)
            .shuffle(self.buffer_size if self.is_train else 1)
            for audio_tars in self.audio_tars_lists
        ]

        # for dynamic batching based on audio duration with max_duration
        batcher = StreamingBucketBatcher(
            max_duration=self.max_duration,
            sample_rate=self.sample_rate,
            is_train=self.is_train,
        )
        dataset = batcher(datasets, weights=self.mux_weights)
        self.stream = iter(dataset)
        batch_count = 0

        while True:
            try:
                if batch_count >= self.epoch_batches and self.is_train:
                    return
                batch_count += 1

                raw_batch = next(self.stream)
                batch_item = {
                    "audio": [],
                    "label": [],
                    "key": [],
                    "audio_len": [],
                }
                batch = [
                    copy.deepcopy(batch_item) for x in range(self.num_copies)
                ]  # container for num copies

                for sample in raw_batch:
                    # sample["audio"]: (1, num_samples)
                    audio = self.augment_audio(sample["audio"])
                    # remove the channel dimension for batching
                    audio = audio.squeeze(0)
                    label = sample["label"]
                    key = sample["__key__"]
                    audio_len = audio.size(0)

                    if audio.numel() == 0:
                        continue

                    if self.num_copies == 1:
                        # noise augmentation with a probability
                        if random.random() < self.noise_prob and self.is_train:
                            audio = self.add_noise(audio)
                        batch[0]["audio"].append(audio.to(self.device))
                        batch[0]["label"].append(label)
                        batch[0]["key"].append(key)
                        batch[0]["audio_len"].append(audio_len)
                    else:
                        for i in range(self.num_copies):
                            # first copy is clean, others are with noise
                            if i == 0:
                                batch[i]["audio"].append(audio.to(self.device))
                            else:
                                batch[i]["audio"].append(
                                    self.add_noise(audio).to(self.device)
                                )
                            batch[i]["label"].append(label)
                            batch[i]["key"].append(key)
                            batch[i]["audio_len"].append(audio_len)
                audios = []
                labels = []
                keys = []
                audio_lens = []
                for i in range(self.num_copies):
                    audios += batch[i]["audio"]
                    labels += batch[i]["label"]
                    keys += batch[i]["key"]
                    audio_lens += batch[i]["audio_len"]
                audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
                audio_lens = torch.tensor(audio_lens, device=audios.device)

                features = None
                frame_lens = None

                if self.feature_extractor is not None:
                    features = self.feature_extractor(
                        audios, self.sample_rate, batch_mode=True
                    )
                    frame_shift = self.feature_extractor.frame_shift
                    frame_lens = (
                        (audio_lens + self.sample_rate * frame_shift // 2)
                        / self.sample_rate
                        / frame_shift
                    ).to(torch.int32)
                    features = features[:, :, 0 : frame_lens.max().item()]

                # Return data to CPU to eliminate the following CUDA ERROR
                # Producer process has been terminated before all shared CUDA tensors released.
                # See Note [Sharing CUDA tensors]
                audios_cpu = audios.to("cpu")
                audio_lens_cpu = audio_lens.to("cpu")
                features_cpu = None if features is None else features.to("cpu")
                frame_lens_cpu = None if frame_lens is None else frame_lens.to("cpu")

                del audios, audio_lens
                if features is not None:
                    del features, frame_lens
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()

                batch_output = {
                    "inputs": audios_cpu,
                    "num_copies": self.num_copies,
                    "supervisions": {
                        "sequence_idx": torch.tensor(
                            list(range(len(labels))), device="cpu"
                        ),
                        "text": labels,
                        "start_frame": torch.zeros(
                            len(labels), dtype=torch.long, device="cpu"
                        ),
                        "num_frames": frame_lens_cpu,
                        "start_sample": torch.zeros(
                            len(labels), dtype=torch.long, device="cpu"
                        ),
                        "num_samples": audio_lens_cpu,
                        "keys": keys,
                    },
                }
                if features_cpu is not None:
                    batch_output["inputs"] = features_cpu
                yield batch_output

            except StopIteration:
                if not self.is_train:
                    return
                self.stream = iter(dataset)
            except RuntimeError as e:
                logging.error(f"Runtime error in data loading: {e}, skipping batch.")
                continue


def create_dataloader(
    manifests: Union[str, List[str]],
    sample_rate: int = 16000,
    max_duration: float = 600.0,
    epoch_hours: Optional[float] = None,
    mux_weights: Optional[List[float]] = None,
    feature_extractor: Optional[Callable] = None,
    noise_tars: Optional[List[str]] = None,
    noise_augment: Tuple = (0.5, 10, 20),  # probs lower_snr_db, upper_snr_db
    speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
    volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
    buffer_size: int = 1000,
    num_workers: int = 2,
    is_train: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """
    Create a dataloader for streaming webdataset.
    Args:
      manifests:
        A list of manifest files containing audio tar files and label files.
      sample_rate:
        Target sample rate for audio.
      max_duration:
        Maximum duration (in seconds) for each batch.
      epoch_hours:
        Number of hours per epoch. If None, will calculate based on manifest durations.
      mux_weights:
        A list of weights for each manifest for muxing, If None, will calculate based on manifest durations.
      feature_extractor:
        Feature extractor to extract features from raw audio.
      noise_tars:
        List of noise tar files for noise augmentation.
      noise_augment:
        Tuple of (probability, lower_snr_db, upper_snr_db) for noise augmentation.
      speed_perturb:
        Tuple of speeds for speed perturbation.
      volume_perturb:
        Tuple of (probability, lower_db, upper_db) for volume perturbation.
      buffer_size:
        Buffer size for shuffling.
      num_workers:
        Number of workers for dataloader.
      is_train:
        Whether the dataloader is for training or not.
      device:
        Device to calculate features.
    """
    dataset = StreamingWebDataset(
        manifests=manifests,
        mux_weights=mux_weights,
        sample_rate=sample_rate,
        max_duration=max_duration,
        epoch_hours=epoch_hours,
        feature_extractor=feature_extractor,
        noise_tars=noise_tars,
        noise_augment=noise_augment,
        speed_perturb=speed_perturb,
        volume_perturb=volume_perturb,
        buffer_size=buffer_size,
        is_train=is_train,
        device=device,
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    )
    return dataloader


def main():
    feature_extractor = FbankExtractor(sample_rate=16000)
    noise_tars = glob.glob("data/musan/audios/musan.*.tar")

    dataset = create_dataloader(
        manifests=[
            "data/tars/aishell_train.lst",
            "data/tars/librispeech_train.lst",
            "data/tars/gigaspeech_XL.lst",
            "data/tars/wenetspeech_L.lst",
        ],
        max_duration=200.0,
        epoch_hours=None,
        mux_weights=None,
        feature_extractor=feature_extractor,
        sample_rate=16000,
        noise_tars=noise_tars,
        noise_augment=(0.5, 10, 20),
        speed_perturb=(0.9, 1.0, 1.1),
        volume_perturb=(0.5, -10, 6),
        buffer_size=1000,
        is_train=True,
        num_workers=2,
        device=torch.device("cuda"),
    )

    start = time.time()
    for i, batch in enumerate(dataset):
        print(
            batch["inputs"].shape,
            batch["supervisions"]["num_frames"].min().item(),
            batch["supervisions"]["num_frames"].max().item(),
            time.time() - start,
        )
        start = time.time()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # The context might already be set.
    main()
