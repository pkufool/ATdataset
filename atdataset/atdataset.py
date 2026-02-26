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

from webdataset.utils import pytorch_worker_info


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
        audio = torch.mean(audio, dim=1, keepdim=True)
    audio = audio.permute(1, 0)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


def audio_augmentation(
    audio,
    sample_rate: int = 16000,
    use_speed_perturb: bool = True,
    speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
    use_volume_perturb: bool = True,
    volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
):
    """
    Apply speed and volume perturbation to the audio tensor.
    Args:
      audio:
        Audio tensor of shape (1, num_samples).
      sample_rate:
        Sample rate of the audio.
      use_speed_perturb:
        Whether to apply speed perturbation.
      speed_perturb:
        Tuple of speeds for speed perturbation.
      use_volume_perturb:
        Whether to apply volume perturbation.
      volume_perturb:
        Tuple of (probability, lower_db, upper_db) for volume perturbation.
    Returns:
      The augmented audio tensor.
    """
    # apply speed perturbation
    if use_speed_perturb and audio.numel() > 0:
        assert isinstance(speed_perturb, (list, tuple))
        speed = random.choice(speed_perturb)
        if speed != 1:
            audio = torchaudio.functional.resample(
                audio, sample_rate, int(sample_rate * speed)
            )

    # apply volume perturbation
    if use_volume_perturb and audio.numel() > 0:
        prob, lower_db, upper_db = volume_perturb
        if random.random() <= prob:
            gain_db = random.uniform(lower_db, upper_db)
            audio = audio * (10 ** (gain_db / 20))
    return audio


def augment_with_noise(audio, noise_sampler, lower_snr_db, upper_snr_db, is_test=False):
    if noise_sampler is None or is_test:
        return audio
    snr_db = random.uniform(lower_snr_db, upper_snr_db)
    noise = noise_sampler.random_noise(audio.size(1))
    audio_rms = audio.pow(2).mean().sqrt()
    noise_rms = noise.pow(2).mean().sqrt()
    snr = 10 ** (snr_db / 20)
    scaled_noise = noise * (audio_rms / (snr * noise_rms + 1e-8))
    audio = audio + scaled_noise
    return audio


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


class AudioDecoder:
    """
    Decode a audio sample from webdataset.
    The returned audio is a tensor of shape (1, num_samples) on CPU.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        audio_formats: Tuple[str] = ("flac", "wav", "mp3"),
    ):
        """
        Args:
          sample_rate:
            Target sample rate for audio.
          audio_formats:
            Tuple of audio file extensions to look for in the sample.
        """
        self.sample_rate = sample_rate
        self.audio_formats = audio_formats

    def __call__(self, sample):
        sample = fix_sample_key(sample)
        audio = torch.empty(0)
        for ext in self.audio_formats:
            if ext in sample:
                # load audio (1, num_samples)
                audio = load_audio(sample[ext], sample_rate=self.sample_rate)
                break
        sample["audio"] = audio
        return sample


class AudioDataset(torch.utils.data.IterableDataset):
    """
    Simple audio dataset built on webdataset tar files.
    """

    def __init__(
        self,
        audio_tars: List[str],
        sample_rate: int = 16000,
        buffer_size: int = 1000,
        nodesplitter: Optional[Any] = wds.split_by_node,
        workersplitter: Optional[Any] = wds.split_by_worker,
        audio_formats: Tuple[str] = ("wav", "flac", "mp3"),
    ):
        super().__init__()
        self.audio_tars = audio_tars
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.nodesplitter = nodesplitter
        self.workersplitter = workersplitter
        self.audio_formats = audio_formats

        self.audio_decoder = AudioDecoder(
            audio_formats=audio_formats,
            sample_rate=sample_rate,
        )

    def _build_dataset(self):
        return (
            wds.WebDataset(
                self.audio_tars,
                shardshuffle=len(self.audio_tars),
                nodesplitter=self.nodesplitter,
                workersplitter=self.workersplitter,
            )
            .decode()
            .map(self.audio_decoder)
            .shuffle(self.buffer_size)
        )

    def __iter__(self):
        # Build a fresh pipeline per worker/process to avoid state sharing issues.
        return iter(self._build_dataset())


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
            sample = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.noise_ds)
            sample = next(self.iterator)

        noise = sample["audio"]
        if noise.size(1) < target_length:
            repeats = (target_length // noise.size(1)) + 1
            noise = noise.repeat(1, repeats)[:, :target_length]
        elif noise.size(1) > target_length:
            start = random.randint(0, noise.size(1) - target_length)
            noise = noise[:, start : start + target_length]
        return noise


# TODO: support multiple labels for one audio, currently only support one label per audio.
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
        audio_formats: Tuple[str] = ("flac", "wav", "mp3"),
    ):
        """
        Args:
          labels_to_audios:
            A dict mapping from audio tar file to label tar file.
          sample_rate:
            Target sample rate for audio.
          audio_formats:
            Tuple of audio file extensions to look for in the sample.
        """
        self.labels = labels_to_audios
        self.sample_rate = sample_rate
        self.label_dataset = None
        self.audio_formats = audio_formats

    def __call__(self, sample):
        sample = fix_sample_key(sample)
        src = sample["__url__"]
        key = sample["__key__"]
        if self.label_dataset is None or self.label_dataset.path != self.labels[src]:
            self.label_dataset = LabelDataset(self.labels[src])

        audio = torch.empty(0)
        for ext in self.audio_formats:
            if ext in sample:
                # load audio (1, num_samples)
                audio = load_audio(sample[ext], sample_rate=self.sample_rate)
                break

        label = self.label_dataset[key]
        sample["audio"] = audio
        sample["label"] = label
        return sample


# TODO: support num_copies > 1
class ATDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        manifest: str,
        sample_rate: int,
        min_length: float = 0.1,
        max_length: float = 30.0,
        filter_func: Optional[Callable] = None,
        map_func: Optional[Callable] = None,
        use_noise_augment: bool = False,
        noise_augment: Tuple = (0.5, 10, 20),  # probs lower_db, upper_db
        noise_manifest: Optional[str] = None,
        use_speed_perturb: bool = True,
        speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
        use_volume_perturb: bool = True,
        volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
        feature_extractor: Optional[Callable] = None,
        buffer_size: int = 1000,
        is_test: bool = False,
        device=torch.device("cpu"),
    ):
        """
        Args:
            manifest:
                Manifest file containing audio tar files and label files.
            sample_rate:
                Target sample rate for audio.
            min_length:
                Minimum length (in seconds) of samples to consider.
            max_length:
                Maximum length (in seconds) of samples to consider.
            filter_func:
                A function to filter samples. It takes a sample dict as input and returns a boolean.
            map_func:
                A function to map samples. It takes a sample dict as input and returns a modified sample dict.
            noise_manifest:
                The filepath containing noise audio tars.
            noise_augment:
                Tuple of (probability, lower_snr_db, upper_snr_db) for noise augmentation.
            speed_perturb:
                Tuple of speeds for speed perturbation.
            volume_perturb:
                Tuple of (probability, lower_db, upper_db) for volume perturbation.
            feature_extractor:
                Feature extractor to extract features from raw audio.
            buffer_size:
                Buffer size for shuffling.
            is_test:
                Whether the dataset is for training or not.
            device:
                Device to calculate features.
        """
        super().__init__()

        self.device = device
        self.is_test = is_test
        self.buffer_size = buffer_size
        self.min_length = min_length
        self.max_length = max_length
        self.use_noise_augment = use_noise_augment
        self.use_speed_perturb = use_speed_perturb
        self.use_volume_perturb = use_volume_perturb

        assert os.path.exists(manifest), f"Manifest file {manifest} does not exist."
        self.manifest = manifest

        if dist.is_initialized():
            if os.environ.get("WORLD_SIZE") is None:
                os.environ["WORLD_SIZE"] = str(dist.get_world_size())
            if os.environ.get("RANK") is None:
                os.environ["RANK"] = str(dist.get_rank())
            if os.environ.get("LOCAL_RANK") is None:
                os.environ["LOCAL_RANK"] = str(
                    dist.get_rank() % torch.cuda.device_count()
                )

        labels_to_audios: Dict[str, str] = {}
        audio_tars: List[str] = []
        duration_hours = 0.0
        log_duration_warning = True  # to avoid too many warnings
        with open(self.manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items = line.split()
                if len(items) < 2:
                    logging.warning(
                        f"Each line in manifest file {self.manifest} should contain "
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
                            f"Manifest file {self.manifest} does not contain duration "
                            f"field, calculating duration from manifests, might be slow."
                            f" Please consider adding duration field to the manifest file."
                        )
                        log_duration_warning = False
                    duration_hours += get_manifest_duration(items[1]) / 3600.0
                else:
                    duration_hours += float(items[2])
                audio_tars.append(items[0])
                labels_to_audios[items[0]] = items[1]

        if duration_hours <= 0.001:
            logging.warning(
                f"Manifest {self.manifest} has very small duration : {duration_hours}, "
                f"please check if the manifest files are correct, especially the duration field."
            )
        logging.info(f"Manifest {self.manifest} duration: {duration_hours:.2f} hours")

        self.audio_tars = audio_tars
        self.duration_hours = duration_hours

        # sample_decoder is to decode audio and assign label
        self.sample_decoder = SampleDecoder(
            labels_to_audios=labels_to_audios,
            sample_rate=sample_rate,
        )

        self.filter_func = filter_func
        self.map_func = map_func

        self.noise_tars = None
        if use_noise_augment and noise_manifest is None:
            logging.warning(
                f"use_noise_augment is True but noise_manifest is not provided, "
                f"noise augmentation will be disabled."
            )
        if not use_noise_augment and noise_manifest is not None:
            logging.warning(
                f"noise_manifest is provided but use_noise_augment is False, "
                f"noise augmentation will be disabled."
            )
        if use_noise_augment and noise_manifest is not None and not is_test:
            assert os.path.exists(
                noise_manifest
            ), f"Noise manifest {noise_manifest} does not exist."
            noise_tars = []
            with open(noise_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if not os.path.exists(line):
                        logging.warning(
                            f"Noise audio tar file {line} does not exist, skipping."
                        )
                        continue
                    noise_tars.append(line)
            if noise_tars:
                self.noise_tars = noise_tars

        self.noise_prob = 0.0
        self.lower_snr_db = 10.0
        self.upper_snr_db = 20.0
        if not is_test and self.noise_tars is not None:
            assert (
                isinstance(noise_augment, (list, tuple)) and len(noise_augment) == 3
            ), (
                "noise_augment should be a tuple of "
                "(probability, lower_snr_db, upper_snr_db)"
            )
            (self.noise_prob, self.lower_snr_db, self.upper_snr_db) = noise_augment

        self.augment_audio = partial(
            audio_augmentation,
            sample_rate=sample_rate,
            speed_perturb=speed_perturb,
            volume_perturb=volume_perturb,
        )

        self.sample_rate = sample_rate
        if feature_extractor is not None:
            feature_extractor = feature_extractor.to(device)
        else:
            self.device = "cpu"
        self.feature_extractor = feature_extractor

    # __iter__ runs on child process, while __init__ runs on main process
    def __iter__(self):
        rank, world_size, worker, num_workers = pytorch_worker_info()
        total_num_workers = world_size * num_workers

        noise_sampler = None
        if self.noise_tars:
            tar_indexs = list(range(len(self.noise_tars)))
            pad_num = total_num_workers - (len(self.noise_tars) % total_num_workers)
            if pad_num != total_num_workers:
                for i in range(pad_num):
                    self.noise_tars.append(self.noise_tars[random.choice(tar_indexs)])
            self.noise_tars = sorted(self.noise_tars)
            noise_ds = AudioDataset(
                self.noise_tars,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            noise_sampler = NoiseSampler(noise_ds)

        self.add_noise = partial(
            augment_with_noise,
            noise_sampler=noise_sampler,
            lower_snr_db=self.lower_snr_db,
            upper_snr_db=self.upper_snr_db,
            is_test=self.is_test,
        )

        # pad audio_tars to be multiple of num_workers (for proper sharding)
        audio_tars = list(self.audio_tars)
        tar_indexs = list(range(len(audio_tars)))
        pad_num = total_num_workers - (len(audio_tars) % total_num_workers)
        if pad_num != total_num_workers:
            for i in range(pad_num):
                audio_tars.append(audio_tars[random.choice(tar_indexs)])
        audio_tars = sorted(audio_tars)

        dataset = (
            wds.WebDataset(
                audio_tars,
                shardshuffle=False if self.is_test else len(audio_tars),
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            .decode()
            .map(self.sample_decoder)
            .shuffle(1 if self.is_test else self.buffer_size)
        )

        stream = iter(dataset)

        while True:
            try:
                sample = next(stream)
            except StopIteration:
                return
            except RuntimeError as e:
                logging.error(f"Runtime error in data loading: {e}, skipping sample.")
                continue

            length = sample["audio"].size(1) / self.sample_rate
            if length < self.min_length or length > self.max_length:
                if self.is_test:
                    logging.warning(
                        f"Sample {sample['__key__']} length {length:.2f} out of "
                        f"range [{self.min_length}, {self.max_length}], skipping."
                    )
                else:
                    logging.debug(
                        f"Sample {sample['__key__']} length {length:.2f} out of "
                        f"range [{self.min_length}, {self.max_length}], skipping."
                    )
                continue

            if self.map_func is not None:
                sample = self.map_func(sample)

            if self.filter_func is not None:
                if not self.filter_func(sample):
                    if self.is_test:
                        logging.warning(
                            f"Sample {sample['__key__']} filtered out by filter_func, skipping."
                        )
                    else:
                        logging.debug(
                            f"Sample {sample['__key__']} filtered out by filter_func, skipping."
                        )
                    continue

            audio = sample["audio"]
            if not self.is_test:
                audio = self.augment_audio(audio)

            if not self.is_test and random.random() < self.noise_prob:
                audio = self.add_noise(audio)
            sample["audio"] = audio

            feature = None
            if self.feature_extractor is not None:
                with torch.no_grad():
                    feature = self.feature_extractor(
                        audio.to(self.device), sample_rate=self.sample_rate
                    ).cpu()
            if feature is not None:
                sample["feature"] = feature
            yield sample


class StreamingBucketBatcher:
    """
    Streaming bucketing batcher using multiple fixed-length buckets.
    Each bucket holds samples with similar durations.
    """

    def __init__(
        self,
        max_duration: float,  # in seconds
        weights: List[float],
        sample_rate: int = 16000,
        max_samples: Optional[int] = None,
        min_length: float = 0.1,  # in seconds
        max_length: float = 30,  # in seconds
        num_buckets: int = 30,
        mux_intra_batch: bool = True,
        is_test: bool = False,
        length_key="audio",
    ):
        """
        Args:
          max_duration:
            Maximum duration (in seconds) for each batch.
          max_samples:
            Maximum number of samples for each batch.
          min_length:
            Minimum length (in seconds) of samples to consider.
          max_length:
            Maximum length (in seconds) of samples to consider.
          num_buckets:
            Number of buckets to use.
          sample_rate:
            Sample rate of the audio samples.
          mux_intra_batch:
            Whether to mix samples from different datasets within the same batch.
          is_test:
            Whether the batcher is for training or not.
          length_key:
            Key in the sample dict to use for length calculation.
        """
        self.max_duration = max_duration
        # approximate max samples based on max_duration (1 second per sample)
        self.max_samples = max_samples if max_samples is not None else int(max_duration)
        self.num_buckets = num_buckets

        self.min_length = min_length
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.length_key = length_key
        self.is_test = is_test
        self.mux_intra_batch = mux_intra_batch
        self.weights = [w / sum(weights) for w in weights]

        if mux_intra_batch:
            if len(self.weights) > 1:
                logging.info(
                    f"Using StreamingBucketBatcher with intra-batch muxing and weights: {self.weights}"
                )
            assert isinstance(min_length, (int, float)) and isinstance(
                max_length, (int, float)
            ), f"When mux_intra_batch is True, min_length and max_length should be single float values."
            self.buckets = collections.defaultdict(collections.deque)
            self.bucket_item_lengths = [
                math.ceil((max_length - max(1, min_length)) / num_buckets) * (i + 1)
                for i in range(num_buckets)
            ]
        else:
            if len(self.weights) > 1:
                logging.info(
                    f"Using StreamingBucketBatcher without inter-batch muxing and weights: {self.weights}"
                )
            self.buckets = [
                collections.defaultdict(collections.deque) for _ in range(len(weights))
            ]
            self.bucket_item_lengths = []
            assert (
                isinstance(self.min_lengths, list)
                and isinstance(self.max_lengths, list)
                and len(self.min_lengths) == len(self.max_lengths) == len(weights)
            ), f"When mux_intra_batch is False, min_length and max_length should be lists with the same length as weights."
            for i in range(len(weights)):
                min_length = self.min_lengths[i]
                max_length = self.max_lengths[i]
                self.bucket_item_lengths.append(
                    [
                        math.ceil((max_length - max(1, min_length)) / num_buckets)
                        * (i + 1)
                        for i in range(num_buckets)
                    ]
                )

    def bucket_id(self, length, min_length, max_length):
        length = max(min_length, min(length, max_length))
        return int(
            (length - min_length) / (max_length - min_length) * (self.num_buckets - 1)
        )

    def __call__(
        self,
        data_streams: List[ATDataset],
    ):
        streams = [iter(data_stream) for data_stream in data_streams]
        weights = self.weights

        stream_idx = 0
        while True:
            # Fill buckets
            full_buckets = []
            try:
                if self.mux_intra_batch:
                    while True:
                        full_buckets = [
                            i
                            for i in range(self.num_buckets)
                            if self.bucket_item_lengths[i] * len(self.buckets[i])
                            > self.max_duration * 1.5
                        ]
                        if full_buckets:
                            break
                        stream_idx = np.random.choice(len(streams), p=weights)
                        sample = next(streams[stream_idx])
                        length = sample[self.length_key].size(1) / self.sample_rate
                        b_id = self.bucket_id(length, self.min_length, self.max_length)
                        self.buckets[b_id].append(sample)
                else:
                    stream_idx = np.random.choice(len(streams), p=weights)
                    while True:
                        full_buckets = [
                            i
                            for i in range(self.num_buckets)
                            if self.bucket_item_lengths[stream_idx][i]
                            * len(self.buckets[stream_idx][i])
                            > self.max_duration * 1.5
                        ]
                        if full_buckets:
                            break
                        sample = next(streams[stream_idx])
                        length = sample[self.length_key].size(1) / self.sample_rate
                        b_id = self.bucket_id(
                            length,
                            self.min_lengths[stream_idx],
                            self.max_lengths[stream_idx],
                        )
                        self.buckets[stream_idx][b_id].append(sample)
            except StopIteration:
                if not self.is_test:
                    # repeat the data stream, the BatchedDataset will handle epoch ending
                    streams[stream_idx] = iter(data_streams[stream_idx])
                    continue

            bucket_range = []

            if full_buckets:
                bucket_range.append(random.choice(full_buckets))
            else:
                # Normally, if self.is_test is False, will not run into this branch
                if self.is_test:
                    # all non-empty buckets
                    bucket_range = [
                        i for i in range(self.num_buckets) if self.buckets[i]
                    ]
                    bucket_range.reverse()

            last_b_id = bucket_range[0] if bucket_range else None

            num_samples = 0
            max_sample_length = 0
            batch = []
            batch_duration = 0

            buckets = self.buckets if self.mux_intra_batch else self.buckets[stream_idx]
            for b_id in bucket_range:
                while buckets[b_id]:
                    if num_samples >= self.max_samples:
                        break
                    sample = buckets[b_id][0]
                    length = sample[self.length_key].size(1) / self.sample_rate
                    tmp_max_sample_length = max(max_sample_length, length)
                    if tmp_max_sample_length * (num_samples + 1) > self.max_duration:
                        if not batch:
                            last_b_id = b_id
                            # for break the outer for loop
                            max_sample_length = length
                            num_samples = 1
                        break
                    else:
                        batch.append(buckets[b_id].popleft())
                        if length > max_sample_length:
                            max_sample_length = length
                        num_samples += 1
                if (
                    max_sample_length * num_samples >= self.max_duration
                    or num_samples >= self.max_samples
                ):
                    break

            if not batch:
                # Has full buckets but could not form a batch within max_duration
                # If a single sample exceeds batch_frames, yield it alone
                if last_b_id and buckets[last_b_id]:
                    batch.append(buckets[last_b_id].popleft())
                else:
                    if self.is_test:
                        return
            yield batch


class BatchedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        datasets: Union[ATDataset, List[ATDataset]],
        sample_rate: int,
        max_duration: float,
        max_samples: Optional[int] = None,
        epoch_hours: Optional[float] = None,
        mux_weights: Optional[List[float]] = None,
        num_copies: int = 1,
        mux_intra_batch: bool = True,
        is_test: bool = False,
        num_buckets: int = 30,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.is_test = is_test
        self.num_buckets = num_buckets
        self.mux_intra_batch = mux_intra_batch
        self.device = device

        if isinstance(datasets, ATDataset):
            self.datasets = [datasets]
        else:
            assert isinstance(datasets, list) and all(
                isinstance(d, ATDataset) for d in datasets
            ), "datasets should be an ATDataset or a list of ATDataset"
            self.datasets = datasets

        calculated_hours = 0.0
        dataset_hours = []
        min_length = 1e10
        min_lengths = []
        max_length = 0.0
        max_lengths = []
        for at in self.datasets:
            calculated_hours += at.duration_hours
            dataset_hours.append(at.duration_hours)
            min_lengths.append(at.min_length)
            max_lengths.append(at.max_length)
            if at.min_length < min_length:
                min_length = at.min_length
            if at.max_length > max_length:
                max_length = at.max_length
        self.min_length = min_length
        self.max_length = max_length
        self.min_lengths = min_lengths
        self.max_lengths = max_lengths

        if epoch_hours is None:
            logging.info(f"Using calculated epoch hours: {calculated_hours:.2f}")
            self.epoch_hours = calculated_hours
        else:
            if abs(calculated_hours - epoch_hours) / epoch_hours > 0.05:
                logging.warning(
                    f"Given epoch hours {epoch_hours} differ from calculated "
                    f"hours {calculated_hours:.2f} by more than 5%. "
                    f"Using given epoch hours, but you may want to double check."
                )
            self.epoch_hours = epoch_hours

        calculated_mux_weights = [dur / calculated_hours for dur in dataset_hours]
        if mux_weights is None and len(self.datasets) > 1:
            logging.info(f"Using calculated mux weights: {calculated_mux_weights}")
            self.mux_weights = calculated_mux_weights
        else:
            if mux_weights is not None:
                assert len(mux_weights) == len(calculated_mux_weights), (
                    f"Length of mux_weights {len(mux_weights)} should match number of datasets "
                    f"{len(calculated_mux_weights)}"
                )
                mux_weights_sum = sum(mux_weights)
                mux_weights = [w / mux_weights_sum for w in mux_weights]
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
                self.mux_weights = mux_weights
            else:
                self.mux_weights = [1]

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.epoch_batches_per_node = math.ceil(
            self.epoch_hours * 3600.0 / world_size / self.max_duration
        )

    def __iter__(self):
        rank, world_size, worker, num_workers = pytorch_worker_info()

        batcher = StreamingBucketBatcher(
            max_duration=self.max_duration,
            max_samples=self.max_samples,
            sample_rate=self.sample_rate,
            min_length=self.min_length if self.mux_intra_batch else self.min_lengths,
            max_length=self.max_length if self.mux_intra_batch else self.max_lengths,
            num_buckets=self.num_buckets,
            mux_intra_batch=self.mux_intra_batch,
            weights=self.mux_weights,
            is_test=self.is_test,
        )
        dataset = batcher(self.datasets)
        stream = iter(dataset)

        epoch_batches = self.epoch_batches_per_node // num_workers
        batch_count = 0

        while True:
            if batch_count >= epoch_batches and not self.is_test:
                return
            try:
                raw_batch = next(stream)
            except StopIteration:
                assert self.is_test, "Data stream should not end during training epoch"
                return

            raw_batch_output = collections.defaultdict(list)
            for sample in raw_batch:
                assert (
                    "audio" in sample
                ), "Each sample should contain 'audio' key after decoding."
                if sample["audio"].numel() == 0:
                    logging.warning(
                        f"Sample {sample['__key__']} has empty audio after decoding, skipping."
                    )
                    continue
                for k, v in sample.items():
                    if k == "audio":
                        raw_batch_output["audio"].append(v.squeeze(0))
                    elif k == "feature":
                        raw_batch_output["feature"].append(v.squeeze(0).transpose(0, 1))
                    else:
                        raw_batch_output[k].append(v)

            if not raw_batch_output["audio"]:
                continue

            batch_size = len(raw_batch_output["audio"])
            batch_output = {}
            for k in raw_batch_output:
                if len(raw_batch_output[k]) != batch_size:
                    logging.warning(
                        f"Inconsistent batch size for key {k}, expected {batch_size} "
                        f"but got {len(raw_batch_output[k])}, please check all data samples "
                        f"have the same keys, dropping {k}."
                    )
                    continue
                if k == "audio":
                    audio_lens = torch.tensor([a.size(0) for a in raw_batch_output[k]])
                    batch_output["audio_lens"] = audio_lens
                    batch_output["audio"] = torch.nn.utils.rnn.pad_sequence(
                        raw_batch_output[k], batch_first=True
                    )
                    continue
                if k == "feature":
                    feature_lens = torch.tensor(
                        [f.size(0) for f in raw_batch_output[k]]
                    )
                    batch_output["feature_lens"] = feature_lens
                    batch_output["feature"] = torch.nn.utils.rnn.pad_sequence(
                        raw_batch_output[k],
                        batch_first=True,
                        padding_value=math.log(1e-10),
                    )
                    continue
                batch_output[k] = raw_batch_output[k]
            batch_count += 1
            yield batch_output


# TODO: support fixed batch size by using only one bucket.
class ATDataloader(wds.WebLoader):
    def __init__(
        self,
        datasets: Union[str, ATDataset, List[Union[str, ATDataset]]],
        sample_rate: int,
        max_duration: float = 600.0,
        max_samples: Optional[int] = None,
        epoch_hours: Optional[float] = None,
        mux_weights: Optional[List[float]] = None,
        min_length: float = 0.1,
        max_length: float = 30.0,
        filter_func: Optional[Callable] = None,
        map_func: Optional[Callable] = None,
        feature_extractor: Optional[Callable] = None,
        use_noise_augment: bool = False,
        noise_augment: Tuple = (0.5, 10, 20),  # prob lower_snr_db, upper_snr_db
        noise_manifest: Optional[str] = None,
        use_speed_perturb: bool = True,
        speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
        use_volume_perturb: bool = True,
        volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
        num_copies: int = 1,
        buffer_size: int = 1000,
        num_workers: int = 4,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        mux_intra_batch: bool = True,
        num_buckets: int = 30,
        is_test: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Create a dataloader for streaming webdataset.
        Args:
            datasets:
                A single manifest path, ATDataset, or a list mixing both. String entries are
                converted into ATDataset with shared arguments from this dataloader.
            sample_rate:
                Target sample rate for audio.
            max_duration:
                Maximum duration (in seconds) for each batch.
            max_samples:
                Maximum number of samples for each batch.
            epoch_hours:
                Number of hours per epoch. If None, will calculate based on manifest durations.
            mux_weights:
                A list of weights for each dataset for muxing. If None, uniform weights are used.
            min_length:
                Minimum length (in seconds) of samples to consider.
            max_length:
                Maximum length (in seconds) of samples to consider.
            feature_extractor:
                Feature extractor to extract features from raw audio.
            filter_func:
                Function to filter samples when creating datasets from string manifests.
            map_func:
                Function to map samples when creating datasets from string manifests.
            noise_manifest:
                The filepath containing noise audio tars (used when creating datasets from strings).
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
            is_test:
                Whether the dataloader is for training or not.
            device:
                Device to calculate features.
            mux_intra_batch:
                If True, mux at sample level (intra-batch). If False, mux per batch.
                Bucketing happens after mux so both modes are supported.
        """

        if not isinstance(datasets, list):
            datasets = [datasets]

        atdatasets: List[ATDataset] = []
        for ds in datasets:
            if isinstance(ds, str):
                atds = ATDataset(
                    manifest=ds,
                    sample_rate=sample_rate,
                    feature_extractor=feature_extractor,
                    min_length=min_length,
                    max_length=max_length,
                    filter_func=filter_func,
                    map_func=map_func,
                    use_noise_augment=use_noise_augment,
                    noise_augment=noise_augment,
                    noise_manifest=noise_manifest,
                    use_speed_perturb=use_speed_perturb,
                    speed_perturb=speed_perturb,
                    use_volume_perturb=use_volume_perturb,
                    volume_perturb=volume_perturb,
                    buffer_size=buffer_size,
                    is_test=is_test,
                    device=device,
                )
                atdatasets.append(atds)
            else:
                assert isinstance(
                    ds, ATDataset
                ), "datasets entries must be str or ATDataset"
                atdatasets.append(ds)

        batched_dataset = BatchedDataset(
            datasets=atdatasets,
            sample_rate=sample_rate,
            max_duration=max_duration,
            max_samples=max_samples,
            epoch_hours=epoch_hours,
            mux_weights=mux_weights,
            mux_intra_batch=mux_intra_batch,
            num_copies=num_copies,
            num_buckets=num_buckets,
            is_test=is_test,
            device=device,
        )

        self.epoch_batches = batched_dataset.epoch_batches_per_node

        super().__init__(
            batched_dataset,
            batch_size=None,
            num_workers=num_workers,
            shuffle=False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )

    def __len__(self):
        return self.epoch_batches

    def __iter__(self):
        for batch in super().__iter__():
            yield batch
