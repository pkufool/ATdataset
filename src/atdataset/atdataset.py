#!/usr/bin/env python3
# Copyright  2025-2026 Wei Kang (wkang@pku.edu.cn)
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


import collections
import copy
import io
import json
import logging
import math
import os
import random

from functools import partial
from typing import Any, List, Tuple, Dict, Callable, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torchaudio
import webdataset as wds

from webdataset.utils import pytorch_worker_info

try:
    from atdataset.feature import Fbank, KaldiFbank, WhisperFbank
except (ImportError, ModuleNotFoundError):
    # feature module is optional — this file can be used standalone without it.
    # Feature extraction via feature_type will raise a clear error at runtime.
    Fbank = KaldiFbank = WhisperFbank = None  # type: ignore[assignment, misc]

_FEATURE_TYPE_MAP = {
    "Fbank": Fbank,
    "KaldiFbank": KaldiFbank,
    "WhisperFbank": WhisperFbank,
}

def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules that zipformer interacts with.
    Includes the ``random`` module, numpy, torch.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)


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


def load_audio(data, sample_rate: int = 16000):
    """
    Load audio from bytes data and resample to the target sample rate if needed.
    Return a tensor of shape (1, num_samples) on CPU.
    """
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio)
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


def get_manifest_duration_num_samples(manifest_path: str) -> Tuple[float, int]:
    """
    Calculate total duration and number of samples of audio files in a manifest file.
    Args:
      manifest_path:
        Path to the manifest file containing audio file paths and durations.
        Each line in the manifest file is in the format of:
        {"audio_filepath": path, "text": text, "duration": duration_in_seconds, "num_samples": num_samples}
    """
    total_duration = 0.0
    total_num_samples = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "duration" in item:
                total_duration += float(item["duration"])
            total_num_samples += 1
    return total_duration, total_num_samples


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
        self.path = manifest_path

        # if the manifest file does not exist, return empty labels
        # for some non speech audios.
        if not os.path.exists(manifest_path):
            logging.warning(f"Label manifest file {manifest_path} does not exist.")
            return
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

        text = self.label_dataset[key]
        sample["audio"] = audio
        sample["text"] = text
        return sample


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
        use_speed_perturb: bool = False,
        speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
        use_volume_perturb: bool = False,
        volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
        feature_type: Optional[str] = "Fbank",
        feature_extractor: Optional[Callable] = None,
        num_copies: int = 1,
        buffer_size: int = 1000,
        is_test: bool = False,
        need_num_samples: bool = False,
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
            use_noise_augment:
                Whether to apply noise augmentation.
            noise_augment:
                Tuple of (probability, lower_snr_db, upper_snr_db) for noise augmentation.
            noise_manifest:
                The filepath containing noise audio tars.
            use_speed_perturb:
                Whether to apply speed perturbation.
            speed_perturb:
                Tuple of speeds for speed perturbation.
            use_volume_perturb:
                Whether to apply volume perturbation.
            volume_perturb:
                Tuple of (probability, lower_db, upper_db) for volume perturbation.
            feature_extractor:
                Feature extractor to extract features from raw audio.
                If provided, takes priority over feature_type.
            feature_type:
                Type of feature to extract when feature_extractor is not provided.
                Supported types: "Fbank", "KaldiFbank", "WhisperFbank".
                A default extractor with the given type will be created using
                the specified sample_rate. Default is "Fbank".
                If neither feature_extractor nor feature_type is provided,
                no feature extraction is performed.
            num_copies:
                Number of copies of samples in one batch with different augmentations.
            buffer_size:
                Buffer size for shuffling.
            is_test:
                Whether the dataset is for training or not.
            need_num_samples:
                Whether to calculate the number of samples in the dataset.
        """
        super().__init__()

        self.is_test = is_test
        self.buffer_size = buffer_size
        self.min_length = min_length
        self.max_length = max_length
        self.use_noise_augment = use_noise_augment
        self.noise_manifest = noise_manifest
        self.noise_augment = noise_augment
        self.use_speed_perturb = use_speed_perturb
        self.speed_perturb = speed_perturb
        self.use_volume_perturb = use_volume_perturb
        self.volume_perturb = volume_perturb
        self.num_copies = num_copies
        self.need_num_samples = need_num_samples

        assert os.path.exists(manifest), f"Manifest file {manifest} does not exist."
        self.manifest = manifest

        if dist.is_initialized():
            if os.environ.get("WORLD_SIZE") is None:
                os.environ["WORLD_SIZE"] = str(dist.get_world_size())
            if os.environ.get("RANK") is None:
                os.environ["RANK"] = str(dist.get_rank())
            if os.environ.get("LOCAL_RANK") is None:
                num_devices = torch.cuda.device_count() or 1
                os.environ["LOCAL_RANK"] = str(
                    dist.get_rank() % num_devices
                )

        labels_to_audios: Dict[str, str] = {}
        audio_tars: List[str] = []
        duration_hours = 0.0
        num_samples = 0
        log_duration_warning = True  # to avoid too many warnings
        log_num_samples_warning = True  # to avoid too many warnings
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
                tmp_num_samples = None
                if len(items) < 3:
                    if log_duration_warning:
                        logging.warning(
                            f"Manifest file {self.manifest} does not contain duration "
                            f"field, calculating duration from manifests, might be slow."
                            f" Please consider adding duration field to the manifest file."
                        )
                        log_duration_warning = False
                    tmp_duration, tmp_num_samples = get_manifest_duration_num_samples(
                        items[1]
                    )
                    duration_hours += tmp_duration / 3600.0
                else:
                    duration_hours += float(items[2])

                if self.need_num_samples:
                    if len(items) < 4:
                        if tmp_num_samples is None:
                            if log_num_samples_warning:
                                logging.warning(
                                    f"Manifest file {self.manifest} does not contain num_samples "
                                    f"field, calculating num_samples from manifests, might be slow."
                                    f" Please consider adding num_samples field to the manifest file."
                                )
                                log_num_samples_warning = False
                            _, tmp_num_samples = get_manifest_duration_num_samples(
                                items[1]
                            )
                        num_samples += tmp_num_samples
                    else:
                        num_samples += int(items[3])

                audio_tars.append(items[0])
                labels_to_audios[items[0]] = items[1]

        if duration_hours <= 0.001:
            logging.warning(
                f"Manifest {self.manifest} has very small duration : {duration_hours}, "
                f"please check if the manifest files are correct, especially the duration field."
            )
        logging.info(
            f"Manifest {self.manifest} duration: {duration_hours:.2f} hours"
            + (f" num_samples: {num_samples}" if self.need_num_samples else "")
        )

        self.audio_tars = audio_tars
        self.duration_hours = duration_hours
        self.num_samples = num_samples

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
            assert os.path.exists(noise_manifest), (
                f"Noise manifest {noise_manifest} does not exist."
            )
            noise_tars = []
            with open(noise_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.split()[0]
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
            use_speed_perturb=use_speed_perturb,
            speed_perturb=speed_perturb,
            use_volume_perturb=use_volume_perturb,
            volume_perturb=volume_perturb,
        )

        self.sample_rate = sample_rate
        self.feature_type = feature_type

        # Resolve feature extractor with priority:
        #   1. user-provided feature_extractor
        #   2. built-in extractor created from feature_type
        #   3. None (no feature extraction)
        if feature_extractor is not None:
            pass  # use the provided extractor as-is
        elif feature_type is not None:
            if feature_type not in _FEATURE_TYPE_MAP:
                raise ValueError(
                    f"Unsupported feature_type: '{feature_type}'. "
                    f"Supported types: {list(_FEATURE_TYPE_MAP.keys())}"
                )
            extractor_cls = _FEATURE_TYPE_MAP[feature_type]
            if extractor_cls is None:
                raise ImportError(
                    f"Feature extractor '{feature_type}' is not available. "
                    f"Please install the required dependencies."
                )
            feature_extractor = extractor_cls(sample_rate=sample_rate)

        self.feature_extractor = feature_extractor

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"ATDataset(manifest={self.manifest}, sample_rate={self.sample_rate}, "
            f"duration_hours={self.duration_hours:.2f}, num_samples={self.num_samples}, buffer_size={self.buffer_size}, "
            f"min_length={self.min_length:.2f}, max_length={self.max_length:.2f}, "
            f"use_speed_perturb={self.use_speed_perturb}, speed_perturb={self.speed_perturb}, "
            f"use_volume_perturb={self.use_volume_perturb}, volume_perturb={self.volume_perturb}, "
            f"use_noise_augment={self.use_noise_augment}, noise_augment={self.noise_augment}, "
            f"noise_manifest={self.noise_manifest}, feature_extractor={self.feature_extractor}, "
            f"filter_func={self.filter_func}, map_func={self.map_func}, num_copies={self.num_copies}, "
            f"is_test={self.is_test})"
        )

    # __iter__ runs on child process, while __init__ runs on main process
    def __iter__(self):
        rank, world_size, worker, num_workers = pytorch_worker_info()
        total_num_workers = world_size * num_workers

        noise_sampler = None
        if self.noise_tars:
            noise_tars = list(self.noise_tars)
            pad_num = total_num_workers - (len(noise_tars) % total_num_workers)
            if pad_num != total_num_workers:
                for i in range(pad_num):
                    noise_tars.append(noise_tars[i % len(self.noise_tars)])
            noise_tars = sorted(noise_tars)
            noise_ds = AudioDataset(
                noise_tars,
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
        pad_num = total_num_workers - (len(audio_tars) % total_num_workers)
        if pad_num != total_num_workers:
            for i in range(pad_num):
                audio_tars.append(audio_tars[i % len(self.audio_tars)])
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

            samples = []
            for _ in range(self.num_copies):
                sample_copy = dict(sample)
                audio = sample["audio"].clone()
                if not self.is_test:
                    audio = self.augment_audio(audio)
                if not self.is_test and random.random() < self.noise_prob:
                    audio = self.add_noise(audio)
                sample_copy["audio"] = audio

                if self.feature_extractor is not None:
                    with torch.no_grad():
                        feature = self.feature_extractor(
                            audio, sample_rate=self.sample_rate
                        )
                    sample_copy["feature"] = feature
                samples.append(sample_copy)
            yield samples if self.num_copies > 1 else samples[0]


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
        batch_size: Optional[int] = None,
        min_length: Union[float, List[float]] = 0.1,  # in seconds
        max_length: Union[float, List[float]] = 30,  # in seconds
        num_buckets: int = 30,
        mux_intra_batch: bool = True,
        is_test: bool = False,
        length_key="audio",
    ):
        """
        Args:
          max_duration:
            Maximum duration (in seconds) for each batch.
          weights:
            List of weights for each dataset stream.
          sample_rate:
            Sample rate of the audio samples.
          max_samples:
            Maximum number of samples for each batch.
          batch_size:
            The num of samples in each batch, if specified, will override max_duration and max_samples.
          min_length:
            Minimum length (in seconds) of samples to consider.
          max_length:
            Maximum length (in seconds) of samples to consider.
          num_buckets:
            Number of buckets to use.
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
        self.batch_size = batch_size

        # use 1 bucket if batch_size is specified, so will not change the bucket assignment logic,
        # and just fill the batch with samples from the same bucket until batch_size is reached.
        self.num_buckets = num_buckets if batch_size is None else 1

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
            ), (
                f"When mux_intra_batch is True, min_length and max_length should be single float values."
            )
            bucket_width = (max_length - min_length) / num_buckets
            self.bucket_item_lengths = [
                min_length + (i + 0.5) * bucket_width
                for i in range(num_buckets)
            ]
        else:
            if len(self.weights) > 1:
                logging.info(
                    f"Using StreamingBucketBatcher without inter-batch muxing and weights: {self.weights}"
                )
            self.bucket_item_lengths = []
            assert (
                isinstance(self.min_length, list)
                and isinstance(self.max_length, list)
                and len(self.min_length) == len(self.max_length) == len(weights)
            ), (
                f"When mux_intra_batch is False, min_length and max_length should be lists with the same length as weights."
            )
            for i in range(len(weights)):
                min_len = self.min_length[i]
                max_len = self.max_length[i]
                bw = (max_len - min_len) / num_buckets
                self.bucket_item_lengths.append(
                    [
                        min_len + (j + 0.5) * bw
                        for j in range(num_buckets)
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

        if self.mux_intra_batch:
            # NOTE on per-batch dataset ratio:
            # With mux_intra_batch=True, samples are drawn from streams according
            # to `weights` and placed into shared buckets by duration. A batch is
            # assembled from a single bucket via FIFO. This means the per-batch
            # dataset ratio depends on the duration overlap between datasets — if
            # datasets have different length distributions, some buckets will be
            # dominated by one dataset.
            #
            # Alternative designs considered and rejected:
            # - Per-dataset independent buckets: would allow exact per-batch ratio
            #   control, but causes memory blowup when dataset×bucket combinations
            #   are sparse — slow-draining buckets block others and accumulate
            #   unboundedly.
            # - Per-sample rebalancing at batch assembly: requires O(n) scanning
            #   of the bucket deque, too expensive for large buckets.
            #
            # The current design guarantees correct global ratio (over the full
            # epoch) and is a deliberate trade-off favoring bounded memory and
            # O(1) batch assembly over strict per-batch ratio adherence.
            buckets = collections.defaultdict(collections.deque)
        else:
            buckets = [
                collections.defaultdict(collections.deque)
                for _ in range(len(weights))
            ]

        stream_idx = 0
        while True:
            # Fill buckets
            full_buckets = []
            try:
                if self.mux_intra_batch:
                    while True:
                        if self.batch_size is not None:
                            if len(buckets[0]) > self.batch_size:
                                full_buckets = [0]
                                break
                        else:
                            full_buckets = [
                                i
                                for i in range(self.num_buckets)
                                if self.bucket_item_lengths[i] * len(buckets[i])
                                > self.max_duration * 1.5
                            ]
                            if full_buckets:
                                break
                        stream_idx = np.random.choice(len(streams), p=weights)
                        samples = next(streams[stream_idx])
                        if isinstance(samples, list):
                            sample = samples[0]
                        else:
                            sample = samples
                        length = sample[self.length_key].size(1) / self.sample_rate
                        b_id = self.bucket_id(length, self.min_length, self.max_length)
                        buckets[b_id].append(samples)
                else:
                    stream_idx = np.random.choice(len(streams), p=weights)
                    while True:
                        if self.batch_size is not None:
                            if len(buckets[stream_idx]) > self.batch_size:
                                full_buckets = [0]
                                break
                        else:
                            full_buckets = [
                                i
                                for i in range(self.num_buckets)
                                if self.bucket_item_lengths[stream_idx][i]
                                * len(buckets[stream_idx][i])
                                > self.max_duration * 1.5
                            ]
                            if full_buckets:
                                break
                        samples = next(streams[stream_idx])
                        if isinstance(samples, list):
                            sample = samples[0]
                        else:
                            sample = samples
                        length = sample[self.length_key].size(1) / self.sample_rate
                        b_id = self.bucket_id(
                            length,
                            self.min_length[stream_idx],
                            self.max_length[stream_idx],
                        )
                        buckets[stream_idx][b_id].append(samples)
            except StopIteration:
                if not self.is_test:
                    # repeat the data stream, the BatchedDataset will handle epoch ending
                    streams[stream_idx] = iter(data_streams[stream_idx])
                    continue
                # Test mode: stream exhausted, flush all remaining buffered samples
                # by treating every non-empty bucket as ready.

            bucket_range = []

            if full_buckets:
                bucket_range.append(random.choice(full_buckets))
            elif self.is_test:
                # Flush remaining samples from all non-empty buckets (longest first)
                active_buckets = buckets if self.mux_intra_batch else buckets[stream_idx]
                bucket_range = [
                    i for i in range(self.num_buckets) if active_buckets[i]
                ]
                bucket_range.reverse()

            last_b_id = bucket_range[0] if bucket_range else None

            num_samples = 0
            max_sample_length = 0
            batch = []

            active_buckets = buckets if self.mux_intra_batch else buckets[stream_idx]
            for b_id in bucket_range:
                while active_buckets[b_id]:
                    if self.batch_size is not None and num_samples >= self.batch_size:
                        break
                    if num_samples >= self.max_samples and self.batch_size is None:
                        break
                    samples = active_buckets[b_id][0]
                    if isinstance(samples, list):
                        sample = samples[0]
                    else:
                        sample = samples
                    length = sample[self.length_key].size(1) / self.sample_rate
                    tmp_max_sample_length = max(max_sample_length, length)
                    if (
                        tmp_max_sample_length * (num_samples + 1) > self.max_duration
                        and self.batch_size is None
                    ):
                        if not batch:
                            last_b_id = b_id
                            # for break the outer for loop
                            max_sample_length = length
                            num_samples = 1
                        break
                    else:
                        batch.append(active_buckets[b_id].popleft())
                        if length > max_sample_length:
                            max_sample_length = length
                        num_samples += 1
                if (
                    max_sample_length * num_samples >= self.max_duration
                    or num_samples >= self.max_samples
                    or (self.batch_size is not None and num_samples >= self.batch_size)
                ):
                    break

            if not batch:
                # Has full buckets but could not form a batch within max_duration
                # If a single sample exceeds batch_frames, yield it alone
                if last_b_id is not None and active_buckets[last_b_id]:
                    batch.append(active_buckets[last_b_id].popleft())
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
        batch_size: Optional[int] = None,
        epoch_hours: Optional[float] = None,
        mux_weights: Optional[List[float]] = None,
        mux_intra_batch: bool = True,
        num_copies: int = 1,
        is_test: bool = False,
        num_buckets: int = 30,
        fill_factor: float = 1.15,
    ):
        """
        Batched dataset that combines multiple ATDatasets with streaming bucketing and optional intra-batch muxing.

        Args:
          datasets:
            An ATDataset or a list of ATDatasets to combine.
          sample_rate:
            Sample rate of the audio samples.
          max_duration:
            Maximum duration (in seconds) for each batch.
          max_samples:
            Maximum number of samples for each batch.
          batch_size:
            If specified, will use fixed batch size instead of max_duration to control batch size.
            When batch_size is specified, max_duration and max_samples will be ignored, and the batcher
            will create batches with batch_size samples.
          epoch_hours:
            Total hours of audio to process in one epoch.
            If None, will use the sum of durations of the datasets.
          mux_weights:
            List of weights for each dataset stream.
            If None, will use the duration-based weights.
          num_copies:
            Number of copies of the sample to return in one batch. Default is 1.
          mux_intra_batch:
            Whether to mix samples from different datasets within the same batch. Default is True.
          is_test:
            Whether the dataset is for training or not. Default is False.
          num_buckets:
            Number of buckets to use for streaming bucketing. Default is 30.
          fill_factor:
            Correction factor for epoch batch count estimation. Batches are
            typically not fully packed to max_duration due to bucketing
            constraints, so the actual number of batches per epoch is higher
            than the naive estimate. Default is 1.15 (i.e. ~87% average fill).
            Use examples/example.py to measure the true value for your data.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.is_test = is_test
        self.num_buckets = num_buckets
        self.mux_intra_batch = mux_intra_batch
        self.num_copies = num_copies
        self.batch_size = batch_size

        if isinstance(datasets, ATDataset):
            self.datasets = [datasets]
        else:
            assert isinstance(datasets, list) and all(
                isinstance(d, ATDataset) for d in datasets
            ), "datasets should be an ATDataset or a list of ATDataset"
            self.datasets = datasets

        calculated_hours = 0.0
        dataset_hours = []
        calculated_samples = 0
        dataset_samples = []
        min_length = 1e10
        min_lengths = []
        max_length = 0.0
        max_lengths = []
        for at in self.datasets:
            calculated_hours += at.duration_hours
            dataset_hours.append(at.duration_hours)
            calculated_samples += at.num_samples
            dataset_samples.append(at.num_samples)
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
            if batch_size is None:
                logging.info(f"Using calculated epoch hours: {calculated_hours:.2f}")
            self.epoch_hours = calculated_hours
        else:
            if (
                abs(calculated_hours - epoch_hours) / epoch_hours > 0.05
                and batch_size is None
            ):
                logging.warning(
                    f"Given epoch hours {epoch_hours} differ from calculated "
                    f"hours {calculated_hours:.2f} by more than 5%. "
                    f"Using given epoch hours, but you may want to double check."
                )
            self.epoch_hours = epoch_hours

        if batch_size is not None:
            calculated_mux_weights = [s / calculated_samples for s in dataset_samples]
        else:
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
        if self.batch_size is not None:
            self.epoch_batches_per_node = math.ceil(
                calculated_samples / self.batch_size / world_size
            )
        else:
            self.epoch_batches_per_node = math.ceil(
                self.epoch_hours * 3600.0 / world_size / self.max_duration
                * fill_factor
            )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"BatchedDataset(sample_rate={self.sample_rate}, max_duration={self.max_duration}, "
            f"max_samples={self.max_samples}, batch_size={self.batch_size}, epoch_hours={self.epoch_hours:.2f}, "
            f"mux_weights={self.mux_weights}, num_copies={self.num_copies}, mux_intra_batch={self.mux_intra_batch}, "
            f"is_test={self.is_test}, num_buckets={self.num_buckets}, "
            f"datasets={self.datasets})"
        )

    def __iter__(self):
        rank, world_size, worker, num_workers = pytorch_worker_info()

        batcher = StreamingBucketBatcher(
            max_duration=self.max_duration,
            max_samples=self.max_samples,
            batch_size=self.batch_size,
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
            for i in range(self.num_copies):
                for sample in raw_batch:
                    if isinstance(sample, list):
                        assert len(sample) == self.num_copies, (
                            f"Expected sample list of length {self.num_copies}, but got {len(sample)}"
                        )
                        sample = sample[i]
                    assert "audio" in sample, (
                        "Each sample should contain 'audio' key after decoding."
                    )
                    if sample["audio"].numel() == 0:
                        logging.warning(
                            f"Sample {sample['__key__']} has empty audio after decoding, skipping."
                        )
                        continue
                    for k, v in sample.items():
                        if k == "audio":
                            raw_batch_output["audio"].append(v.squeeze(0))
                        elif k == "feature":
                            raw_batch_output["feature"].append(
                                v.squeeze(0).transpose(0, 1)
                            )
                        elif k == "__key__":
                            raw_batch_output["ids"].append(v)
                            raw_batch_output[k].append(v)
                        else:
                            raw_batch_output[k].append(v)

            if not raw_batch_output["audio"]:
                continue

            batch_size = len(raw_batch_output["audio"])
            batch_output = {}
            for k in raw_batch_output:
                if len(raw_batch_output[k]) != batch_size and k not in ("mp3", "flac", "wav"):
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
                values = raw_batch_output[k]
                if isinstance(values[0], torch.Tensor) and values[0].dim() >= 1:
                    batch_output[f"{k}_lens"] = torch.tensor(
                        [v.size(0) for v in values]
                    )
                    batch_output[k] = torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True
                    )
                else:
                    batch_output[k] = values
            batch_count += 1
            yield batch_output


class ATDataloader(wds.WebLoader):
    def __init__(
        self,
        datasets: Union[str, ATDataset, List[Union[str, ATDataset]]],
        sample_rate: int,
        max_duration: float = 600.0,
        max_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        epoch_hours: Optional[float] = None,
        mux_weights: Optional[List[float]] = None,
        min_length: float = 0.1,
        max_length: float = 60.0,
        filter_func: Optional[Callable] = None,
        map_func: Optional[Callable] = None,
        feature_type: Optional[str] = "Fbank",
        feature_extractor: Optional[Callable] = None,
        use_noise_augment: bool = False,
        noise_augment: Tuple = (0.5, 10, 20),  # prob, lower_snr_db, upper_snr_db
        noise_manifest: Optional[str] = None,
        use_speed_perturb: bool = False,
        speed_perturb: Tuple = (0.9, 1.0, 1.1),  # speeds
        use_volume_perturb: bool = False,
        volume_perturb: Tuple = (0.5, -10, 6),  # prob, lower_db, upper_db
        num_copies: int = 1,
        buffer_size: int = 1000,
        num_workers: int = 4,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        seed: Optional[int] = 1015,
        mux_intra_batch: bool = True,
        num_buckets: int = 30,
        fill_factor: float = 1.15,
        is_test: bool = False,
    ):
        """
        Create a dataloader for streaming webdataset that supports multiple datasets with
        different muxing strategies and bucketing for variable length audio samples.

        Args:
            datasets:
                A single manifest path, ATDataset, or a list mixing both. String entries are
                converted into ATDataset with shared arguments from this dataloader.
            sample_rate:
                Target sample rate for audio.
            max_duration:
                Maximum duration (in seconds) for each batch.
            max_samples:
                Maximum number of samples for each batch, used together with max_duration
                to control batch size.
            batch_size:
                If specified, will use fixed batch size instead of max_duration to control batch size.
                When batch_size is specified, max_duration and max_samples will be ignored, and the batcher
                will create batches with batch_size samples.
            epoch_hours:
                Number of hours per epoch. If None, will calculate based on manifest durations.
            mux_weights:
                A list of weights for each dataset for muxing. If None, will calculate
                based on dataset durations.
            min_length:
                Minimum length (in seconds) of samples to consider, used when creating datasets
                from string manifests.
            max_length:
                Maximum length (in seconds) of samples to consider, used when creating datasets
                from string manifests.
            filter_func:
                Function to filter samples, used when creating datasets from string manifests.
            map_func:
                Function to map samples, used when creating datasets from string manifests.
            feature_type:
                Type of feature to extract, used when creating datasets from string manifests.
                Supported types: "Fbank", "KaldiFbank", "WhisperFbank".
                A default extractor with the given type will be created.
                Only used when feature_extractor is not provided.
                If feature_extractor is provided, this argument will be ignored.
                If neither is provided, no feature extraction is performed.
                Default is "Fbank".
            feature_extractor:
                Feature extractor to extract features from raw audio, used when creating datasets
                from string manifests. Takes priority over feature_type.
            use_noise_augment:
                Whether to use noise augmentation, used when creating datasets from string manifests.
            noise_manifest:
                The filepath containing noise audio tars, used when creating datasets from strings manifests.
            noise_augment:
                Tuple of (probability, lower_snr_db, upper_snr_db) for noise augmentation,
                used when creating datasets from string manifests.
            use_speed_perturb:
                Whether to use speed perturbation, used when creating datasets from string manifests.
            speed_perturb:
                Tuple of speeds for speed perturbation, used when creating datasets from string manifests.
            use_volume_perturb:
                Whether to use volume perturbation, used when creating datasets from string manifests.
            volume_perturb:
                Tuple of (probability, lower_db, upper_db) for volume perturbation, used when
                creating datasets from string manifests.
            num_copies:
                Number of copies of the samples to return, used when creating datasets from string manifests.
            buffer_size:
                Buffer size for shuffling, used when creating datasets from string manifests.
            num_workers:
                Number of workers for dataloader.
            worker_init_fn:
                Worker init function for dataloader.
            prefetch_factor:
                Prefetch factor for dataloader, used when num_workers > 0.
            seed:
                Optional integer seed for reproducibility. Each worker is seeded with ``seed + worker_id``
                so that different workers produce different (but deterministic) sequences.
                When None, no seeding is applied and behaviour is non-deterministic.
            num_buckets:
                Number of buckets for bucketing variable length audio samples.
            fill_factor:
                Correction factor for epoch batch count estimation. Due to
                bucketing constraints, batches are typically not fully packed.
                Default 1.15 means ~87% average fill. Measure the true value
                with examples/example.py and adjust accordingly.
            is_test:
                Whether the dataloader is for testing or not, for training and validation, set is_test to False
                to enable shuffling and data augmentation.
            mux_intra_batch:
                If True, mux at sample level (intra-batch). If False, mux per batch.
        """

        if not isinstance(datasets, list):
            datasets = [datasets]

        # When num_copies > 1, each raw sample produces num_copies augmented
        # variants in the final batch. Scale down the batching budget so the
        # total batch size after expansion matches what the user requested.
        max_duration = max_duration / num_copies if max_duration is not None else None
        max_samples = max_samples // num_copies if max_samples is not None else None
        batch_size = batch_size // num_copies if batch_size is not None else None

        atdatasets: List[ATDataset] = []
        for ds in datasets:
            if isinstance(ds, str):
                atds = ATDataset(
                    manifest=ds,
                    sample_rate=sample_rate,
                    feature_extractor=feature_extractor,
                    feature_type=feature_type,
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
                    num_copies=num_copies,
                    is_test=is_test,
                    need_num_samples=batch_size is not None,
                )
                atdatasets.append(atds)
            else:
                assert isinstance(ds, ATDataset), (
                    "datasets entries must be str or ATDataset"
                )
                assert ds.sample_rate == sample_rate, (
                    f"Dataset sample rate {ds.sample_rate} does not match dataloader sample rate {sample_rate}"
                )
                assert ds.num_copies == num_copies, (
                    f"Dataset num_copies {ds.num_copies} does not match dataloader num_copies {num_copies}"
                )
                assert ds.is_test == is_test, (
                    f"Dataset is_test {ds.is_test} does not match dataloader is_test {is_test}"
                )
                assert ds.need_num_samples == (batch_size is not None), (
                    f"Dataset need_num_samples {ds.need_num_samples} does not match dataloader"
                    f" need_num_samples {batch_size is not None}"
                )
                if feature_type is None:
                    assert ds.feature_type is None, (
                        f"Dataset feature_type {ds.feature_type} should be None when dataloader feature_type is None"
                    )
                else:
                    assert ds.feature_type == feature_type, (
                        f"Dataset feature_type {ds.feature_type} does not match dataloader feature_type {feature_type}"
                    )
                if feature_extractor is None:
                    assert ds.feature_extractor is None, (
                        f"Dataset feature_extractor should be None when dataloader feature_extractor is None"
                    )
                else:
                    assert type(ds.feature_extractor) is type(feature_extractor), (
                        f"Dataset feature_extractor should be of type {type(feature_extractor)} when dataloader "
                        f"feature_extractor is of type {type(feature_extractor)}"
                    )
                atdatasets.append(ds)

        batched_dataset = BatchedDataset(
            datasets=atdatasets,
            sample_rate=sample_rate,
            max_duration=max_duration,
            max_samples=max_samples,
            batch_size=batch_size,
            epoch_hours=epoch_hours,
            mux_weights=mux_weights,
            num_copies=num_copies,
            mux_intra_batch=mux_intra_batch,
            num_buckets=num_buckets,
            fill_factor=fill_factor,
            is_test=is_test,
        )

        self.seed = seed
        self.num_copies = num_copies
        self._epoch = 0

        self.dataset = batched_dataset
        self.epoch_batches = batched_dataset.epoch_batches_per_node
        self.num_workers = num_workers if not is_test else 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

        if seed is not None:
            fix_random_seed(seed)

            _user_init = worker_init_fn
            _seed = seed
            # Use a mutable container so the closure captures the current epoch
            # at iteration time rather than at construction time.
            _epoch_ref = [0]
            self._epoch_ref = _epoch_ref

            def _worker_init_fn(worker_id: int):
                worker_seed = _seed + worker_id + _epoch_ref[0] * 10000
                fix_random_seed(worker_seed)
                if _user_init is not None:
                    _user_init(worker_id)

            worker_init_fn = _worker_init_fn
        else:
            self._epoch_ref = None

        super().__init__(
            batched_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=worker_init_fn,
        )

    def __repr__(self):
        return (
            f"ATDataloader(dataset={self.dataset}, length={self.epoch_batches}, "
            f"num_workers={self.num_workers}, prefetch_factor={self.prefetch_factor}, "
            f"seed={self.seed})"
        )

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.epoch_batches

    def __iter__(self):
        if self._epoch_ref is not None:
            self._epoch_ref[0] = self._epoch
        for batch in super().__iter__():
            yield batch
        self._epoch += 1
