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
import logging
import math
import os
import random
import time
import threading

from functools import partial
from typing import Any, List, Tuple, Dict, Callable, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
import torch.multiprocessing as mp


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


def _simple_filter_keys(sample, exts=("wav", "flac", "mp3")):
    return any(k in sample for k in exts)


def _simple_decode_audio(
    sample, exts=("wav", "flac", "mp3"), sample_rate=16000, device="cpu"
):
    for ext in exts:
        if ext in sample:
            return load_audio(sample[ext], sample_rate=sample_rate, device=device)
    raise RuntimeError("No noise audio found in sample")


def create_simple_audio_dataset(
    audio_tars: List[str],
    sample_rate: int = 16000,
    exts: Tuple[str] = ("wav", "flac", "mp3"),
    buffer_size: int = 200,
    nodesplitter: Optional[Any] = None,
    workersplitter: Optional[Any] = None,
    device: str = "cpu",
):
    """
    Create a simple audio dataset from webdataset tar files.
    Args:
      audio_tars:
        List of audio tar files.
      sample_rate:
        Target sample rate for audio.
      exts:
        Tuple of audio file extensions to look for in the sample.
      buffer_size:
        Buffer size for shuffling.
      nodesplitter:
        Node splitter for webdataset.
      workersplitter:
        Worker splitter for webdataset.
      device:
        Device to load audio tensors.
    """

    simple_filter_keys = partial(_simple_filter_keys, exts=exts)
    simple_decode_audio = partial(
        _simple_decode_audio, exts=exts, sample_rate=sample_rate, device=device
    )
    audio_tars = list(audio_tars)

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

    def __init__(
        self,
        noise_ds,
        stop_event: Optional[threading.Event] = None,
        buffer_size: int = 200,
    ):
        self.noise_ds = noise_ds
        self.iterator = None
        self.buffer_size = max(buffer_size, 200)
        self.buffer = collections.deque()
        self.stop_event = stop_event
        self.read_thread = None
        self.buffer_lock = None

    def _preload_noise(self):
        while True:
            try:
                bucke_full = False
                with self.buffer_lock:
                    bucke_full = len(self.buffer) >= self.buffer_size
                if bucke_full:
                    time.sleep(0.5)
                    continue
                if self.stop_event.is_set():
                    logging.info("Stop event set, stopping noise preload thread.")
                    break
                noise = next(self.iterator)
                with self.buffer_lock:
                    self.buffer.append(noise)
            except StopIteration:
                self.iterator = iter(self.noise_ds)
            except Exception as e:
                logging.warning(f"Error in noise preloading thread: {e}, will continue")
                continue
            except KeyboardInterrupt:
                self.stop_event.set()
                break

    def random_noise(self, target_length):
        if self.iterator is None:
            self.iterator = iter(self.noise_ds)

        if self.buffer_lock is None:
            self.buffer_lock = threading.Lock()

        if self.stop_event is None:
            self.stop_event = threading.Event()

        if self.read_thread is None or not self.read_thread.is_alive():
            self.read_thread = threading.Thread(target=self._preload_noise, daemon=True)
            self.read_thread.start()

        noise = None
        while noise is None:
            with self.buffer_lock:
                if len(self.buffer) > 0:
                    noise = self.buffer.popleft()
            if noise is None:
                time.sleep(0.1)

        if noise.size(1) < target_length:
            repeats = (target_length // noise.size(1)) + 1
            noise = noise.repeat(1, repeats)[:, :target_length]
        elif noise.size(1) > target_length:
            start = random.randint(0, noise.size(1) - target_length)
            noise = noise[:, start : start + target_length]
        return noise

    def __del__(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.read_thread is not None and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)


class LabelDataset:
    def __init__(self, tar_path):
        """
        Load labels from a webdataset tar file.
        Args:
          tar_path:
            Path to the tar file containing labels.
        """
        self._labels = {}

        # if the tar file does not exist, return empty labels
        # for some non speech audios.
        if not os.path.exists(tar_path):
            logging.warning(f"Label tar file {tar_path} does not exist.")
            return

        def decode_text(sample):
            sample = fix_sample_key(sample)
            return {"txt": sample["txt"], "key": sample["__key__"]}

        # No need to split on nodes and workers here, audio tar and label tar is one one mapping
        dataset = (
            wds.WebDataset(
                tar_path, shardshuffle=False, nodesplitter=None, workersplitter=None
            )
            .decode()
            .map(decode_text)
            .to_tuple("key", "txt")
        )
        self.path = tar_path
        for (key, txt) in dataset:
            self._labels[key] = txt

    def __getitem__(self, key):
        if key not in self._labels:
            return "<|empty|>"
        return self._labels[key]


class SampleDecoder:
    """
    Decode a sample from webdataset, including loading audio and fetching label.
    And also apply augmentations if needed.
    """

    def __init__(
        self,
        config: Dict,
        labels_to_audios: Dict,
        sample_rate: int = 16000,
        is_train: bool = True,
        device: torch.device = torch.device("cpu"),
        audio_ext: Tuple[str] = ("flac", "wav", "mp3"),
    ):
        """
        Args:
          config:
            A dict containing configuration for data loading and augmentation.
          labels_to_audios:
            A dict mapping from audio tar file to label tar file.
          sample_rate:
            Target sample rate for audio.
          is_train:
            Whether the dataset is for training or not.
          audio_ext:
            Tuple of audio file extensions to look for in the sample.
        """
        self.config = config
        self.labels = labels_to_audios
        self.sample_rate = sample_rate
        self.label_dataset = None
        self.audio_ext = audio_ext
        self.is_train = is_train
        self.device = device

    def __call__(self, sample):
        sample = fix_sample_key(sample)
        src = sample["__url__"]
        key = sample["__key__"]
        if self.label_dataset is None or self.label_dataset.path != self.labels[src]:
            logging.info(f"Loading labels from {self.labels[src]}")
            self.label_dataset = LabelDataset(self.labels[src])

        audio = torch.empty(0, device=self.device)
        for ext in self.audio_ext:
            if ext in sample:
                # load audio (1, num_samples)
                audio = load_audio(
                    sample[ext], sample_rate=self.sample_rate, device=self.device
                )
                break

        # apply speed perturbation
        if (
            self.is_train
            and self.config.get("speed_perturb", False)
            and audio.numel() > 0
        ):
            speed = random.choice(self.config["speed_perturb"])
            if speed != 1:
                audio = torchaudio.functional.resample(
                    audio, self.sample_rate, int(self.sample_rate * speed)
                )

        # apply volume perturbation
        if (
            self.is_train
            and self.config.get("volume_perturb", False)
            and audio.numel() > 0
        ):
            prob, lower_db, upper_db = self.config["volume_perturb"]
            if random.random() <= prob:
                gain_db = random.uniform(lower_db, upper_db)
                audio = audio * (10 ** (gain_db / 20))

        label = self.label_dataset[key]
        return {
            "audio": audio,
            "label": label,
        }


def augment_with_noise(audio, noise_sampler, lower_snr_db, upper_snr_db, is_train=True):
    if noise_sampler is None or not is_train:
        return audio
    snr_db = random.uniform(lower_snr_db, upper_snr_db)
    start = time.time()
    noise = noise_sampler.random_noise(audio.size(0)).squeeze(0)
    # logging.warning(f"Noise sampling time: {time.time()-start} seconds")
    start = time.time()
    audio_rms = audio.pow(2).mean().sqrt()
    noise_rms = noise.pow(2).mean().sqrt()
    snr = 10 ** (snr_db / 20)
    scaled_noise = noise * (audio_rms / (snr * noise_rms + 1e-8))
    audio = audio + scaled_noise
    # logging.warning(f"Noise adding time: {time.time()-start} seconds")
    return audio


class StreamingBucketBatcher:
    """
    Streaming bucketing batcher using multiple fixed-length buckets.
    Each bucket holds samples with similar durations.
    """

    def __init__(
        self,
        max_duration: float = 500.0,  # in seconds
        num_buckets: int = 30,
        min_length: float = 1,  # in seconds
        max_length: float = 30,  # in seconds
        sample_rate: int = 16000,
        length_key="audio",
        stop_event: Optional[threading.Event] = None,
    ):
        """
        Args:
          max_duration:
            Maximum duration (in seconds) for each batch.
          num_buckets:
            Number of buckets to use.
          min_length:
            Minimum length (in seconds) of samples to consider.
          max_length:
            Maximum length (in seconds) of samples to consider.
          sample_rate:
            Sample rate of the audio samples.
          length_key:
            Key in the sample dict to use for length calculation.
        """
        self.max_duration = max_duration
        self.num_buckets = num_buckets
        self.min_length = min_length
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.length_key = length_key
        # max number of samples per bucket, calculated based on max_duration and min_length
        self.buffer_per_bucket = math.ceil(max_duration / max(1, min_length) * 1.5)
        self.buckets = collections.defaultdict(collections.deque)
        self.bucket_item_lengths = [
            math.ceil((max_length - min_length) / num_buckets) * (i + 1)
            for i in range(num_buckets)
        ]
        self.stop_event = stop_event
        self.data_stream = None
        self.read_thread = None
        self.bucket_locks = None

    def bucket_id(self, length):
        length = max(self.min_length, min(length, self.max_length))
        return int(
            (length - self.min_length)
            / (self.max_length - self.min_length)
            * (self.num_buckets - 1)
        )

    def _fill_buckets_worker(self):
        while True:
            try:
                sample = next(self.data_stream)

                if self.stop_event.is_set():
                    logging.info("Stop event set, stopping bucket filling thread.")
                    break

                length = sample[self.length_key].size(1) / self.sample_rate
                b_id = self.bucket_id(length)

                bucket_full = False
                with self.bucket_locks:
                    self.buckets[b_id].append(sample)
                    bucket_full = len(self.buckets[b_id]) >= self.buffer_per_bucket
                if bucket_full:
                    time.sleep(0.5)

            except StopIteration:
                logging.warning(
                    "Data stream exhausted, stopping bucket filling thread."
                )
                break
            except Exception as e:
                logging.warning(f"Error in bucket filling thread: {e}, will continue")
                continue

    def __call__(self, data_stream):
        self.data_stream = iter(data_stream)
        if self.bucket_locks is None:
            self.bucket_locks = threading.Lock()
        if self.stop_event is None:
            self.stop_event = threading.Event()
        else:
            self.stop_event.clear()

        if self.read_thread is None or not self.read_thread.is_alive():
            self.read_thread = threading.Thread(
                target=self._fill_buckets_worker, daemon=True
            )
            self.read_thread.start()

        while True:
            if (
                not self.read_thread.is_alive()
                and all(len(q) == 0 for q in self.buckets.values())
            ) or self.stop_event.is_set():
                break

            if self.read_thread.is_alive():
                with self.bucket_locks:
                    bucket_duration = [
                        self.bucket_item_lengths[i] * len(self.buckets[i])
                        for i in range(self.num_buckets)
                    ]
                bucket_range = [
                    i
                    for i in range(self.num_buckets)
                    if bucket_duration[i] > self.max_duration * 2
                ]
                if not bucket_range:
                    bucket_range = [
                        i
                        for i in range(self.num_buckets)
                        if bucket_duration[i] > self.max_duration
                    ]
                    if bucket_range:
                        bucket_range = bucket_range[
                            0 : random.randint(1, len(bucket_range))
                        ]
                if not bucket_range:
                    time.sleep(0.5)
                    continue
            else:
                with self.bucket_locks:
                    bucket_range = [
                        i for i in range(self.num_buckets) if len(self.buckets[i]) > 0
                    ]

            bucket_range.reverse()

            batch = []
            batch_duration = 0
            last_b_id = bucket_range[0]
            return_one_element = False

            for b_id in bucket_range:
                while self.buckets[b_id]:
                    with self.bucket_locks:
                        sample = self.buckets[b_id][0]
                    length = sample[self.length_key].size(1) / self.sample_rate
                    if batch_duration + length > self.max_duration:
                        if not batch:
                            last_b_id = b_id
                            return_one_element = True
                            batch_duration += length
                        break
                    with self.bucket_locks:
                        batch.append(self.buckets[b_id].popleft())
                    batch_duration += length
                if batch_duration >= self.max_duration:
                    break

            if not batch:
                will_continue = False
                with self.bucket_locks:
                    if return_one_element and self.buckets[last_b_id]:
                        batch.append(self.buckets[last_b_id].popleft())
                    else:
                        will_continue = True
                if will_continue:
                    time.sleep(0.5)
                    continue
            yield batch

        self.stop_event.set()
        if self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)

    def __del__(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.read_thread is not None and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)


class StreamingWebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        audio_tars: List[str],
        labels_to_audios: Dict,
        config: Dict,
        sample_rate: int = 16000,
        max_duration: float = 1000.0,
        feature_extractor: Optional[Callable] = None,
        noise_tars: Optional[List[str]] = None,
        buffer_size: int = 200,
        is_train: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Streaming webdataset for ASR training with dynamic bucketing batching.
        Args:
          audio_tars:
            List of audio tar files.
          labels_to_audios:
            A dict mapping from audio tar file to label tar file.
          config:
            A dict containing configuration for data loading and augmentation.
          sample_rate:
            Target sample rate for audio.
          max_duration:
            Maximum duration (in seconds) for each batch.
          feature_extractor:
            Feature extractor to extract features from raw audio.
          noise_tars:
            List of noise tar files for noise augmentation.
          buffer_size:
            Buffer size for shuffling.
          is_train:
            Whether the dataset is for training or not.
        """
        super().__init__()

        self.device = device
        self.is_train = is_train
        self.buffer_size = buffer_size
        self.audio_tars = list(audio_tars)

        # for stopping the background io threads (audios and noise)
        self.stop_event = None

        # noise dataset for noise augmentation
        self.noise_ds = None
        if noise_tars is not None and config.get("noise_augment", False) and is_train:
            self.noise_ds = create_simple_audio_dataset(
                noise_tars,
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                device=torch.device("cpu"),
                workersplitter=wds.split_by_worker,
            )

        # sample_decoder is to decode audio and assign label
        self.sample_decoder = SampleDecoder(
            labels_to_audios=labels_to_audios,
            config=config,
            sample_rate=sample_rate,
            is_train=is_train,
            device=torch.device("cpu"),
        )

        self.num_copies = 1
        self.noise_prob = 0.0
        self.lower_snr_db = 10.0
        self.upper_snr_db = 20.0
        if (
            is_train
            and config.get("noise_augment", False)
            and self.noise_ds is not None
        ):
            self.num_copies = 3
            self.noise_prob, self.lower_snr_db, self.upper_snr_db = config[
                "noise_augment"
            ]

        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_extractor = feature_extractor

    def __iter__(self):
        if self.stop_event is None:
            self.stop_event = threading.Event()

        # read raw audio data and label
        dataset = (
            wds.WebDataset(
                self.audio_tars,
                shardshuffle=len(self.audio_tars) if self.is_train else False,
                nodesplitter=wds.split_by_node,
            )
            .decode()
            .map(self.sample_decoder)
            .shuffle(self.buffer_size if self.is_train else 1)
        )

        noise_sampler = NoiseSampler(
            self.noise_ds, stop_event=self.stop_event, buffer_size=self.buffer_size
        )
        self.add_noise = partial(
            augment_with_noise,
            noise_sampler=noise_sampler,
            lower_snr_db=self.lower_snr_db,
            upper_snr_db=self.upper_snr_db,
            is_train=self.is_train,
        )

        # for dynamic batching based on audio duration with max_duration
        self.batcher = StreamingBucketBatcher(
            max_duration=self.max_duration,
            sample_rate=self.sample_rate,
            stop_event=self.stop_event,
        )

        # apply bucketing batcher
        dataset = self.batcher(dataset)

        self.stream = iter(dataset)

        while True:
            try:
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

                start = time.time()
                for sample in raw_batch:
                    # sample["audio"]: (1, num_samples)
                    audio = sample["audio"].squeeze(0)
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
                # logging.warning(f"Batch processing time: {time.time()-start} seconds")
                start = time.time()

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

                # logging.warning(f"Padding time: {time.time()-start} seconds")
                start = time.time()

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

                # logging.warning(f"Feature extraction time: {time.time()-start} seconds")

                # move tensors to cpu before yielding to avoid IPC issue
                batch_output = {
                    "inputs": audios.to("cpu"),
                    "num_copies": self.num_copies,
                    "supervisions": {
                        "sequence_idx": torch.tensor(
                            list(range(len(labels))), device="cpu"
                        ),
                        "text": labels,
                        "start_frame": torch.zeros(
                            len(labels), dtype=torch.long, device="cpu"
                        ),
                        "num_frames": frame_lens.to("cpu"),
                        "start_sample": torch.zeros(
                            len(labels), dtype=torch.long, device="cpu"
                        ),
                        "num_samples": audio_lens.to("cpu"),
                        "keys": keys,
                    },
                }
                if features is not None:
                    batch_output["inputs"] = features.to("cpu")
                yield batch_output

            except StopIteration:
                self.stop_event.set()
                break
            except RuntimeError as e:
                logging.warning(f"Runtime error: {e}, continue to next batch.")
                continue
            except KeyboardInterrupt:
                self.stop_event.set()
                break


def create_dataloader(
    audio_tars: List[str],
    labels_to_audios: Dict,
    config: Dict,
    sample_rate: int = 16000,
    max_duration: float = 600.0,
    feature_extractor: Optional[Callable] = None,
    noise_tars: Optional[List[str]] = None,
    buffer_size: int = 200,
    num_workers: int = 4,
    is_train: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """
    Create a dataloader for streaming webdataset.
    Args:
      audio_tars:
        List of audio tar files.
      labels_to_audios:
        A dict mapping from audio tar file to label tar file.
      config:
        A dict containing configuration for data loading and augmentation.
      sample_rate:
        Target sample rate for audio.
      max_duration:
        Maximum duration (in seconds) for each batch.
      feature_extractor:
        Feature extractor to extract features from raw audio.
      noise_tars:
        List of noise tar files for noise augmentation.
      buffer_size:
        Buffer size for shuffling.
      num_workers:
        Number of workers for dataloader.
      is_train:
        Whether the dataloader is for training or not.
    """
    dataset = StreamingWebDataset(
        audio_tars=audio_tars,
        labels_to_audios=labels_to_audios,
        config=config,
        sample_rate=sample_rate,
        max_duration=max_duration,
        feature_extractor=feature_extractor,
        noise_tars=noise_tars,
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
    audio_tars = glob.glob("data/tars/audios/librispeech_train-clean-360.*.tar")
    annot_tars = glob.glob("data/tars/txts/librispeech_train-clean-360.*.tar")
    noise_tars = glob.glob("data/musan/audios/musan.*.tar")

    audio_tars.sort()
    annot_tars.sort()

    labels = {}
    for x, y in zip(audio_tars, annot_tars):
        labels[x] = y

    config = {
        "noise_augment": (0.5, 10, 20),  # lower_snr_db, upper_snr_db
        "speed_perturb": (0.9, 1.0, 1.1),  # speeds
        "volume_perturb": (0.5, -10, 6),  # prob, lower_db, upper_db
        "sample_rate": 16000,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = FbankExtractor(
        sample_rate=config.get("sample_rate", 16000),
        device=device,
    )

    dataset = create_dataloader(
        audio_tars=audio_tars,
        labels_to_audios=labels,
        max_duration=100.0,
        feature_extractor=feature_extractor,
        sample_rate=config.get("sample_rate", 16000),
        config=config,
        noise_tars=noise_tars,
        buffer_size=200,
        is_train=True,
        device=device,
        num_workers=4,
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
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # The context might already be set.

    main()
