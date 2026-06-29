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


"""
Example script demonstrating how to use ATDataloader.

Usage:
    # Basic: batch_size mode with 2 datasets
    python example.py \
        --datasets data/tars/aishell_train.lst data/tars/aishell2_train.lst \
        --sample-rate 16000 \
        --batch-size 32

    # max_duration mode with noise augmentation
    python example.py \
        --datasets data/tars/aishell_train.lst data/tars/aishell2_train.lst \
        --sample-rate 16000 \
        --max-duration 100.0 \
        --max-samples 100 \
        --use-noise-augment \
        --noise-manifest data/tars/musan.lst \
        --use-speed-perturb \
        --use-volume-perturb \
        --num-copies 2

    # Use KaldiFbank feature extractor
    python example.py \
        --datasets data/tars/aishell_train.lst \
        --sample-rate 16000 \
        --batch-size 64 \
        --feature-type KaldiFbank

    # No feature extraction
    python example.py \
        --datasets data/tars/aishell_train.lst \
        --sample-rate 16000 \
        --batch-size 64 \
        --feature-type none

    # Test mode
    python example.py \
        --datasets data/tars/aishell_test.lst \
        --sample-rate 16000 \
        --batch-size 1 \
        --is-test
"""

import argparse
import logging
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from atdataset import ATDataloader


def filter_func(sample):
    """Filter out samples shorter than 5 seconds."""
    if sample["audio"].size(1) < 16000 * 5:
        return False
    return True


def worker_init_fn(worker_id):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="ATDataloader example script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset ----
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Manifest file paths for the datasets.",
    )

    # ---- Audio ----
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Target sample rate for audio.",
    )

    # ---- Batching ----
    parser.add_argument(
        "--max-duration", type=float, default=600.0,
        help="Maximum duration (in seconds) for each batch.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples for each batch.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Fixed batch size. When specified, max_duration and max_samples are ignored.",
    )
    parser.add_argument(
        "--epoch-hours", type=float, default=None,
        help="Number of hours per epoch. If None, calculated from manifest durations.",
    )
    parser.add_argument(
        "--mux-weights", nargs="+", type=float, default=None,
        help="Weights for each dataset for muxing. If None, calculated from durations.",
    )
    parser.add_argument(
        "--mux-intra-batch", action="store_true", default=True,
        help="Mix samples from different datasets within the same batch.",
    )
    parser.add_argument(
        "--no-mux-intra-batch", dest="mux_intra_batch", action="store_false",
        help="Disable intra-batch muxing (mux per batch instead).",
    )
    parser.add_argument(
        "--num-buckets", type=int, default=30,
        help="Number of buckets for bucketing variable length audio samples.",
    )

    # ---- Sample length filter ----
    parser.add_argument(
        "--min-length", type=float, default=0.1,
        help="Minimum length (in seconds) of samples to consider.",
    )
    parser.add_argument(
        "--max-length", type=float, default=60.0,
        help="Maximum length (in seconds) of samples to consider.",
    )

    # ---- Feature ----
    parser.add_argument(
        "--feature-type", type=str, default=None,
        choices=["Fbank", "KaldiFbank", "WhisperFbank", "none"],
        help="Type of feature to extract. Use 'none' to disable feature extraction.",
    )

    # ---- Augmentation ----
    parser.add_argument(
        "--use-noise-augment", action="store_true", default=False,
        help="Enable noise augmentation.",
    )
    parser.add_argument(
        "--noise-manifest", type=str, default=None,
        help="Manifest file containing noise audio tars.",
    )
    parser.add_argument(
        "--noise-augment", nargs=3, type=float, default=[0.5, 10, 20],
        metavar=("PROB", "LOWER_DB", "UPPER_DB"),
        help="Noise augmentation params: probability, lower_snr_db, upper_snr_db.",
    )
    parser.add_argument(
        "--use-speed-perturb", action="store_true", default=False,
        help="Enable speed perturbation.",
    )
    parser.add_argument(
        "--speed-perturb", nargs="+", type=float, default=[0.9, 1.0, 1.1],
        help="Speeds for speed perturbation.",
    )
    parser.add_argument(
        "--use-volume-perturb", action="store_true", default=False,
        help="Enable volume perturbation.",
    )
    parser.add_argument(
        "--volume-perturb", nargs=3, type=float, default=[0.5, -10, 6],
        metavar=("PROB", "LOWER_DB", "UPPER_DB"),
        help="Volume perturbation params: probability, lower_db, upper_db.",
    )

    # ---- Data loading ----
    parser.add_argument(
        "--num-copies", type=int, default=1,
        help="Number of copies of samples with different augmentations.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=1000,
        help="Buffer size for shuffling.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Prefetch factor for dataloader.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )

    # ---- Misc ----
    parser.add_argument(
        "--is-test", action="store_true", default=False,
        help="Run in test mode (no shuffling, no augmentation).",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Resolve feature_type: "none" → None
    feature_type = args.feature_type if args.feature_type != "none" else None

    dl = ATDataloader(
        datasets=args.datasets,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epoch_hours=args.epoch_hours,
        mux_weights=args.mux_weights,
        mux_intra_batch=args.mux_intra_batch,
        min_length=args.min_length,
        max_length=args.max_length,
        feature_type=feature_type,
        use_noise_augment=args.use_noise_augment,
        noise_manifest=args.noise_manifest,
        noise_augment=tuple(args.noise_augment),
        use_speed_perturb=args.use_speed_perturb,
        speed_perturb=tuple(args.speed_perturb),
        use_volume_perturb=args.use_volume_perturb,
        volume_perturb=tuple(args.volume_perturb),
        num_copies=args.num_copies,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        num_buckets=args.num_buckets,
        is_test=args.is_test,
        filter_func=filter_func,
    )

    logging.info(f"Dataloader initialized: {dl}.")

    start = time.time()
    total_batch_duration = 0.0
    num_batches = 0
    for i, batch in enumerate(tqdm(dl, total=len(dl))):
        logging.info(f"Batch {i}: ids={batch['ids']}")
        # Measure actual batch duration for fill_factor estimation
        if "audio" in batch and "audio_lens" in batch:
            batch_dur = batch["audio_lens"].sum().item() / args.sample_rate
            total_batch_duration += batch_dur
            num_batches += 1
    elapsed = time.time() - start
    logging.info(f"Finished {i + 1} batches in {elapsed:.2f}s.")

    # Report fill_factor: ratio of actual batch duration to max_duration.
    # Use this value as the fill_factor when setting epoch_hours to get
    # more accurate epoch length estimation:
    #   epoch_batches ≈ epoch_hours * 3600 / max_duration / fill_factor
    if num_batches > 0 and args.max_duration and args.batch_size is None:
        avg_batch_duration = total_batch_duration / num_batches
        fill_factor = avg_batch_duration / args.max_duration
        logging.info(
            f"Fill factor estimation: {fill_factor:.3f} "
            f"(avg_batch_duration={avg_batch_duration:.1f}s, "
            f"max_duration={args.max_duration:.1f}s). "
            f"Recommended: set epoch_hours = actual_hours / {fill_factor:.3f} "
            f"for more accurate epoch length."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # The context might already be set.
    main()
