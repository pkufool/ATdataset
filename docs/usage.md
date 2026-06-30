# Usage Guide

## Basic Usage (Single Dataset)

```python
from atdataset import ATDataloader

dl = ATDataloader(
    datasets="data/tars/train.lst",
    sample_rate=16000,
    max_duration=600.0,
    num_workers=4,
)

for batch in dl:
    # batch["audio"]: (B, T) padded waveform
    # batch["audio_lens"]: (B,) actual lengths in samples
    # batch["feature"]: (B, T, F) padded log-mel features
    # batch["feature_lens"]: (B,) actual frame counts
    # batch["text"]: list of transcription strings
    # batch["ids"]: list of utterance IDs
    train_step(batch)
```

## Multi-Dataset Muxing

The `datasets` parameter accepts a flexible mix of inputs — `.lst` file paths, `ATDataset` instances, or both in the same list. String paths are automatically converted to `ATDataset` with the shared arguments from `ATDataloader`. This lets you customize some datasets while keeping others simple:

```python
# Mixed input: string path (uses shared defaults) + ATDataset (custom settings)
dl = ATDataloader(
    datasets=["simple_train.lst", custom_dataset_instance],
    sample_rate=16000,
    max_duration=600.0,
)
```

Each dataset can have its own `filter_func`, `map_func`, and augmentation settings when passed as `ATDataset` objects:

```python
from atdataset import ATDataset, ATDataloader

# Dataset A: short utterances, no augmentation
ds_a = ATDataset(
    manifest="data/tars/short.lst",
    sample_rate=16000,
    min_length=0.5,
    max_length=10.0,
    filter_func=lambda s: len(s["text"]) > 0,
)

# Dataset B: long utterances, with speed perturbation
ds_b = ATDataset(
    manifest="data/tars/long.lst",
    sample_rate=16000,
    min_length=5.0,
    max_length=30.0,
    use_speed_perturb=True,
    speed_perturb=(0.9, 1.0, 1.1),
    map_func=lambda s: {**s, "text": s["text"].upper()},
)

dl = ATDataloader(
    datasets=[ds_a, ds_b],
    sample_rate=16000,
    max_duration=600.0,
    mux_weights=[0.3, 0.7],  # 30% short, 70% long
    mux_intra_batch=True,
)
```

### mux_weights

- If not specified, weights are proportional to dataset durations
- Normalized internally (e.g. `[1, 3]` becomes `[0.25, 0.75]`)
- Use `mux_intra_batch=False` when datasets have very different length distributions

```python
# Per-batch muxing: each batch from one dataset only
dl = ATDataloader(
    datasets=[ds_a, ds_b],
    sample_rate=16000,
    max_duration=600.0,
    mux_intra_batch=False,
)
```

## Augmentation

All augmentation is applied only during training (`is_test=False`).

```python
dl = ATDataloader(
    datasets="train.lst",
    sample_rate=16000,
    max_duration=600.0,
    # Speed perturbation
    use_speed_perturb=True,
    speed_perturb=(0.9, 1.0, 1.1),
    # Volume perturbation
    use_volume_perturb=True,
    volume_perturb=(0.5, -10, 6),  # (prob, lower_db, upper_db)
    # Noise augmentation
    use_noise_augment=True,
    noise_manifest="noise.lst",
    noise_augment=(0.5, 10, 20),  # (prob, lower_snr_db, upper_snr_db)
)
```

## Feature Extractor

By default, `ATDataloader` uses `feature_type="Fbank"` (torchaudio MelSpectrogram, 80-dim, hop_length=160). You can change or disable it:

Three built-in extractors are available via `feature_type`:

```python
# Default: torchaudio MelSpectrogram
dl = ATDataloader(..., feature_type="Fbank")

# Kaldi-compatible fbank
dl = ATDataloader(..., feature_type="KaldiFbank")

# Whisper-style log-mel
dl = ATDataloader(..., feature_type="WhisperFbank")

# No feature extraction (raw audio only)
dl = ATDataloader(..., feature_type=None)
```

Or provide a custom extractor:

```python
from atdataset import Fbank

extractor = Fbank(sample_rate=16000, n_mels=128, hop_length=160)
dl = ATDataloader(..., feature_extractor=extractor, feature_type=None)
```

## batch_size vs max_duration

Two batching strategies are available:

### max_duration mode (default)

Batches are formed to fit within a total duration budget. Batch size varies per batch depending on sample lengths. Better GPU utilization for variable-length audio.

```python
dl = ATDataloader(
    ...,
    max_duration=600.0,   # ~600 seconds per batch
    max_samples=100,      # also cap at 100 samples max
)
```

### batch_size mode

Fixed number of samples per batch, regardless of duration. Simpler but may waste GPU memory on short batches or OOM on long batches.

```python
dl = ATDataloader(
    ...,
    batch_size=32,  # exactly 32 samples per batch
)
```

## fill_factor Tuning

The `fill_factor` parameter controls epoch length estimation accuracy. Due to bucketing, batches are typically not fully packed to `max_duration`.

Default is `1.15` (assumes ~87% average fill). To measure your actual fill factor:

```bash
python examples/example.py \
    --datasets data/tars/train.lst \
    --sample-rate 16000 \
    --max-duration 600.0
```

The script reports:
```
Fill factor estimation: 1.35 (avg_batch_duration=444.2s, max_duration=600.0s)
```

Then set `fill_factor=1.35` in your training config for accurate epoch lengths.

## Test / Evaluation Mode

```python
dl = ATDataloader(
    datasets="test.lst",
    sample_rate=16000,
    batch_size=1,
    is_test=True,   # no shuffle, no augmentation, no looping
)

for batch in dl:
    # processes all samples exactly once, then stops
    decode(batch)
```

## Reproducibility

```python
dl = ATDataloader(
    ...,
    seed=42,  # deterministic across runs
)
```

Each worker is seeded with `seed + worker_id + epoch * 10000`, ensuring:
- Different augmentation per worker
- Different augmentation per epoch
- Same result given same (seed, epoch, worker)
