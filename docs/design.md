# Design & Architecture

## Pipeline Overview

```
ATDataloader (WebLoader, controls num_workers/prefetch)
 └── BatchedDataset (IterableDataset, epoch control + collation)
       └── StreamingBucketBatcher (bucketing + batch assembly)
             └── ATDataset.__iter__() × N streams
                   └── WebDataset → decode → SampleDecoder → shuffle
                         → length filter → map/filter → augment → feature extract
```

## Data Flow

1. **Shard loading**: WebDataset reads tar shards, splits across nodes and workers
2. **Decoding**: `SampleDecoder` loads audio bytes → tensor, attaches text label from JSONL manifest
3. **Shuffle**: Buffer-based sample shuffling within each worker
4. **Filtering**: Length filter + user-provided `filter_func` / `map_func`
5. **Augmentation**: Speed perturb → volume perturb → noise mixing (training only)
6. **Feature extraction**: Fbank / KaldiFbank / WhisperFbank (CPU, per-sample)
7. **Bucketing**: Samples grouped by duration into fixed-width buckets
8. **Batching**: Bucket contents assembled into batches respecting `max_duration`
9. **Collation**: Pad sequences, produce `{audio, audio_lens, feature, feature_lens, text, ids}`

## Dynamic Batching with max_duration

The core batching strategy is **duration-based dynamic batching** (`max_duration`). Instead of a fixed batch size, each batch is filled until the total duration budget is reached:

```
batch_duration = max_sample_length × num_samples ≤ max_duration
```

**Why this matters for speech training:**

- **Maximizes GPU utilization**: Short utterances pack more samples per batch; long utterances pack fewer — GPU memory usage stays constant regardless of utterance length distribution.
- **No OOM surprises**: Fixed `batch_size=32` can OOM when all 32 samples happen to be 30s. `max_duration` guarantees bounded memory.
- **Better gradient quality**: Duration-based batching means each gradient step sees roughly the same amount of speech data, reducing variance across steps.
- **Works with bucketing**: Combined with same-length bucketing, padding waste is minimal (<5% typically), so the "rectangle" budget closely matches real computation.

For evaluation or simple cases, a fixed `batch_size` mode is also available, but `max_duration` is the recommended default for training.

## Bucketing Strategy

Samples are assigned to one of `num_buckets` (default 30) fixed-width buckets based on duration. Each bucket covers a uniform duration range between `min_length` and `max_length`.

A bucket is considered "full" when its estimated total duration exceeds `max_duration * 1.5`. At that point, a batch is assembled from that single bucket — ensuring all samples in a batch have similar lengths, minimizing padding waste.

```
bucket_id = int((length - min_length) / (max_length - min_length) * (num_buckets - 1))
```

## Muxing Strategies

When training with multiple datasets (e.g. LibriSpeech + AISHELL), two muxing modes are available:

### Intra-batch muxing (`mux_intra_batch=True`, default)

Samples from different datasets are mixed **within** the same batch. All datasets share a single set of buckets. This produces more diverse batches.

```
Batch: [libri_001, aishell_042, libri_003, aishell_015, ...]
```

### Per-batch muxing (`mux_intra_batch=False`)

Each batch contains samples from only **one** dataset. The batcher selects which dataset to draw from based on `mux_weights`. Each dataset has its own set of buckets with independent `min_length` / `max_length`.

```
Batch 1: [libri_001, libri_003, libri_007, ...]
Batch 2: [aishell_042, aishell_015, aishell_088, ...]
```

### mux_weights

Controls the sampling probability for each dataset stream. If not specified, weights are calculated proportionally from dataset durations (or sample counts when `batch_size` mode is used).

```python
# 70% from dataset A, 30% from dataset B
ATDataloader(datasets=["a.lst", "b.lst"], mux_weights=[0.7, 0.3], ...)
```

## num_copies

When `num_copies > 1`, each raw sample produces multiple augmented copies in the final batch. This is useful for contrastive learning or multi-view training.

Implementation:
- The batching budget (`max_duration` / `batch_size`) is divided by `num_copies` internally
- In `ATDataset.__iter__`, each sample yields a list of `num_copies` independently augmented copies
- In `BatchedDataset.__iter__`, the copies are unrolled back into the final batch

Example with `num_copies=2, max_duration=600`:
- Batcher targets 300s per raw batch
- Each raw sample produces 2 augmented versions
- Final batch ≈ 600s total (300s × 2 copies)

## Epoch Control

An "epoch" is defined by `epoch_batches_per_node`:

```
epoch_batches = ceil(epoch_hours * 3600 / world_size / max_duration * fill_factor)
```

- `epoch_hours`: Total audio hours per epoch (default: sum of all dataset durations)
- `fill_factor`: Correction for imperfect bucket packing (default: 1.2)
- Data streams loop infinitely; epoch ends when `batch_count >= epoch_batches`

In `batch_size` mode:
```
epoch_batches = ceil(total_samples / batch_size / world_size)
```

## Worker Seeding

When `seed` is provided:
- Each worker gets `seed + worker_id + epoch * 10000`
- Ensures different augmentation across workers and epochs
- Deterministic within a given (seed, epoch, worker) triple
