# Data Format

## Overview

ATdataset uses a three-level data organization:

```
.lst file  →  tar shards + JSONL manifests  →  audio samples
```

## LST File (Dataset List)

The `.lst` file is what you pass to `ATDataloader`. Each line describes one tar shard:

```
/abs/path/to/shard.tar /abs/path/to/shard.jsonl duration_hours num_samples
```

| Column | Type | Description |
|--------|------|-------------|
| 1 | path | Absolute path to audio tar shard |
| 2 | path | Absolute path to JSONL transcript manifest |
| 3 | float | Total audio duration in hours (optional but recommended) |
| 4 | int | Number of samples (optional, needed for `batch_size` mode) |

Example:
```
/data/tars/audios/train.000000.tar /data/tars/manifests/train.000000.jsonl 1.234 1500
/data/tars/audios/train.000001.tar /data/tars/manifests/train.000001.jsonl 1.198 1480
```

If columns 3-4 are missing, ATdataset will scan the JSONL manifests to compute them (slow for large datasets).

Generate with:
```bash
atdataset gen_lst --audio-pattern 'data/tars/audios/*.tar' \
    --txt-dir data/tars/manifests --output train.lst
```

Run `atdataset gen_lst --help` for full option details.

## Tar Shards

Each tar shard is a standard tar archive containing audio files:

```
train.000000.tar
├── utt_00001.flac
├── utt_00002.flac
├── utt_00003.flac
└── ...
```

Supported audio formats: FLAC, WAV, MP3.

The filename (without extension) is the utterance ID, used to look up the transcript in the JSONL manifest.

## JSONL Manifest

Each JSONL manifest corresponds to one tar shard. One JSON object per line:

```jsonl
{"audio_filepath": "utt_00001.flac", "text": "hello world", "duration": 2.5}
{"audio_filepath": "utt_00002.flac", "text": "good morning", "duration": 1.8}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_filepath` | string | yes | Filename matching the tar entry |
| `text` | string | yes | Transcription text |
| `duration` | float | recommended | Audio duration in seconds |

The `audio_filepath` field (without extension) is used as the lookup key for matching audio in the tar.

## Input TSV Manifest (for `atdataset build`)

The `atdataset build` command accepts a tab-separated input with either 3 or 5 fields:

### 3-field format (whole file)
```
id\taudio_path\ttext
```

### 5-field format (segment)
```
id\taudio_path\ttext\tstart\tduration
```

| Field | Description |
|-------|-------------|
| `id` | Unique utterance identifier |
| `audio_path` | Path to source audio file |
| `text` | Transcription text |
| `start` | Segment start time in seconds (5-field only) |
| `duration` | Segment duration in seconds (5-field only) |

Example:
```
utt001	/corpus/audio/001.wav	hello world
utt002	/corpus/audio/002.wav	good morning	1.5	3.2
```

## Building a Dataset

Full pipeline from raw audio to training-ready format:

```bash
# 1. Prepare a TSV manifest (tab-separated: id, audio_path, text)
# 2. Build tar shards
atdataset build \
    --input train.tsv \
    --output-dir data/tars \
    --num-tars 64 \
    --format FLAC \
    --sample-rate 16000

# 3. Output structure:
#    data/tars/
#    ├── audios/train.000000.tar ... train.000063.tar
#    ├── manifests/train.000000.jsonl ... train.000063.jsonl
#    └── data.lst   (auto-generated)
```

The generated `data.lst` is ready to use directly with `ATDataloader`.

Run `atdataset build --help` for full option details.

## Distributed Build

For very large datasets, you can split the tar building across multiple machines:

```bash
# Step 1: Create the split plan (no tar building)
atdataset build --input train.tsv --output-dir data/tars --num-tars 64 --plan-only

# Step 2: Copy the split files to all machines, then build in parallel (It's better that all the machines share the same HOME)
# Machine 1:
atdataset build --input train.tsv --output-dir data/tars --split-start 0 --split-end 32
# Machine 2:
atdataset build --input train.tsv --output-dir data/tars --split-start 32 --split-end 64

# Step 3: Generate the list file after all machines finish
atdataset gen_lst --audio-pattern 'data/tars/audios/*.tar' \
    --txt-dir data/tars/manifests --output data/tars/data.lst
```

All machines must use the same `--input`, `--output-dir`, `--num-tars` (or other split parameters) so the split plan is consistent.
