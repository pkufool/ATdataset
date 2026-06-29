# ATdataset

A streaming audio-text dataloader for PyTorch training, built on [WebDataset](https://github.com/webdataset/webdataset).

Designed for large-scale speech recognition training with multi-dataset muxing, dynamic batching, bucketing, and on-the-fly augmentation.

## Install

```bash
pip install atdataset
```

For development:

```bash
git clone https://github.com/pkufool/ATdataset.git
cd ATdataset
pip install -e ".[dev]"
```

## Quick Start

```python
from atdataset import ATDataloader

dl = ATDataloader(
    datasets="data/tars/train.lst",
    sample_rate=16000,
    max_duration=600.0,
)

for batch in dl:
    audio = batch["audio"]          # (B, T), padded
    audio_lens = batch["audio_lens"]  # (B,)
    feature = batch["feature"]      # (B, T, F), padded
    feature_lens = batch["feature_lens"]  # (B,)
    texts = batch["text"]           # list of str
    break
```

## CLI Tools

After installation, the `atdataset` command provides two subcommands:

```bash
# Build tar shards from a TSV manifest
atdataset build --input train.tsv --output-dir data/tars --num-tars 64

# Generate a .lst file from existing tar shards
atdataset gen_lst --audio-pattern 'data/tars/audios/*.tar' \
    --txt-dir data/tars/manifests --output data/tars/train.lst
```

Run `atdataset build --help` or `atdataset gen_lst --help` for full option details.

## Documentation

- [Design & Architecture](docs/design.md) — pipeline architecture, bucketing, muxing, epoch control
- [Usage Guide](docs/usage.md) — detailed examples for all features
- [Data Format](docs/data_format.md) — manifest, tar shard, and lst file formats

## License

Apache-2.0
